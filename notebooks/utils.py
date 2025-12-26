from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from catboost import CatBoostRegressor
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def plot_time_series(df, columns, step=10, rolling_window=None):
    """
    Clean and fast Plotly plot:
    - Subplots stacked vertically
    - Optional rolling median trend (black)
    - Only shows every `step`th tick
    
    Parameters
    ----------
    df : pd.DataFrame
        Time series dataframe with a DateTimeIndex.
    columns : list of str
        List of column names to plot.
    step : int
        Subsampling step for faster plotting.
    rolling_window : int or None
        Window size for rolling median.
        If None → no rolling median plotted.
    """

    # Subsample for speed
    df_small = df.iloc[::step]

    # Create subplot layout
    fig = make_subplots(
        rows=len(columns),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        subplot_titles=columns
    )

    # Precompute rolling only if requested
    if rolling_window is not None:
        df_roll = (
            df[columns]
            .rolling(rolling_window, min_periods=1)
            .median()
            .iloc[::step]
        )

    for i, col in enumerate(columns, start=1):

        # Original series
        fig.add_trace(
            go.Scatter(
                x=df_small.index,
                y=df_small[col],
                mode="lines",
                name=col,
                line=dict(width=1)
            ),
            row=i, col=1
        )

        # Rolling trend only if rolling_window is given
        if rolling_window is not None:
            fig.add_trace(
                go.Scatter(
                    x=df_roll.index,
                    y=df_roll[col],
                    mode="lines",
                    name=f"{col} (rolling median)",
                    line=dict(width=1, color="black"),
                ),
                row=i, col=1
            )

    fig.update_xaxes(tickmode="auto")

    fig.update_layout(
        height=250 * len(columns),
        showlegend=False,
        title_text="Time Series Overview",
        margin=dict(l=50, r=30, t=50, b=50)
    )

    fig.show()


def make_sequences(
    x_np: np.ndarray,
    y_np: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for time series data.

    Creates sequences where each input sequence contains `seq_len` consecutive
    time steps, and the corresponding target is the value at the next time step.

    Parameters
    ----------
    x_np : np.ndarray
        Input features array of shape (n_samples, n_features).
    y_np : np.ndarray
        Target values array of shape (n_samples,).
    seq_len : int
        Length of each input sequence (number of time steps to look back).

    Returns
    -------
    X_seq : np.ndarray
        Sequence inputs of shape (n_samples - seq_len, seq_len, n_features).
    y_seq : np.ndarray
        Sequence targets of shape (n_samples - seq_len,).
    """
    X_seq, y_seq = [], []
    for i in range(len(x_np) - seq_len):
        X_seq.append(x_np[i:i + seq_len])
        y_seq.append(y_np[i + seq_len])
    return np.stack(X_seq), np.array(y_seq)


class LSTMRegressor(nn.Module):
    """
    Many-to-one LSTM regressor for time series prediction.

    Architecture
    ------------
    - LSTM layers with configurable hidden size and number of layers
    - Fully connected head: hidden_size -> hidden_size//2 -> 1
    - Uses last hidden state from final LSTM layer for prediction
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize LSTM regressor.

        Parameters
        ----------
        n_features : int
            Number of input features per time step.
        hidden_size : int, default=128
            Hidden size of LSTM layers.
        num_layers : int, default=2
            Number of stacked LSTM layers.
        dropout : float, default=0.0
            Dropout rate applied between LSTM layers (only if num_layers > 1).
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # a slightly richer head than just one Linear
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM regressor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, n_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1).
        """
        out, (h_n, c_n) = self.lstm(x)      # h_n: (num_layers, B, H)
        last_hidden = h_n[-1]               # (B, H)
        return self.head(last_hidden)       # (B, 1)


class MLPRegressor(nn.Module):
    def __init__(self, n_features, hidden_sizes=[128, 64], dropout=0.1):
        """
        Parameters
        ----------
        n_features : int
            Number of input features.
        hidden_sizes : list[int]
            Example: [256, 128, 64] => 3 hidden layers.
        dropout : float
            Dropout applied AFTER each hidden layer. Set 0 for no dropout.
        """
        super().__init__()

        layers = []
        in_dim = n_features

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_dim = h

        # final output layer
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_lstm(
    x_train_df: pd.DataFrame,
    y_train_ser: pd.Series,
    x_val_df: pd.DataFrame,
    y_val_ser: pd.Series,
    model_params: dict[str, Any] | None = None,
    SEED: int = 42,
) -> tuple[np.ndarray, np.ndarray, nn.Module, StandardScaler]:
    """
    Fit a many-to-one LSTM on (x_train, y_train) and predict on x_val.

    Steps
    -----
    1. Standardize X with StandardScaler
    2. Standardize y with StandardScaler
    3. Create sequences of length seq_len
    4. Build LSTM (hidden layers, dropout) on GPU if available
    5. Train with MSE loss + Adam
    6. Predict on validation data
    7. Invert scaling for predictions

    Parameters
    ----------
    x_train_df : pd.DataFrame
        Training features.
    y_train_ser : pd.Series
        Training target.
    x_val_df : pd.DataFrame
        Validation features.
    y_val_ser : pd.Series
        Validation target.
    model_params : dict, optional
        Dictionary containing hyperparameters:
        - seq_len: int, default=48
            Sequence length for LSTM input
        - hidden_size: int, default=128
            Hidden size of LSTM layers
        - num_layers: int, default=2
            Number of LSTM layers
        - dropout: float, default=0.0
            Dropout rate (only applied if num_layers > 1)
        - lr: float, default=1e-3
            Learning rate for Adam optimizer
        - batch_size: int, default=128
            Batch size for training
        - epochs: int, default=40
            Number of training epochs
        - verbose: bool, default=True
            Whether to print training progress
    SEED : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    y_val_aligned : np.ndarray
        Validation target (original scale, aligned with predictions).
    y_pred_val : np.ndarray
        Validation predictions (original scale).
    model : nn.Module
        Trained LSTM model.
    x_scaler : StandardScaler
        Fitted StandardScaler for X features.
    """

    if model_params is None:
        model_params = {}

    # ---- hyperparams (with sensible defaults) ----
    seq_len = model_params.get("seq_len", 48)
    hidden_size = model_params.get("hidden_size", 128)
    num_layers = model_params.get("num_layers", 2)
    dropout = model_params.get("dropout", 0.0)
    lr = model_params.get("lr", 1e-3)
    batch_size = model_params.get("batch_size", 128)
    epochs = model_params.get("epochs", 40)
    verbose = model_params.get("verbose", True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- scale X ----
    x_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train_df.values.astype(np.float32))
    x_val_scaled = x_scaler.transform(x_val_df.values.astype(np.float32))

    # ---- scale y ----
    y_train = y_train_ser.values.astype(np.float32).reshape(-1, 1)
    y_val = y_val_ser.values.astype(np.float32).reshape(-1, 1)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train).flatten()
    y_val_scaled = y_scaler.transform(y_val).flatten()

    # ---- make sequences (in scaled space) ----
    X_train_seq, y_train_seq = make_sequences(x_train_scaled, y_train_scaled, seq_len)
    X_val_seq, y_val_seq = make_sequences(x_val_scaled, y_val_scaled, seq_len)

    # We'll also keep the ORIGINAL-scale validation y for metrics later:
    _, y_val_orig_seq = make_sequences(x_val_scaled, y_val.flatten(), seq_len)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_seq),
        torch.from_numpy(y_train_seq).view(-1, 1)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_seq),
        torch.from_numpy(y_val_seq).view(-1, 1)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ---- model ----
    n_features = x_train_df.shape[1]
    model = LSTMRegressor(
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- training loop with progress ----
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_ds)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_ds)

        if verbose:
            print(f"Epoch {epoch:03d}/{epochs} | train MSE (scaled): {train_loss:.4f} | "
                  f"val MSE (scaled): {val_loss:.4f}")

    # ---- predict on validation ----
    model.eval()
    preds_list = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy().flatten()
            preds_list.append(preds)

    y_pred_val_scaled = np.concatenate(preds_list).reshape(-1, 1)
    y_pred_val = y_scaler.inverse_transform(y_pred_val_scaled).flatten()

    # original-scale y for metrics (same length as y_pred_val)
    y_val_aligned = y_val_orig_seq

    return y_val_aligned, y_pred_val, model, x_scaler


def fit_mlp(
    x_train_df: pd.DataFrame,
    y_train_ser: pd.Series,
    x_test_df: pd.DataFrame,
    y_test_ser: pd.Series,
    model_params: dict[str, Any] | None = None,
    SEED: int = 42,
) -> tuple[np.ndarray, np.ndarray, nn.Module, StandardScaler]:
    """
    Fit a feed-forward MLP for regression on (X_train, y_train) and
    predict on X_test using PyTorch.

    Steps
    -----
    1. Standardize X with StandardScaler
    2. Standardize y with StandardScaler
    3. Build MLP (hidden layers, dropout) on GPU if available
    4. Train with MSE loss + Adam
    5. Predict on test data
    6. Invert scaling for predictions

    Parameters
    ----------
    x_train_df : pd.DataFrame
        Training features.
    y_train_ser : pd.Series
        Training target.
    x_test_df : pd.DataFrame
        Test features.
    y_test_ser : pd.Series
        Test target.
    model_params : dict, default=None
        - hidden_sizes, dropout, lr, batch_size, epochs, verbose
    SEED : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    y_test_orig : np.ndarray
        Test target (original scale).
    y_pred_test : np.ndarray
        Test predictions (original scale).
    model : nn.Module
        Trained model.
    x_scaler : StandardScaler
        Fitted StandardScaler for X features.
    """

    if model_params is None:
        model_params = {}

    hidden_sizes = model_params.get("hidden_sizes", [128, 64])
    dropout = model_params.get("dropout", 0.1)
    lr = model_params.get("lr", 1e-3)
    batch_size = model_params.get("batch_size", 128)
    epochs = model_params.get("epochs", 40)
    verbose = model_params.get("verbose", True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- scale X ----
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(x_train_df.values.astype(np.float32))
    X_test = x_scaler.transform(x_test_df.values.astype(np.float32))

    # ---- scale y ----
    y_train = y_train_ser.values.astype(np.float32).reshape(-1, 1)
    y_test = y_test_ser.values.astype(np.float32).reshape(-1, 1)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train).flatten()
    y_test_scaled = y_scaler.transform(y_test).flatten()

    # ---- tensors & loaders ----
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train_scaled).view(-1, 1)

    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test_scaled).view(-1, 1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ---- model ----
    n_features = x_train_df.shape[1]
    model = MLPRegressor(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- training ----
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_ds)

        # evaluate on test
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                test_loss += loss.item() * X_batch.size(0)

        test_loss /= len(test_ds)

        if verbose:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train MSE (scaled): {train_loss:.4f} | "
                f"test MSE (scaled): {test_loss:.4f}"
            )

    # ---- predict on test ----
    model.eval()
    preds_list = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy().flatten()
            preds_list.append(preds)

    y_pred_test_scaled = np.concatenate(preds_list).reshape(-1, 1)
    y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled).flatten()

    y_test_orig = y_test.flatten()
    return y_test_orig, y_pred_test, model, x_scaler


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """
    Compute MAE, RMSE, and MAPE for regression predictions.
    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    tuple of floats
        (mae, rmse, mape)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
    return mae, rmse, mape


def eval_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    n_splits: int = 3,
    model_name: Literal["RF", "LinReg", "CatBoost", "LSTM", "MLP"] = "RF",
    model_params: dict[str, Any] | None = None,
    SEED: int = 42,
) -> dict[str, Any]:
    """
    Evaluate a time series model using TimeSeriesSplit cross-validation
    on the training set, then refit on the full train data and evaluate on test.

    Parameters
    ----------
    x_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    x_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    n_splits : int, default=3
        Number of time-series CV folds.
    model_name : {'RF','LinReg','CatBoost','LSTM','MLP'}, default='RF'
        Model identifier.
    model_params : dict or None
        Keyword arguments for the selected model.
    SEED : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        - 'cv_mae'   : average MAE over CV folds
        - 'cv_rmse'  : average RMSE over CV folds
        - 'cv_mape'  : average MAPE over CV folds
        - 'test_mae' : MAE on the final test set
        - 'test_rmse': RMSE on the final test set
        - 'test_mape': MAPE on the final test set
        - 'y_pred_test': model predictions on test set
        - 'model'      : fitted final model
        - 'x_scaler'   : fitted StandardScaler for X features
    """
    np.random.seed(SEED)

    if model_params is None:
        model_params = {}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_mae_list: list[float] = []
    cv_rmse_list: list[float] = []
    cv_mape_list: list[float] = []

    # --- Cross-validation ---
    for fold, (train_idx, val_idx) in enumerate(tscv.split(x_train), 1):
        x_train_cv = x_train.iloc[train_idx, :].copy()
        x_val_cv = x_train.iloc[val_idx, :].copy()
        y_train_cv = y_train.iloc[train_idx].copy()
        y_val_cv = y_train.iloc[val_idx].copy()

        if model_name == "LSTM":
            y_val_aligned, y_pred_cv, model, _ = fit_lstm(
                x_train_cv, y_train_cv, x_val_cv, y_val_cv, model_params, SEED
            )
            mae_err, rmse_err, mape_err = compute_metrics(y_val_aligned, y_pred_cv)

        elif model_name == "MLP":
            y_val_aligned, y_pred_cv, model, _ = fit_mlp(
                x_train_cv, y_train_cv, x_val_cv, y_val_cv, model_params, SEED
            )
            mae_err, rmse_err, mape_err = compute_metrics(y_val_aligned, y_pred_cv)

        else:
            # Scale features
            x_scaler = StandardScaler()
            x_scaled_cv_train = x_scaler.fit_transform(x_train_cv)
            x_scaled_cv_val = x_scaler.transform(x_val_cv)

            y_train_cv_vals = y_train_cv.values.ravel()
            y_val_cv_vals = y_val_cv.values.ravel()

            # Construct model
            if model_name == "RF":
                model = RF(**model_params)
            elif model_name == "LinReg":
                model = Ridge(**model_params)
            elif model_name == "CatBoost":
                params = dict(model_params)
                params.setdefault("verbose", False)
                params.setdefault("random_seed", SEED)
                model = CatBoostRegressor(**params)
            else:
                raise ValueError(f"Unknown model_name: {model_name}")

            model.fit(x_scaled_cv_train, y_train_cv_vals)
            y_pred_cv = model.predict(x_scaled_cv_val)

            mae_err, rmse_err, mape_err = compute_metrics(y_val_cv_vals, y_pred_cv)

        cv_mae_list.append(float(mae_err))
        cv_rmse_list.append(float(rmse_err))
        cv_mape_list.append(float(mape_err))

    cv_mae = float(np.mean(cv_mae_list))
    cv_rmse = float(np.mean(cv_rmse_list))
    cv_mape = float(np.mean(cv_mape_list))

    # --- Final model training ---
    if model_name == "LSTM":
        y_test_aligned, y_pred_test_aligned, model, x_scaler = fit_lstm(
            x_train, y_train, x_test, y_test, model_params, SEED
        )
        seq_len = model_params.get("seq_len", 48)

        y_pred_test = np.full(len(y_test), np.nan, dtype=float)
        y_pred_test[seq_len:] = y_pred_test_aligned

        mae_err_test, rmse_err_test, mape_err_test = compute_metrics(
            y_test.values[seq_len:], y_pred_test[seq_len:]
        )

    elif model_name == "MLP":
        y_test_aligned, y_pred_test, model, x_scaler = fit_mlp(
            x_train, y_train, x_test, y_test, model_params, SEED
        )
        mae_err_test, rmse_err_test, mape_err_test = compute_metrics(
            y_test_aligned, y_pred_test
        )

    else:
        if model_name == "RF":
            model = RF(**model_params)
        elif model_name == "LinReg":
            model = Ridge(**model_params)
        elif model_name == "CatBoost":
            params = dict(model_params)
            params.setdefault("verbose", False)
            params.setdefault("random_seed", SEED)
            model = CatBoostRegressor(**params)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        x_scaler = StandardScaler()
        x_scaled_train = x_scaler.fit_transform(x_train)
        x_scaled_test = x_scaler.transform(x_test)

        model.fit(x_scaled_train, y_train.values.ravel())

        y_pred_test = model.predict(x_scaled_test)
        mae_err_test, rmse_err_test, mape_err_test = compute_metrics(y_test, y_pred_test)

    return {
        "cv_mae": round(cv_mae, 2),
        "cv_rmse": round(cv_rmse, 2),
        "cv_mape": round(cv_mape, 2),
        "test_mae": round(float(mae_err_test), 2),
        "test_rmse": round(float(rmse_err_test), 2),
        "test_mape": round(float(mape_err_test), 2),
        "y_pred_test": y_pred_test,
        "model": model,
        "x_scaler": x_scaler,
    }


def plot_predictions(y_true: pd.Series, y_pred: pd.Series) -> None:
    """
    Plots interactive predictions with Plotly
    """
    # Assume y_test and y_pred are pandas Series with a datetime index
    fig = go.Figure()

    # Add actual values
    fig.add_trace(go.Scatter(
        x=y_true.index, y=y_true.values,
        mode='lines',
        name='Actual',
        line=dict(width=2)
    ))

    # Add predicted values
    fig.add_trace(go.Scatter(
        x=y_true.index, y=y_pred,
        mode='lines',
        name='Predicted',
        line=dict(width=1)
    ))

    # Customize layout
    fig.update_layout(
        title='Actual vs Predicted Values',
        xaxis_title='Date',
        yaxis_title='Value',
        legend=dict(x=0, y=1),
        height=500
    )

    fig.show()


def remove_outliers_zscore(
    df: pd.DataFrame,
    threshold: float = 3.0,
    nan_treatment: str = 'ffill',
    stats: dict[str, tuple[float, float]] | None = None
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """
    Replace outliers (per-column z-score > threshold) with NaN.
    Optionally forward-fills or drops the resulting NaNs.
    If stats are provided, uses them; otherwise computes and returns them
    (so the same parameters can be applied to test data).

    Parameters
    ----------
    df : pd.DataFrame
        Input data (numeric/mixed).
    threshold : float
        |z| cutoff.
    nan_treatment : {'ffill','drop'}
        How to handle NaNs.
    stats : dict or None
        Precomputed {col: (mean, std)}.

    Returns
    -------
    df_masked : pd.DataFrame
        DataFrame with outliers replaced and NaN treatment applied.
    fit_stats : Dict[str,(float,float)]
        Mean and std used (save for test).
    """
    df_masked = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    fit_stats: dict[str, tuple[float, float]] = {}

    if stats is None:
        for col in numeric_cols:
            fit_stats[col] = (df[col].mean(), df[col].std(ddof=0))
    else:
        fit_stats = stats

    total_changed = 0

    for col in numeric_cols:
        mean, std = fit_stats[col]
        if std == 0:
            print(f"{col}: std==0, skipped")
            continue

        z = np.abs((df[col] - mean) / std)
        changed = int((z > threshold).sum())
        total_changed += changed

        print(f"{col}: {changed} replaced")

        df_masked.loc[z > threshold, col] = np.nan

    print(f"TOTAL replaced: {total_changed}")

    if nan_treatment == 'ffill':
        df_masked = df_masked.ffill()
    elif nan_treatment == 'drop':
        df_masked = df_masked.dropna()
    else:
        raise ValueError(f"nan_treatment '{nan_treatment}' not recognized.")

    return df_masked, fit_stats


def plot_errors(
    x_true: pd.DataFrame,
    y_true: pd.Series | pd.DataFrame,
    y_pred: np.ndarray | list,
    error: Literal["mae", "mape"],
    error_threshold: float,
    rolling_window: int
) -> None:
    """
    Plot prediction errors with detected anomalies and threshold lines.

    Parameters
    ----------
    x_true : pd.DataFrame
        Original input data (with datetime index).
    y_true : pd.Series or pd.DataFrame
        True target values.
    y_pred : array-like
        Predicted target values.
    error : {"mae", "mape"}
        Type of error to visualize.
    error_threshold : float
        Threshold above which points are flagged as anomalies.
    rolling_window: int
        Rolling window for the error rolling aggregation
    """

    # Create a DataFrame with true values and compute the error
    y_test_err = pd.DataFrame(y_true)
    if error == "mae":
        y_test_err['Error'] = abs(y_test_err['Power'] - y_pred)
    elif error == 'rmse':
        y_test_err['Error'] = np.sqrt((y_test_err['Power'] - y_pred)**2)
    elif error == 'mape':
        y_test_err['Error'] = abs(y_test_err['Power'] - y_pred) / y_test_err['Power'] * 100

    # Ensure both y_test_err and x_true have datetime indices
    y_test_err.index = pd.to_datetime(y_test_err.index)
    x_true.index = pd.to_datetime(x_true.index)

    # Initialize Plotly figure
    fig = go.Figure()

    # Plot the error over time
    fig.add_trace(go.Scatter(
        x=y_test_err.index,
        y=y_test_err['Error'].values,
        mode='lines',
        name='Error',
        line=dict(width=2)
    ))

    # Add horizontal line for static 95th percentile threshold
    fig.add_hline(
        y=error_threshold,
        line_color="black",
        annotation_text="Upper Threshold",
        annotation_position="bottom right"
    )

    # Plot rolling median error for smoothed trend
    fig.add_trace(go.Scatter(
        x=y_test_err.index,
        y=y_test_err['Error'].rolling(rolling_window).median(),
        mode='lines',
        name='Rolling Median Error',
        line=dict(width=2)
    ))

    # Configure layout settings
    fig.update_layout(
        title='Prediction Error and Anomaly Detection',
        xaxis_title='Date',
        yaxis_title='Error',
        legend=dict(x=0, y=1),
        height=500,
        template='plotly_white'
    )

    fig.show()


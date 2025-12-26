# Lesson: Writing the First Kedro Pipeline

## Part 1: Feature Engineering Pipeline

### Step 1: Create Pipeline Structure

In the `pipelines` directory, create:
- `feature_eng/` folder
- `pipeline.py` file
- `nodes.py` file

### Step 2: Prepare Local Testing Data

Create a separate notebook: `05 - Local Data Preparation.ipynb`

This notebook will help you:
- Test functions before adding them to the pipeline
- Debug data transformations
- Verify expected outputs

### Step 3: Start with Data Cleaning

Start going step-by-step through `get_clean_data()` function from your notebooks.

### Step 4: Add remove_diff_outliers() Node

1. Copy `remove_diff_outliers()` function to `nodes.py`
2. Modify the function to take only a dict with thresholds (from config)
3. Also mention that we would usually want to rename data, so let's make this function `rename_data()`
4. Add this to the pipeline

### Step 5: Create Dataset Configuration

Create dataset: `input_df`

**Show how to check which datasets are available:**
https://docs.kedro.org/projects/kedro-datasets/en/kedro-datasets-5.1.0/api/kedro_datasets.pandas.ParquetDataset.html

Add to `conf/base/catalog.yml`:
```yaml
input_df:
  type: pandas.ParquetDataset
  filepath: data/01_raw/input_data.parquet
```

### Step 6: Register Pipeline

Go to `pipeline_registry.py` and modify it to include the feature engineering pipeline.

### Step 7: Run Pipeline

Make `kedro run`

**Also, configure debugger in PyCharm:**
- Module name: `kedro`
- Parameters: `run`
- Working directory: `<project_root>`

### Step 8: Add drop_columns() Node

Add `drop_columns()` node to the pipeline.

### Step 9: Add smooth_signal() Function

1. Add `smooth_signal()` function
2. Change it for multi-features (handle multiple columns)
3. Run `kedro run` to test

### Step 10: Add Lagged Features

Add `lagged_features`:
- Change so that each feature has a lag
- Run `kedro run`
- Run in debug to check the output

### Step 11: Add Rolling Features

Add `add_rolling_features()` function:
- Change the function so that for each feature you can add stat and window
- Make it configurable via parameters

### Step 12: Add get_features_and_target() Function

Add `get_features_and_target()` function and add this to the pipeline.

This function:
- Separates features from target variable
- Returns both as separate outputs
- Prepares data for model training

---

## Part 2: Training Pipeline

### Step 1: Create Training Pipeline Structure

Add `create_pipeline()` function in `pipelines/training/pipeline.py`

Add `training_pipeline` key to `params.yml` config.

### Step 2: Create train_test_split() Node

Create `train_test_split()` node that:
- Takes features and target
- Splits into train/test sets
- Returns x_train, y_train, x_test, y_test

### Step 3: Create tune_hyperparameters() Node

Create `tune_hyperparameter()` node.

**Important:** We need to start from `study = optuna.create_study` because this is the top layer inside which we run everything.

### Step 4: Create objective() Function

1. Copy paste objective from Notebook
2. Create this function in `utils.py` under the "training" directory

**We need to write the objective function in a generic way**, so that it fits not only CatBoost, but other algorithms.

**Refactoring steps:**

1. **First, remove the MLflow stuff for now** (we'll add it back later)

2. **Also, for now, remove Neural Networks.** Because:
   - The code has grown a lot
   - We are not using it
   - If we want to use it, from a code perspective, we would need to make the interface like the sklearn style

3. **Remove test set metrics**, because we will do only cross-validation

4. **Also, remove model re-training on the whole dataset**; we will make it a separate node

**By this step, we are left with:**
- `params_dict` for CatBoost
- A shortened `eval_model()` function

### Step 5: Create Generic Parameter Sampling

Now, we need to create a way to specify the parameters generically.

1. **Write `sample_optuna_params()` function**
2. **In the config file, create a way we specify the grid:**
   ```yaml
   training_pipeline:
     optuna_params:
       catboost:
         learning_rate: [0.01, 0.1]
         depth: [4, 8]
   ```
3. **Write the `sample_optuna_params()` function** that reads from config
4. **Remove optuna params Dict from the objective**
5. **Finalize the `objective()` function**

### Step 6: Add Training Pipeline to Default Pipelines

Add training pipeline to the default pipelines and run `kedro`.

**Note:** There will be a lot of iterations, so we need to add verbose fixed number of iterations in addition to the sampled Optuna params.

### Step 7: Add fit_best_model() Node

**fit_best_model() node:**
1. Write the function (try to ask Cursor to create it, check against the prepared code)
2. For now, let's return the model and the scaler only
3. Add `fit_best_model()` node to the pipeline
4. Make `kedro run` and then `kedro-viz`

### Step 8: Save Model and Scaler

To proceed with inference, we need to save the model and scaler. For now, let's save it in `02_models`.

**For this, create datasets:**
- `best_model` with a joblib backend
- `x_scaler` dataset

**Emphasize:** You don't need to write any code for it - just add to catalog.yml:
```yaml
best_model:
  type: joblib.JoblibDataset
  filepath: data/02_models/best_model.pkl

x_scaler:
  type: joblib.JoblibDataset
  filepath: data/02_models/x_scaler.pkl
```

Re-run `kedro run`

### Step 9: Register Training Pipeline

Add `training_pipeline` to `pipeline_registry.py` and run it with:
```bash
kedro run --pipeline=training
```

---

## Part 3: Inference Pipeline

### Step 1: Create predict() Node

Create `predict()` node and add this to the pipeline.

### Step 2: Register Inference Pipeline

Add `inference_pipeline` to the pipeline registry, add to the return dict as `inference_pipeline`.

### Step 3: Handle Data Loading

**Problem:** The `feature_eng` pipeline takes the training data as an input, so we need to either split the current pipeline or parameterize it.

**Solution:** For simplicity, we will create 2 additional loading pipelines and then combine with feature engineering pipeline.

1. Create `load_db_training_data()` pipeline
2. Create `load_inference_data_from_db()` pipeline
3. Add new pipelines to the `pipeline_registry`

### Step 4: Add Input Datasets to Catalog

Make sure to add the input datasets to the catalog.

### Step 5: Add compute_metrics() Node

Add `compute_metrics()` node, print the metrics.

**Note:** Later, we need to add rolling_statistics to compute the metrics.

### Step 6: Add MLflow Integration

**Problem:** We don't log any training parameters, metrics, etc to MLflow. Also, we save the model locally and not in the MLflow server (even local for now).

**Let's add mlflow stuff:**

#### Adding MLflow Training Tracking + Model Registry

**Step 6.1: Create MLflow Hook**

Every time when we train the model, we need to point to the MLflow server. To set up such background processes, there's a mechanism in Kedro called **hooks**.

1. To add a hook, create `hooks.py` file
2. Add MLflow Hook that runs before any pipeline run
3. Add hook to `settings.py` file

**Emphasize:** It's convenient to run the code for specific pipelines and nodes.

After adding, run `kedro run`.

**Step 6.2: Create log_to_mlflow() Node**

Create `log_to_mlflow()` node.

**Step 6.3: Create MLModelWrapper**

When coming to `mlflow.pyfunc.log_model()`, say that we need to create `MLModelWrapper`, say that we need a general wrapper. Create this in `utils.py`.

**Step 6.4: Explain the log_to_mlflow() Function**

Explain the function step by step:

1. **Create a local directory for model artifacts** if it does not exist
2. **Start a new MLflow run** (all logs belong to this run)
3. **Log training metrics and best hyperparameters**
4. **Build an MLflow input–output signature** from a small input example
5. **Save the trained model and fitted scaler** as local artifact files
6. **Log a PyFunc model** that bundles model + scaler logic
7. **MLflow copies local artifact files** into its artifact store (local or remote)
8. **Local files can be deleted safely** after the run completes

Run `kedro training pipeline`, see the logged metrics and artifacts.

**Step 6.5: Create register_model() Node**

Create `register_model()` node.

**Tell about model registry stages and migration from them to aliases and tags:**
https://github.com/mlflow/mlflow/issues/10336

**Return version** to force promotion node run after the registry node.

**Step 6.6: Create validate_challenger() Node**

Create `validate_challenger()` node, explain its logic. Run the pipeline.

**Step 6.7: Load Model from Registry in Inference**

For the inference pipeline, `load_from_registry()`.

**Note:** In kedro-mlflow plugin, there's such a dataset. However, mlflow develops quickly and it can be the case that at some point these dataset might not match. Also, it's easy to do and we get more control over it.

In the `predict()` node, remove `x_scaler` because we load it with the model.

**Step 6.8: Check Metrics**

Check the metrics. Demonstrate that often incorrect shapes in metrics can give crazy metrics that is why we write:

```python
y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)
```

This ensures:
- Consistent array shapes
- Proper broadcasting
- Avoids unexpected behavior from pandas Series/DataFrame


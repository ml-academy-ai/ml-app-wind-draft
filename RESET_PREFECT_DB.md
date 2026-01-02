# Quick Fix: Reset Prefect Database

## Problem
SQLAlchemy database error when starting Prefect server - database is locked or corrupted.

## Quick Fix Steps

### 1. Stop any running Prefect processes
```bash
# Check if Prefect is running
lsof -i :4200

# If something is using port 4200, kill it
pkill -f "prefect server"
pkill -f "prefect worker"
```

### 2. Backup and reset Prefect database
```bash
# Backup existing database (optional - saves your deployments for reference)
cp ~/.prefect/prefect.db ~/.prefect/prefect.db.backup

# Reset Prefect database (WARNING: This deletes all deployments and run history)
rm ~/.prefect/prefect.db
```

### 3. Restart Prefect server
```bash
prefect server start
```

### 4. Re-register your deployments

**Note:** After resetting the database, you'll need to recreate everything:

```bash
# Terminal 1: Start Prefect Server (keep running)
prefect server start

# Terminal 2: Create work queues (one-time)
prefect work-queue create training-queue
prefect work-queue create inference-queue

# Terminal 3: Create work pool and start worker (keep running)
prefect work-pool create default-worker-pool --type process
export PREFECT_API_URL=http://127.0.0.1:4200/api
prefect worker start --pool default-worker-pool --work-queue training-queue --work-queue inference-queue

# Terminal 4: Register training deployment
export MLFLOW_TRACKING_URI=file:///$(pwd)/mlflow/mlruns  # Optional but recommended
python entrypoint/training_prefect.py

# Terminal 5: Register inference deployment
export MLFLOW_TRACKING_URI=file:///$(pwd)/mlflow/mlruns  # Optional but recommended
python entrypoint/inference_prefect.py
```

## Alternative: Try to recover without deleting

If you want to try recovering the database first:

```bash
# Try to repair SQLite database
sqlite3 ~/.prefect/prefect.db ".recover" | sqlite3 ~/.prefect/prefect.db.new
mv ~/.prefect/prefect.db.new ~/.prefect/prefect.db
```

Then try starting Prefect server again. If it still fails, proceed with the reset steps above.


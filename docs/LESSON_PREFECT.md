# Lesson: Prefect Orchestration Guide

## Overview

This comprehensive guide covers everything you need to know about using Prefect for orchestrating your ML training and inference pipelines. You'll learn:
- Prefect concepts and architecture
- How to test flows locally before deployment
- How to create and deploy Prefect flows using `flow.serve()`
- Complete setup workflow from testing to production
- Monitoring, troubleshooting, and best practices

## Current Implementation Approach

This project uses **`flow.serve()`** for deployment, which:
- ✅ Registers the deployment with Prefect server
- ✅ Runs continuously to manage scheduled executions
- ✅ Works without complex storage configuration
- ✅ Perfect for Docker Compose and VM deployments

## Prerequisites

- Python 3.11 or 3.12 installed
- Your ML application working locally (Kedro pipelines)
- Basic understanding of Python and command line
- Prefect dependency installed (already in `pyproject.toml`)

## Part 1: Understanding Prefect

### What is Prefect?

Prefect is a workflow orchestration tool that helps you:
- **Schedule** pipelines to run at specific intervals
- **Monitor** pipeline execution and view logs
- **Retry** failed runs automatically
- **Scale** by running workers on multiple machines
- **Track** execution history and performance metrics

### Key Concepts

#### 1. **Flows** (`@flow`)
- Top-level functions that define a workflow
- Can contain multiple tasks
- Handle orchestration logic

#### 2. **Tasks** (`@task`)
- Individual units of work
- Can be retried independently
- Represent discrete operations (e.g., "run training pipeline")

#### 3. **Deployments**
- Configuration that defines how a flow should run
- Includes: schedule, parameters, work queue assignment
- Registered with Prefect server (stored in database)

#### 4. **Work Pools**
- Logical groups for organizing workers
- Different pools for different environments (dev, prod)

#### 5. **Work Queues**
- Queues that hold scheduled flow runs
- Workers pull jobs from queues
- One flow can be assigned to multiple queues

#### 6. **Workers**
- Processes that execute scheduled flows
- Pull jobs from work queues
- Can run on same machine or different machines

#### 7. **Prefect Server**
- API server and database
- Stores deployments, run history, logs
- Provides UI at http://localhost:4200

### Architecture Diagram

```
┌─────────────────┐
│ Prefect Server  │  ← Stores deployments, schedules, run history
│  (localhost:    │
│      4200)      │
└────────┬────────┘
         │
         │ API calls
         │
┌────────▼────────┐     ┌──────────────┐
│  Worker         │     │  Deployment  │
│  (Process)      │ ←── │  Script      │
│                 │     │              │
│  - Pulls jobs   │     │  - Registers │
│    from queues  │     │    deployment│
│  - Executes     │     │  - Defines   │
│    flows        │     │    schedule  │
└─────────────────┘     └──────────────┘
```

## Part 2: Project Structure

Your Prefect flows are located in:
```
entrypoint/prefect/
├── training_flow.py    # Training pipeline orchestration
└── inference_flow.py   # Inference pipeline orchestration
```

### Example: Training Flow Structure

```python
"""Prefect flow for training pipeline orchestration."""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from loguru import logger
from prefect import flow, task

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
sys.path.append(str(project_root))
os.chdir(project_root)

from app_data_manager.utils import read_config
from common.mlflow_utils import get_latest_model_timestamp

# Read configuration
parameters_path = project_root / "conf" / "base" / "parameters.yml"
config = read_config(parameters_path)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")


@task(name="check-latest-model-timestamp")
def check_latest_model_timestamp_task(
    mlflow_tracking_uri: str, model_name: str
) -> datetime | None:
    """Check MLflow for the latest model timestamp (champion or challenger)."""
    return get_latest_model_timestamp(mlflow_tracking_uri, model_name)


@task(name="should-train")
def should_train_task(
    latest_timestamp: datetime | None, training_frequency: float
) -> bool:
    """Determine if training should run based on the latest model timestamp."""
    if latest_timestamp is None:
        return True
    
    time_elapsed = datetime.now() - latest_timestamp
    time_elapsed_minutes = time_elapsed.total_seconds() / 60.0
    
    return time_elapsed_minutes >= training_frequency


@task(name="training-task")
def training_task(env: str = "local", pipeline_name: str = "training"):
    """Prefect task to run the Kedro training pipeline."""
    logger.info("Starting training pipeline...")
    with open(project_root / "pyproject.toml", "rb") as f:
        package_name = tomllib.load(f)["tool"]["kedro"]["package_name"]

    configure_project(package_name)
    bootstrap_project(project_root)

    with KedroSession.create(project_path=project_root, env=env) as session:
        session.run(pipeline_name=pipeline_name)
    
    logger.info("Training completed successfully")


@flow(name="training-flow")
def training_flow(env: str = "local"):
    """Prefect flow for training that checks MLflow and runs training if needed."""
    training_config = config["training_pipeline"]["training_real_time"]
    training_frequency = training_config["training_frequency"]
    model_name = config["mlflow"]["registered_model_name"]
    
    latest_timestamp = check_latest_model_timestamp_task(
        MLFLOW_TRACKING_URI, model_name
    )
    
    should_train = should_train_task(latest_timestamp, training_frequency)
    
    if should_train:
        training_task(env=env)
    else:
        logger.info("Training skipped - not enough time has passed since last model.")


if __name__ == "__main__":
    os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")

    training_config = config["training_pipeline"]["training_real_time"]
    check_frequency_minutes = training_config["check_frequency"]

    training_flow.serve(
        name="training-flow",
        interval=timedelta(minutes=check_frequency_minutes),
        parameters={"env": os.getenv("KEDRO_ENV", "local")},
    )

# Test section (uncomment for local testing)
# if __name__ == "__main__":
#     training_flow(env="local")
```

### Key Components Explained

1. **Path Setup**: `parents[2]` resolves to project root from `entrypoint/prefect/training_flow.py`
2. **Task Definition**: `@task` decorator wraps your Kedro pipeline execution
3. **Flow Definition**: `@flow` decorator orchestrates the tasks
4. **Deployment**: `flow.serve()` registers and serves the deployment
5. **Testing**: Commented section for direct flow execution (testing)

## Part 3: Testing Flows Locally (Before Deployment)

**Always test your flow locally before deploying!**

### Step 1: Comment Out the Deployment, Uncomment the Test Section

Before deploying, test your flow locally first:

```python
# Comment out the deployment section
# if __name__ == "__main__":
#     os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")
#     training_flow.serve(...)

# Uncomment the test section
if __name__ == "__main__":
    training_flow(env="local")
```

### Step 2: Run the Flow Directly

```bash
cd /path/to/ml-app-wind-draft
export MLFLOW_TRACKING_URI=http://localhost:5001  # Optional, if using MLflow
export KEDRO_ENV=local

python entrypoint/prefect/training_flow.py
```

**What this does:**
- Runs the flow **directly** without Prefect server
- Tests that your Kedro pipeline works correctly
- Verifies all dependencies and paths are correct
- No Prefect server needed for this step

**Expected behavior:**
- Flow executes immediately
- You see Kedro pipeline execution logs
- Flow completes and script exits

### Step 3: Verify the Flow Works

Check that:
- ✅ No import errors
- ✅ Kedro pipeline runs successfully
- ✅ All data paths are correct
- ✅ MLflow tracking works (if configured)

## Part 4: Deploying with Prefect

### Step 1: Restore the Deployment Code

After testing, restore the deployment code:

```python
if __name__ == "__main__":
    os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")

    training_flow.serve(
        name="training-flow",
        interval=timedelta(minutes=check_frequency_minutes),
        parameters={"env": os.getenv("KEDRO_ENV", "local")},
    )

# Comment out the test section again
# if __name__ == "__main__":
#     training_flow(env="local")
```

### Step 2: Start Prefect Server

**Terminal 1:**
```bash
prefect server start
```

**Keep this running!** You should see:
```
Starting Prefect API server at http://127.0.0.1:4200/api
Starting Prefect UI at http://127.0.0.1:4200
```

Wait until the server fully starts before proceeding.

### Step 3: Set Up Work Pool and Queue (One-time Setup)

**Terminal 2:**
```bash
export PREFECT_API_URL=http://127.0.0.1:4200/api

# Create work pool (ignore error if already exists)
prefect work-pool create default-worker-pool --type process 2>/dev/null || true

# Create work queues (ignore errors if already exist)
prefect work-queue create training-queue --pool default-worker-pool 2>/dev/null || true
prefect work-queue create inference-queue --pool default-worker-pool 2>/dev/null || true
```

**Verify:**
```bash
prefect work-pool ls
prefect work-queue ls
```

### Step 4: Start Prefect Worker

**Terminal 3:**
```bash
export PREFECT_API_URL=http://127.0.0.1:4200/api

prefect worker start \
  --pool default-worker-pool \
  --work-queue training-queue \
  --work-queue inference-queue
```

**Keep this running!** The worker will:
- Connect to Prefect server
- Listen to the specified queues
- Execute scheduled flow runs

**Expected output:**
```
Connected to Prefect server at http://127.0.0.1:4200/api
Polling for runs from queues: training-queue, inference-queue
```

### Step 5: Deploy Your Flow

**Terminal 4:**
```bash
cd /path/to/ml-app-wind-draft
export PREFECT_API_URL=http://127.0.0.1:4200/api
export KEDRO_ENV=local
export MLFLOW_TRACKING_URI=http://localhost:5001  # If using MLflow

# Deploy training flow
python entrypoint/prefect/training_flow.py

# In another terminal, deploy inference flow
python entrypoint/prefect/inference_flow.py
```

**What happens:**
- Script registers the deployment with Prefect server
- Deployment becomes active and scheduled
- Script continues running (serving the deployment)
- Worker picks up scheduled runs and executes them

**Expected output:**
```
Serving deployment 'training-flow'...
Press CTRL+C to exit.
```

**Keep this terminal running!** The script manages the deployment.

## Part 5: Complete Workflow Summary

### Testing → Deployment Workflow

```bash
# 1. TEST FIRST: Comment deployment, uncomment test section
# Edit training_flow.py: Uncomment test, comment deployment

# 2. Run flow directly (no Prefect needed)
python entrypoint/prefect/training_flow.py

# 3. If successful, restore deployment code
# Edit training_flow.py: Comment test, uncomment deployment

# 4. Start Prefect Server (Terminal 1)
prefect server start

# 5. Setup work pool/queues (Terminal 2, one-time)
export PREFECT_API_URL=http://127.0.0.1:4200/api
prefect work-pool create default-worker-pool --type process 2>/dev/null || true
prefect work-queue create training-queue --pool default-worker-pool 2>/dev/null || true

# 6. Start Worker (Terminal 3)
export PREFECT_API_URL=http://127.0.0.1:4200/api
prefect worker start --pool default-worker-pool --work-queue training-queue

# 7. Deploy Flow (Terminal 4)
export PREFECT_API_URL=http://127.0.0.1:4200/api
python entrypoint/prefect/training_flow.py
```

## Part 6: Configuration Options

### Flow.serve() Parameters

```python
training_flow.serve(
    name="training-flow",                    # Deployment name (unique identifier)
    interval=timedelta(minutes=1),           # Schedule: how often to run
    parameters={"env": "local"},             # Default parameters for flow
    tags=["training", "production"],         # Optional tags for organization
    description="Training pipeline",         # Optional description
)
```

### Common Schedule Options

```python
from datetime import timedelta

# Every minute
interval=timedelta(minutes=1)

# Every 5 minutes
interval=timedelta(minutes=5)

# Every hour
interval=timedelta(hours=1)

# Every day
interval=timedelta(days=1)

# Every 30 seconds
interval=timedelta(seconds=30)

# No schedule (manual only)
# Omit interval parameter
```

### Environment Variables

```bash
# Required
export PREFECT_API_URL=http://127.0.0.1:4200/api

# Optional (project-specific)
export KEDRO_ENV=local
export MLFLOW_TRACKING_URI=http://localhost:5001
```

### Configuration from parameters.yml

Your flows read configuration from `conf/base/parameters.yml`:

```yaml
training_pipeline:
  training_real_time:
    training_frequency: 60.0  # Minutes since last model before triggering training
    check_frequency: 1.0      # Minutes between checks

inference_pipeline:
  inference_frequency: 30    # Seconds since last inference before triggering
```

## Part 7: Monitoring and Management

### Prefect UI

Open in browser: **http://localhost:4200**

**Key sections:**
- **Dashboard**: Overview of recent runs
- **Deployments**: View all registered deployments
- **Flow Runs**: See execution history and logs
- **Work Pools**: Check worker pool status
- **Work Queues**: Monitor queue status and pending runs
- **Logs**: Detailed execution logs for each run

### CLI Commands

```bash
# List deployments
prefect deployment ls

# Inspect a specific deployment
prefect deployment inspect training-flow/training-flow

# List flow runs
prefect flow-run ls --limit 10

# Inspect a flow run
prefect flow-run inspect <flow-run-id>

# List work pools
prefect work-pool ls

# List work queues
prefect work-queue ls

# Check worker status (in worker terminal)
# Workers show status in their terminal output
```

### Managing Workers

**Find running workers:**
```bash
ps aux | grep "[p]refect worker"
```

**Stop all workers:**
```bash
pkill -f "prefect worker"
```

**Force stop (if needed):**
```bash
pkill -9 -f "prefect worker"
```

## Part 8: Troubleshooting

### Issue 1: Flow Not Executing

**Symptoms:**
- Deployment registered but no runs appearing
- Worker running but idle

**Solutions:**
1. **Check worker is connected:**
   ```bash
   prefect work-pool inspect default-worker-pool
   ```

2. **Verify queue names match:**
   - Deployment queue name must match worker queue name
   - Check deployment: `prefect deployment inspect training-flow/training-flow`

3. **Check worker logs:**
   - Look at worker terminal output
   - Should show "Polling for runs from queues: ..."

### Issue 2: Import Errors

**Symptoms:**
- `ModuleNotFoundError` when flow runs
- Path resolution errors

**Solutions:**
1. **Verify path calculation:**
   ```python
   # From entrypoint/prefect/training_flow.py
   project_root = Path(__file__).resolve().parents[2]  # Goes up 2 levels
   ```

2. **Test locally first** using the commented test section

3. **Check working directory:**
   ```python
   os.chdir(project_root)  # Must be set before imports
   ```

### Issue 3: Prefect Server Connection Refused

**Symptoms:**
- `ConnectionRefusedError` when deploying
- `PREFECT_API_URL` errors

**Solutions:**
1. **Start Prefect server first:**
   ```bash
   prefect server start
   ```

2. **Set environment variable:**
   ```bash
   export PREFECT_API_URL=http://127.0.0.1:4200/api
   ```

3. **Wait for server to fully start** before running deployment script

### Issue 4: Connection Refused to MLflow (Port 5001)

**Symptoms:**
```
WARNING  Retrying (Retry(total=4, ...)) after connection broken by
'NewConnectionError("HTTPConnection(host='localhost', port=5001): 
Failed to establish a new connection: [Errno 61] Connection refused")'
```

**Solutions:**

**Option A: Use File-based MLflow (Recommended)**
```bash
export MLFLOW_TRACKING_URI=file:///$(pwd)/mlflow/mlruns
```

**Option B: Start MLflow Server**
```bash
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri file:///$(pwd)/mlflow/mlruns
export MLFLOW_TRACKING_URI=http://localhost:5001
```

**Note:** The warnings won't stop execution, but MLflow logging will fail.

### Issue 5: Multiple Workers/Duplicate Executions

**Symptoms:**
- Same flow running multiple times
- Duplicate executions

**Solutions:**
1. **Stop extra workers:**
   ```bash
   pkill -f "prefect worker"
   ```

2. **Start only one worker per queue:**
   - Or use different queues for different deployments

### Issue 6: Prefect Server Port Already in Use

**Symptoms:**
```
Address already in use: 4200
```

**Solution:**
```bash
# Find process using port 4200
lsof -i :4200

# Kill the process
kill -9 <PID>

# Or use different port
prefect server start --port 4201
# Update PREFECT_API_URL accordingly
export PREFECT_API_URL=http://127.0.0.1:4201/api
```

### Issue 7: Flow Run Fails Immediately

**Symptoms:**
- Flow runs appear but fail immediately
- Logs show import errors or path issues

**Possible Causes:**

1. **Python path issues:**
   - Ensure script is run from project root or proper working directory
   - Check `project_root` path in script

2. **Missing dependencies:**
   ```bash
   # Verify all dependencies installed
   uv sync
   # or
   pip install -e .
   ```

3. **Kedro project not configured:**
   - Ensure `bootstrap_project()` is called before `KedroSession.create()`

### Debugging Tips

1. **Check Worker Logs:**
   - Worker terminal shows execution logs
   - Look for Python errors or exceptions

2. **Check Prefect UI Logs:**
   - Click on failed flow run
   - View "Logs" tab for detailed error messages

3. **Test Flow Locally:**
   ```python
   # In Python shell or script
   from entrypoint.prefect.training_flow import training_flow
   training_flow(env="local")  # Run flow directly (no Prefect server needed)
   ```

4. **Check Deployment Configuration:**
   ```bash
   prefect deployment inspect training-flow/training-flow
   ```

## Part 9: Advanced Configuration

### Environment Variables

Create a `.env` file or export variables:

```bash
# .env file
MLFLOW_TRACKING_URI=file:///$(pwd)/mlflow/mlruns
KEDRO_ENV=local
PREFECT_API_URL=http://127.0.0.1:4200/api
```

Load with:
```bash
# Using direnv (if installed)
direnv allow

# Or manually
export $(cat .env | xargs)
```

### Separate Workers for Training and Inference

For better resource isolation:

```bash
# Terminal: Training Worker
export PREFECT_API_URL=http://127.0.0.1:4200/api
prefect worker start --pool default-worker-pool --work-queue training-queue

# Terminal: Inference Worker
export PREFECT_API_URL=http://127.0.0.1:4200/api
prefect worker start --pool default-worker-pool --work-queue inference-queue
```

### Custom Retry Logic

Add retries to tasks:

```python
@task(name="training-task", retries=3, retry_delay_seconds=60)
def training_task(env: str = "local", pipeline_name: str = "training"):
    ...
```

### Error Handling and Notifications

Add error handling:

```python
@flow(name="training-flow")
def training_flow(env: str = "local"):
    try:
        training_task(env=env)
    except Exception as e:
        # Log error, send notification, etc.
        logger.error(f"Training failed: {e}")
        raise
```

### Docker Compose Deployment

For Docker deployments, the setup is similar but uses service names:

```python
if __name__ == "__main__":
    # IMPORTANT: Use service names from docker-compose
    os.environ.setdefault("PREFECT_API_URL", "http://prefect-server:4200/api")
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://mlflow:5001")

    training_flow.serve(
        name="training-flow",
        interval=timedelta(minutes=1),
        parameters={"env": os.getenv("KEDRO_ENV", "local")},
        tags=["docker", "training"],
    )
```

**Key differences:**
- `prefect-server` instead of `localhost`
- `mlflow` instead of `localhost`
- Service names match your `docker-compose.yml`

## Part 10: Best Practices

### 1. Always Test Locally First

```python
# Step 1: Test with commented section
# if __name__ == "__main__":
#     training_flow(env="local")

# Step 2: Deploy only after testing works
if __name__ == "__main__":
    training_flow.serve(...)
```

### 2. Use Descriptive Names

```python
# Good
name="training-production-daily"
tags=["training", "production", "daily"]

# Bad
name="train"
tags=["t"]
```

### 3. Set Appropriate Intervals

- **Training**: Usually daily or weekly (longer intervals)
- **Inference**: Often minutes or hours (shorter intervals)
- **Testing**: Use short intervals (1-5 minutes)

### 4. Environment Variables

- Set in deployment scripts for Docker
- Use `.env` files for local development
- Never hardcode sensitive values

### 5. Monitor Regularly

- Check Prefect UI for failed runs
- Review logs for errors
- Monitor worker status

### 6. Version Control

- Commit Prefect scripts to git
- Document schedule changes in commit messages
- Keep deployment names consistent across environments

### 7. Environment Separation

Use different work pools for different environments:

```bash
# Development
prefect work-pool create dev-worker-pool --type process

# Production
prefect work-pool create prod-worker-pool --type process
```

### 8. Clean Up Old Deployments

```bash
# List deployments
prefect deployment ls

# Delete unused deployment
prefect deployment delete <flow-name>/<deployment-name>
```

## Part 11: Production Considerations

### Using Prefect Cloud (Optional)

For production, consider Prefect Cloud:
- Managed Prefect server
- Better monitoring and alerts
- Team collaboration
- Better scalability

Setup:
```bash
prefect cloud login
prefect cloud workspace set --workspace <workspace-name>
```

### Using PostgreSQL (For Self-Hosted)

For production self-hosted setup, use PostgreSQL instead of SQLite:

```bash
# Set Prefect database URL
export PREFECT_API_DATABASE_CONNECTION_URL=postgresql://user:pass@localhost/prefect

# Start server (will use PostgreSQL)
prefect server start
```

### Security

- Use environment variables for secrets
- Don't commit credentials
- Use Prefect blocks for secure secret storage
- Restrict API access in production

## Part 12: Quick Reference

### Complete Setup Checklist

- [ ] Test flow locally (commented section)
- [ ] Restore deployment code
- [ ] Start Prefect server
- [ ] Create work pool and queues
- [ ] Start worker
- [ ] Deploy flow
- [ ] Verify in Prefect UI
- [ ] Monitor first execution

### Common Commands

```bash
# Server
prefect server start

# Setup (one-time)
prefect work-pool create default-worker-pool --type process
prefect work-queue create training-queue --pool default-worker-pool

# Worker
prefect worker start --pool default-worker-pool --work-queue training-queue

# Deployment
python entrypoint/prefect/training_flow.py

# Monitoring
prefect deployment ls
prefect flow-run ls --limit 10

# Cleanup
pkill -f "prefect worker"
```

### File Locations

- **Prefect Scripts**: `entrypoint/prefect/training_flow.py`, `entrypoint/prefect/inference_flow.py`
- **Prefect Database**: `~/.prefect/prefect.db` (SQLite default)
- **Prefect UI**: http://localhost:4200
- **MLflow Data**: `mlflow/mlruns/` (if using file-based)

### Key URLs

- **Prefect UI**: http://localhost:4200
- **Prefect API**: http://localhost:4200/api
- **MLflow UI** (if running): http://localhost:5001

## Summary

This implementation uses `flow.serve()` for a simple, effective deployment:
1. **Test locally first** using the commented test section
2. **Deploy** using `flow.serve()` after testing
3. **Manage** via Prefect UI and CLI commands
4. **Scale** by adding more workers as needed

The approach works seamlessly with Docker Compose and VM deployments, requiring minimal configuration while providing full orchestration capabilities.

## Next Steps

- Explore Prefect UI features (filters, search, run details)
- Experiment with different schedules
- Add retry logic and error handling
- Set up separate workers for different workloads
- Consider Prefect Cloud for production deployments
- Integrate with monitoring/alerting systems

## Additional Resources

- [Prefect Documentation](https://docs.prefect.io/)
- [Prefect Tutorials](https://docs.prefect.io/tutorials/)
- [Prefect API Reference](https://docs.prefect.io/api-ref/)
- [Prefect Community](https://discourse.prefect.io/)


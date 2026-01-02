# Lesson: Prefect Orchestration

## Overview

This lesson covers how to configure and use Prefect for orchestrating your ML training and inference pipelines. You'll learn how to:
- Understand Prefect concepts (flows, tasks, deployments, workers)
- Set up Prefect server and workers
- Schedule periodic training and inference pipelines
- Monitor pipeline execution through Prefect UI
- Handle common configuration issues (MLflow, environment variables)

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

## Part 2: Installation and Setup

### Step 1: Verify Prefect Installation

Prefect should already be in your `pyproject.toml`:

```toml
dependencies = [
    ...
    "prefect>=3.6.8",
]
```

If not installed, run:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install prefect>=3.6.8
```

Verify installation:

```bash
prefect version
```

### Step 2: Project Structure

Your Prefect scripts are located in the `entrypoint/` directory:

```
entrypoint/
├── training_prefect.py    # Training pipeline orchestration
└── inference_prefect.py   # Inference pipeline orchestration
```

## Part 3: Understanding the Code

### Training Pipeline (`entrypoint/training_prefect.py`)

Let's break down the code:

```python
"""Minimal Prefect training example - runs every 2 minutes."""

import os
import sys
from datetime import timedelta
from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from prefect import flow, task
from prefect.client.schemas.schedules import IntervalSchedule

# Set up project paths
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
os.chdir(project_root)
```

**Key Points:**
- Sets up Python path to find the Kedro project
- Changes working directory to project root

```python
@task(name="training-task")
def training_task(env: str = "local", pipeline_name: str = "training"):
    """Prefect task wrapper."""
    package_name = "ml_app_wind_draft"
    configure_project(package_name)
    bootstrap_project(project_root)

    with KedroSession.create(project_path=project_root, env=env) as session:
        session.run(pipeline_name=pipeline_name)
```

**Key Points:**
- `@task` decorator marks this as a Prefect task
- Configures Kedro project and creates a session
- Runs the Kedro training pipeline

```python
@flow(name="training-flow")
def training_flow(env: str = "local"):
    """Prefect flow for training."""
    training_task(env=env)
```

**Key Points:**
- `@flow` decorator marks this as a Prefect flow
- Orchestrates the training task
- Can be extended with multiple tasks, retries, error handling

```python
if __name__ == "__main__":
    # Schedule: Every 2 minutes
    deployment = training_flow.to_deployment(
        name="training-2m",
        schedule=IntervalSchedule(interval=timedelta(minutes=2)),
        parameters={"env": os.getenv("KEDRO_ENV", "local")},
        work_pool_name="default-worker-pool",
        work_queue_name="training-queue",
    )
    deployment.apply()
    print("Training deployment registered: Every 2 minutes")
    print("Deployment will be executed by worker on schedule")
    print("Prefect UI: http://localhost:4200")
```

**Key Points:**
- `to_deployment()` creates deployment configuration
- `IntervalSchedule` defines how often to run (every 2 minutes)
- `work_pool_name` and `work_queue_name` assign to specific queue
- `apply()` registers deployment with Prefect server
- Script exits after registration (worker executes scheduled runs)

### Inference Pipeline (`entrypoint/inference_prefect.py`)

Similar structure but for inference:

```python
# Schedule: Every 10 seconds
deployment = inference_flow.to_deployment(
    name="inference-10s",
    schedule=IntervalSchedule(interval=timedelta(seconds=10)),
    parameters={"env": os.getenv("KEDRO_ENV", "local")},
    work_pool_name="default-worker-pool",
    work_queue_name="inference-queue",
)
```

**Key Differences:**
- Runs every 10 seconds (more frequent than training)
- Uses `inference-queue` instead of `training-queue`
- Can run independently with different schedules

## Part 4: Configuration Steps

### Step 1: Set Up MLflow Tracking URI (Optional but Recommended)

Your Kedro pipelines use MLflow for experiment tracking. To avoid connection errors, set the MLflow tracking URI:

**Option A: File-based (No Server Required)**

```bash
export MLFLOW_TRACKING_URI=file:///$(pwd)/mlflow/mlruns
```

**Option B: Start MLflow Server**

If you want a running MLflow server:

```bash
# In a separate terminal
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri file:///$(pwd)/mlflow/mlruns
```

Then set:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
```

**Option C: Skip MLflow (Not Recommended)**

If you don't need tracking for test runs, the pipeline will still work but MLflow logging will fail. You'll see connection errors, but they won't stop execution.

### Step 2: Set Kedro Environment (Optional)

By default, scripts use `"local"` environment. To use a different one:

```bash
export KEDRO_ENV=base  # or any other environment
```

## Part 5: Running Prefect Setup

### Step-by-Step Execution

Follow these steps **in order**:

#### Terminal 1: Start Prefect Server

```bash
prefect server start
```

**What this does:**
- Starts Prefect API server (default port 4200)
- Creates SQLite database for storing deployments and run history
- Provides web UI at http://localhost:4200

**Keep this running!** All other components depend on it.

**Expected output:**
```
Starting Prefect server...
Starting Prefect API server at http://127.0.0.1:4200/api
Starting Prefect UI at http://127.0.0.1:4200
```

#### Terminal 2: Create Work Queues

```bash
# Create training queue
prefect work-queue create training-queue

# Create inference queue
prefect work-queue create inference-queue
```

**What this does:**
- Creates named queues that workers will poll
- Queues are stored in Prefect server database

**Note:** If queues already exist, you'll see an error. That's fine - they can be reused.

#### Terminal 3: Create Work Pool and Start Worker

```bash
# Create work pool (one-time setup)
prefect work-pool create default-worker-pool --type process

# Set Prefect API URL (required for worker)
export PREFECT_API_URL=http://127.0.0.1:4200/api

# Start worker (listens to both queues)
prefect worker start --pool default-worker-pool --work-queue training-queue --work-queue inference-queue
```

**What this does:**
- Creates a work pool named `default-worker-pool`
- Starts a worker process that:
  - Connects to Prefect server
  - Polls both queues for scheduled runs
  - Executes flows when scheduled
  - Reports results back to server

**Keep this running!** This is what actually executes your pipelines.

**Expected output:**
```
Starting worker process...
Connected to Prefect server at http://127.0.0.1:4200/api
Polling for runs from queues: training-queue, inference-queue
```

**Note:** Use `--work-queue` (singular) multiple times, not `--work-queues` (plural).

#### Terminal 4: Register Training Deployment

```bash
# Optional: Set MLflow tracking URI
export MLFLOW_TRACKING_URI=file:///$(pwd)/mlflow/mlruns

# Register training deployment
python entrypoint/training_prefect.py
```

**What this does:**
- Registers the training deployment with Prefect server
- Script exits immediately after registration
- Deployment is now scheduled to run every 2 minutes
- Worker will pick up scheduled runs from the queue

**Expected output:**
```
Training deployment registered: Every 2 minutes
Deployment will be executed by worker on schedule
Prefect UI: http://localhost:4200
```

#### Terminal 5: Register Inference Deployment

```bash
# Optional: Set MLflow tracking URI (if not already set)
export MLFLOW_TRACKING_URI=file:///$(pwd)/mlflow/mlruns

# Register inference deployment
python entrypoint/inference_prefect.py
```

**What this does:**
- Registers the inference deployment with Prefect server
- Script exits immediately after registration
- Deployment is now scheduled to run every 10 seconds
- Worker will pick up scheduled runs from the queue

**Expected output:**
```
Inference deployment registered: Every 10 seconds
Deployment will be executed by worker on schedule
Prefect UI: http://localhost:4200
```

### Summary: Complete Setup Commands

Here's the complete sequence in one place:

```bash
# Terminal 1: Prefect Server (keep running)
prefect server start

# Terminal 2: Create Work Queues (one-time, can skip if already exist)
prefect work-queue create training-queue
prefect work-queue create inference-queue

# Terminal 3: Create Work Pool and Start Worker (keep running)
prefect work-pool create default-worker-pool --type process
export PREFECT_API_URL=http://127.0.0.1:4200/api
prefect worker start --pool default-worker-pool --work-queue training-queue --work-queue inference-queue

# Terminal 4: Register Training Deployment (runs once, then exits)
export MLFLOW_TRACKING_URI=file:///$(pwd)/mlflow/mlruns  # Optional but recommended
python entrypoint/training_prefect.py

# Terminal 5: Register Inference Deployment (runs once, then exits)
export MLFLOW_TRACKING_URI=file:///$(pwd)/mlflow/mlruns  # Optional but recommended
python entrypoint/inference_prefect.py
```

### What to Keep Running

**Keep running:**
- Terminal 1: Prefect Server
- Terminal 3: Worker

**Can close after registration:**
- Terminal 4: Training deployment script (exits after registration)
- Terminal 5: Inference deployment script (exits after registration)

## Part 6: Monitoring and Verification

### Prefect UI

Open your browser and go to:

```
http://localhost:4200
```

**What you can see:**
- **Dashboard**: Overview of recent runs
- **Deployments**: List of registered deployments
- **Flows**: Flow run history
- **Work Queues**: Queue status and pending runs
- **Work Pools**: Worker pool status
- **Logs**: Detailed execution logs for each run

### Check Deployment Status

In Prefect UI:
1. Navigate to **Deployments**
2. You should see:
   - `training-flow/training-2m` (scheduled every 2 minutes)
   - `inference-flow/inference-10s` (scheduled every 10 seconds)

### Check Flow Runs

1. Navigate to **Flow Runs**
2. You should see runs appearing:
   - Training runs every 2 minutes
   - Inference runs every 10 seconds
3. Click on a run to see:
   - Execution status (Running, Completed, Failed)
   - Start/end time
   - Logs
   - Task details

### Command Line Status Check

```bash
# List deployments
prefect deployment ls

# List work queues
prefect work-queue ls

# List flow runs (last 10)
prefect flow-run ls --limit 10

# View specific flow run details
prefect flow-run inspect <flow-run-id>
```

## Part 7: Understanding Schedules

### IntervalSchedule

Used in both scripts:

```python
IntervalSchedule(interval=timedelta(minutes=2))
IntervalSchedule(interval=timedelta(seconds=10))
```

**Options:**
- `timedelta(seconds=X)` - Every X seconds
- `timedelta(minutes=X)` - Every X minutes
- `timedelta(hours=X)` - Every X hours
- `timedelta(days=X)` - Every X days

### CronSchedule (Alternative)

For more complex schedules:

```python
from prefect.client.schemas.schedules import CronSchedule

# Run every day at 2 AM
CronSchedule(cron="0 2 * * *")

# Run every Monday at 9 AM
CronSchedule(cron="0 9 * * 1")

# Run every 30 minutes during business hours (9-17)
CronSchedule(cron="*/30 9-17 * * 1-5")
```

### Modifying Schedules

To change the schedule, edit the script and re-register:

1. Edit `entrypoint/training_prefect.py`:
   ```python
   schedule=IntervalSchedule(interval=timedelta(hours=1))  # Change to 1 hour
   ```

2. Re-run the deployment script:
   ```bash
   python entrypoint/training_prefect.py
   ```

3. Prefect will update the existing deployment with new schedule.

## Part 8: Common Issues and Troubleshooting

### Issue 1: Connection Refused to MLflow (Port 5001)

**Symptoms:**
```
WARNING  Retrying (Retry(total=4, ...)) after connection broken by
'NewConnectionError("HTTPConnection(host='localhost', port=5001): 
Failed to establish a new connection: [Errno 61] Connection refused")'
```

**Cause:**
- Kedro pipeline tries to log to MLflow server
- MLflow server is not running on port 5001

**Solution:**

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

### Issue 2: Worker Cannot Connect to Server

**Symptoms:**
```
ValueError: PREFECT_API_URL must be set to start a Worker.
```

**Solution:**
```bash
export PREFECT_API_URL=http://127.0.0.1:4200/api
prefect worker start ...
```

### Issue 3: Deployment Not Executing

**Symptoms:**
- Deployment registered successfully
- No flow runs appearing in UI
- Worker running but idle

**Possible Causes:**

1. **Worker not polling correct queue:**
   ```bash
   # Check queue name matches deployment
   prefect deployment inspect training-flow/training-2m
   
   # Ensure worker is listening to that queue
   prefect worker start --pool default-worker-pool --work-queue training-queue
   ```

2. **Work pool mismatch:**
   - Deployment uses `work_pool_name="default-worker-pool"`
   - Worker must use same pool: `--pool default-worker-pool`

3. **Schedule hasn't triggered yet:**
   - Check next scheduled run time in Prefect UI
   - Wait for next scheduled interval

### Issue 4: Multiple Deployments Registered

**Symptoms:**
- Same deployment appears multiple times in UI
- Multiple scheduled runs executing simultaneously

**Solution:**
- Deployments are identified by `flow_name/deployment_name`
- If you run the script multiple times with same names, it updates existing deployment
- To create new deployment, change the `name` parameter in `to_deployment()`

### Issue 5: Prefect Server Port Already in Use

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

### Issue 6: Work Queue Not Found

**Symptoms:**
```
Work queue 'training-queue' not found
```

**Solution:**
```bash
# Create the queue first
prefect work-queue create training-queue

# Or list existing queues
prefect work-queue ls
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
   from entrypoint.training_prefect import training_flow
   training_flow(env="local")  # Run flow directly (no Prefect server needed)
   ```

4. **Check Deployment Configuration:**
   ```bash
   prefect deployment inspect training-flow/training-2m
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
        print(f"Training failed: {e}")
        raise
```

### Conditional Execution

Skip training if conditions not met:

```python
@flow(name="training-flow")
def training_flow(env: str = "local"):
    # Check if training should run (e.g., new data available)
    should_train = check_training_condition()
    
    if should_train:
        training_task(env=env)
    else:
        print("Skipping training: conditions not met")
```

## Part 10: Best Practices

### 1. Use Descriptive Names

```python
# Good
deployment = training_flow.to_deployment(
    name="training-production-daily",
    ...
)

# Bad
deployment = training_flow.to_deployment(
    name="train",
    ...
)
```

### 2. Set Appropriate Schedules

- **Training**: Less frequent (minutes to hours)
- **Inference**: More frequent (seconds to minutes)
- **Balance**: Consider resource usage vs. freshness

### 3. Monitor Resource Usage

- Check CPU/memory usage of workers
- Adjust number of workers based on load
- Use separate workers for heavy (training) vs. light (inference) workloads

### 4. Version Control

- Commit Prefect scripts to git
- Document schedule changes in commit messages
- Keep deployment names consistent across environments

### 5. Environment Separation

Use different work pools for different environments:

```bash
# Development
prefect work-pool create dev-worker-pool --type process

# Production
prefect work-pool create prod-worker-pool --type process
```

### 6. Logging and Monitoring

- Monitor Prefect UI regularly
- Set up alerts for failed runs (if Prefect Cloud used)
- Review logs for patterns or issues

### 7. Clean Up Old Deployments

```bash
# List deployments
prefect deployment ls

# Delete unused deployment
prefect deployment delete <flow-name>/<deployment-name>
```

### 8. Test Before Production

- Test deployments locally first
- Verify schedules work as expected
- Check resource requirements
- Validate error handling

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

### Docker Deployment

For containerized deployments:
- Run Prefect server in container
- Run workers in separate containers
- Use Docker Compose for orchestration
- Mount volumes for persistent storage

### Security

- Use environment variables for secrets
- Don't commit credentials
- Use Prefect blocks for secure secret storage
- Restrict API access in production

## Part 12: Quick Reference

### Common Commands

```bash
# Server
prefect server start

# Work Queues
prefect work-queue create <queue-name>
prefect work-queue ls
prefect work-queue inspect <queue-name>

# Work Pools
prefect work-pool create <pool-name> --type process
prefect work-pool ls

# Workers
export PREFECT_API_URL=http://127.0.0.1:4200/api
prefect worker start --pool <pool-name> --work-queue <queue-name>

# Deployments
prefect deployment ls
prefect deployment inspect <flow-name>/<deployment-name>
prefect deployment delete <flow-name>/<deployment-name>

# Flow Runs
prefect flow-run ls
prefect flow-run inspect <flow-run-id>
```

### File Locations

- **Prefect Scripts**: `entrypoint/training_prefect.py`, `entrypoint/inference_prefect.py`
- **Prefect Database**: `~/.prefect/prefect.db` (SQLite default)
- **Prefect UI**: http://localhost:4200
- **MLflow Data**: `mlflow/mlruns/` (if using file-based)

### Key URLs

- **Prefect UI**: http://localhost:4200
- **Prefect API**: http://localhost:4200/api
- **MLflow UI** (if running): http://localhost:5001

## Summary

You've learned how to:
- ✅ Understand Prefect concepts (flows, tasks, deployments, workers)
- ✅ Set up Prefect server and workers
- ✅ Create and register deployments with schedules
- ✅ Monitor pipeline execution through Prefect UI
- ✅ Troubleshoot common configuration issues
- ✅ Apply best practices for production use

Your training pipeline now runs every 2 minutes, and inference runs every 10 seconds, all orchestrated by Prefect!

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


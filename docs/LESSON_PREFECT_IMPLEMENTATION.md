# Lesson: Prefect Implementation Guide

## Overview

This lesson provides a practical guide for implementing Prefect orchestration in this project using `flow.serve()`. You'll learn:
- How to test flows locally before deployment
- How to create and deploy Prefect flows using `flow.serve()`
- The complete setup workflow from testing to production
- How to manage workers and deployments

## Current Implementation Approach

This project uses **`flow.serve()`** for deployment, which:
- ✅ Registers the deployment with Prefect server
- ✅ Runs continuously to manage scheduled executions
- ✅ Works without complex storage configuration
- ✅ Perfect for Docker Compose and VM deployments

## Project Structure

Your Prefect flows are located in:
```
entrypoint/prefect/
├── training_flow.py    # Training pipeline orchestration
└── inference_flow.py   # Inference pipeline orchestration
```

## Part 1: Understanding the Flow Structure

### Example: Training Flow (`entrypoint/prefect/training_flow.py`)

```python
"""Test script to run training flow with deployment."""

import os
import sys
from datetime import timedelta
from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from prefect import flow, task

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
os.chdir(project_root)


@task(name="training-task")
def training_task(env: str = "local", pipeline_name: str = "training"):
    """Prefect task wrapper."""
    package_name = "ml_app_wind_draft"
    configure_project(package_name)
    bootstrap_project(project_root)

    with KedroSession.create(project_path=project_root, env=env) as session:
        session.run(pipeline_name=pipeline_name)


@flow(name="training-flow")
def training_flow(env: str = "local"):
    """Prefect flow for training."""
    training_task(env=env)


if __name__ == "__main__":
    os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")

    training_flow.serve(
        name="training-flow",
        interval=timedelta(minutes=1),
        parameters={"env": os.getenv("KEDRO_ENV", "local")},
    )

# if __name__ == "__main__":
#     os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
#     training_flow(env="local")
```

### Key Components Explained

1. **Path Setup**: `parents[2]` resolves to project root from `entrypoint/prefect/training_flow.py`
2. **Task Definition**: `@task` decorator wraps your Kedro pipeline execution
3. **Flow Definition**: `@flow` decorator orchestrates the task
4. **Deployment**: `flow.serve()` registers and serves the deployment
5. **Testing**: Commented section for direct flow execution (testing)

## Part 2: Testing Flows Locally (Before Deployment)

### Step 1: Comment Out the Deployment, Uncomment the Test Section

Before deploying, always test your flow locally first:

```python
# Comment out the deployment section
# if __name__ == "__main__":
#     os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")
#     training_flow.serve(...)

# Uncomment the test section
if __name__ == "__main__":
    os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
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

## Part 3: Deploying with Prefect

### Step 1: Restore the Deployment Code

After testing, restore the deployment code:

```python
if __name__ == "__main__":
    os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")

    training_flow.serve(
        name="training-flow",
        interval=timedelta(minutes=1),
        parameters={"env": os.getenv("KEDRO_ENV", "local")},
    )

# Comment out the test section again
# if __name__ == "__main__":
#     os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
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

## Part 4: Complete Workflow Summary

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

## Part 5: Configuration Options

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

## Part 6: Monitoring and Management

### Prefect UI

Open in browser: **http://localhost:4200**

**Key sections:**
- **Deployments**: View all registered deployments
- **Flow Runs**: See execution history and logs
- **Work Pools**: Check worker pool status
- **Work Queues**: Monitor queue status and pending runs

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

## Part 7: Docker Compose Deployment

For Docker deployments, the setup is similar but uses service names:

### Example: Training Flow for Docker

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

## Part 8: Troubleshooting

### Issue: Flow Not Executing

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

### Issue: Import Errors

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

### Issue: Prefect Server Connection Refused

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

### Issue: Multiple Workers/Duplicate Executions

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

## Part 9: Best Practices

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
name="training-daily-production"
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

## Part 10: Quick Reference

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

## Summary

This implementation uses `flow.serve()` for a simple, effective deployment:
1. **Test locally first** using the commented test section
2. **Deploy** using `flow.serve()` after testing
3. **Manage** via Prefect UI and CLI commands
4. **Scale** by adding more workers as needed

The approach works seamlessly with Docker Compose and VM deployments, requiring minimal configuration while providing full orchestration capabilities.


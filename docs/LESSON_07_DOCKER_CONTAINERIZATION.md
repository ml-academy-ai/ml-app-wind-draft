# Lesson 7: Docker Containerization

## Overview

This lesson covers how to containerize your ML application using Docker and Docker Compose. You'll learn how to:
- Create a Dockerfile for your application
- Set up Docker Compose for multi-service orchestration
- Run MLflow, Kedro Viz, training, inference, and UI services in containers
- Manage volumes and networking between services

## Prerequisites

- Docker Desktop installed and running
- Basic understanding of Docker concepts (containers, images, volumes)
- Your ML application is working locally

## Part 1: Understanding the Docker Setup

### Architecture Overview

Our Docker setup consists of 6 services:

1. **mlflow**: MLflow tracking server for experiment tracking and model registry
2. **app-ml-train**: Trains the ML model (runs once, then stops)
3. **app-ml-inference**: Runs inference on new data (monitors for new data continuously)
4. **app-stream-data**: Streams data to database for real-time inference
5. **app-ui**: Web dashboard for visualizing predictions and model metrics
6. **kedro-viz**: Pipeline visualization tool

### Key Concepts

- **Volumes**: Shared folders between host and containers (persist data)
- **Ports**: Expose container ports to host machine (access services)
- **Networks**: Allow services to communicate using service names
- **Depends_on**: Service startup order dependencies

## Part 2: Creating the Dockerfile

### Step 1: Understand the Base Image

Our Dockerfile uses `python:3.12-slim` as the base image:
- Lightweight Debian-based image
- Python 3.12 pre-installed
- Suitable for production deployments

### Step 2: Install System Dependencies

```dockerfile
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
```

**Why?** Some Python packages (like `catboost`, `torch`) require compilation tools.

### Step 3: Install uv Package Manager

```dockerfile
RUN pip install --no-cache-dir uv
```

**Why uv?** 
- Faster than pip for dependency resolution
- Better lock file support
- Handles virtual environments automatically

### Step 4: Copy Dependency Files

```dockerfile
COPY pyproject.toml uv.lock ./
```

**Why both?**
- `pyproject.toml`: Project metadata and dependencies
- `uv.lock`: Locked versions for reproducible builds

### Step 5: Copy Entire Project

```dockerfile
COPY . .
```

**Important:** This must be done BEFORE `uv sync` because:
- `pyproject.toml` uses `dynamic = ["version"]`
- Version is read from `src/ml_app_wind_draft/__init__.py`
- Source code must be present for version detection

### Step 6: Install Python Dependencies

```dockerfile
RUN uv sync --frozen --no-dev
```

**Flags explained:**
- `--frozen`: Use exact versions from `uv.lock` (reproducible)
- `--no-dev`: Skip development dependencies (smaller image)

### Step 7: Set Environment Variables

```dockerfile
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV PATH="/app/.venv/bin:$PATH"
```

**Why?**
- `PYTHONPATH`: Allows imports from `src/` directory
- `PATH`: Makes `python`, `kedro`, etc. available in PATH

## Part 3: Understanding Docker Compose Files

### Step 1: Local vs Production Docker Compose

This project has **two** Docker Compose files for different use cases:

**1. `docker-compose.local.yml` - Local Development**
- Builds images from source code
- Use when developing and testing locally
- Changes to source code require rebuilding
- Command: `docker compose -f docker-compose.local.yml up`

**2. `docker-compose.yml` - Production Deployment**
- Pulls pre-built images from Docker Hub
- Use for production deployments
- Requires environment variables to be set
- Command: `docker compose up`

**Key Differences:**

| Feature | Local (`docker-compose.local.yml`) | Production (`docker-compose.yml`) |
|---------|-----------------------------------|-----------------------------------|
| Image Source | Builds from source | Pulls from Docker Hub |
| Environment Variables | Optional (has defaults) | Required |
| Use Case | Development, testing | Production, CI/CD |
| Speed | Slower (builds first) | Faster (pulls pre-built) |

### Step 2: Service Structure

Each service in Docker Compose has:
- `build` or `image`: How to build or which image to use
- `command`: What command to run
- `volumes`: What directories to mount
- `ports`: What ports to expose
- `environment`: Environment variables
- `networks`: Which network to join
- `depends_on`: What services to wait for

### Step 3: Environment Variables Setup

**For Production (`docker-compose.yml`):**

You need to set environment variables before running Docker Compose. There are two methods:

**Method 1: Create a `.env` file (Recommended)**

Create a `.env` file in the project root:

```bash
# .env file
DOCKERHUB_USERNAME=your-dockerhub-username
MLFLOW_UI_URI=http://your-server-ip:5001
KEDRO_VIZ_URI=http://your-server-ip:4141
```

**Method 2: Export in Shell**

Export variables in your terminal session:

```bash
export DOCKERHUB_USERNAME=your-dockerhub-username
export MLFLOW_UI_URI=http://your-server-ip:5001
export KEDRO_VIZ_URI=http://your-server-ip:4141
```

**Why these variables?**
- `DOCKERHUB_USERNAME`: Used to pull images from Docker Hub (e.g., `username/ml-app-wind-draft:latest`)
- `MLFLOW_UI_URI`: External URL for MLflow UI (used by app-ui service for links)
- `KEDRO_VIZ_URI`: External URL for Kedro Viz (used by app-ui service for iframe embedding)

**Important:** The `environment:` section in `docker-compose.yml` passes these variables **into** the container. The application code reads them using `os.getenv()`.

### Step 4: Volume Mounts

**Why mount volumes?**
- Persist data between container restarts
- Share data between services
- Access files from host machine

**Common mounts:**
```yaml
volumes:
  - ./data:/app/data           # Data and database
  - ./conf:/app/conf           # Configuration files
  - ./mlflow:/app/mlflow       # MLflow artifacts
```

**Note:** Production compose does NOT mount source code (it's in the image). Local compose can optionally mount source for development.

### Step 5: Network Configuration

All services use the `ml-app-network` bridge network:
- Services can communicate using service names (e.g., `http://mlflow:5001`)
- Isolated from other Docker networks
- No need to expose internal ports

### Step 6: Service Dependencies

**Startup order:**
1. `mlflow` starts first (no dependencies)
2. Other services wait for `mlflow` via `depends_on`
3. `app-ml-train` runs once, then `app-ml-inference` can start

## Part 4: MLflow Configuration

### Step 1: MLflow Server Command

```yaml
command: >
  sh -c "mlflow server 
  --host 0.0.0.0 
  --port 5001 
  --default-artifact-root file:///app/mlflow/mlartifacts
  --serve-artifacts
  --allowed-hosts mlflow,mlflow:5001,localhost,localhost:5001,127.0.0.1,127.0.0.1:5001,0.0.0.0,0.0.0.0:5001
  --cors-allowed-origins '*'"
```

**Key flags:**
- `--host 0.0.0.0`: Listen on all interfaces (required for Docker)
- `--port 5001`: Use port 5001 (5000 conflicts with macOS AirPlay)
- `--default-artifact-root`: Where to store model artifacts
- `--allowed-hosts`: Fixes "Invalid Host header" errors in Docker
- `--cors-allowed-origins`: Allows cross-origin requests

### Step 2: File-Based Storage

We use file-based storage only (no database):
- Simpler setup
- All data in `./mlflow/` directory
- Easy to backup and restore

### Step 3: Connecting to MLflow

Services connect using:
```yaml
environment:
  - MLFLOW_TRACKING_URI=http://mlflow:5001
```

**Why `mlflow:5001`?** Docker Compose creates DNS entries for service names.

## Part 5: Running Services

### Step 1: Local Development Setup

**For local development using `docker-compose.local.yml`:**

1. **Navigate to project directory:**
   ```bash
   cd /path/to/ml-app-wind-draft
   ```

2. **Build and start services:**
   ```bash
   docker compose -f docker-compose.local.yml up --build
   ```

   **What happens:**
   - Builds Docker images from source code
   - Installs all dependencies
   - Starts all services in dependency order
   - Shows logs from all services

3. **Start in background:**
   ```bash
   docker compose -f docker-compose.local.yml up --build -d
   ```

4. **Access services:**
   - MLflow UI: http://localhost:5001
   - App UI: http://localhost:8050
   - Kedro Viz: http://localhost:4141

### Step 2: Production Deployment Setup

**For production using `docker-compose.yml`:**

1. **Set up environment variables (choose one method):**

   **Option A: Create `.env` file (Recommended)**
   ```bash
   cd /path/to/ml-app-wind-draft
   cat > .env << EOF
   DOCKERHUB_USERNAME=your-dockerhub-username
   MLFLOW_UI_URI=http://your-server-ip:5001
   KEDRO_VIZ_URI=http://your-server-ip:4141
   EOF
   ```

   **Option B: Export in shell**
   ```bash
   export DOCKERHUB_USERNAME=your-dockerhub-username
   export MLFLOW_UI_URI=http://your-server-ip:5001
   export KEDRO_VIZ_URI=http://your-server-ip:4141
   ```

2. **Pull latest images from Docker Hub:**
   ```bash
   docker compose pull
   ```

3. **Start all services:**
   ```bash
   docker compose up -d
   ```

   **What happens:**
   - Pulls pre-built images from Docker Hub
   - Starts all services in dependency order
   - Runs in background (detached mode)

4. **Verify services are running:**
   ```bash
   docker compose ps
   ```

5. **Access services:**
   - MLflow UI: http://your-server-ip:5001
   - App UI: http://your-server-ip:8050
   - Kedro Viz: http://your-server-ip:4141

### Step 3: View Logs

```bash
# All services
docker compose logs

# Specific service
docker compose logs app-ui

# Follow logs (like tail -f)
docker compose logs -f app-ui
```

### Step 4: Start Specific Services

**Local development:**
```bash
# Start only MLflow
docker compose -f docker-compose.local.yml up mlflow

# Start MLflow and UI
docker compose -f docker-compose.local.yml up mlflow app-ui
```

**Production:**
```bash
# Start only MLflow
docker compose up -d mlflow

# Start MLflow and UI
docker compose up -d mlflow app-ui
```

### Step 5: Stop Services

```bash
# Stop all services (local)
docker compose -f docker-compose.local.yml down

# Stop all services (production)
docker compose down

# Stop and remove volumes (deletes data!)
docker compose down -v
```

## Part 6: Service-Specific Configuration

### Training Service (app-ml-train)

**Characteristics:**
- Runs once, then stops (`restart: "no"`)
- Trains model and saves to MLflow
- Must complete before inference can run

**Run manually:**
```bash
docker-compose up app-ml-train
```

### Inference Service (app-ml-inference)

**Characteristics:**
- Runs continuously (`restart: unless-stopped`)
- Monitors for new data every 5 seconds
- Runs inference pipeline when new data detected
- Initializes predictions table automatically

**Important:** Ensure predictions table is initialized in the inference node:
```python
# In src/ml_app_wind_draft/pipelines/inference/nodes.py
data_manager.init_predictions_db_table()
```

### Data Streaming Service (app-stream-data)

**Characteristics:**
- Streams data point-by-point to database
- Simulates real-time data ingestion
- Required for inference to have data to process

### UI Service (app-ui)

**Characteristics:**
- Web dashboard on port 8050
- Auto-refreshes every 5 seconds
- Displays predictions, errors, and model info

**Access:** http://localhost:8050

### Kedro Viz Service (kedro-viz)

**Characteristics:**
- Pipeline visualization on port 4141
- Shows data pipeline structure
- Uses `uv run kedro viz` command

**Access:** http://localhost:4141

## Part 7: Common Workflows

### Workflow 1: Full Pipeline (Training + Inference)

```bash
# Start MLflow
docker-compose up -d mlflow

# Run training (waits for MLflow)
docker-compose up app-ml-train

# Start inference (waits for training)
docker-compose up -d app-ml-inference

# Start data streaming
docker-compose up -d app-stream-data

# Start UI
docker-compose up app-ui
```

### Workflow 2: Development Mode

```bash
# Start only infrastructure
docker-compose up -d mlflow kedro-viz

# Run training/inference locally
kedro run --pipeline training
kedro run --pipeline inference

# Access services
# MLflow: http://localhost:5001
# Kedro Viz: http://localhost:4141
```

### Workflow 3: Production Mode

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## Part 8: Troubleshooting

### Issue 1: "Invalid Host header" in MLflow

**Solution:** Already fixed with `--allowed-hosts` flag in docker-compose.yml

### Issue 2: Port Already in Use

**Error:** `Bind for 0.0.0.0:5001 failed: port is already allocated`

**Solution:**
```bash
# Find process using port
lsof -i :5001

# Kill process or change port in docker-compose.yml
```

### Issue 3: "No such table: predictions"

**Solution:** Ensure `init_predictions_db_table()` is called in inference node before saving predictions.

### Issue 4: Services Can't Connect to MLflow

**Check:**
1. MLflow service is running: `docker-compose ps`
2. Network is correct: `networks: - ml-app-network`
3. Service name is correct: `http://mlflow:5001` (not `localhost`)

### Issue 5: Volume Permissions

**Issue:** Container can't write to mounted volumes

**Solution:**
```bash
# Fix permissions (macOS/Linux)
chmod -R 777 ./data ./mlflow

# Or run container with user ID
# Add to docker-compose.yml:
user: "${UID}:${GID}"
```

### Issue 6: Build Fails with "ModuleNotFoundError"

**Cause:** Source code not copied before `uv sync`

**Solution:** Ensure `COPY . .` comes before `RUN uv sync` in Dockerfile

### Issue 7: Kedro Command Not Found

**Error:** `exec /app/.venv/bin/kedro: no such file or directory`

**Solution:** Use `uv run kedro` instead of `kedro` directly:
```yaml
command: ["uv", "run", "kedro", "viz", "--host", "0.0.0.0", "--port", "4141"]
```

## Part 9: Best Practices

### 1. Use .dockerignore

Create `.dockerignore` to exclude unnecessary files:
```
.git
.venv
__pycache__
*.pyc
.pytest_cache
notebooks/.ipynb_checkpoints
mlflow/
data/
```

### 2. Multi-Stage Builds (Optional)

For smaller images, use multi-stage builds:
```dockerfile
# Build stage
FROM python:3.12-slim as builder
# ... install dependencies ...

# Runtime stage
FROM python:3.12-slim
COPY --from=builder /app/.venv /app/.venv
# ... copy application ...
```

### 3. Health Checks

Add health checks for services:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8050"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### 4. Resource Limits

Set resource limits for production:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

### 5. Environment Variables

Use `.env` file for sensitive data:
```bash
# .env
MLFLOW_TRACKING_URI=http://mlflow:5001
KEDRO_ENV=local
```

Then in docker-compose.yml:
```yaml
environment:
  - MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
```

## Part 10: Useful Commands Reference

### Docker Compose Commands

```bash
# Build images
docker-compose build

# Start services
docker-compose up
docker-compose up -d              # Background
docker-compose up <service>       # Specific service

# Stop services
docker-compose down
docker-compose stop               # Stop but don't remove
docker-compose restart <service>  # Restart service

# View logs
docker-compose logs
docker-compose logs -f            # Follow logs
docker-compose logs <service>     # Specific service

# Execute commands
docker-compose exec <service> <command>
docker-compose exec app-ui python --version

# Check status
docker-compose ps                 # Running services
docker-compose ps -a               # All services

# Rebuild and restart
docker-compose up --build          # Rebuild and start
docker-compose up --build <service> # Rebuild specific service
```

### Docker Commands

```bash
# List containers
docker ps
docker ps -a

# List images
docker images

# Remove containers
docker rm <container_id>
docker rm -f <container_id>       # Force remove

# Remove images
docker rmi <image_id>

# Clean up
docker system prune               # Remove unused resources
docker volume prune                # Remove unused volumes
```

## Summary

You've learned how to:
- ✅ Create a Dockerfile for your ML application
- ✅ Set up Docker Compose for multi-service orchestration
- ✅ Configure MLflow, Kedro Viz, and application services
- ✅ Manage volumes, networks, and dependencies
- ✅ Run services individually or together
- ✅ Troubleshoot common Docker issues

**Next Steps:**
- Experiment with different service combinations
- Add health checks and resource limits
- Set up CI/CD to build and deploy Docker images
- Consider using Docker Swarm or Kubernetes for production


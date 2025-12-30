# Lesson: Continuous Deployment to DigitalOcean

## Overview

This lesson covers how to deploy your ML application to a DigitalOcean droplet using Docker and Docker Compose. You'll learn how to:
- Set up DockerHub for container image storage
- Configure GitHub Actions for automated builds
- Create and configure a DigitalOcean droplet
- Deploy your application manually using Docker Compose

## Prerequisites

- GitHub repository with your ML application
- Docker Desktop installed locally (for testing)
- Basic understanding of Docker and Docker Compose
- SSH access to a server (DigitalOcean droplet)

---

## Part 1: Setting Up DockerHub and GitHub Actions

### Step 1: Create DockerHub Account

1. Go to [DockerHub](https://hub.docker.com/)
2. Click **Sign Up** and create a new account
3. Choose a username (remember this - you'll need it later)
4. Verify your email address

**Important:** Write down your DockerHub username and password. You'll need these for authentication.

### Step 2: Create DockerHub Access Token

DockerHub tokens are more secure than passwords for automated workflows:

1. Log in to DockerHub
2. Click on your profile icon (top right) → **Account Settings**
3. Navigate to **Security** → **New Access Token**
4. Give it a name (e.g., "GitHub Actions")
5. Set permissions to **Read & Write**
6. Click **Generate**
7. **Copy the token immediately** - you won't be able to see it again!

**Security Note:** Treat this token like a password. Never commit it to your repository.

### Step 3: Add Secrets to GitHub Repository

GitHub Actions needs your DockerHub credentials to push images:

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add the following secrets:

   **Secret 1: DOCKERHUB_USERNAME**
   - Name: `DOCKERHUB_USERNAME`
   - Value: Your DockerHub username

   **Secret 2: DOCKERHUB_TOKEN**
   - Name: `DOCKERHUB_TOKEN`
   - Value: The access token you created in Step 2

### Step 4: Create GitHub Actions Workflow

Create a new workflow file for Continuous Deployment:

1. In your repository, create `.github/workflows/` directory if it doesn't exist
2. Create a new file: `.github/workflows/cd.yml`

### Step 5: Configure the CD Workflow

Add the following content to `cd.yml`:

```yaml
name: Continuous Deployment

on:
  push:
    branches:
      - main  # Trigger on pushes to main branch

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to DockerHub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ secrets.DOCKERHUB_USERNAME }}/ml-app-wind-draft:latest
        cache-from: type=registry,ref=${{ secrets.DOCKERHUB_USERNAME }}/ml-app-wind-draft:buildcache
        cache-to: type=registry,ref=${{ secrets.DOCKERHUB_USERNAME }}/ml-app-wind-draft:buildcache,mode=max
```

### Understanding the Workflow

Let's break down each section:

**Workflow Triggers:**
```yaml
on:
  push:
    branches:
      - main
```
- Runs automatically when code is pushed to the `main` branch
- You can add more branches or trigger conditions as needed

**Job: build-and-push**
```yaml
jobs:
  build-and-push:
    runs-on: ubuntu-latest
```
- Defines a job that runs on Ubuntu
- This job will build and push your Docker image

**Step 1: Checkout code**
```yaml
- name: Checkout code
  uses: actions/checkout@v4
```
- Downloads your repository code to the GitHub Actions runner

**Step 2: Set up Docker Buildx**
```yaml
- name: Set up Docker Buildx
  uses: docker/setup-buildx-action@v3
```
- Sets up Docker Buildx for advanced building features
- Enables caching and multi-platform builds

**Step 3: Log in to DockerHub**
```yaml
- name: Log in to DockerHub
  uses: docker/login-action@v3
  with:
    username: ${{ secrets.DOCKERHUB_USERNAME }}
    password: ${{ secrets.DOCKERHUB_TOKEN }}
```
- Authenticates with DockerHub using your secrets
- `${{ secrets.DOCKERHUB_USERNAME }}` references the secret you created
- `${{ secrets.DOCKERHUB_TOKEN }}` uses your access token

**Step 4: Build and push Docker image**
```yaml
- name: Build and push Docker image
  uses: docker/build-push-action@v5
  with:
    context: .
    push: true
    tags: ${{ secrets.DOCKERHUB_USERNAME }}/ml-app-wind-draft:latest
    cache-from: type=registry,ref=${{ secrets.DOCKERHUB_USERNAME }}/ml-app-wind-draft:buildcache
    cache-to: type=registry,ref=${{ secrets.DOCKERHUB_USERNAME }}/ml-app-wind-draft:buildcache,mode=max
```

**Parameters explained:**
- `context: .` - Build context is the repository root (where Dockerfile is)
- `push: true` - Automatically push the image to DockerHub after building
- `tags: ...` - Tags the image with your DockerHub username and `latest` tag
- `cache-from/cache-to` - Uses DockerHub registry for build cache (faster builds)

### Step 6: Commit and Push

1. Add the workflow file:
   ```bash
   git add .github/workflows/cd.yml
   ```

2. Commit the changes:
   ```bash
   git commit -m "Add CD workflow for DockerHub image building"
   ```

3. Push to trigger the workflow:
   ```bash
   git push origin main
   ```

4. Check the workflow status:
   - Go to your GitHub repository
   - Click on **Actions** tab
   - You should see the workflow running
   - Once complete, verify the image in DockerHub

### Verifying the Build

1. Go to [DockerHub](https://hub.docker.com/)
2. Log in and navigate to your repositories
3. You should see `ml-app-wind-draft` repository
4. Click on it to see the `latest` tag

**Troubleshooting:**
- If the workflow fails, check the Actions tab for error messages
- Verify your secrets are correctly set in GitHub
- Ensure your Dockerfile is in the repository root
- Check that your DockerHub username matches the secret

---

## Part 2: Setting Up DigitalOcean and Manual Deployment

### Step 1: Move EDA Libraries to EDA Group (Optional)

If you have EDA (Exploratory Data Analysis) dependencies that are only needed for notebooks:

1. Check your `pyproject.toml` for EDA dependencies
2. Ensure they're in the `[project.optional-dependencies.eda]` section
3. This keeps the Docker image smaller for production

Example:
```toml
[project.optional-dependencies]
eda = [
    "ipython>=8.10",
    "jupyterlab>=4.5.0",
    "matplotlib==3.8.2",
    "seaborn==0.13.1",
    # ... other EDA dependencies
]
```

### Step 2: Generate SSH Key

You'll need an SSH key to securely connect to your DigitalOcean droplet:

1. **Check if you already have an SSH key:**
   ```bash
   ls -la ~/.ssh/
   ```
   Look for `id_ed25519` or `id_rsa` files

2. **If you don't have one, generate a new key:**
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```
   - Press Enter to accept default location
   - Optionally set a passphrase (recommended for security)

3. **Display your public key:**
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
   - Copy the entire output (starts with `ssh-ed25519`)

4. **Display your private key (for GitHub secret):**
   ```bash
   cat ~/.ssh/id_ed25519
   ```
   - Copy the entire output (starts with `-----BEGIN OPENSSH PRIVATE KEY-----`)
   - **Keep this secret!** Never share it publicly

### Step 3: Add SSH Key to GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add:
   - Name: `DO_SSH_KEY`
   - Value: Paste your **private key** (from `cat ~/.ssh/id_ed25519`)

### Step 4: Create DigitalOcean Account and Droplet

1. **Sign up for DigitalOcean:**
   - Go to [DigitalOcean](https://www.digitalocean.com/)
   - Create an account (you may get free credits)

2. **Create a new Droplet:**
   - Click **Create** → **Droplets**
   - Choose **Ubuntu 22.04 (LTS)**
   - Select a plan (minimum 2GB RAM recommended for Docker)
   - Choose a datacenter region
   - Under **Authentication**, select **SSH keys**
   - Click **New SSH Key**
   - Paste your **public key** (from `cat ~/.ssh/id_ed25519.pub`)
   - Give it a name (e.g., "My Laptop")
   - Click **Create Droplet**

3. **Wait for droplet creation** (usually 1-2 minutes)

4. **Note the IP address:**
   - Once created, you'll see the droplet's IP address
   - Example: `139.59.86.152`
   - Copy this IP address

### Step 5: Add DigitalOcean Secrets to GitHub

1. Go to your GitHub repository → **Settings** → **Secrets and variables** → **Actions**

2. Add the following secrets:

   **Secret 1: DO_HOST**
   - Name: `DO_HOST`
   - Value: Your droplet's IP address (e.g., `139.59.86.152`)

   **Secret 2: DO_USER**
   - Name: `DO_USER`
   - Value: `root` (default user for DigitalOcean droplets)

### Step 6: Test SSH Connection

Test that you can connect to your droplet:

```bash
ssh root@YOUR_DROPLET_IP
```

Replace `YOUR_DROPLET_IP` with your actual IP (e.g., `ssh root@139.59.86.152`)

**First connection:**
- You'll see a message about host authenticity - type `yes`
- If successful, you'll see the Ubuntu welcome message

**If connection fails:**
- Verify your public key was added correctly in DigitalOcean
- Check that your droplet is running
- Ensure your IP address is correct

### Step 7: Install Docker on DigitalOcean Droplet

Follow the official DigitalOcean guide to install Docker:

1. **SSH into your droplet:**
   ```bash
   ssh root@YOUR_DROPLET_IP
   ```

2. **Update package index:**
   ```bash
   apt update
   ```

3. **Install prerequisites:**
   ```bash
   apt install -y ca-certificates curl gnupg lsb-release
   ```

4. **Add Docker's official GPG key:**
   ```bash
   install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   chmod a+r /etc/apt/keyrings/docker.gpg
   ```

5. **Set up Docker repository:**
   ```bash
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

6. **Install Docker:**
   ```bash
   apt update
   apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

7. **Verify Docker installation:**
   ```bash
   docker --version
   docker compose version
   ```

8. **Remove sudo requirement (optional but recommended):**
   ```bash
   usermod -aG docker $USER
   ```
   - Log out and log back in for this to take effect
   - Or use `newgrp docker` to apply immediately

9. **Test Docker without sudo:**
   ```bash
   docker run hello-world
   ```

**Reference:** [DigitalOcean Docker Installation Guide](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-22-04)

### Step 8: Clone Repository and Deploy

1. **Navigate to `/opt` directory:**
   ```bash
   cd /opt
   ```

2. **Clone your repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   ```
   Replace with your actual GitHub repository URL

   **If your repo is private:**
   - Use SSH: `git clone git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git`
   - Or use a personal access token in the URL

3. **Navigate to the repository:**
   ```bash
   cd YOUR_REPO_NAME
   ```

4. **Build and start services:**
   ```bash
   docker compose up --build
   ```

   **What this does:**
   - `--build` - Rebuilds all images before starting
   - Builds all services defined in `docker-compose.yml`
   - Starts services in the correct order based on dependencies
   - Shows logs from all services

5. **Run in detached mode (background):**
   ```bash
   docker compose up --build -d
   ```
   - `-d` flag runs containers in the background
   - Services continue running after you disconnect

### Step 9: Verify Deployment

1. **Check running containers:**
   ```bash
   docker compose ps
   ```

2. **View logs:**
   ```bash
   docker compose logs
   docker compose logs -f  # Follow logs in real-time
   docker compose logs mlflow  # Specific service
   ```

3. **Test services:**
   - **MLflow UI:** `http://YOUR_DROPLET_IP:5001`
   - **App UI:** `http://YOUR_DROPLET_IP:8050`
   - **Kedro Viz:** `http://YOUR_DROPLET_IP:4141`

4. **Check if ports are accessible:**
   ```bash
   curl http://localhost:5001/health  # MLflow health check
   ```

### Step 10: Configure Firewall (Important!)

DigitalOcean droplets have a firewall that may block incoming connections:

1. **Check current firewall status:**
   ```bash
   ufw status
   ```

2. **Allow required ports:**
   ```bash
   ufw allow 22/tcp    # SSH
   ufw allow 5001/tcp # MLflow
   ufw allow 8050/tcp # App UI
   ufw allow 4141/tcp # Kedro Viz
   ```

3. **Enable firewall:**
   ```bash
   ufw enable
   ```

4. **Or configure via DigitalOcean Dashboard:**
   - Go to **Networking** → **Firewalls**
   - Create a new firewall
   - Add inbound rules for ports 5001, 8050, 4141
   - Apply to your droplet

### Common Commands for Managing Deployment

```bash
# View running services
docker compose ps

# View logs
docker compose logs
docker compose logs -f              # Follow logs
docker compose logs app-ui          # Specific service

# Stop services
docker compose down

# Restart a service
docker compose restart app-ui

# Rebuild and restart
docker compose up --build -d

# Execute command in container
docker compose exec app-ui python --version

# View resource usage
docker stats
```

### Troubleshooting

**Issue: Cannot connect to services from browser**
- Check firewall rules (see Step 10)
- Verify services are running: `docker compose ps`
- Check logs for errors: `docker compose logs`
- Ensure ports are correctly mapped in `docker-compose.yml`

**Issue: Services fail to start**
- Check logs: `docker compose logs <service-name>`
- Verify all dependencies are met
- Check if ports are already in use: `netstat -tuln | grep <port>`
- Ensure MLflow starts before other services

**Issue: Out of memory**
- DigitalOcean droplets have limited RAM
- Check usage: `free -h`
- Consider upgrading droplet size
- Or reduce number of services running

**Issue: Cannot pull from DockerHub**
- Verify DockerHub credentials
- Check network connectivity: `ping docker.io`
- Try pulling manually: `docker pull <image>`

**Issue: Permission denied errors**
- Ensure Docker is installed correctly
- Check user is in docker group: `groups`
- Try: `newgrp docker`

## Summary

You've learned how to:
- ✅ Set up DockerHub account and access tokens
- ✅ Configure GitHub Actions for automated image building
- ✅ Create and configure a DigitalOcean droplet
- ✅ Install Docker on Ubuntu
- ✅ Deploy your application using Docker Compose
- ✅ Manage and troubleshoot your deployment

**Next Steps:**
- Set up automated deployment from GitHub Actions (Part 3)
- Configure domain names and SSL certificates
- Set up monitoring and logging
- Implement backup strategies
- Add health checks and auto-restart policies

**Security Best Practices:**
- Use SSH keys instead of passwords
- Keep Docker and system packages updated
- Use firewall rules to restrict access
- Regularly rotate access tokens
- Monitor logs for suspicious activity
- Use environment variables for sensitive data

---

## Additional Resources

- [DockerHub Documentation](https://docs.docker.com/docker-hub/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [DigitalOcean Docker Tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-22-04)
- [Docker Compose Documentation](https://docs.docker.com/compose/)


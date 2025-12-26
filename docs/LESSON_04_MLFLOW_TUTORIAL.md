# Lesson 4: MLflow Tutorial

## Step 1: Configure .gitignore

Add this to `.gitignore` to exclude MLflow local runs:

```
# mlflow local runs
mlruns/
mlartifacts/
```

**Why?**
- `mlruns/` contains MLflow experiment tracking data
- `mlartifacts/` contains model artifacts
- These are generated locally and shouldn't be committed to version control
- Each developer/CI system should have their own MLflow tracking server

## Step 2: Understanding MLflow Artifacts

**Artifact saving example:**
https://mlflow.org/blog/pyfunc-in-practice#creating-the-ensemble-model

This article demonstrates:
- How to save custom models as MLflow artifacts
- Creating PyFunc models for deployment
- Bundling model dependencies
- Model versioning and registry

**Key concepts:**
- **Artifacts**: Files associated with an MLflow run (models, plots, data)
- **PyFunc**: MLflow's generic Python function model format
- **Model Registry**: Centralized model store with versioning and stage management


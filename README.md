# MLOps Assignment 2: Olivetti Face Dataset Classifier

### 1. Project Overview
This project implements an end-to-end MLOps pipeline for classifying face images from the Olivetti dataset. The pipeline demonstrates the full lifecycle: Design, Model Development, and Operations.

### 2. Technical Architecture
* **Framework**: Python, Flask
* **Version Control**: Git/GitHub
* **Containerization**: Docker
* **Orchestration**: Kubernetes

### 3. Workflow Execution
1. **Training**: The Decision Tree algorithm is trained on the Olivetti dataset using `train.py`, generating `savedmodel.pth`.
2. **Deployment**: The application is containerized and deployed using `deployment.yaml` and `service.yaml` on Kubernetes.
3. **Inference**: Users can upload a test image to the web interface to retrieve a "Predicted Subject ID".

### 4. Pipeline Validation
* **CI/CD**: Branching strategy with `main` and `dev` branches for production and development.
* **Deployment Status**: Kubernetes pods are confirmed to be in a "Running" state.
* **Functional Test**: The model successfully returns predictions for uploaded images.

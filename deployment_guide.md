# Deployment Guide: Continuous ML & Streamlit Application

## 1. Prerequisites
- Docker & Docker Compose
- AWS CLI configured with ECR access
- Python 3.10+

## 2. Local Testing
Before deploying, verify the models work locally:
```bash
python train_model.py
streamlit run app.py
```
Check that `best_model.pkl` and `feature_meta.pkl` exist in the root directory.

## 3. Dockerizing the Application

Create a `Dockerfile` in the root:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && apt-get install -y build-essential
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build the image:
```bash
docker build -t hr-attrition-engine .
```

## 4. Production Deployment to Enterprise Cloud (AWS ECS)

1. **Tag the Docker Image**:
   ```bash
   docker tag hr-attrition-engine:latest <account_id>.dkr.ecr.<region>.amazonaws.com/hr-engine:latest
   ```
2. **Push to Elastic Container Registry (ECR)**:
   ```bash
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account_id>.dkr.ecr.<region>.amazonaws.com
   docker push <account_id>.dkr.ecr.<region>.amazonaws.com/hr-engine:latest
   ```

3. **Deploy to Fargate/ECS**:
   Update your ECS task definition to point to the new image URI and expose port 8501. Put the service behind an Application Load Balancer with HTTPS configured for secure HTTPS access.

## 5. Security & RBAC Configuration
As this contains sensitive HR PII, ensure the stream is behind a strict VPC. Integrate AWS Cognito or your Enterprise Okta SSO directly into the Load Balancer layer so only authorized HR Business Partners can access the Streamlit UI.

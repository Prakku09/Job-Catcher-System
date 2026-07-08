# Deployment Guide

This document outlines how to deploy the `Job-Catcher-System` API to production.

## 1. Local Deployment with Docker (Recommended)
You can deploy the system locally using Docker Compose, which packages the API and its environment.

```bash
# Build and run the container
docker-compose up --build -d
```
The API will be available at `http://localhost:8000`.

## 2. Local Deployment without Docker
If you prefer not to use Docker, you can run the API directly using Uvicorn.

```bash
# Install dependencies
pip install -r requirements.txt

# Start the ASGI server
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

## 3. Cloud Deployment (Active on Render)
The API is currently deployed live on Render. You can access the public endpoints here:
**[https://job-catcher-system.onrender.com](https://job-catcher-system.onrender.com)**

To deploy your own instance on a PaaS provider like Render, Railway, or Heroku:
1. Link your GitHub repository to your PaaS provider.
2. Specify the Build Command: `pip install -r requirements.txt`
3. Specify the Start Command: `uvicorn src.app:app --host 0.0.0.0 --port $PORT`
4. The provider will automatically build and expose your API on a public URL.

## Accessing the Live Demo (Swagger UI)
Once deployed (either locally or on cloud), navigate to the `/docs` endpoint (e.g., `https://job-catcher-system.onrender.com/docs`) to interact with the live Swagger UI. Here you can execute live tests against `/health`, `/metadata`, and `/predict`.

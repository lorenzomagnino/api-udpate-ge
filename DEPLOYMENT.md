# Google Cloud Deployment Guide

This guide explains how to deploy the Veyt data retrieval service to Google Cloud Platform using Cloud Run and Cloud Scheduler.

## Prerequisites

1. Google Cloud SDK (`gcloud`) installed and configured
2. Docker installed (for local testing)
3. A GCP project with billing enabled
4. Required APIs enabled:
   - Cloud Run API
   - Cloud Build API
   - Cloud Scheduler API
   - Cloud Storage API

## Setup

### 1. Enable Required APIs

```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
gcloud services enable storage-api.googleapis.com
```

### 2. Set Up Authentication

The service needs access to Google Cloud Storage. Ensure your Cloud Run service account has the necessary permissions:

```bash
# Get your project number
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format="value(projectNumber)")

# Grant Storage Admin role to Cloud Run service account
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/storage.admin"
```

### 3. Build and Deploy

#### Option A: Using Cloud Build (Recommended)

```bash
# Submit build to Cloud Build
gcloud builds submit --config cloudbuild.yaml
```

#### Option B: Manual Build and Deploy

```bash
# Set your project ID
export PROJECT_ID=$(gcloud config get-value project)

# Build the Docker image
docker build -t gcr.io/${PROJECT_ID}/veyt-data-retrieval:latest .

# Push to Container Registry
docker push gcr.io/${PROJECT_ID}/veyt-data-retrieval:latest

# Deploy to Cloud Run
gcloud run deploy veyt-data-retrieval \
    --image gcr.io/${PROJECT_ID}/veyt-data-retrieval:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --timeout 3600 \
    --max-instances 1
```

### 4. Set Up Cloud Scheduler

Create a scheduled job to run the service daily (or as needed):

```bash
# Get the Cloud Run service URL
SERVICE_URL=$(gcloud run services describe veyt-data-retrieval \
    --platform managed \
    --region us-central1 \
    --format 'value(status.url)')

# Create a Cloud Scheduler job (runs daily at 2 AM UTC)
gcloud scheduler jobs create http veyt-data-retrieval-daily \
    --location=us-central1 \
    --schedule="0 2 * * *" \
    --uri="${SERVICE_URL}" \
    --http-method=GET \
    --time-zone="UTC" \
    --description="Daily Veyt data retrieval job"
```

#### Custom Schedule Examples

```bash
# Run every 6 hours
--schedule="0 */6 * * *"

# Run every Monday at 3 AM UTC
--schedule="0 3 * * 1"

# Run at specific time (9 AM UTC daily)
--schedule="0 9 * * *"
```

### 5. Test the Deployment

```bash
# Get the service URL
SERVICE_URL=$(gcloud run services describe veyt-data-retrieval \
    --platform managed \
    --region us-central1 \
    --format 'value(status.url)')

# Test with curl
curl "${SERVICE_URL}"
```

## Configuration

### Environment Variables (Optional)

If you need to customize behavior, you can set environment variables in Cloud Run:

```bash
gcloud run services update veyt-data-retrieval \
    --platform managed \
    --region us-central1 \
    --set-env-vars="LOG_LEVEL=INFO"
```

### Custom Arguments

To pass custom arguments to the script via Cloud Scheduler, you can modify the scheduler job to use POST with a body, or update the Dockerfile CMD to include default arguments.

## Monitoring

### View Logs

```bash
# View Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=veyt-data-retrieval" \
    --limit 50 \
    --format json
```

### View Scheduler Job History

```bash
# List scheduler jobs
gcloud scheduler jobs list --location=us-central1

# View job execution history
gcloud scheduler jobs describe veyt-data-retrieval-daily --location=us-central1
```

## Troubleshooting

### Check Service Status

```bash
gcloud run services describe veyt-data-retrieval \
    --platform managed \
    --region us-central1
```

### Check Build Logs

```bash
gcloud builds list --limit=5
```

### Common Issues

1. **Authentication Errors**: Ensure the service account has Storage Admin permissions
2. **Timeout Errors**: Increase the timeout value in Cloud Run settings (max 3600s)
3. **Memory Issues**: Increase memory allocation if processing large datasets
4. **Token Cache**: The `.veyt_token_cache.json` file is created at runtime and stored in the container's ephemeral storage

## Updating the Service

To update the service after making code changes:

```bash
# Rebuild and redeploy
gcloud builds submit --config cloudbuild.yaml
```

Or manually:

```bash
docker build -t gcr.io/${PROJECT_ID}/veyt-data-retrieval:latest .
docker push gcr.io/${PROJECT_ID}/veyt-data-retrieval:latest
gcloud run deploy veyt-data-retrieval \
    --image gcr.io/${PROJECT_ID}/veyt-data-retrieval:latest \
    --platform managed \
    --region us-central1
```

## Cost Optimization

- Cloud Run only charges for actual execution time
- Set `--max-instances 1` to prevent multiple concurrent runs
- Use appropriate memory allocation (2Gi is a good starting point)
- Consider using Cloud Scheduler with appropriate frequency to balance freshness vs cost


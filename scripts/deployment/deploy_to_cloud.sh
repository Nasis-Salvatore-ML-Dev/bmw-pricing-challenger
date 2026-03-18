#!/bin/bash
# BMW Pricing API - Google Cloud Run Deployment
# No local Docker required - builds in cloud!

set -e

PROJECT_ID="portfolio-bmw-pricing-v1"
SERVICE_NAME="bmw-pricing-api"
REGION="europe-west1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "============================================================"
echo "BMW PRICING API - CLOUD DEPLOYMENT"
echo "============================================================"
echo "Project: ${PROJECT_ID}"
echo "Service: ${SERVICE_NAME}"
echo "Region:  ${REGION}"
echo ""

# Set project
gcloud config set project ${PROJECT_ID}

# Enable APIs
echo "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com

# Build in cloud (no local Docker!)
echo ""
echo "Building image in cloud (3-5 minutes)..."
gcloud builds submit --tag ${IMAGE_NAME} --timeout=10m .

# Deploy
echo ""
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 10 \
  --port 8000

# Get URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format='value(status.url)')

echo ""
echo "============================================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "============================================================"
echo ""
echo "🌐 URL: ${SERVICE_URL}"
echo ""
echo "Test:"
echo "  curl ${SERVICE_URL}/health"
echo ""
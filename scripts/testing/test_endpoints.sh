#!/bin/bash

# URL of the service
API_URL="http://localhost:8000"

echo "🧪 Starting API Smoke Tests against $API_URL..."

# 1. Health Check Loop
echo "   Waiting for service to be healthy..."
max_retries=10
count=0
while [ $count -lt $max_retries ]; do
    status=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/health)
    if [ "$status" -eq 200 ]; then
        echo "✅ Service is UP!"
        break
    fi
    echo "   ...waiting (attempt $((count+1))/$max_retries)"
    sleep 2
    count=$((count+1))
done

if [ $count -eq $max_retries ]; then
    echo "❌ Service failed to start."
    exit 1
fi

# 2. Prediction Test
echo "   Testing /predict endpoint..."
PAYLOAD='{
  "model_key": "320",
  "mileage": 50000,
  "engine_power": 135,
  "registration_date": "2018-05-01",
  "fuel": "diesel",
  "paint_color": "black",
  "car_type": "sedan"
}'

response=$(curl -s -X POST "$API_URL/api/v1/predict" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD")

echo "   Response: $response"

# Simple validation: Check if response contains "predicted_price"
if [[ "$response" == *"predicted_price"* ]]; then
    echo "✅ /predict test PASSED"
else
    echo "❌ /predict test FAILED"
    exit 1
fi
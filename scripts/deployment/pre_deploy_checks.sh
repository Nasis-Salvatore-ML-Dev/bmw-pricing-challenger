#!/bin/bash
set -e

echo "🚀 Starting Pre-Deployment Checks..."

# 1. Model Check
if [ ! -f "data/models/xgboost_pipeline.pkl" ]; then
    echo "❌ Model artifact missing!"
    exit 1
fi
echo "✅ Model exists."

# 2. Unit Tests
echo "🔍 Running Unit Tests..."
pytest tests/unit/ -v
echo "✅ Unit tests passed."

# 3. Code Quality
echo "🧹 Running Linter..."
flake8 src/
echo "✅ Linting passed."

# 4. Security Scan
echo "🔒 Running Security Scan..."
bandit -r src/ -f quiet
echo "✅ Security scan passed."

# 5. Integration Tests (Dry Run)
echo "🔌 Running Integration Tests..."
pytest tests/integration/ -v
echo "✅ Integration tests passed."

echo "========================================"
echo "✅ ALL CHECKS PASSED. Ready to Deploy."
echo "========================================"
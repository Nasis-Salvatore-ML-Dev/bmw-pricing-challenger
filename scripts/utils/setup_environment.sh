#!/bin/bash

# Script: create_env.sh
# Purpose: Create conda environment and verify critical packages

set -e  # Exit immediately on any error

ENV_NAME="tf_env_clean"
REQUIREMENTS_FILE="requirements.txt"

echo "=== Creating Conda Environment '$ENV_NAME' ==="

# 1. Check conda availability
if ! command -v conda &> /dev/null; then
    echo "ERROR: 'conda' command not found. Install Miniconda/Anaconda first."
    exit 1
fi

# 2. Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo "Removing existing environment: $ENV_NAME"
    conda remove --name $ENV_NAME --all -y
fi

# 3. Create fresh environment
echo "Creating new environment: $ENV_NAME"
conda create --name $ENV_NAME python=3.10 -y

# 4. Source conda and activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# 5. Install from requirements.txt
echo "Installing packages from $REQUIREMENTS_FILE..."
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "ERROR: $REQUIREMENTS_FILE not found in current directory."
    echo "Please create a requirements.txt file with your packages."
    exit 1
fi

pip install -r "$REQUIREMENTS_FILE"

# 6. Verify critical packages
echo ""
echo "=== Verifying Critical Packages ==="

verify_package() {
    local pkg_name=$1
    local import_cmd=$2
    
    echo -n "Testing $pkg_name... "
    python -c "$import_cmd" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ OK"
        return 0
    else
        echo "✗ FAILED"
        return 1
    fi
}

# Test each critical package
declare -A PACKAGE_TESTS=(
    ["xgboost"]="import xgboost; print('xgboost version:', xgboost.__version__)"
    ["fastapi"]="import fastapi; print('fastapi imported')"
    ["pandas"]="import pandas; print('pandas version:', pandas.__version__)"
    ["sklearn"]="import sklearn; print('sklearn version:', sklearn.__version__)"
)

FAILED_COUNT=0
for pkg in "${!PACKAGE_TESTS[@]}"; do
    if ! verify_package "$pkg" "${PACKAGE_TESTS[$pkg]}"; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

# 7. Final report
echo ""
echo "=== Summary ==="
if [ $FAILED_COUNT -eq 0 ]; then
    echo "✅ All critical packages verified successfully!"
    echo ""
    echo "To activate this environment, run:"
    echo "  conda activate $ENV_NAME"
else
    echo "❌ $FAILED_COUNT package(s) failed verification."
    echo "Check your requirements.txt and try:"
    echo "  pip install --upgrade -r $REQUIREMENTS_FILE"
    exit 1
fi

echo ""
echo "Environment ready. Current Python: $(python --version)"
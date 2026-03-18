#!/bin/bash
# Ensures that the src module is found by python
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "🧪 Running full test suite..."
pytest tests/ -v
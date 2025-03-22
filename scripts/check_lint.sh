# This scripts checks the code for linting errors
# Run this script from the root of the project
echo "Running flake8..."
flake8 .
echo "Running black..."
black --check .
echo "Running isort..."
isort --check-only .
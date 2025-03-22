# This script will autoformat all the code in the project
# Run this script from the root of the project
echo "Running black..."
black .
echo "Running isort..."
isort .
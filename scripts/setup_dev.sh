#!/bin/bash
set -e

echo "🚀 Setting up development environment for eval-annotate-scale..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment from environment.yml
echo "📦 Creating conda environment..."
conda env create -f environment.yml

echo "🔧 Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eval-annotate-scale

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Build the project with Bazel
echo "🏗️ Building project with Bazel..."
bazel build //...

# Run tests to verify setup
echo "🧪 Running tests to verify setup..."
bazel test //tests:test_annotation --test_output=errors

echo "✅ Development environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate eval-annotate-scale"
echo ""
echo "To deactivate, run:"
echo "  conda deactivate"
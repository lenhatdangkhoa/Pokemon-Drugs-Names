#!/bin/bash
# Environment setup for Pokemon experiments
# This script helps set up the required environment variables

# Get the project root directory (parent of bash directory)
POKEMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$POKEMON_DIR/.env"

echo "Pokemon-Drugs-Names Environment Setup"
echo "===================================="

# Check if .env file exists
if [[ -f "$ENV_FILE" ]]; then
    echo "Found existing .env file: $ENV_FILE"
    read -p "Overwrite existing .env file? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing .env file"
        exit 0
    fi
fi

echo "Setting up environment variables..."
echo "# Pokemon-Drugs-Names Environment Variables" > "$ENV_FILE"
echo "# Generated on $(date)" >> "$ENV_FILE"
echo "" >> "$ENV_FILE"

# OpenAI API Key
echo "OpenAI API Key:"
echo "You can find your API key at: https://platform.openai.com/api-keys"
read -p "Enter your OpenAI API key (or press Enter to skip): " OPENAI_KEY
if [[ -n "$OPENAI_KEY" ]]; then
    echo "OPENAI_API_KEY=$OPENAI_KEY" >> "$ENV_FILE"
    echo "✓ OpenAI API key configured"
else
    echo "# OPENAI_API_KEY=your_openai_api_key_here" >> "$ENV_FILE"
    echo "⚠ OpenAI API key not configured (add manually to .env file)"
fi

echo "" >> "$ENV_FILE"

# Azure OpenAI (optional)
echo "Azure OpenAI Configuration (optional):"
read -p "Do you want to configure Azure OpenAI? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter Azure OpenAI endpoint: " AZURE_ENDPOINT
    if [[ -n "$AZURE_ENDPOINT" ]]; then
        echo "AZURE_OPENAI_ENDPOINT=$AZURE_ENDPOINT" >> "$ENV_FILE"
        echo "✓ Azure OpenAI endpoint configured"
    else
        echo "# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/" >> "$ENV_FILE"
        echo "⚠ Azure OpenAI endpoint not configured"
    fi
else
    echo "# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/" >> "$ENV_FILE"
    echo "ℹ Azure OpenAI not configured (optional)"
fi

echo "" >> "$ENV_FILE"
echo "# Experiment Configuration" >> "$ENV_FILE"
echo "# Set to 'true' to enable advanced features" >> "$ENV_FILE"
echo "# RUN_GPT5=true" >> "$ENV_FILE"
echo "# RUN_AZURE_GPT5=true" >> "$ENV_FILE"
echo "# UPDATE_LABELS=true" >> "$ENV_FILE"

echo "" >> "$ENV_FILE"
echo "# VLLM Model Selection (set to 'false' to disable)" >> "$ENV_FILE"
echo "# RUN_GEMMA=true" >> "$ENV_FILE"
echo "# RUN_LLAMA=true" >> "$ENV_FILE"
echo "# RUN_QWEN=true" >> "$ENV_FILE"

echo ""
echo "Environment setup completed!"
echo "Environment file created: $ENV_FILE"
echo ""
echo "Next steps:"
echo "1. Review and edit the .env file if needed"
echo "2. Load the environment: source $ENV_FILE"
echo "3. Test with: ./bash/test_run.sh"
echo "4. Run full experiments: ./bash/run_pokemon_experiments.sh [mode]"
echo ""
echo "Available modes:"
echo "  openai    - OpenAI models only"
echo "  azure     - Azure OpenAI models only"
echo "  vllm      - Local VLLM models only"
echo "  all       - Complete pipeline"
echo "  evaluation - Generate tables from existing results"
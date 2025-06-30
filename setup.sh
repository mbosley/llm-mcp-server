#!/bin/bash

echo "Setting up LLM MCP Server..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for .env file
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys!"
else
    echo "✓ .env file already exists"
fi

# Make server executable
chmod +x server.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. Add this to your Claude Code settings (⚙️ → MCP):"
echo ""
echo '{'
echo '  "mcpServers": {'
echo '    "llm-tools": {'
echo '      "command": "python",'
echo '      "args": ["'$(pwd)'/server.py"]'
echo '    }'
echo '  }'
echo '}'
echo ""
echo "3. Restart Claude Code to load the MCP server"
echo "4. The LLM tools will be available in your sessions!"
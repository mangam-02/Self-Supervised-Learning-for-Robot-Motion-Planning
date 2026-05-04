# Self-Supervised-Learning-for-Robot-Motion-Planning

# Installation

## Requirements

For the **Recommended Workflow**, you need:

- **Python 3.11** (required for `python3.11 -m venv .venv` command)
- **Git** (for cloning the repository)
- **pip** (included with Python, for installing dependencies)
- **requirements.txt** file (present in the repository)

**Alternative:** Python 3.10 or later can be used, but Python 3.11 is specifically recommended for this project.

## Python 3.11 Setup (Recommended)

This project is tested and recommended to run with Python 3.11. Follow the platform-specific instructions below to install Python 3.11 and create a virtual environment.

### macOS (Homebrew)

Install Python 3.11 using Homebrew:

```bash
brew install python@3.11
```

Verify the installation:

```bash
python3.11 --version
```

If `python3.11` command is not found after installation, link it:

```bash
brew link python@3.11 --force
```

### Linux (Ubuntu/Debian)

Install Python 3.11 and required dependencies:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

Verify the installation:

```bash
python3.11 --version
```

### Windows

1. Download Python 3.11 from the [official Python website](https://www.python.org/downloads/release/python-3110/)
2. Run the installer
3. **Important:** Check the box "Add Python to PATH" during installation
4. Verify in Command Prompt or PowerShell:

```cmd
python --version
```

### Creating a Virtual Environment with Python 3.11

Once Python 3.11 is installed, create a virtual environment:

```bash
python3.11 -m venv .venv
```

**For Windows (if `python3.11` is not available):**

```cmd
python -m venv .venv
```

Then activate the environment:

**macOS/Linux:**

```bash
source .venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

### Fallback: Using Default Python

If Python 3.11 is not available, you can use the default Python 3.x installation:

```bash
python3 -m venv .venv
```

**Note:** This is not recommended as the environment may use a different Python version. Verify the Python version within the virtual environment after activation:

```bash
python --version
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Recommended Workflow

For a clean and consistent setup, follow these steps:

```bash
# Clone repository
git clone <repository-url>
cd Self-Supervised-Learning-for-Robot-Motion-Planning

# Create virtual environment with Python 3.11
python3.11 -m venv .venv

# Activate environment
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate            # Windows CMD
# .venv\Scripts\Activate.ps1        # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

## Deactivating the Virtual Environment

To exit the virtual environment when finished:

```bash
deactivate
```

## Troubleshooting

### Command Not Found: `python3`

- **macOS/Linux:** Use `python` instead, or install Python via Homebrew: `brew install python3`
- **Windows:** Install Python from [python.org](https://www.python.org/downloads/) and ensure "Add Python to PATH" is selected

### Permission Denied (macOS/Linux)

If you encounter permission errors, try:

```bash
chmod +x .venv/bin/activate
source .venv/bin/activate
```

### pip install Fails

- Ensure your virtual environment is activated
- Upgrade pip: `pip install --upgrade pip`
- Clear pip cache: `pip cache purge`


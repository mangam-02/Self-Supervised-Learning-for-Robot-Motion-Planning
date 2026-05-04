# Self-Supervised-Learning-for-Robot-Motion-Planning

# Installation

## Requirements

- **Python 3.10** or **3.11** (recommended)
- Git
- pip (included with Python)

## Prerequisites

Before starting, ensure you have Python installed on your system:

```bash
python3 --version
```

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Self-Supervised-Learning-for-Robot-Motion-Planning
```

### 2. Create a Virtual Environment

Create a Python virtual environment to isolate project dependencies:

```bash
python3 -m venv .venv
```

### 3. Activate the Virtual Environment

**For macOS/Linux:**

```bash
source .venv/bin/activate
```

**For Windows (Command Prompt):**

```cmd
.venv\Scripts\activate
```

**For Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the beginning of your terminal prompt when the environment is active.

### 4. Install Dependencies

Install all required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 5. Verify Installation (Optional)

Confirm that your environment is properly configured:

```bash
python --version
pip list
```

This will display your Python version and all installed packages.

## Quick Start

For a rapid setup, run these minimal commands:

```bash
git clone <repository-url>
cd Self-Supervised-Learning-for-Robot-Motion-Planning
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate            # Windows
pip install -r requirements.txt
```


### Deactivating the Virtual Environment

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


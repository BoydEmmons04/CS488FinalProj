# Project Setup Guide

This guide explains how to clone the repository, set up your local development environment, and run the project.

---

## Prerequisites

Make sure you have the following installed:

* Python 3.9 or newer
* Git
* (Optional) VS Code or another code editor
* At least 1–2 GB of free disk space

  The virtual environment and required Python dependencies typically require between 500 MB and 1.5 GB. Additional space may be needed for datasets and generated outputs.

Verify installations:

```bash
python3 --version
git --version
```

---

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

---

## 2. Create Virtual Environment

Create a local Python virtual environment:

```bash
python3 -m venv .venv
```

---

## 3. Activate Virtual Environment

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)

```cmd
.venv\Scripts\activate.bat
```

You should see `(.venv)` in your terminal after activation.

---

## 4. Install Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

---

## 5. Verify Setup

Run a simple check:

```bash
python -c "import pandas, numpy, sklearn"
```

If no errors appear, your environment is ready.

---

## 6. Running the Project

### Run main pipeline

```bash
python main.py
```

### Run dashboard (if applicable)

```bash
streamlit run dashboard.py
```

---

## 7. Development Workflow

### Pull latest changes

```bash
git pull origin main
```

### Create a new branch

```bash
git checkout -b feature/your-feature-name
```

### After making changes

```bash
git add .
git commit -m "Describe your changes"
git push origin feature/your-feature-name
```

---

## 8. Adding New Dependencies

If you install a new package:

```bash
pip install <package>
pip freeze > requirements.txt
```

Commit the updated `requirements.txt`.

---

## 9. Common Issues

### Virtual environment not activated

* Make sure `(.venv)` appears in your terminal
* Re-run activation command if needed

### Module not found errors

```bash
pip install -r requirements.txt
```

### Python version issues

* Ensure Python 3.9+ is being used
* Try:

```bash
python3 -m venv .venv
```

---

## 10. Project Structure (Overview)

```text
project/
├── main.py
├── ingest.py
├── cleaning.py
├── features.py
├── modeling.py
├── evaluation.py
├── dashboard.py
├── requirements.txt
└── .venv/
```

---

## Summary

1. Clone the repository
2. Create and activate virtual environment
3. Install dependencies
4. Run the project

After setup, you can begin working on your assigned features.

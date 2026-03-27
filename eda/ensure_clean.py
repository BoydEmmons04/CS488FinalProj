# Run cleaning.py if clean data folder don't exist yet
import os
import subprocess
import sys

def ensure_cleaned_data():
    cleaned_dir = os.path.join(os.path.dirname(__file__), "..", "cleaned_data")
    if not os.path.isdir(cleaned_dir):
        print("Cleaned data not found, running preprocessing...")
        cleaning_script = os.path.join(os.path.dirname(__file__), "..", "cleaning.py")
        result = subprocess.run([sys.executable, cleaning_script])
        if result.returncode != 0:
            print("ERROR: cleaning.py failed")
            sys.exit(1)
    return cleaned_dir

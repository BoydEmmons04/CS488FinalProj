import subprocess
import sys
import os

# Run cleaning once before all EDA scripts
print("Running preprocessing...")
result = subprocess.run([sys.executable, os.path.join("eda", "..", "cleaning.py")])
if result.returncode != 0:
    print("ERROR: cleaning.py failed with code " + str(result.returncode))
    sys.exit(1)

scripts = [
    "eda/eda_fuel.py",
    "eda/eda_delays.py",
    "eda/eda_competition.py",
    "eda/eda_t100.py",
    "eda/eda_db1b.py",
]

for script in scripts:

    print("\nRunning " + script)

    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print("ERROR: " + script + " failed with code " + str(result.returncode))
        sys.exit(1)

print("\nAll EDA scripts completed successfully.")

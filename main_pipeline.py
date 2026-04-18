import subprocess

scripts = [
    "notebook/class distribution.py",
    "notebook/reactive isolation forest.py",
    "notebook/predictive random forest.py",
    "notebook/predictive xgboost.py",
    "notebook/deep learning analysis.py",
    "notebook/model comparison and importance.py",
    "notebook/roc curve comparison.py"
]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script], check=True)

print("Pipeline completed successfully.")

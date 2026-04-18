import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_PATH = r"D:/CU/Dissertation/Final DS Project/data/processed/cicids2018_processed.csv"
OUT_FIG = os.path.join(PROJECT_ROOT, "output", "figures")

os.makedirs(OUT_FIG, exist_ok=True)

df = pd.read_csv(DATA_PATH)

counts = df["is_attack"].value_counts().sort_index()

labels = ["Benign", "Attack"]
values = counts.values

total = sum(values)
percentages = [(v / total) * 100 for v in values]

plt.figure(figsize=(6,5))
ax = sns.barplot(x=labels, y=values)

for i, v in enumerate(values):
    ax.text(i, v, f"{v}\n({percentages[i]:.1f}%)", ha='center')

plt.title("Dataset Class Distribution")
plt.ylabel("Number of Samples")
plt.xlabel("Class")

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "class distribution.png"))
plt.show()
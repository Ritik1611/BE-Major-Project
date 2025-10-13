#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load CSV
csv_path = "dp_noise_mechanism_comparison_full.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found")

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} entries from {csv_path}")
print(df.head())

# --- Clean data ---
df = df.dropna(subset=["mechanism", "noise_multiplier", "silhouette_score"])

# Ensure proper numeric types
for col in ["noise_multiplier", "silhouette_score", "l2_before", "l2_after", "distortion_ratio"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --- 1️⃣ Silhouette vs Noise Multiplier ---
plt.figure(figsize=(10,6))
sns.lineplot(
    data=df,
    x="noise_multiplier",
    y="silhouette_score",
    hue="mechanism",
    marker="o"
)
plt.title("Silhouette Score vs Noise Multiplier per Mechanism")
plt.xlabel("Noise Multiplier")
plt.ylabel("Silhouette Score")
plt.grid(True, alpha=0.3)
plt.legend(title="DP Mechanism")
plt.tight_layout()
plt.savefig("silhouette_vs_noise_multiplier.png", dpi=300)
plt.show()

# --- 2️⃣ Distortion ratio vs Noise Multiplier ---
plt.figure(figsize=(10,6))
sns.lineplot(
    data=df,
    x="noise_multiplier",
    y="distortion_ratio",
    hue="mechanism",
    marker="o"
)
plt.title("Distortion Ratio vs Noise Multiplier per Mechanism")
plt.xlabel("Noise Multiplier")
plt.ylabel("Distortion Ratio (L2_after / L2_before)")
plt.grid(True, alpha=0.3)
plt.legend(title="DP Mechanism")
plt.tight_layout()
plt.savefig("distortion_vs_noise_multiplier.png", dpi=300)
plt.show()

# --- 3️⃣ Correlation between Distortion and Silhouette ---
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="distortion_ratio",
    y="silhouette_score",
    hue="mechanism",
    style="mechanism",
    s=100
)
plt.title("Privacy-Utility Tradeoff (Distortion vs Silhouette)")
plt.xlabel("Distortion Ratio (Higher = More Noise)")
plt.ylabel("Silhouette Score (Higher = Better Clustering)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("privacy_utility_tradeoff.png", dpi=300)
plt.show()

# --- 4️⃣ Average silhouette by mechanism ---
plt.figure(figsize=(8,6))
sns.barplot(
    data=df.groupby("mechanism", as_index=False)["silhouette_score"].mean(),
    x="mechanism",
    y="silhouette_score",
    palette="Set2"
)
plt.title("Average Silhouette Score by Mechanism")
plt.ylabel("Mean Silhouette Score")
plt.xlabel("DP Mechanism")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("avg_silhouette_per_mechanism.png", dpi=300)
plt.show()

print("✅ Saved plots: silhouette_vs_noise_multiplier.png, distortion_vs_noise_multiplier.png, privacy_utility_tradeoff.png, avg_silhouette_per_mechanism.png")

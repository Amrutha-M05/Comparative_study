from __future__ import annotations

import os
import json
import pandas as pd
from tabulate import tabulate


# ================= CONFIG ================= #
OUTPUT_DIR = "outputs"
SORT_METRIC = "roc_auc"   # change if needed


# ================= MAIN ================= #
def main():
    records = []

    if not os.path.exists(OUTPUT_DIR):
        raise FileNotFoundError(f"{OUTPUT_DIR} not found!")

    model_names = [
        d for d in os.listdir(OUTPUT_DIR)
        if os.path.isdir(os.path.join(OUTPUT_DIR, d))
    ]

    print(f"Found models: {model_names}\n")

    for model_name in model_names:
        metrics_path = os.path.join(
            OUTPUT_DIR, model_name, "test_metrics.json"
        )

        if not os.path.exists(metrics_path):
            print(f"⚠️ Skipping {model_name} (no test_metrics.json)")
            continue

        try:
            with open(metrics_path) as f:
                metrics = json.load(f)

            metrics["model"] = model_name
            records.append(metrics)

        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")

    if len(records) == 0:
        print("❌ No valid model metrics found!")
        return

    df = pd.DataFrame(records)

    # ===== SORT =====
    if SORT_METRIC in df.columns:
        df = df.sort_values(by=SORT_METRIC, ascending=False)
    else:
        print(f"⚠️ Metric '{SORT_METRIC}' not found. Skipping sort.")

    # ===== REORDER COLUMNS =====
    cols = ["model"] + [c for c in df.columns if c != "model"]
    df = df[cols]

    # ===== PRINT TABLE =====
    print("\n📊 Model Comparison:\n")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))

    # ===== SAVE CSV =====
    save_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    df.to_csv(save_path, index=False)

    print(f"\n💾 Saved comparison → {save_path}")


# ================= ENTRY ================= #
if __name__ == "__main__":
    main() 
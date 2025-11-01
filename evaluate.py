import os
import pandas as pd

OUT = "outputs"


def main():
    p = os.path.join(OUT, "results_all.csv")
    if not os.path.exists(p):
        raise FileNotFoundError("Run train.py first.")

    df = pd.read_csv(p)

    refs = pd.DataFrame([
        {"dataset": "Big-Vul", "model": "CodeBERT", "accuracy": 0.755, "precision": 0.775, "recall": 0.719, "f1": 0.746, "roc_auc": None, "source": "[74]"},
        {"dataset": "Big-Vul", "model": "GraphCodeBERT", "accuracy": 0.733, "precision": 0.769, "recall": 0.667, "f1": 0.714, "roc_auc": None, "source": "[74]"},
    ])
    big = df[df["dataset"] == "Big-Vul"].copy()
    big["source"] = "This work"

    t32 = pd.concat([refs, big], ignore_index=True)
    t32.to_csv(os.path.join(OUT, "table_3_2.csv"), index=False)

    print("[OK] saved outputs/table_3_2.csv")
    print(t32)


if __name__ == "__main__":
    main()
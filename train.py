import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils import set_seed, load_yaml, ensure_dir, load_jsonl
from src.data import VulnDataset
from src.models.fcrm import FCRMBase, FCRMAWAF
from src.engine.trainer import train_engine, evaluate


def build_model(key, cfg):
    m = cfg["models"]
    if key == "fcrm":
        return FCRMBase(m["codebert_name"], m["graphcodebert_name"], m["hidden_dim"], m["dropout"], tuple(m["mlp_hidden_dims"]))
    return FCRMAWAF(m["codebert_name"], m["graphcodebert_name"], m["hidden_dim"], m["dropout"], tuple(m["mlp_hidden_dims"]))


def get_loaders(split_root, ds, cfg):
    tr = load_jsonl(os.path.join(split_root, ds, "train.jsonl"))
    va = load_jsonl(os.path.join(split_root, ds, "val.jsonl"))
    te = load_jsonl(os.path.join(split_root, ds, "test.jsonl"))

    m, t, p = cfg["models"], cfg["train"], cfg["paths"]
    tr_ds = VulnDataset(tr, m["codebert_name"], m["graphcodebert_name"], p["tree_sitter_lib"], t["max_len_code"], t["max_len_graph"])
    va_ds = VulnDataset(va, m["codebert_name"], m["graphcodebert_name"], p["tree_sitter_lib"], t["max_len_code"], t["max_len_graph"])
    te_ds = VulnDataset(te, m["codebert_name"], m["graphcodebert_name"], p["tree_sitter_lib"], t["max_len_code"], t["max_len_graph"])

    return (
        DataLoader(tr_ds, batch_size=t["batch_size"], shuffle=True, num_workers=2),
        DataLoader(va_ds, batch_size=t["batch_size"], shuffle=False, num_workers=2),
        DataLoader(te_ds, batch_size=t["batch_size"], shuffle=False, num_workers=2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])

    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    ensure_dir(cfg["paths"]["checkpoint_dir"])
    ensure_dir(cfg["paths"]["output_dir"])

    datasets = [("juliet", "Juliet"), ("bigvul", "Big-Vul"), ("nvd", "NVD")]
    models = [("fcrm", "FCRM"), ("fcrm_awaf", "FCRM-AWAF")]

    rows = []
    for mk, mn in models:
        for dk, dn in datasets:
            sdir = os.path.join(cfg["paths"]["split_data_dir"], dk)
            if not os.path.exists(sdir):
                print(f"[WARN] missing split dir: {sdir}")
                continue

            tr_ld, va_ld, te_ld = get_loaders(cfg["paths"]["split_data_dir"], dk, cfg)
            model = build_model(mk, cfg).to(device)

            ckpt = os.path.join(cfg["paths"]["checkpoint_dir"], f"{dk}_{mk}.pt")
            model = train_engine(model, tr_ld, va_ld, cfg["train"], device, ckpt)
            test_loss, test_m = evaluate(model, te_ld, device)

            rows.append({
                "dataset": dn,
                "model": mn,
                "accuracy": test_m["accuracy"],
                "precision": test_m["precision"],
                "recall": test_m["recall"],
                "f1": test_m["f1"],
                "roc_auc": test_m["roc_auc"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg["paths"]["output_dir"], "results_all.csv"), index=False)
    df.to_csv(os.path.join(cfg["paths"]["output_dir"], "table_3_1.csv"), index=False)
    print("[OK] saved outputs/results_all.csv and outputs/table_3_1.csv")


if __name__ == "__main__":
    main()
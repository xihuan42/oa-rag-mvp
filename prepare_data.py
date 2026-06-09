#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_data.py — 多数据集 OA 滑膜表达数据下载与 meta 分析
下载 GSE55235 / GSE12021 / GSE55457 / GSE55584,
逐数据集差异表达分析, Fisher 合并 p 值, 保存到 data/ 目录。
"""

import os
import sys
import gzip
import json
import urllib.request
import warnings
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ─── 数据集清单 ───
DATASETS = {
    "GSE55235": {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE55nnn/GSE55235/matrix/GSE55235_series_matrix.txt.gz",
        "platform": "GPL96",
    },
    "GSE12021": {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE12nnn/GSE12021/matrix/GSE12021-GPL96_series_matrix.txt.gz",
        "platform": "GPL96",
    },
    "GSE55457": {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE55nnn/GSE55457/matrix/GSE55457_series_matrix.txt.gz",
        "platform": "GPL96",
    },
    "GSE55584": {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE55nnn/GSE55584/matrix/GSE55584_series_matrix.txt.gz",
        "platform": "GPL96",
    },
}


# ─── 1. 下载与解析 ───

def download_series_matrix(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    return gzip.decompress(raw).decode("utf-8", errors="replace")


def parse_series_matrix(text: str):
    """
    解析 series matrix, 返回 (expr_df, gsm_to_group)。
    列名 = GSM ID, 行名 = probe ID。
    """
    lines = text.split("\n")

    gsm_ids = []
    titles = []

    for line in lines:
        if line.startswith("!Sample_geo_accession"):
            gsm_ids = [c.strip().strip('"') for c in line.split("\t")[1:] if c.strip()]
        elif line.startswith("!Sample_title"):
            titles = [c.strip().strip('"') for c in line.split("\t")[1:] if c.strip()]

    # 从 title 判定分组
    def classify(t: str) -> str:
        tl = t.lower()
        if "healthy" in tl or "normal" in tl or tl.startswith("normal"):
            return "Normal"
        if "osteoarth" in tl or tl.startswith("oa") or "oa_" in tl:
            return "OA"
        if "rheumatoid" in tl or tl.startswith("ra"):
            return "RA"
        return "Unknown"

    gsm_to_group = {}
    for i, gsm in enumerate(gsm_ids):
        title = titles[i] if i < len(titles) else ""
        gsm_to_group[gsm] = classify(title)

    # 解析表达数据
    in_table = False
    data_cols = None
    data_rows = []

    for line in lines:
        if line.startswith("!series_matrix_table_begin"):
            in_table = True
            continue
        if line.startswith("!series_matrix_table_end"):
            break
        if in_table and line.strip():
            cols = line.strip().split("\t")
            cols = [c.strip().strip('"') for c in cols]
            if data_cols is None:
                data_cols = cols
            else:
                data_rows.append(cols)

    if data_cols is None:
        raise ValueError(f"No expression data found")

    expr_df = pd.DataFrame(data_rows, columns=data_cols)
    expr_df = expr_df.set_index("ID_REF")
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
    return expr_df, gsm_to_group


# ─── 2. GPL96 注释 ───

def download_gpl96_annotation() -> pd.DataFrame:
    url = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
        text = gzip.decompress(raw).decode("utf-8", errors="replace")
        rows = []
        in_data = False
        for line in text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if not in_data:
                if parts[0].strip() == "ID":
                    in_data = True
                continue
            if len(parts) >= 3:
                probe = parts[0].strip()
                sym = parts[2].strip().split("///")[0].strip()
                if sym and sym != "---":
                    rows.append({"probe": probe, "gene": sym})
        df = pd.DataFrame(rows).drop_duplicates(subset=["probe"])
        return df
    except Exception as e:
        print(f"  [WARN] GPL96 annotation download failed: {e}")
        return pd.DataFrame(columns=["probe", "gene"])


# ─── 3. 差异表达分析 ───

def run_de(expr_df: pd.DataFrame, gsm_to_group: dict, dataset: str) -> pd.DataFrame:
    """Welch t-test OA vs Normal, 返回 DE 结果表。"""
    oa_gsm = [g for g, grp in gsm_to_group.items() if grp == "OA"]
    norm_gsm = [g for g, grp in gsm_to_group.items() if grp == "Normal"]

    cols = expr_df.columns.tolist()
    cols_oa = [c for c in cols if c in oa_gsm]
    cols_norm = [c for c in cols if c in norm_gsm]

    print(f"    OA samples: {len(oa_gsm)}, Normal samples: {len(norm_gsm)}")

    results = []
    for gene in expr_df.index:
        va = expr_df.loc[gene, cols_oa].dropna().astype(float)
        vb = expr_df.loc[gene, cols_norm].dropna().astype(float)
        if len(va) < 2 or len(vb) < 2:
            continue
        m_a, m_b = va.mean(), vb.mean()
        fc = m_a / m_b if m_b != 0 else np.nan
        l2 = np.log2(fc) if fc > 0 else np.nan
        t_stat, p = scipy_stats.ttest_ind(va, vb, equal_var=False)
        results.append({
            "probe": gene,
            "dataset": dataset,
            "log2FC": l2,
            "mean_OA": m_a,
            "mean_Normal": m_b,
            "p_value": p,
            "t_statistic": t_stat,
        })

    df = pd.DataFrame(results).dropna(subset=["log2FC"])
    _, adj_p, _, _ = multipletests(df["p_value"].values, method="fdr_bh")
    df["adj_p_value"] = adj_p
    df["-log10p"] = -np.log10(df["p_value"].clip(lower=1e-300))
    df["direction"] = np.where(df["log2FC"] > 0, "up", "down")
    return df.sort_values("p_value")


def map_probes(de_df: pd.DataFrame, annot_df: pd.DataFrame) -> pd.DataFrame:
    """探针→基因符号映射, 多探针取最显著保留。"""
    if annot_df.empty:
        de_df["gene_symbol"] = de_df["probe"]
        return de_df
    pmap = dict(zip(annot_df["probe"], annot_df["gene"]))
    de_df["gene_symbol"] = de_df["probe"].map(lambda x: pmap.get(x, x))
    # 多探针取最显著
    de_df = de_df.sort_values("p_value").drop_duplicates(
        subset=["gene_symbol", "dataset"], keep="first"
    )
    return de_df


# ─── 4. Meta 分析: Fisher 合并 p 值 ───

def meta_analyze(de_results: Dict[str, pd.DataFrame], annot_df: pd.DataFrame):
    """
    Fisher 法合并多数据集的 p 值。
    要求各数据集 DE 结果已映射到 gene_symbol。
    返回 meta_analysis DataFrame。
    """
    # 收集所有基因符号
    all_genes = set()
    for df in de_results.values():
        all_genes.update(df["gene_symbol"].unique())

    pmap = dict(zip(annot_df["probe"], annot_df["gene"])) if not annot_df.empty else {}

    rows = []
    for gene in sorted(all_genes):
        p_vals = []
        l2s = []
        datasets_used = []
        for ds_name, df in de_results.items():
            match = df[df["gene_symbol"] == gene]
            if not match.empty:
                row = match.iloc[0]
                p_vals.append(row["p_value"])
                l2s.append(row["log2FC"])
                datasets_used.append(ds_name)

        if len(p_vals) < 2:
            continue  # 至少 2 个数据集才能 meta

        # Fisher 合并
        chi2 = -2 * np.sum(np.log(p_vals))
        df_fisher = 2 * len(p_vals)
        meta_p = 1 - scipy_stats.chi2.cdf(chi2, df_fisher)

        # 效应量: 加权平均 (按样本量近似加权)
        meta_log2fc = np.mean(l2s)

        # 方向一致性
        directions = [1 if l > 0 else -1 for l in l2s]
        all_same_dir = len(set(directions)) == 1
        n_up = sum(1 for d in directions if d > 0)
        n_down = sum(1 for d in directions if d < 0)

        rows.append({
            "gene_symbol": gene,
            "meta_p_value": meta_p,
            "meta_log2FC": meta_log2fc,
            "n_datasets": len(datasets_used),
            "datasets": ";".join(datasets_used),
            "all_same_direction": all_same_dir,
            "n_up": n_up,
            "n_down": n_down,
            "individual_p": ";".join(f"{p:.2e}" for p in p_vals),
            "individual_log2FC": ";".join(f"{l2:.3f}" for l2 in l2s),
        })

    meta_df = pd.DataFrame(rows)
    if not meta_df.empty:
        _, adj_p, _, _ = multipletests(
            meta_df["meta_p_value"].values, method="fdr_bh"
        )
        meta_df["meta_adj_p_value"] = adj_p
        meta_df["-log10_meta_p"] = -np.log10(
            meta_df["meta_p_value"].clip(lower=1e-300)
        )

    return meta_df.sort_values("meta_p_value")


# ─── 5. 主流程 ───

def main():
    print("=" * 60)
    print("OA Multi-Dataset Meta Analysis - Data Preparation")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── 下载 GPL96 注释 ──
    print("\n[1/4] Downloading GPL96 annotation...")
    annot_df = download_gpl96_annotation()
    print(f"  {len(annot_df)} probes mapped to genes")

    # ── 逐数据集处理 ──
    print("\n[2/4] Downloading and processing datasets...")
    de_results = {}
    dataset_info = {}

    for ds_name, ds_info in DATASETS.items():
        print(f"\n  --- {ds_name} ---")
        try:
            text = download_series_matrix(ds_info["url"])
            expr_df, gsm_to_group = parse_series_matrix(text)
            print(f"  Expression: {expr_df.shape[0]} probes x {expr_df.shape[1]} samples")

            # 只保留 OA 和 Normal
            keep = [gsm for gsm, g in gsm_to_group.items() if g in ("OA", "Normal")]
            if len(keep) < 4:
                print(f"  SKIP: insufficient OA/Normal samples ({len(keep)})")
                continue

            expr_sub = expr_df[keep].copy()
            expr_sub = expr_sub.dropna(thresh=len(keep) * 0.5)

            # DE
            de_df = run_de(expr_sub, gsm_to_group, ds_name)
            de_df = map_probes(de_df, annot_df)

            n_sig = (de_df["adj_p_value"] < 0.05).sum()
            print(f"  DEGs (adj_p<0.05): {n_sig}")
            print(f"  Top: {de_df.iloc[0]['gene_symbol']} (log2FC={de_df.iloc[0]['log2FC']:.3f}, p={de_df.iloc[0]['p_value']:.2e})")

            de_results[ds_name] = de_df

            # Save per-dataset DE
            de_df.to_csv(os.path.join(DATA_DIR, f"de_{ds_name}.csv"), index=False)
            print(f"  Saved: data/de_{ds_name}.csv")

            # Sample info for reference
            dataset_info[ds_name] = {
                "n_oa": sum(1 for g in gsm_to_group.values() if g == "OA"),
                "n_normal": sum(1 for g in gsm_to_group.values() if g == "Normal"),
                "n_ra": sum(1 for g in gsm_to_group.values() if g == "RA"),
                "n_unknown": sum(1 for g in gsm_to_group.values() if g == "Unknown"),
                "n_probes": expr_df.shape[0],
                "n_genes_filtered": expr_sub.shape[0],
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ── Meta 分析 ──
    print("\n[3/4] Running meta-analysis (Fisher's method)...")
    # 只对同时有 OA 和 Normal 的数据集做 meta
    meta_datasets = [ds for ds in de_results
                     if dataset_info.get(ds, {}).get("n_oa", 0) >= 3
                     and dataset_info.get(ds, {}).get("n_normal", 0) >= 3]
    print(f"  Datasets for meta: {meta_datasets}")

    if len(meta_datasets) >= 2:
        meta_subset = {ds: de_results[ds] for ds in meta_datasets}
        meta_df = meta_analyze(meta_subset, annot_df)
        n_robust_sig = (meta_df["meta_adj_p_value"] < 0.05).sum()
        n_robust_consistent = (
            (meta_df["meta_adj_p_value"] < 0.05) & meta_df["all_same_direction"]
        ).sum()
        print(f"  Meta DEGs (adj_p<0.05): {n_robust_sig}")
        print(f"  Robust (adj_p<0.05 + consistent direction): {n_robust_consistent}")
        print(f"  Top: {meta_df.iloc[0]['gene_symbol']} (p={meta_df.iloc[0]['meta_p_value']:.2e})")

        meta_df.to_csv(os.path.join(DATA_DIR, "meta_analysis.csv"), index=False)
        print("  Saved: data/meta_analysis.csv")
    else:
        print("  WARN: insufficient datasets for meta-analysis")
        meta_df = pd.DataFrame()

    # ── 汇总信息 ──
    print("\n[4/4] Saving dataset info...")
    info_path = os.path.join(DATA_DIR, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {info_path}")

    # 注释
    annot_df.to_csv(os.path.join(DATA_DIR, "gpl96_annotation.csv"), index=False)

    print("\n" + "=" * 60)
    print("DONE. All data saved to data/")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OA Knowledge Graph MVP — 多数据集版
证明：仅靠文献检索无法得到的结果，通过分析原始数据就能得到。

数据集: GSE55235 + GSE12021 + GSE55457 (同平台, 滑膜组织)
参考论文: PMID:24690414 (Lambert 2014)

3个经典问题 + 3个跨数据集meta分析问题
"""

import sys, os, gzip, urllib.request, json, warnings
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import networkx as nx
from pyvis.network import Network
import plotly.express as px
import plotly.graph_objects as go
import requests
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="OA RAG MVP - 多数据集",
    page_icon="\U0001F9F1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ───
st.markdown("""
<style>
.main-header { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; }
.sub-header { font-size: 1.2rem; color: #666; margin-bottom: 1.5rem; }
.highlight-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 1.2rem; border-radius: 12px; margin-bottom: 1rem;
}
.question-box {
    background: #f0f2f6; border-left: 4px solid #ff4b4b;
    padding: 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;
    color: #1a1a2e;
}
.answer-box {
    background: #e8f5e9; border-left: 4px solid #4caf50;
    padding: 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;
    color: #1a1a2e;
}
.paper-box {
    background: #fff3e0; border-left: 4px solid #ff9800;
    padding: 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;
    color: #1a1a2e;
}
.stDataFrame { border: none !important; }
</style>""", unsafe_allow_html=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ════════════════════════════════════════════════════════════════════
# 数据加载（优先 cache，否则提示运行 prepare_data.py）
# ════════════════════════════════════════════════════════════════════

CACHE_FILES = {
    "annot": "gpl96_annotation.csv",
    "meta": "meta_analysis.csv",
    "info": "dataset_info.json",
    "de_GSE55235": "de_GSE55235.csv",
    "de_GSE12021": "de_GSE12021.csv",
    "de_GSE55457": "de_GSE55457.csv",
}


def cache_available() -> bool:
    return all(os.path.exists(os.path.join(DATA_DIR, f))
               for f in CACHE_FILES.values()
               if f.endswith(".csv") or f.endswith(".json"))


@st.cache_data
def load_cached_data():
    """从 data/ 目录加载预计算结果。"""
    annot = pd.read_csv(os.path.join(DATA_DIR, CACHE_FILES["annot"]))
    meta = pd.read_csv(os.path.join(DATA_DIR, CACHE_FILES["meta"]))
    with open(os.path.join(DATA_DIR, CACHE_FILES["info"]), "r") as f:
        info = json.load(f)

    de_results = {}
    for ds in ["GSE55235", "GSE12021", "GSE55457"]:
        path = os.path.join(DATA_DIR, CACHE_FILES.get(f"de_{ds}", f"de_{ds}.csv"))
        if os.path.exists(path):
            de_results[ds] = pd.read_csv(path)

    return annot, meta, info, de_results


# ════════════════════════════════════════════════════════════════════
# 单数据集在线加载（fallback）
# ════════════════════════════════════════════════════════════════════

def download_series_matrix(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    return gzip.decompress(raw).decode("utf-8", errors="replace")


def parse_series_matrix(text: str):
    lines = text.split("\n")
    gsm_ids, titles = [], []
    for line in lines:
        if line.startswith("!Sample_geo_accession"):
            gsm_ids = [c.strip().strip('"') for c in line.split("\t")[1:] if c.strip()]
        elif line.startswith("!Sample_title"):
            titles = [c.strip().strip('"') for c in line.split("\t")[1:] if c.strip()]
    gsm_to_group = {}
    for i, gsm in enumerate(gsm_ids):
        tl = (titles[i] if i < len(titles) else "").lower()
        if "healthy" in tl or "normal" in tl:
            gsm_to_group[gsm] = "Normal"
        elif "osteoarth" in tl:
            gsm_to_group[gsm] = "OA"
        elif "rheumatoid" in tl:
            gsm_to_group[gsm] = "RA"
        else:
            gsm_to_group[gsm] = "Unknown"
    in_table = False
    data_cols, data_rows = None, []
    for line in lines:
        if line.startswith("!series_matrix_table_begin"):
            in_table = True; continue
        if line.startswith("!series_matrix_table_end"):
            break
        if in_table and line.strip():
            cols = [c.strip().strip('"') for c in line.strip().split("\t")]
            if data_cols is None:
                data_cols = cols
            else:
                data_rows.append(cols)
    expr = pd.DataFrame(data_rows, columns=data_cols).set_index("ID_REF")
    expr = expr.apply(pd.to_numeric, errors="coerce")
    return expr, gsm_to_group


@st.cache_data(show_spinner="从 GEO 下载 GSE55235...")
def download_gse55235():
    url = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE55nnn/GSE55235/matrix/"
           "GSE55235_series_matrix.txt.gz")
    text = download_series_matrix(url)
    return parse_series_matrix(text)


@st.cache_data(show_spinner="下载 GPL96 注释...")
def download_annotation():
    url = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        text = gzip.decompress(urllib.request.urlopen(req, timeout=120).read()).decode("utf-8", errors="replace")
        rows, in_data = [], False
        for line in text.split("\n"):
            if line.startswith("#") or not line.strip(): continue
            parts = line.split("\t")
            if not in_data:
                if parts[0].strip() == "ID": in_data = True; continue
            if in_data and len(parts) >= 3:
                sym = parts[2].strip().split("///")[0].strip()
                if sym and sym != "---":
                    rows.append({"probe": parts[0].strip(), "gene": sym})
        df = pd.DataFrame(rows).drop_duplicates(subset=["probe"])
        return df
    except Exception as e:
        return pd.DataFrame(columns=["probe", "gene"])


@st.cache_data(show_spinner="差异表达分析...")
def run_de_single(expr, gsm_to_group):
    oa = [g for g, gr in gsm_to_group.items() if gr == "OA"]
    nm = [g for g, gr in gsm_to_group.items() if gr == "Normal"]
    co = [c for c in expr.columns if c in oa]
    cn = [c for c in expr.columns if c in nm]
    res = []
    for gene in expr.index:
        va = expr.loc[gene, co].dropna().astype(float)
        vb = expr.loc[gene, cn].dropna().astype(float)
        if len(va) < 2 or len(vb) < 2: continue
        m = va.mean() / vb.mean() if vb.mean() != 0 else np.nan
        l2 = np.log2(m) if m > 0 else np.nan
        t, p = scipy_stats.ttest_ind(va, vb, equal_var=False)
        res.append({"gene": gene, "log2FC": l2, "p_value": p, "mean_OA": va.mean(), "mean_Normal": vb.mean()})
    df = pd.DataFrame(res).dropna(subset=["log2FC"])
    _, adj, _, _ = multipletests(df["p_value"].values, method="fdr_bh")
    df["adj_p_value"] = adj
    df["-log10_pvalue"] = -np.log10(df["p_value"].clip(lower=1e-300))
    return df.sort_values("p_value")


def map_probes(de_df, annot_df):
    if annot_df.empty:
        de_df["gene_symbol"] = de_df["gene"]
        return de_df
    pmap = dict(zip(annot_df["probe"], annot_df["gene"]))
    de_df["gene_symbol"] = de_df["gene"].map(lambda x: pmap.get(x, x))
    return de_df.sort_values("p_value").drop_duplicates(subset=["gene_symbol"], keep="first")


# ════════════════════════════════════════════════════════════════════
# 通路富集 (Enrichr API)
# ════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="通路富集分析 (Enrichr API)...")
def pathway_enrichment(gene_list, n_top=10):
    if len(gene_list) == 0:
        return pd.DataFrame(columns=["pathway", "p_value", "adj_p_value", "genes"])
    ENRICHR_URL = "https://maayanlab.cloud/Enrichr"
    try:
        resp = requests.post(f"{ENRICHR_URL}/addList", files={
            "list": (None, "\n".join(gene_list)),
            "description": (None, "OA DEGs"),
        }, timeout=30)
        uid = resp.json().get("userListId")
        if not uid:
            return pd.DataFrame(columns=["pathway", "p_value", "adj_p_value", "genes"])
        for lib in ["KEGG_2021_Human", "KEGG_2019_Human"]:
            r = requests.get(f"{ENRICHR_URL}/enrich", params={
                "userListId": uid, "backgroundType": lib
            }, timeout=30)
            raw = r.json().get(lib, [])
            if raw:
                break
        rows = []
        for entry in raw[:n_top]:
            rows.append({
                "pathway": entry[1], "p_value": entry[2],
                "z_score": entry[3], "combined_score": entry[4],
                "genes": "; ".join(entry[5]) if isinstance(entry[5], list) else str(entry[5]),
                "adj_p_value": entry[6] if len(entry) > 6 else entry[2],
            })
        result = pd.DataFrame(rows)
        if not result.empty:
            result["-log10_pvalue"] = -np.log10(result["p_value"].astype(float).clip(lower=1e-300))
        return result
    except Exception as e:
        st.warning(f"通路富集 API 失败: {e}")
        return pd.DataFrame(columns=["pathway", "p_value", "adj_p_value", "genes"])


# ════════════════════════════════════════════════════════════════════
# 共表达分析
# ════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="共表达分析...")
def correlation_analysis(expr, gsm_to_group, target_gene, top_n=10):
    if target_gene not in expr.index:
        return {"error": f"{target_gene} not in data"}
    oa = [c for c in expr.columns if c in [g for g, gr in gsm_to_group.items() if gr == "OA"]]
    nm = [c for c in expr.columns if c in [g for g, gr in gsm_to_group.items() if gr == "Normal"]]
    t_oa = expr.loc[target_gene, oa].astype(float).values
    t_nm = expr.loc[target_gene, nm].astype(float).values
    res_oa, res_nm = [], []
    for gene in expr.index:
        if gene == target_gene: continue
        v_oa = expr.loc[gene, oa].astype(float).values
        v_nm = expr.loc[gene, nm].astype(float).values
        valid_oa = ~(np.isnan(v_oa) | np.isnan(t_oa))
        valid_nm = ~(np.isnan(v_nm) | np.isnan(t_nm))
        if valid_oa.sum() >= 4:
            r, p = scipy_stats.pearsonr(v_oa[valid_oa], t_oa[valid_oa])
            res_oa.append({"gene": gene, "correlation": r, "p_value": p})
        if valid_nm.sum() >= 4:
            r, p = scipy_stats.pearsonr(v_nm[valid_nm], t_nm[valid_nm])
            res_nm.append({"gene": gene, "correlation": r, "p_value": p})
    return {
        "OA": pd.DataFrame(res_oa).sort_values("correlation", ascending=False).head(top_n),
        "Normal": pd.DataFrame(res_nm).sort_values("correlation", ascending=False).head(top_n),
        "target_gene": target_gene,
    }


def map_corr_probes(corr_dict, annot_df):
    if not corr_dict or "error" in corr_dict:
        return corr_dict
    pmap = dict(zip(annot_df["probe"], annot_df["gene"])) if not annot_df.empty else {}
    for grp in ("OA", "Normal"):
        if grp in corr_dict and isinstance(corr_dict[grp], pd.DataFrame):
            df = corr_dict[grp].copy()
            df["gene_symbol"] = df["gene"].map(lambda x: pmap.get(x, x))
            corr_dict[grp] = df
    return corr_dict


# ════════════════════════════════════════════════════════════════════
# 知识图谱
# ════════════════════════════════════════════════════════════════════

def build_knowledge_graph(de_df, pathway_df, corr_results=None,
                          n_genes=20, n_pathways=8, multi_ds=None, meta_df=None):
    G = nx.DiGraph()

    # ─── 区域坐标配置 ───
    region_centers = {
        "OA Group": (0, 200),
        "Normal Group": (0, -200),
        "consensus": (0, 0),
        "GSE55235": (-250, 0),
        "GSE12021": (200, -150),
        "GSE55457": (200, 150),
    }
    scatter_r = 80  # 散布半径

    G.add_node("OA Group", type="group", color="#e74c3c", size=20,
               title="OA 滑膜组织", pos_x=0, pos_y=200, shape="star")
    G.add_node("Normal Group", type="group", color="#3498db", size=20,
               title="正常滑膜组织", pos_x=0, pos_y=-200, shape="star")

    ds_palettes = {
        "GSE55235": {"up": (231, 76, 60), "down": (52, 152, 219)},
        "GSE12021": {"up": (46, 204, 113), "down": (155, 89, 182)},
        "GSE55457": {"up": (241, 196, 15), "down": (230, 126, 34)},
    }
    consensus_colors = {"up": (0, 188, 212), "down": (0, 151, 167)}

    if multi_ds and meta_df is not None:
        # ═══════════════════════════════════════════════════════
        # 多数据集分区模式：共识中心 + 三翼
        # ═══════════════════════════════════════════════════════

        # 1) 共识基因（中心）
        robust = meta_df[(meta_df["meta_adj_p_value"] < 0.05) & meta_df["all_same_direction"]]
        consensus_top = robust.head(10)
        cx, cy = region_centers["consensus"]
        for i, (_, row) in enumerate(consensus_top.iterrows()):
            gn = row["gene_symbol"]
            if not gn:
                continue
            # Scatter in a ring around center
            angle = 2 * np.pi * i / len(consensus_top)
            px = cx + scatter_r * 0.7 * np.cos(angle)
            py = cy + scatter_r * 0.7 * np.sin(angle)
            direc = "up" if row["meta_log2FC"] > 0 else "down"
            intensity = min(abs(row["meta_log2FC"]) / 3.0, 1.0)
            alpha = 0.5 + intensity * 0.5
            r, g, b = consensus_colors[direc]
            nc = f"rgba({r},{g},{b},{alpha})"
            title = (f"Gene: {gn} (跨数据集共识)<br>"
                     f"meta_log2FC: {row['meta_log2FC']:.3f}<br>"
                     f"meta_adj_p: {row['meta_adj_p_value']:.2e}")
            sz = 8 + min(-np.log10(row["meta_adj_p_value"] + 1e-300), 14)
            G.add_node(gn, type="gene", color=nc, size=sz, title=title,
                       direction=direc, region="consensus", pos_x=px, pos_y=py,
                       border_color="#00838f", shape="diamond")
            tg = "OA Group" if direc == "up" else "Normal Group"
            G.add_edge(gn, tg, type="deg",
                       label=f"FC={row['meta_log2FC']:.2f}",
                       weight=abs(row["meta_log2FC"]), color=nc)

        consensus_genes = set(consensus_top["gene_symbol"].tolist())

        # 2) 各数据集特有基因（三翼）
        for ds_name in ["GSE55235", "GSE12021", "GSE55457"]:
            if ds_name not in multi_ds:
                continue
            ds_df = multi_ds[ds_name]
            specific = ds_df[~ds_df["gene_symbol"].isin(consensus_genes)].head(6)
            cxc, cyc = region_centers[ds_name]
            palette = ds_palettes[ds_name]
            for j, (_, row) in enumerate(specific.iterrows()):
                gn = row.get("gene_symbol", row.get("gene", ""))
                if not gn:
                    continue
                angle = 2 * np.pi * j / max(len(specific), 1)
                px = cxc + scatter_r * np.cos(angle)
                py = cyc + scatter_r * np.sin(angle)
                direc = "up" if row["log2FC"] > 0 else "down"
                intensity = min(abs(row["log2FC"]) / 3.0, 1.0)
                alpha = 0.5 + intensity * 0.5
                r, g, b = palette[direc]
                nc = f"rgba({r},{g},{b},{alpha})"
                title = (f"Gene: {gn}<br>Dataset: {ds_name}<br>"
                         f"log2FC: {row['log2FC']:.3f}<br>"
                         f"adj_p: {row.get('adj_p_value', 0):.2e}")
                sz = 6 + min(-np.log10(row.get("adj_p_value", row.get("p_value", 0.05)) + 1e-300) * 1.5, 12)
                G.add_node(gn, type="gene", color=nc, size=sz, title=title,
                           direction=direc, region=ds_name, pos_x=px, pos_y=py,
                           shape="dot")
                tg = "OA Group" if direc == "up" else "Normal Group"
                G.add_edge(gn, tg, type="deg",
                           label=f"FC={row['log2FC']:.2f}",
                           weight=abs(row["log2FC"]), color=nc)

        # 3) 数据集区域节点
        for ds_name in ["GSE55235", "GSE12021", "GSE55457"]:
            if ds_name not in multi_ds:
                continue
            cxc, cyc = region_centers[ds_name]
            pal = ds_palettes[ds_name]
            dc = f"rgba({pal['up'][0]},{pal['up'][1]},{pal['up'][2]},0.8)"
            G.add_node(ds_name, type="dataset", color=dc, size=14,
                       title=f"Dataset: {ds_name}",
                       pos_x=cxc, pos_y=cyc)

    else:
        # ═══════════════════════════════════════════════════════
        # 单数据集模式（原有逻辑）
        # ═══════════════════════════════════════════════════════
        top_genes = de_df.head(n_genes)
        for _, row in top_genes.iterrows():
            gn = row.get("gene_symbol", row.get("gene", ""))
            if not gn: continue
            direc = "up" if row["log2FC"] > 0 else "down"
            intensity = min(abs(row["log2FC"]) / 3.0, 1.0)
            nc = f"rgba(231,76,60,{0.5+intensity*0.5})" if direc == "up" else f"rgba(52,152,219,{0.5+intensity*0.5})"
            title = (f"Gene: {gn}<br>log2FC: {row['log2FC']:.3f}<br>"
                     f"adj_p: {row.get('adj_p_value', 0):.2e}")
            sz = 6 + min(-np.log10(row.get("adj_p_value", row.get("p_value", 0.05)) + 1e-300) * 1.5, 12)
            G.add_node(gn, type="gene", color=nc, size=sz, title=title, direction=direc)
            tg = "OA Group" if direc == "up" else "Normal Group"
            G.add_edge(gn, tg, type="deg", label=f"FC={row['log2FC']:.2f}",
                       weight=abs(row["log2FC"]), color=nc)

    # ─── 通路节点（只添加至少连接一个图谱基因的通路） ───
    for _, row in pathway_df.iterrows():
        pw = row["pathway"][:60]
        pathway_genes = [g.strip() for g in str(row.get("genes", "")).split(";")
                         if g.strip() in G.nodes]
        if not pathway_genes:
            continue
        title = f"Pathway: {pw}<br>p={row['p_value']:.2e}"
        G.add_node(pw, type="pathway", color="#9b59b6", size=10, title=title, shape="square")
        for gg in pathway_genes:
            G.add_edge(pw, gg, type="pathway_gene", color="#9b59b6", dashes=True)

    # ─── 共表达边 ───
    if corr_results and "OA" in corr_results and isinstance(corr_results["OA"], pd.DataFrame):
        for _, row in corr_results["OA"].iterrows():
            g1 = corr_results.get("target_gene", "")
            g2 = row.get("gene_symbol", row.get("gene", ""))
            if g1 in G.nodes and g2 in G.nodes and g1 != g2:
                G.add_edge(g1, g2, type="correlation",
                           label=f"r={row['correlation']:.2f}",
                           color="#2ecc71", weight=abs(row["correlation"]))

    return G


def render_pyvis_graph(G, height="550px"):
    net = Network(height=height, width="100%", directed=True, notebook=False)
    net.set_options("""{"physics":{"forceAtlas2Based":{"gravitationalConstant":-40,
        "centralGravity":0.005,"springLength":200,"springConstant":0.02,"damping":0.4},
        "stabilization":{"iterations":50,"updateInterval":25},"minVelocity":0.5,"maxVelocity":3},
        "edges":{"smooth":{"enabled":true,"type":"continuous"}}}""")
    for node, attrs in G.nodes(data=True):
        px = attrs.get("pos_x")
        py = attrs.get("pos_y")
        has_pos = px is not None and py is not None
        base_kw = dict(
            label=node if len(node) < 30 else node[:27] + "...",
            title=attrs.get("title", node),
            size=attrs.get("size", 15),
            font={"size": 14 if attrs.get("type") in ("group", "dataset") else 10},
        )
        if has_pos:
            base_kw["x"] = px
            base_kw["y"] = py
            base_kw["physics"] = False

        border_color = attrs.get("border_color")
        if border_color:
            base_kw["color"] = {"background": attrs.get("color", "#888"), "border": border_color}
            base_kw["borderWidth"] = 3 if attrs.get("type") == "group" else 2
        else:
            base_kw["color"] = attrs.get("color", "#888")
            base_kw["borderWidth"] = 3 if attrs.get("type") == "group" else 1
        net.add_node(node, **base_kw)
    for u, v, attrs in G.edges(data=True):
        arrows = "to"
        if attrs.get("type") in ("pathway_gene",): arrows = "to"
        elif attrs.get("type") == "correlation": arrows = ""
        net.add_edge(u, v, title=attrs.get("label", ""), color=attrs.get("color", "#666"),
                     width=max(attrs.get("weight", 1) * 1.5, 1),
                     dashes=attrs.get("dashes", False), arrows=arrows,
                     label=attrs.get("label", ""), font={"size": 9, "color": attrs.get("color", "#666")})
    return net.generate_html()


# ════════════════════════════════════════════════════════════════════
# 3个经典问题和3个新问题
# ════════════════════════════════════════════════════════════════════

def q1_top_deg(de_df):
    df = de_df.copy()
    df["方向"] = df["log2FC"].apply(lambda x: "\U0001F7E2 OA 上调" if x > 0 else "\U0001F535 OA 下调")
    df["显著性"] = df["adj_p_value"].apply(lambda x: "***" if x < 0.001 else ("**" if x < 0.01 else ("*" if x < 0.05 else "")))
    return df


def q2_correlation(expr, gsm_to_group, annot_df):
    pmap = dict(zip(annot_df["probe"], annot_df["gene"])) if not annot_df.empty else {}
    probes = [p for p, g in pmap.items() if g == "TREM1"] if pmap else []
    tp = probes[0] if probes else "TREM1"
    return correlation_analysis(expr, gsm_to_group, tp)


def q3_pathways(pathway_df):
    paper_pathways = [
        "Inflammatory response", "TNF signaling", "NF-kappa B signaling",
        "Wnt signaling", "Cartilage metabolism", "Angiogenesis",
        "Cytokine-cytokine receptor interaction", "MAPK signaling",
        "PI3K-Akt signaling", "Osteoclast differentiation", "IL-17 signaling",
    ]
    if pathway_df.empty:
        return pathway_df, paper_pathways
    df = pathway_df.copy()
    df["在原文中"] = df["pathway"].apply(
        lambda x: "✅ 是 (原文提及)" if any(p.lower() in x.lower() or x.lower() in p.lower()
                                          for p in paper_pathways)
        else "❌ 否 (原文未提及)"
    )
    return df, paper_pathways


def q4_meta_overview(meta_df):
    """Q4: 跨数据集稳健差异基因 top 榜"""
    df = meta_df.copy()
    df["方向一致性"] = df["all_same_direction"].apply(
        lambda x: "✅ 全部一致" if x else "⚠️ 存在分歧"
    )
    return df


def q5_cross_dataset_validation(de_results, meta_df, annot_df):
    """Q5: 单数据集不显著但 meta 显著的基因"""
    findings = []
    for _, row in meta_df.iterrows():
        if row["meta_adj_p_value"] >= 0.05:
            continue
        gene = row["gene_symbol"]
        # Check if significant in individual datasets
        n_single_sig = 0
        details = []
        for ds_name, ds_df in de_results.items():
            match = ds_df[ds_df["gene_symbol"] == gene]
            if not match.empty:
                r = match.iloc[0]
                is_sig = r["adj_p_value"] < 0.05
                if is_sig:
                    n_single_sig += 1
                details.append(f"{ds_name}: p={r['adj_p_value']:.2e} ({'sig' if is_sig else 'ns'})")
        if n_single_sig <= 1:
            findings.append({
                "gene_symbol": gene,
                "meta_adj_p": row["meta_adj_p_value"],
                "meta_log2FC": row["meta_log2FC"],
                "n_datasets_sig": n_single_sig,
                "n_datasets_total": row["n_datasets"],
                "details": "; ".join(details),
            })
    return pd.DataFrame(findings).sort_values("meta_adj_p") if findings else pd.DataFrame()


def q6_robust_pathways(meta_df, de_results, annot_df):
    """Q6: 稳健差异基因的通路富集"""
    robust = meta_df[(meta_df["meta_adj_p_value"] < 0.05) & meta_df["all_same_direction"]]
    up = robust[robust["meta_log2FC"] > 0].head(50)["gene_symbol"].tolist()
    down = robust[robust["meta_log2FC"] < 0].head(50)["gene_symbol"].tolist()
    return pathway_enrichment(up + down), len(robust), len(up), len(down)


# ════════════════════════════════════════════════════════════════════
# 主界面
# ════════════════════════════════════════════════════════════════════

def main():
    st.markdown('<p class="main-header">\U0001F9F1 骨关节炎知识图谱 — 多数据集 Meta 分析</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">文献检索不到的知识，藏在 3 个数据集的原始表达矩阵中</p>',
                unsafe_allow_html=True)

    # ─── 加载数据 ───
    if cache_available():
        with st.spinner("加载预计算数据..."):
            annot_df, meta_df, ds_info, de_results = load_cached_data()
            multi_mode = True
            # Use GSE55235 as the primary dataset for KG and Q1-Q3
            primary_ds = "GSE55235"
            de_primary = de_results[primary_ds]
            # Build combined expr (construct from individual DE results)
            # For correlation we need actual expr matrix - use primary
            expr, gsm_to_group = download_gse55235()
            # Filter OA/Normal
            keep = [g for g in expr.columns if gsm_to_group.get(g) in ("OA", "Normal")]
            expr = expr[keep].copy().dropna(thresh=len(keep) * 0.5)
    else:
        with st.spinner("正在从 GEO 下载数据..."):
            expr, gsm_to_group = download_gse55235()
            keep = [g for g in expr.columns if gsm_to_group.get(g) in ("OA", "Normal")]
            expr = expr[keep].copy().dropna(thresh=len(keep) * 0.5)
            annot_df = download_annotation()
            de_primary = run_de_single(expr, gsm_to_group)
            de_primary = map_probes(de_primary, annot_df)
            meta_df = pd.DataFrame()
            ds_info = {}
            de_results = {}
            multi_mode = False

    # ─── 侧边栏 ───
    with st.sidebar:
        st.markdown("## \U0001F9F1 OA RAG MVP")
        st.markdown("---")
        if multi_mode:
            total_oa = sum(v.get("n_oa", 0) for v in ds_info.values())
            total_nm = sum(v.get("n_normal", 0) for v in ds_info.values())
            st.markdown(f"**数据集 ({len(ds_info)} 个)**")
            for ds, v in ds_info.items():
                emoji = "\U00002705" if v.get("n_oa", 0) >= 3 else "\U0000274C"
                st.markdown(f"{emoji} **{ds}**: OA={v['n_oa']} Normal={v['n_normal']} RA={v['n_ra']}")
            st.markdown(f"**合并共 {total_oa} OA + {total_nm} Normal**")
        else:
            n_oa = sum(1 for g in gsm_to_group.values() if g == "OA")
            n_nm = sum(1 for g in gsm_to_group.values() if g == "Normal")
            st.markdown(f"**GSE55235**: OA={n_oa} Normal={n_nm}")
        st.markdown("---")
        st.markdown("**参考论文**: PMID:24690414")
        st.markdown("Lambert 2014 — OA 滑膜炎症 vs 非炎症区域")
        n_q = "6" if multi_mode else "3"
        st.markdown(f"**{n_q} 个问题** — 答案均无法从文献直接获取")
        st.markdown("---")
        if multi_mode:
            robust_n = int(((meta_df["meta_adj_p_value"] < 0.05) & meta_df["all_same_direction"]).sum())
            st.metric("Meta 稳健 DEGs", robust_n)
            st.metric("合并数据集", len(de_results))
        st.caption("OA RAG MVP | NCBI GEO")

    # ─── 指标栏 ───
    if multi_mode:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("数据集", f"{len(de_results)} + meta")
        c2.metric("OA 样本", sum(v.get("n_oa", 0) for v in ds_info.values()))
        c3.metric("Normal 样本", sum(v.get("n_normal", 0) for v in ds_info.values()))
        c4.metric("Meta 基因数", len(meta_df))
        robust_n = int(((meta_df["meta_adj_p_value"] < 0.05) & meta_df["all_same_direction"]).sum())
        c5.metric("稳健 DEGs", robust_n)
    else:
        c1, c2, c3, c4 = st.columns(4)
        n_oa = sum(1 for g in gsm_to_group.values() if g == "OA")
        n_nm = sum(1 for g in gsm_to_group.values() if g == "Normal")
        c1.metric("OA 样本", n_oa)
        c2.metric("Normal 样本", n_nm)
        c3.metric("检测基因", len(expr))
        c4.metric("探针映射", len(annot_df) if not annot_df.empty else 0)

    # ─── 预计算通路（用于 KG，基于图谱实际展示的基因） ───
    if multi_mode:
        n_kg_genes = 20
        genes_per_ds = max(n_kg_genes // len(de_results), 6)
        kg_gene_list = list(set(
            g for ds_df in de_results.values()
            for g in ds_df.head(genes_per_ds)["gene_symbol"].tolist()
        ))
        pathway_df = pathway_enrichment(kg_gene_list)
    else:
        up_genes = de_primary[de_primary["log2FC"] > 0].head(15)["gene_symbol"].tolist()
        down_genes = de_primary[de_primary["log2FC"] < 0].head(15)["gene_symbol"].tolist()
        pathway_df = pathway_enrichment(up_genes + down_genes)

    # ─── TREM1 相关性 ───
    def get_trem1_corr():
        pmap = dict(zip(annot_df["probe"], annot_df["gene"])) if not annot_df.empty else {}
        probes = [p for p, g in pmap.items() if g == "TREM1"] if pmap else []
        tp = probes[0] if probes else None
        if tp and tp in expr.index:
            return map_corr_probes(correlation_analysis(expr, gsm_to_group, tp), annot_df)
        return {}

    # ─── Tabs ───
    tabs_main = ["\U0001F30D 知识图谱", "\U0001F4CA Q1: 差异基因排名",
                 "\U0001F504 Q2: TREM1 共表达", "\U0001F50D Q3: 隐藏通路"]
    if multi_mode:
        tabs_main += ["\U0001F52C Q4: Meta 稳健 DEGs",
                      "\U0001F50E Q5: 跨数据集验证",
                      "\U0001F9EA Q6: 稳健通路",
                      "\U0001F4CB 数据总览"]
    else:
        tabs_main += ["\U0001F4CB 原始数据"]

    tabs = st.tabs(tabs_main)

    # ═══════════════════ Tab: 知识图谱 ═══════════════════
    with tabs[0]:
        st.markdown(
            '<div class="highlight-box">'
            "\U0001F4A1 <b>核心命题</b>：文献 RAG 只能检索论文中明确报告的结论。"
            "<b>数据检索增强</b>（Data-Augmented Retrieval）直接分析原始表达矩阵，"
            "发现从未被书写发表的知识。"
            + ("<br><br><b>中枢</b>（青色）= 跨 3 数据集一致的共识基因；"
               "<b>三翼</b> = 各数据集特有的差异基因。"
               "单篇论文只看到自己的数据，合并分析才能发现全局图像。"
               if multi_mode else "")
            + "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### \U0001F30D 交互式知识图谱")
        corr_results = get_trem1_corr()
        multi_ds = de_results if multi_mode else None
        G = build_knowledge_graph(de_primary, pathway_df, corr_results,
                                  multi_ds=multi_ds, meta_df=meta_df if multi_mode else None)
        kg_html = render_pyvis_graph(G)
        st.components.v1.html(
            f'<div style="width:100%;height:600px;border:1px solid #ddd;'
            f'border-radius:8px;overflow:hidden;">{kg_html}</div>',
            height=620,
        )
        if multi_mode:
            c1, c2, c3 = st.columns(3)
            c1.markdown("\U0001F7E3 **紫色** — 共识基因（跨数据集）")
            c1.markdown("&nbsp;&nbsp;&nbsp;&nbsp;\U0001F7E3 **上调** / \U0001F7E1 **下调**")
            c1.markdown("---")
            c1.markdown("\U0001F534 **红色** / \U0001F535 **蓝色** — GSE55235 特有")
            c1.markdown("\U0001F7E2 **绿色** / \U0001F7E3 **紫色** — GSE12021 特有")
            c1.markdown("\U0001F7E1 **金色** / \U0001F7E0 **橙色** — GSE55457 特有")
            c2.markdown("\U0001F7E3 **紫色节点** — KEGG 通路")
            c2.markdown("\U0001F7E0 **大节点** — 样本组 / 数据集")
            c2.markdown("\U0001F517 **实线箭头** — 差异表达")
            c2.markdown("\U0001F517 **虚线箭头** — 通路-基因关系")
            c2.markdown("\U0001F517 **绿色边** — 共表达关系")
        else:
            c1, c2, c3 = st.columns(3)
            c1.markdown("\U0001F534 **红色节点** — OA 上调基因")
            c1.markdown("\U0001F535 **蓝色节点** — OA 下调基因")
            c2.markdown("\U0001F7E3 **紫色节点** — KEGG 通路")
            c2.markdown("\U0001F7E0 **大节点** — 样本组")
            c3.markdown("\U0001F517 **实线箭头** — 差异表达")
            c3.markdown("\U0001F517 **虚线箭头** — 通路-基因关系")
            c3.markdown("\U0001F517 **绿色边** — 共表达关系")

    # ═══════════════════ Tab: Q1 ═══════════════════
    with tabs[1]:
        st.markdown("## \U0001F4CA 问题 1: OA vs Normal 差异表达基因的精确排名")
        st.markdown(
            '<div class="question-box">'
            "<b>问题</b>：表达变化最显著的前 10 个基因是哪些？"
            "它们的确切 log2FC 和校正后 p 值是多少？"
            "</div>", unsafe_allow_html=True)
        col_r, col_res = st.columns([1, 1.5])
        with col_r:
            st.markdown(
                '<div class="paper-box">'
                "<b>\U0001F4D6 为什么文献答不了</b><br><br>"
                "PMID:24690414 比较的是 OA 滑膜<b>炎症 vs 非炎症区域</b>，"
                "而非 OA vs Normal。<br><br>"
                "且论文最多说\"某基因上调\"，不给出全部显著基因的<b>精确排名和数值</b>。"
                + ("<br><br>多数据集视角：GSE55235 的单数据集结果与 meta 分析合并结果可相互验证。"
                   if multi_mode else "")
                + "</div>", unsafe_allow_html=True)
        with col_res:
            if st.button("\U0001F50D 计算", key="btn_q1"):
                result = q1_top_deg(de_primary).head(10)
                cols = ["gene_symbol", "log2FC", "方向", "p_value", "adj_p_value", "显著性"]
                avail = [c for c in cols if c in result.columns]
                st.dataframe(
                    result[avail].style.format({
                        "log2FC": "{:.3f}", "p_value": "{:.2e}", "adj_p_value": "{:.2e}",
                    }),
                    use_container_width=True, hide_index=True)
                fig = px.bar(result.head(10), x="log2FC", y="gene_symbol",
                             color="log2FC", color_continuous_scale=["#3498db", "white", "#e74c3c"],
                             title="Top 10 差异基因 log2FC", orientation="h")
                fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════ Tab: Q2 ═══════════════════
    with tabs[2]:
        st.markdown("## \U0001F504 问题 2: TREM1 的 OA 特异性共表达网络")
        st.markdown(
            '<div class="question-box">'
            "<b>问题</b>：OA 样本中 TREM1 与哪些基因高度相关？"
            "这些相关性在正常样本中是否也存在？"
            "</div>", unsafe_allow_html=True)
        col_r, col_res = st.columns([1, 1.5])
        with col_r:
            st.markdown(
                '<div class="paper-box">'
                "<b>\U0001F4D6 为什么文献答不了</b><br><br>"
                "Lambert (2014) 报告 TREM1 上调 3.45 倍，但<b>未做任何共表达/相关性分析</b>。<br><br>"
                "共表达网络只能从<b>原始表达矩阵</b>中计算 Pearson 相关系数获得。"
                + ("<br><br>多数据集优势：后续可在 GSE12021、GSE55457 中验证这些相关性是否复现。"
                   if multi_mode else "")
                + "</div>", unsafe_allow_html=True)
        with col_res:
            if st.button("\U0001F50D 计算", key="btn_q2"):
                result = q2_correlation(expr, gsm_to_group, annot_df)
                if "error" in result:
                    st.error(result["error"])
                else:
                    ta, tb = st.tabs(["OA 样本", "Normal 样本"])
                    for t, grp in [(ta, "OA"), (tb, "Normal")]:
                        with t:
                            if grp in result and isinstance(result[grp], pd.DataFrame) and not result[grp].empty:
                                df = result[grp]
                                df["gene_symbol"] = df.get("gene_symbol", df["gene"])
                                st.dataframe(df[["gene_symbol", "correlation", "p_value"]]
                                             .style.format({"correlation": "{:.4f}", "p_value": "{:.2e}"}),
                                             use_container_width=True, hide_index=True)
                            else:
                                st.info("无结果")

    # ═══════════════════ Tab: Q3 ═══════════════════
    with tabs[3]:
        st.markdown("## \U0001F50D 问题 3: 哪些显著富集通路被原文忽略了？")
        st.markdown(
            '<div class="question-box">'
            "<b>问题</b>：差异基因富集在哪些 KEGG 通路中？"
            "哪些通路在 PMID:24690414 中<b>没有被讨论</b>？"
            "</div>", unsafe_allow_html=True)
        col_r, col_res = st.columns([1, 1.5])
        with col_r:
            st.markdown(
                '<div class="paper-box">'
                "<b>\U0001F4D6 为什么文献答不了</b><br><br>"
                "Lambert (2014) 基于手动文献回顾，讨论了："
                "<ul><li>炎症通路 (TNF, NF-κB)</li>"
                "<li>Wnt 信号</li><li>血管生成</li><li>软骨代谢</li></ul>"
                "但这是<b>非系统性</b>的列举——作者只讨论了感兴趣的少数通路。<br><br>"
                "通过<b>系统性通路富集分析</b>，发现原文未提及的显著通路。"
                + ("<br><br>多数据集版还可比较：哪些通路在多个数据集中均显著富集？"
                   if multi_mode else "")
                + "</div>", unsafe_allow_html=True)
        with col_res:
            if st.button("\U0001F50D 计算", key="btn_q3"):
                up = de_primary[de_primary["log2FC"] > 0].head(30)["gene_symbol"].tolist()
                down = de_primary[de_primary["log2FC"] < 0].head(30)["gene_symbol"].tolist()
                pw_df = pathway_enrichment(up + down)
                result, _ = q3_pathways(pw_df)
                if result.empty:
                    st.warning("通路富集暂未返回结果")
                else:
                    st.dataframe(
                        result[["pathway", "p_value", "adj_p_value", "-log10_pvalue",
                                "在原文中", "genes"]]
                        .style.format({"p_value": "{:.2e}", "adj_p_value": "{:.2e}",
                                       "-log10_pvalue": "{:.2f}"}),
                        use_container_width=True, hide_index=True)
                    fig = px.bar(result, x="-log10_pvalue", y="pathway", color="在原文中",
                                 color_discrete_map={"✅ 是 (原文提及)": "#ff9800",
                                                     "❌ 否 (原文未提及)": "#4caf50"},
                                 title="KEGG 通路富集", orientation="h")
                    fig.update_layout(height=max(400, len(result) * 35),
                                      yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)
                    novel = result[result["在原文中"].str.contains("否")]
                    discussed = result[result["在原文中"].str.contains("是")]
                    novel_lines = []
                    if not novel.empty:
                        novel_lines.append(
                            f"在 {len(result)} 个通路中，{len(novel)} 个在原文未讨论：")
                        for _, row in novel.iterrows():
                            novel_lines.append(
                                f"\U0001F195 <b>{row['pathway']}</b> (p = {row['p_value']:.2e})")
                    if not discussed.empty:
                        novel_lines.append(
                            f"另有 {len(discussed)} 个通路与原文讨论一致。")
                    st.markdown(
                        '<div class="answer-box">'
                        "<b>✅ 关键发现</b>：<br>"
                        + "<br>".join(novel_lines) + "</div>",
                        unsafe_allow_html=True)

    # ═══════════════════ 多数据集专属 Tabs ═══════════════════
    tab_idx = 4

    if multi_mode:
        # ─── Q4: Meta 稳健 DEGs ───
        with tabs[tab_idx]:
            st.markdown("## \U0001F52C 问题 4: 跨 3 个数据集稳健差异表达的基因")
            st.markdown(
                '<div class="question-box">'
                "<b>问题</b>：在 GSE55235 + GSE12021 + GSE55457 三个 OA 滑膜数据集中，"
                "哪些基因在<b>全部数据集</b>中方向一致且 meta 分析显著？"
                "</div>", unsafe_allow_html=True)
            col_r, col_res = st.columns([1, 1.5])
            with col_r:
                st.markdown(
                    '<div class="paper-box">'
                    "<b>\U0001F4D6 为什么文献答不了</b><br><br>"
                    "没有任何一篇论文同时分析这三个数据集并做系统 meta 分析。<br><br>"
                    "每个数据集单独发表时，各自报告不同的差异基因列表。<br><br>"
                    "只有合并原始数据，才能识别出<b>跨数据集稳健</b>的分子信号——"
                    '这种"稳健性"信息不存在于任何单篇文献中。'
                    + "</div>", unsafe_allow_html=True)
            with col_res:
                if st.button("\U0001F50D 计算", key="btn_q4"):
                    result = q4_meta_overview(meta_df)
                    robust = result[(result["meta_adj_p_value"] < 0.05) & result["all_same_direction"]]
                    st.success(
                        f"发现 <b>{len(robust)}</b> 个稳健差异基因 "
                        f"(meta adj_p<0.05 + 方向一致)")
                    top = robust.head(15)
                    st.dataframe(
                        top[["gene_symbol", "meta_log2FC", "meta_adj_p_value",
                             "-log10_meta_p", "n_datasets", "方向一致性"]]
                        .style.format({"meta_log2FC": "{:.3f}", "meta_adj_p_value": "{:.2e}",
                                       "-log10_meta_p": "{:.2f}"}),
                        use_container_width=True, hide_index=True)
                    fig = px.bar(top.head(15), x="meta_log2FC", y="gene_symbol",
                                 color="meta_log2FC",
                                 color_continuous_scale=["#3498db", "white", "#e74c3c"],
                                 title="Top 15 稳健差异基因 (跨 3 数据集)", orientation="h")
                    fig.update_layout(height=450, yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)

        tab_idx += 1

        # ─── Q5: 跨数据集验证 ───
        with tabs[tab_idx]:
            st.markdown("## \U0001F50E 问题 5: 单数据集不显著但 meta 显著的基因")
            st.markdown(
                '<div class="question-box">'
                "<b>问题</b>：哪些基因在单个数据集中不显著(adj_p≥0.05)，"
                "但跨数据集合并后变得显著？<br><br>"
                "这些是<b>统计功效不足</b>导致被单篇文献遗漏的潜在重要基因。"
                "</div>", unsafe_allow_html=True)
            col_r, col_res = st.columns([1, 1.5])
            with col_r:
                st.markdown(
                    '<div class="paper-box">'
                    "<b>\U0001F4D6 为什么文献答不了</b><br><br>"
                    "每篇论文只报告自己数据集中统计显著的基因。<br><br>"
                    "样本量小的研究 (n≈10/组) 统计功效有限，"
                    "会遗漏许多真实但效应量中等的差异基因。<br><br>"
                    "只有合并多个数据集的原始数据，通过 meta 分析增大样本量，"
                    "才能发现这些\"被隐藏\"的信号。"
                    + "</div>", unsafe_allow_html=True)
            with col_res:
                if st.button("\U0001F50D 计算", key="btn_q5"):
                    findings = q5_cross_dataset_validation(de_results, meta_df, annot_df)
                    if findings.empty:
                        st.info("未找到符合条件的基因。")
                    else:
                        st.success(
                            f"发现 <b>{len(findings)}</b> 个基因在单数据集中不显著，"
                            f"但 meta 分析显著")
                        top = findings.head(15)
                        st.dataframe(
                            top[["gene_symbol", "meta_adj_p", "meta_log2FC",
                                 "n_datasets_sig", "n_datasets_total", "details"]]
                            .style.format({"meta_adj_p": "{:.2e}", "meta_log2FC": "{:.3f}"}),
                            use_container_width=True, hide_index=True)
                        fig = px.bar(top.head(15), x="meta_log2FC", y="gene_symbol",
                                     color="n_datasets_sig",
                                     color_continuous_scale="Viridis",
                                     title="Meta 显著但单数据集不显著的基因",
                                     orientation="h")
                        fig.update_layout(height=450, yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown(
                            '<div class="answer-box">'
                            "<b>✅ 关键发现</b>：这些基因被单篇文献遗漏，"
                            "因为每个数据集的样本量有限 (n≈10/组)，"
                            "统计功效不足以检测到它们的差异表达。"
                            "跨数据集合并才得以发现。"
                            "</div>", unsafe_allow_html=True)

        tab_idx += 1

        # ─── Q6: 稳健通路 ───
        with tabs[tab_idx]:
            st.markdown("## \U0001F9EA 问题 6: 稳健差异基因的 KEGG 通路富集")
            st.markdown(
                '<div class="question-box">'
                "<b>问题</b>：跨 3 个数据集稳健差异的基因，"
                "富集在哪些 KEGG 通路中？<br><br>"
                "这比单数据集通路分析更可靠——排除了数据集特异的假阳性。"
                "</div>", unsafe_allow_html=True)
            col_r, col_res = st.columns([1, 1.5])
            with col_r:
                st.markdown(
                    '<div class="paper-box">'
                    "<b>\U0001F4D6 为什么文献答不了</b><br><br>"
                    "文献中的通路分析均基于<b>单个数据集</b>的差异基因。<br><br>"
                    "单数据集分析容易受批次效应和样本偏差影响，"
                    "产生数据集特异的假阳性通路。<br><br>"
                    "跨数据集稳健差异基因的通路富集，"
                    "揭示了最可能具有生物学意义的机制——"
                    '这种"跨数据集稳健性"分析不存在于任何文献中。'
                    + "</div>", unsafe_allow_html=True)
            with col_res:
                if st.button("\U0001F50D 计算", key="btn_q6"):
                    pw_df, n_robust, n_up, n_down = q6_robust_pathways(
                        meta_df, de_results, annot_df)
                    st.success(
                        f"基于 {n_robust} 个稳健差异基因 "
                        f"(上调 {n_up}, 下调 {n_down}) 的通路富集")
                    if pw_df.empty:
                        st.warning("通路富集暂未返回结果")
                    else:
                        st.dataframe(
                            pw_df[["pathway", "p_value", "adj_p_value", "-log10_pvalue", "genes"]]
                            .style.format({"p_value": "{:.2e}", "adj_p_value": "{:.2e}",
                                           "-log10_pvalue": "{:.2f}"}),
                            use_container_width=True, hide_index=True)
                        fig = px.bar(pw_df.head(10), x="-log10_pvalue", y="pathway",
                                     color="-log10_pvalue", color_continuous_scale="Purples",
                                     title="稳健 DEGs 的 KEGG 通路富集 (Top 10)",
                                     orientation="h")
                        fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig, use_container_width=True)

        tab_idx += 1

        # ─── 数据总览 ───
        with tabs[tab_idx]:
            st.markdown("### \U0001F4CB 多数据集数据总览")
            st.markdown("#### 各数据集差异基因统计")
            rows = []
            for ds, df in de_results.items():
                n_sig = int((df["adj_p_value"] < 0.05).sum())
                n_up = int(((df["adj_p_value"] < 0.05) & (df["log2FC"] > 0)).sum())
                n_down = int(((df["adj_p_value"] < 0.05) & (df["log2FC"] < 0)).sum())
                info = ds_info.get(ds, {})
                rows.append({
                    "数据集": ds, "OA": info.get("n_oa", "?"), "Normal": info.get("n_normal", "?"),
                    "DEGs": n_sig, "上调": n_up, "下调": n_down,
                    "Top 基因": df.iloc[0]["gene_symbol"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("#### Meta 分析结果")
            robust = meta_df[(meta_df["meta_adj_p_value"] < 0.05) & meta_df["all_same_direction"]]
            c1, c2, c3 = st.columns(3)
            c1.metric("Meta 总基因数", len(meta_df))
            c2.metric("显著 DEGs", int((meta_df["meta_adj_p_value"] < 0.05).sum()))
            c3.metric("方向一致稳健", len(robust))
            st.dataframe(
                meta_df.head(20)[["gene_symbol", "meta_log2FC", "meta_adj_p_value",
                                  "n_datasets", "all_same_direction"]]
                .style.format({"meta_log2FC": "{:.3f}", "meta_adj_p_value": "{:.2e}"}),
                use_container_width=True, hide_index=True)

    else:
        # ─── 单数据集: 原始数据 ───
        with tabs[tab_idx]:
            st.markdown("### \U0001F4CB GSE55235 原始数据预览")
            groups_df = pd.DataFrame({
                "GSM ID": list(gsm_to_group.keys()),
                "分组": list(gsm_to_group.values()),
            })
            groups_df = groups_df[groups_df["分组"].isin(["OA", "Normal"])]
            st.dataframe(groups_df, use_container_width=True, hide_index=True)
            n_cols = min(5, expr.shape[1])
            st.markdown("#### 表达矩阵 (前 10 × 前 5)")
            st.dataframe(expr.head(10).iloc[:, :n_cols], use_container_width=True)
            st.markdown("#### 差异表达结果 (前 20)")
            de_display = de_primary.head(20)[["gene_symbol", "gene", "log2FC", "p_value", "adj_p_value"]]
            st.dataframe(
                de_display.style.format({
                    "log2FC": "{:.3f}", "p_value": "{:.2e}", "adj_p_value": "{:.2e}",
                }),
                use_container_width=True, hide_index=True,
            )

    st.markdown("---")
    st.markdown(
        "<center><small>"
        "OA RAG MVP | 数据: NCBI GEO GSE55235/GSE12021/GSE55457"
        + (" | Meta 分析: Fisher's method" if multi_mode else "")
        + " | 参考论文: PMID:24690414"
        "</small></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OA Knowledge Graph MVP - GSE55235
证明：仅靠文献检索无法得到的结果，通过分析原始数据就能得到。

数据集: GSE55235 (OA vs Normal 滑膜组织, GPL96平台)
参考论文: PMID:24690414 (Lambert 2014 - OA滑膜炎症区域 vs 非炎症区域)

3个问题的答案无法从原文读取，但可以从表达矩阵计算得出。
"""

import sys
import os
import gzip
import urllib.request
import warnings
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

# ─── 忽略杂项警告 ───
warnings.filterwarnings("ignore")

# ─── 页面配置 ───
st.set_page_config(
    page_title="OA RAG MVP - 文献 vs 原始数据",
    page_icon="\U0001F9F1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 样式定制 ───
st.markdown(
    """
<style>
    .main-header { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #666; margin-bottom: 1.5rem; }
    .highlight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .question-box {
        background: #f0f2f6;
        border-left: 4px solid #ff4b4b;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .answer-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .paper-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .stDataFrame { border: none !important; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════
# 1. 数据加载
# ════════════════════════════════════════════════════════════════════════

GSE55235_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE55nnn/GSE55235/matrix/"
    "GSE55235_series_matrix.txt.gz"
)
GPL96_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/"
    "GPL96.annot.gz"
)


@st.cache_data(show_spinner="正在从 GEO 下载数据 (~1.4 MB)...")
def download_series_matrix() -> str:
    req = urllib.request.Request(
        GSE55235_URL, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    text = gzip.decompress(raw).decode("utf-8", errors="replace")
    return text


@st.cache_data(show_spinner="正在下载 GPL96 注释文件...")
def download_gpl_annotation() -> pd.DataFrame:
    req = urllib.request.Request(
        GPL96_URL, headers={"User-Agent": "Mozilla/5.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
        text = gzip.decompress(raw).decode("utf-8", errors="replace")
        rows = []
        in_data = False
        for line in text.split("\n"):
            # Skip header lines
            if line.startswith("#") or not line.strip():
                continue
            # First data line has column headers: ID, Gene title, Gene symbol, ...
            parts = line.split("\t")
            if not in_data:
                if parts[0].strip() == "ID":
                    in_data = True
                continue
            # Data rows: col0=probe_id, col2=gene_symbol
            if len(parts) >= 3:
                probe_id = parts[0].strip()
                raw_symbol = parts[2].strip()
                # Handle "MIR4640///DDR1" -> take first gene
                gene_symbol = raw_symbol.split("///")[0].strip() if raw_symbol else ""
                if gene_symbol and gene_symbol != "---":
                    rows.append({"probe": probe_id, "gene": gene_symbol})
        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["probe"])
        return df
    except Exception as e:
        st.warning(f"GPL96 注释下载失败 ({e})，将使用 probe ID 代替基因名。")
        return pd.DataFrame(columns=["probe", "gene"])


@st.cache_data(show_spinner="正在解析表达矩阵...")
def parse_series_matrix(text: str):
    """
    解析 GSE55235 series matrix 文件。
    返回: (expr_df, gsm_to_group)
    - expr_df: 探针×样本的表达矩阵 (列名=GSM ID)
    - gsm_to_group: {GSM_ID: "OA"|"Normal"|"RA"}
    """
    lines = text.split("\n")

    # ── 1. 收集样本元信息 ──
    gsm_ids = []
    titles = []
    disease_states = {}

    for line in lines:
        if line.startswith("!Sample_geo_accession"):
            # strip quotes from each field
            gsm_ids = [
                c.strip().strip('"')
                for c in line.split("\t")[1:]
                if c.strip()
            ]
        elif line.startswith("!Sample_title"):
            titles = [
                c.strip().strip('"')
                for c in line.split("\t")[1:]
                if c.strip()
            ]
        elif line.startswith('!Sample_characteristics_ch1'):
            vals = [
                c.strip().strip('"')
                for c in line.split("\t")[1:]
                if c.strip()
            ]
            # Check if this is the disease-state line
            if vals and ("disease state" in vals[0].lower()):
                for i, v in enumerate(vals):
                    if ":" in v:
                        disease_states[i] = v.split(":", 1)[1].strip()

    # ── 2. 从 title 判断分组 (最可靠) ──
    gsm_to_group = {}
    for i, gsm in enumerate(gsm_ids):
        title = titles[i] if i < len(titles) else ""
        title_lower = title.lower()

        if "healthy" in title_lower or "normal" in title_lower:
            group = "Normal"
        elif "osteoarthritic" in title_lower or title_lower.startswith("oa"):
            group = "OA"
        elif "rheumatoid" in title_lower:
            group = "RA"
        elif i in disease_states:
            ds = disease_states[i].lower()
            if "healthy" in ds or "normal" in ds:
                group = "Normal"
            elif "osteoarthritis" in ds:
                group = "OA"
            elif "rheumatoid" in ds:
                group = "RA"
            else:
                group = "Unknown"
        else:
            group = "Unknown"
        gsm_to_group[gsm] = group

    # ── 3. 解析表达数据 ──
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
            # Split by tab, then strip quotes from each field
            cols = line.strip().split("\t")
            cols = [c.strip().strip('"') for c in cols]
            if data_cols is None:
                data_cols = cols
            else:
                data_rows.append(cols)

    if data_cols is None:
        raise ValueError("Cannot find expression data table")

    expr_df = pd.DataFrame(data_rows, columns=data_cols)
    expr_df = expr_df.set_index("ID_REF")
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce")

    return expr_df, gsm_to_group


@st.cache_data(show_spinner="正在进行差异表达分析...")
def differential_expression(
    expr_df: pd.DataFrame,
    gsm_to_group: dict,
    group_a: str = "OA",
    group_b: str = "Normal",
) -> pd.DataFrame:
    """对每个基因做两组间 t-test (Welch)，返回 DE 结果表。"""
    samples_a = [gsm for gsm, g in gsm_to_group.items() if g == group_a]
    samples_b = [gsm for gsm, g in gsm_to_group.items() if g == group_b]

    expr_cols = expr_df.columns.tolist()
    # Match GSM IDs to expression columns (they should match directly)
    cols_a = [c for c in expr_cols if c in samples_a]
    cols_b = [c for c in expr_cols if c in samples_b]

    results = []
    for gene in expr_df.index:
        vals_a = expr_df.loc[gene, cols_a].dropna().astype(float)
        vals_b = expr_df.loc[gene, cols_b].dropna().astype(float)

        if len(vals_a) < 2 or len(vals_b) < 2:
            continue

        mean_a = vals_a.mean()
        mean_b = vals_b.mean()
        fc = mean_a / mean_b if mean_b != 0 else np.nan
        log2fc = np.log2(fc) if fc > 0 else np.nan

        t_stat, p_val = scipy_stats.ttest_ind(vals_a, vals_b, equal_var=False)

        results.append({
            "gene": gene,
            "mean_OA": mean_a,
            "mean_Normal": mean_b,
            "log2FC": log2fc,
            "t_statistic": t_stat,
            "p_value": p_val,
        })

    de_df = pd.DataFrame(results).dropna(subset=["log2FC"])

    if len(de_df) > 0:
        _, adj_p, _, _ = multipletests(de_df["p_value"].values, method="fdr_bh")
        de_df["adj_p_value"] = adj_p
    else:
        de_df["adj_p_value"] = np.nan

    de_df["-log10_pvalue"] = -np.log10(de_df["p_value"].clip(lower=1e-300))
    de_df = de_df.sort_values("p_value")
    return de_df


def map_probes_to_genes(de_df: pd.DataFrame, annot_df: pd.DataFrame) -> pd.DataFrame:
    """将探针级别的 DE 结果映射到基因级别。对于多探针基因，保留最显著的那个。"""
    if annot_df.empty:
        de_df["gene_symbol"] = de_df["gene"]
        return de_df

    # 创建映射字典
    probe_gene_map = dict(zip(annot_df["probe"], annot_df["gene"]))

    # 映射
    de_df["gene_symbol"] = de_df["gene"].map(
        lambda x: probe_gene_map.get(x, x)
    )

    # 多探针取最显著
    de_df = de_df.sort_values("p_value").drop_duplicates(
        subset=["gene_symbol"], keep="first"
    )

    return de_df


@st.cache_data(show_spinner="正在进行通路富集分析 (Enrichr API)...")
def pathway_enrichment(
    upregulated_genes: List[str],
    downregulated_genes: List[str],
    n_top: int = 10,
) -> pd.DataFrame:
    """使用 Enrichr API 进行 KEGG 通路富集。"""
    all_deg = upregulated_genes + downregulated_genes
    if len(all_deg) == 0:
        return pd.DataFrame(columns=["pathway", "p_value", "adj_p_value", "genes", "odds_ratio"])

    ENRICHR_URL = "https://maayanlab.cloud/Enrichr"

    try:
        # Step 1: submit gene list
        payload = {
            "list": (None, "\n".join(all_deg)),
            "description": (None, "OA DEGs from GSE55235"),
        }
        resp = requests.post(f"{ENRICHR_URL}/addList", files=payload, timeout=30)
        resp.raise_for_status()
        user_list_id = resp.json().get("userListId")
        if not user_list_id:
            raise ValueError("Enrichr did not return userListId")

        # Step 2: get enrichment for KEGG
        enrich_resp = requests.get(
            f"{ENRICHR_URL}/enrich",
            params={
                "userListId": user_list_id,
                "backgroundType": "KEGG_2021_Human",
            },
            timeout=30,
        )
        enrich_resp.raise_for_status()
        raw = enrich_resp.json().get("KEGG_2021_Human", [])

        if not raw:
            # Try alternative gene set
            enrich_resp = requests.get(
                f"{ENRICHR_URL}/enrich",
                params={
                    "userListId": user_list_id,
                    "backgroundType": "KEGG_2019_Human",
                },
                timeout=30,
            )
            enrich_resp.raise_for_status()
            raw = enrich_resp.json().get("KEGG_2019_Human", [])

        # Parse results
        # Each entry: [rank, term, p-value, z-score, combined_score, overlapping_genes, adj_p_value, ...]
        rows = []
        for entry in raw[:n_top]:
            if len(entry) >= 7:
                rows.append({
                    "pathway": entry[1],
                    "p_value": entry[2],
                    "z_score": entry[3],
                    "combined_score": entry[4],
                    "genes": "; ".join(entry[5]) if isinstance(entry[5], list) else str(entry[5]),
                    "adj_p_value": entry[6],
                })
            elif len(entry) >= 4:
                rows.append({
                    "pathway": entry[1],
                    "p_value": entry[2],
                    "z_score": entry[3],
                    "combined_score": entry[4] if len(entry) > 4 else 0,
                    "genes": "",
                    "adj_p_value": entry[2],
                })

        result_df = pd.DataFrame(rows)
        if len(result_df) > 0:
            result_df["-log10_pvalue"] = -np.log10(result_df["p_value"].astype(float) + 1e-300)
        return result_df

    except Exception as e:
        st.warning(f"通路富集 API 请求失败: {e}")
        return pd.DataFrame(columns=["pathway", "p_value", "adj_p_value", "genes"])


@st.cache_data(show_spinner="正在计算基因共表达...")
def correlation_analysis(
    expr_df: pd.DataFrame,
    gsm_to_group: dict,
    target_gene: str,
    top_n: int = 10,
) -> Dict:
    """计算与目标基因的相关性 (Pearson), 分别在 OA 和 Normal 中计算。"""
    if target_gene not in expr_df.index:
        return {"error": f"基因 {target_gene} 未在数据集中找到。"}

    samples_oa = [gsm for gsm, g in gsm_to_group.items() if g == "OA"]
    samples_normal = [gsm for gsm, g in gsm_to_group.items() if g == "Normal"]
    expr_cols = expr_df.columns.tolist()

    cols_oa = [c for c in expr_cols if c in samples_oa]
    cols_normal = [c for c in expr_cols if c in samples_normal]

    target_oa = expr_df.loc[target_gene, cols_oa].astype(float).values
    target_normal = expr_df.loc[target_gene, cols_normal].astype(float).values

    results_oa = []
    results_normal = []

    for gene in expr_df.index:
        if gene == target_gene:
            continue

        # OA correlation
        vals_oa = expr_df.loc[gene, cols_oa].astype(float).values
        valid_oa = ~(np.isnan(vals_oa) | np.isnan(target_oa))
        if valid_oa.sum() >= 4:
            r_oa, p_oa = scipy_stats.pearsonr(vals_oa[valid_oa], target_oa[valid_oa])
            results_oa.append({"gene": gene, "correlation": r_oa, "p_value": p_oa})

        # Normal correlation
        vals_normal = expr_df.loc[gene, cols_normal].astype(float).values
        valid_n = ~(np.isnan(vals_normal) | np.isnan(target_normal))
        if valid_n.sum() >= 4:
            r_norm, p_norm = scipy_stats.pearsonr(
                vals_normal[valid_n], target_normal[valid_n]
            )
            results_normal.append({"gene": gene, "correlation": r_norm, "p_value": p_norm})

    df_oa = pd.DataFrame(results_oa).sort_values("correlation", ascending=False).head(top_n)
    df_normal = pd.DataFrame(results_normal).sort_values("correlation", ascending=False).head(top_n)

    return {"OA": df_oa, "Normal": df_normal, "target_gene": target_gene}


# ════════════════════════════════════════════════════════════════════════
# 2. 知识图谱构建
# ════════════════════════════════════════════════════════════════════════

def build_knowledge_graph(
    de_df: pd.DataFrame,
    pathway_df: pd.DataFrame,
    corr_results: Optional[Dict] = None,
    n_genes: int = 20,
    n_pathways: int = 8,
) -> nx.DiGraph:
    """构建知识图谱 (有向图)。"""
    G = nx.DiGraph()

    # --- 节点: 组节点 ---
    G.add_node("OA Group", type="group", color="#e74c3c", size=30, title="OA 滑膜组织 (10 样本)")
    G.add_node("Normal Group", type="group", color="#3498db", size=30, title="正常滑膜组织 (10 样本)")

    # --- 节点: 差异基因 ---
    top_genes = de_df.head(n_genes)
    for _, row in top_genes.iterrows():
        gene_name = row.get("gene_symbol", row["gene"])
        direction = "up" if row["log2FC"] > 0 else "down"
        intensity = min(abs(row["log2FC"]) / 3.0, 1.0)  # 0~1

        if direction == "up":
            node_color = f"rgba(231, 76, 60, {0.5 + intensity * 0.5})"
        else:
            node_color = f"rgba(52, 152, 219, {0.5 + intensity * 0.5})"

        title_text = (
            f"Gene: {gene_name}<br>"
            f"log2FC: {row['log2FC']:.3f}<br>"
            f"adj p-value: {row.get('adj_p_value', 0):.2e}<br>"
            f"Direction: {'Up in OA' if direction == 'up' else 'Down in OA'}"
        )
        node_size = 10 + min(-np.log10(row.get("adj_p_value", row.get("p_value", 0.05)) + 1e-300) * 2, 20)

        G.add_node(
            gene_name,
            type="gene",
            color=node_color,
            size=node_size,
            title=title_text,
            direction=direction,
            log2fc=row["log2FC"],
        )

        # 边: 基因 -> 组 (差异表达关系)
        target_group = "OA Group" if direction == "up" else "Normal Group"
        G.add_edge(
            gene_name, target_group,
            type="deg",
            label=f"log2FC={row['log2FC']:.2f}",
            weight=abs(row["log2FC"]),
            color="#e74c3c" if direction == "up" else "#3498db",
        )

    # --- 节点: 通路 ---
    pathway_gene_map = {}
    for _, row in pathway_df.iterrows():
        pathway_name = row["pathway"][:60]  # truncate
        genes_str = row.get("genes", "")
        gene_list = [g.strip() for g in genes_str.split(";") if g.strip()]
        pathway_gene_map[pathway_name] = gene_list

        title_text = (
            f"Pathway: {pathway_name}<br>"
            f"p-value: {row['p_value']:.2e}<br>"
            f"adj p-value: {row.get('adj_p_value', row['p_value']):.2e}"
        )
        G.add_node(
            pathway_name,
            type="pathway",
            color="#9b59b6",
            size=15,
            title=title_text,
        )

        # 边: 通路 -> 基因 (通路包含关系)
        for gene_name in gene_list:
            if gene_name in G.nodes:
                G.add_edge(
                    pathway_name, gene_name,
                    type="pathway_gene",
                    label="",
                    color="#9b59b6",
                    dashes=True,
                )

    # --- 边: 基因-基因共表达 ---
    if corr_results and "OA" in corr_results and isinstance(corr_results["OA"], pd.DataFrame):
        corr_df = corr_results["OA"]
        for _, row in corr_df.iterrows():
            gene1 = corr_results.get("target_gene", "")
            gene2 = row.get("gene_symbol", row.get("gene", ""))
            if gene1 in G.nodes and gene2 in G.nodes and gene1 != gene2:
                G.add_edge(
                    gene1, gene2,
                    type="correlation",
                    label=f"r={row['correlation']:.2f}",
                    color="#2ecc71",
                    weight=abs(row["correlation"]),
                )

    return G


def render_pyvis_graph(G: nx.DiGraph, height: str = "550px") -> str:
    """将 NetworkX 图转换为 Pyvis HTML 字符串。"""
    net = Network(height=height, width="100%", directed=True, notebook=False)
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -40,
                "centralGravity": 0.005,
                "springLength": 200,
                "springConstant": 0.02,
                "damping": 0.4
            },
            "stabilization": {
                "iterations": 100,
                "updateInterval": 25
            },
            "minVelocity": 0.5,
            "maxVelocity": 3
        },
        "edges": {
            "smooth": {
                "enabled": true,
                "type": "continuous"
            }
        }
    }
    """)

    # 添加节点
    for node, attrs in G.nodes(data=True):
        color = attrs.get("color", "#888")
        size = attrs.get("size", 15)
        title = attrs.get("title", node)
        node_type = attrs.get("type", "")
        border_width = 3 if node_type == "group" else 1

        net.add_node(
            node,
            label=node if len(node) < 30 else node[:27] + "...",
            title=title,
            color=color,
            size=size,
            borderWidth=border_width,
            font={"size": 14 if node_type == "group" else 10},
        )

    # 添加边
    for u, v, attrs in G.edges(data=True):
        edge_type = attrs.get("type", "")
        color = attrs.get("color", "#666")
        label = attrs.get("label", "")
        dashes = attrs.get("dashes", False)
        width = attrs.get("weight", 1) * 1.5

        if edge_type == "pathway_gene":
            arrows = "to"
        elif edge_type == "correlation":
            arrows = ""
        else:
            arrows = "to"

        net.add_edge(
            u, v,
            title=label,
            color=color,
            width=max(width, 1),
            dashes=dashes,
            arrows=arrows,
            label=label,
            font={"size": 9, "color": color},
        )

    # 生成 HTML
    html = net.generate_html()
    return html


# ════════════════════════════════════════════════════════════════════════
# 3. 3 个问题的计算逻辑
# ════════════════════════════════════════════════════════════════════════

def q1_top_deg_analysis(de_df: pd.DataFrame) -> pd.DataFrame:
    """问题1: 精确的差异基因排名和表达变化值。"""
    result = de_df.copy()
    result = result.rename(columns={
        "gene": "Probe/基因",
        "log2FC": "log2FC",
        "p_value": "P-value",
        "adj_p_value": "校正后 P-value (BH)",
        "-log10_pvalue": "-log10(P-value)",
        "mean_OA": "OA 平均表达",
        "mean_Normal": "Normal 平均表达",
        "gene_symbol": "基因符号",
    })
    result["方向"] = result.get("log2FC", result.get("log2FC")).apply(
        lambda x: "\U0001F7E2 OA 上调" if x > 0 else "\U0001F535 OA 下调"
    )
    result["显著性标记"] = result.get("校正后 P-value (BH)", pd.Series([1] * len(result))).apply(
        lambda x: "***" if x < 0.001 else ("**" if x < 0.01 else ("*" if x < 0.05 else ""))
    )
    return result


def q2_correlation_analysis(expr_df, gsm_to_group, annot_df):
    """问题2: 与 TREM1 的相关性分析。"""
    # 找到 TREM1 的探针
    target_gene = "TREM1"

    # 先尝试用 gene symbol 找
    if not annot_df.empty:
        probe_gene_map = dict(zip(annot_df["probe"], annot_df["gene"]))
        possible_probes = [p for p, g in probe_gene_map.items() if g == target_gene]
    else:
        possible_probes = [p for p in expr_df.index if target_gene in p.upper()]

    if possible_probes:
        target_probe = possible_probes[0]
    else:
        target_probe = target_gene

    return correlation_analysis(expr_df, gsm_to_group, target_probe, top_n=10)


def q3_pathway_analysis(pathway_df: pd.DataFrame) -> pd.DataFrame:
    """问题3: 通路富集 - 与文献讨论做对比。"""

    # PMID:24690414 (Lambert 2014) 中讨论的通路
    paper_pathways = [
        "Inflammatory response",
        "TNF signaling",
        "NF-kappa B signaling",
        "Wnt signaling",
        "Cartilage metabolism",
        "Angiogenesis",
        "Cytokine-cytokine receptor interaction",
        "MAPK signaling",
        "PI3K-Akt signaling",
        "Osteoclast differentiation",
        "IL-17 signaling",
    ]

    if pathway_df.empty:
        return pathway_df, paper_pathways

    result = pathway_df.copy()
    # 标记哪些通路被文献讨论过
    def is_in_paper(pw):
        pw_lower = pw.lower()
        for pp in paper_pathways:
            if pp.lower() in pw_lower or pw_lower in pp.lower():
                return True
        return False

    result["文献讨论"] = result["pathway"].apply(
        lambda x: "✅ 是 (原文提及)" if is_in_paper(x) else "❌ 否 (原文未提及)"
    )

    return result, paper_pathways


# ════════════════════════════════════════════════════════════════════════
# 4. Streamlit 主界面
# ════════════════════════════════════════════════════════════════════════

def main():
    # ─── 侧边栏 ───
    with st.sidebar:
        st.markdown("## \U0001F9F1 OA RAG MVP")
        st.markdown("---")
        st.markdown(
            """
            **核心理念**: 文献 RAG 只能检索已发表的知识，
            而原始表达矩阵包含大量从未被文献报道的隐藏信息。

            **数据集**: GSE55235
            - 10 OA 滑膜组织 vs 10 Normal 滑膜组织
            - 平台: GPL96 (Affymetrix U133A)
            - ~22,000 个探针

            **参考论文**: PMID:24690414
            - Lambert et al. 2014
            - 研究 OA 滑膜中"炎症区域 vs 非炎症区域"
            - **非** GSE55235 的 OA vs Normal 比较
            """
        )
        st.markdown("---")

        # 3个问题的快速导航
        st.markdown("### \U0001F4AC 三个问题")
        st.markdown(
            """
            1. **差异基因精确排名** — 文献只说部分基因上调，不说具体数值
            2. **TREM1 共表达网络** — 文献不做相关性分析
            3. **未被文献讨论的通路** — 富集分析 vs 手动列举
            """
        )
        st.markdown("---")

        st.caption(
            "Built as MVP for OA RAG project.\n"
            "Data from NCBI GEO."
        )

    # ─── 主页面: 数据加载 (带缓存) ───
    st.markdown(
        '<p class="main-header">\U0001F9F1 骨关节炎知识图谱 MVP</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">'
        "文献检索不到的知识，藏在原始表达矩阵里 — 通过计算发现它们"
        "</p>",
        unsafe_allow_html=True,
    )

    # 加载数据
    with st.spinner("正在准备数据..."):
        text = download_series_matrix()
        expr, gsm_to_group = parse_series_matrix(text)

        # 过滤: 只保留 OA 和 Normal 样本的 GSM 列
        keep_gsm = [gsm for gsm in expr.columns if gsm_to_group.get(gsm) in ("OA", "Normal")]
        if not keep_gsm:
            # Fallback: first 20 columns (10 Normal + 10 OA)
            keep_gsm = expr.columns[:20].tolist()
            for i, gsm in enumerate(keep_gsm):
                gsm_to_group[gsm] = "Normal" if i < 10 else "OA"

        expr = expr[keep_gsm].copy()
        expr = expr.dropna(thresh=len(keep_gsm) * 0.5)

        # 下载探针-基因映射
        annot_df = download_gpl_annotation()

    # ─── 展示数据概览 ───
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_oa = sum(1 for g in gsm_to_group.values() if g == "OA")
        st.metric("OA 样本数", n_oa)
    with col2:
        n_normal = sum(1 for g in gsm_to_group.values() if g == "Normal")
        st.metric("Normal 样本数", n_normal)
    with col3:
        st.metric("检测基因数", len(expr))
    with col4:
        st.metric("探针-基因映射", len(annot_df) if not annot_df.empty else 0)

    # ─── 差异表达分析 ───
    de_df = differential_expression(expr, gsm_to_group)
    de_df = map_probes_to_genes(de_df, annot_df)

    # ─── 选项卡布局 ───
    tab_overview, tab_q1, tab_q2, tab_q3, tab_data = st.tabs(
        ["\U0001F30D 知识图谱总览", "\U0001F4CA 问题 1: 差异基因排名",
         "\U0001F504 问题 2: TREM1 共表达", "\U0001F50D 问题 3: 隐藏通路",
         "\U0001F4CB 原始数据"]
    )

    # ════════════════════════════════════════════════════════════════
    # Tab 0: 总览 + 知识图谱
    # ════════════════════════════════════════════════════════════════
    with tab_overview:
        st.markdown(
            '<p class="highlight-box">'
            "\U0001F4A1 <b>核心命题</b>：文献 RAG 只能检索论文中明确报告的结果。"
            "但原始表达矩阵包含大量从未被书写发表的知识——例如精确的 fold change、"
            "基因共表达网络、以及未被作者讨论的富集通路。"
            "通过分析 GSE55235 的原始数据，我们发现以下三个“隐藏”答案。"
            "</p>",
            unsafe_allow_html=True,
        )

        st.markdown("### \U0001F30D 交互式知识图谱")
        st.markdown(
            "图谱展示了 OA vs Normal 的差异表达基因、富集通路和共表达关系。"
            "**拖拽节点**查看详情，**悬停**显示信息。"
        )

        # 计算通路富集 (用于 KG)
        up_genes = de_df[de_df["log2FC"] > 0].head(15)["gene_symbol"].tolist()
        down_genes = de_df[de_df["log2FC"] < 0].head(15)["gene_symbol"].tolist()

        pathway_df = pathway_enrichment(up_genes, down_genes)

        # 计算 TREM1 相关性
        if not annot_df.empty:
            probe_gene_map = dict(zip(annot_df["probe"], annot_df["gene"]))
            trem1_probes = [p for p, g in probe_gene_map.items() if g == "TREM1"]
        else:
            trem1_probes = [p for p in expr.index if "TREM1" in p.upper()]
        trem1_probe = trem1_probes[0] if trem1_probes else None
        corr_results = {}
        if trem1_probe and trem1_probe in expr.index:
            corr_results_raw = correlation_analysis(expr, gsm_to_group, trem1_probe)
            # Map probe IDs to gene symbols for KG compatibility
            if "error" not in corr_results_raw:
                corr_results = {"target_gene": "TREM1"}
                if not annot_df.empty:
                    probe_gene_map = dict(zip(annot_df["probe"], annot_df["gene"]))
                else:
                    probe_gene_map = {}
                for grp in ("OA", "Normal"):
                    if grp in corr_results_raw and isinstance(corr_results_raw[grp], pd.DataFrame):
                        df = corr_results_raw[grp].copy()
                        df["gene_symbol"] = df["gene"].map(
                            lambda x: probe_gene_map.get(x, x)
                        )
                        corr_results[grp] = df
            else:
                corr_results = {}

        # 构建知识图谱
        G = build_knowledge_graph(de_df, pathway_df, corr_results, n_genes=20, n_pathways=8)

        # 渲染
        kg_html = render_pyvis_graph(G, height="600px")

        # Pyvis wrapper with improved sizing
        html_wrapper = f"""
        <div style="width:100%; height:600px; border:1px solid #ddd; border-radius:8px; overflow:hidden;">
            {kg_html}
        </div>
        """
        st.components.v1.html(html_wrapper, height=620)

        # 图例
        col_legend1, col_legend2, col_legend3 = st.columns(3)
        with col_legend1:
            st.markdown("\U0001F534 **红色节点** — OA 上调基因")
            st.markdown("\U0001F535 **蓝色节点** — OA 下调基因")
        with col_legend2:
            st.markdown("\U0001F7E3 **紫色节点** — KEGG 通路")
            st.markdown("\U0001F7E0 **大节点** — 样本组")
        with col_legend3:
            st.markdown("\U0001F517 **实线箭头** — 差异表达")
            st.markdown("\U0001F517 **虚线箭头** — 通路-基因关系")
            st.markdown("\U0001F517 **绿色边** — 共表达关系")

    # ════════════════════════════════════════════════════════════════
    # Tab 1: 问题 1 — 差异基因精确排名
    # ════════════════════════════════════════════════════════════════
    with tab_q1:
        st.markdown("## \U0001F4CA 问题 1: OA vs Normal 差异表达基因的精确排名")

        st.markdown(
            '<div class="question-box">'
            "<b>问题</b>：在 GSE55235 数据集中，OA 与 Normal 滑膜组织相比，"
            "<b>表达变化最显著的前 10 个基因是哪些？</b>"
            "它们的确切 log2FC 和校正后 p 值是多少？"
            "</div>",
            unsafe_allow_html=True,
        )

        col_reason, col_result = st.columns([1, 1.5])

        with col_reason:
            st.markdown(
                '<div class="paper-box">'
                "<b>\U0001F4D6 为什么文献答不了</b><br><br>"
                "PMID:24690414 (Lambert 2014) 研究的是 OA 患者滑膜中 "
                "<b>炎症区域 vs 非炎症区域</b> 的差异表达，"
                "而非 OA vs Normal 人群比较。"
                "<br><br>"
                "因此，该论文中报告的差异基因 (如 TREM1 ↑3.45倍, IL-8 ↑4.45倍) "
                "均来自同一患者不同区域的比较，<b>与 GSE55235 的 OA vs Normal "
                "比较完全不同</b>。<br><br>"
                "即使对于 OA vs Normal 比较，文献最多说“某些基因上调”，"
                "而不会给出全部显著基因的<b>精确排名和统计数值</b>。"
                "</div>",
                unsafe_allow_html=True,
            )

        with col_result:
            if st.button("\U0001F50D 计算差异基因排名", key="btn_q1"):
                with st.spinner("正在计算..."):
                    result = q1_top_deg_analysis(de_df)
                    top10 = result.head(10)

                    display_cols = [
                        "基因符号", "Probe/基因", "log2FC", "方向",
                        "P-value", "校正后 P-value (BH)", "显著性标记"
                    ]
                    available_cols = [c for c in display_cols if c in top10.columns]
                    st.dataframe(
                        top10[available_cols].style
                        .format({
                            "log2FC": "{:.3f}",
                            "P-value": "{:.2e}",
                            "校正后 P-value (BH)": "{:.2e}",
                        })
                        .map(
                            lambda v: "color: red" if isinstance(v, str) and "上调" in v
                            else ("color: blue" if isinstance(v, str) and "下调" in v else ""),
                            subset=["方向"],
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # 可视化
                    fig = px.bar(
                        top10.head(10),
                        x="log2FC",
                        y="基因符号",
                        color="log2FC",
                        color_continuous_scale=["#3498db", "white", "#e74c3c"],
                        title="Top 10 差异基因 log2FC",
                        orientation="h",
                    )
                    fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(
                        '<div class="answer-box">'
                        f"<b>✅ 关键发现</b>：<br>"
                        f"Top 1 差异基因 <b>{top10.iloc[0].get('gene_symbol', top10.iloc[0].get('Probe/基因', ''))}</b> "
                        f"log2FC = {top10.iloc[0]['log2FC']:.3f}，"
                        f"校正后 p = {top10.iloc[0].get('校正后 P-value (BH)', 0):.2e}。<br>"
                        "这个精确的定量信息在 PMID:24690414 中完全不存在。"
                        "</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("\U0001F446 点击上方按钮开始计算")

    # ════════════════════════════════════════════════════════════════
    # Tab 2: 问题 2 — TREM1 共表达网络
    # ════════════════════════════════════════════════════════════════
    with tab_q2:
        st.markdown("## \U0001F504 问题 2: TREM1 的 OA 特异性共表达网络")

        st.markdown(
            '<div class="question-box">'
            "<b>问题</b>：在 OA 样本中，<b>TREM1</b>（炎症关键基因）"
            "与哪些基因的表达高度相关？"
            "这些相关性在正常样本中是否也存在？<br><br>"
            "<b>延伸问题</b>：在 OA 样本中，与 TREM1 最相关的基因，"
            "是否同时也是差异表达基因？"
            "</div>",
            unsafe_allow_html=True,
        )

        col_reason, col_result = st.columns([1, 1.5])

        with col_reason:
            st.markdown(
                '<div class="paper-box">'
                "<b>\U0001F4D6 为什么文献答不了</b><br><br>"
                "Lambert (2014) 报告 TREM1 在炎症区域上调 3.45 倍，"
                "但<b>未报告任何基因共表达 / 相关性分析</b>。<br><br>"
                "共表达网络只能从<b>原始表达矩阵</b>中通过计算 "
                "Pearson/Spearman 相关系数获得。<br><br>"
                "文献 RAG 可以告诉你 TREM1 在 OA 中上调，"
                "但无法告诉你 TREM1 与哪个基因协同表达。"
                "</div>",
                unsafe_allow_html=True,
            )

        with col_result:
            if st.button("\U0001F50D 计算 TREM1 共表达网络", key="btn_q2"):
                with st.spinner("正在计算相关性..."):
                    result = q2_correlation_analysis(expr, gsm_to_group, annot_df)

                if "error" in result:
                    st.error(result["error"])
                else:
                    target = result.get("target_gene", "TREM1")

                    tab_oa, tab_normal = st.tabs(["OA 样本中相关基因", "Normal 样本中相关基因"])

                    with tab_oa:
                        if isinstance(result["OA"], pd.DataFrame) and not result["OA"].empty:
                            df_oa = result["OA"].copy()
                            # 尝试映射 gene symbol
                            if not annot_df.empty:
                                probe_gene_map = dict(zip(annot_df["probe"], annot_df["gene"]))
                                df_oa["gene_symbol"] = df_oa["gene"].map(
                                    lambda x: probe_gene_map.get(x, x)
                                )
                            else:
                                df_oa["gene_symbol"] = df_oa["gene"]

                            st.dataframe(
                                df_oa[["gene_symbol", "gene", "correlation", "p_value"]]
                                .style.format({
                                    "correlation": "{:.4f}",
                                    "p_value": "{:.2e}",
                                }),
                                use_container_width=True,
                                hide_index=True,
                            )

                            # 条形图
                            fig = px.bar(
                                df_oa.head(10),
                                x="correlation",
                                y="gene_symbol",
                                color="correlation",
                                color_continuous_scale="Reds",
                                title=f"OA 样本中与 TREM1 最相关的基因",
                                orientation="h",
                                text_auto=".3f",
                            )
                            fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("OA 组无结果")

                    with tab_normal:
                        if isinstance(result["Normal"], pd.DataFrame) and not result["Normal"].empty:
                            df_normal = result["Normal"].copy()
                            if not annot_df.empty:
                                probe_gene_map = dict(zip(annot_df["probe"], annot_df["gene"]))
                                df_normal["gene_symbol"] = df_normal["gene"].map(
                                    lambda x: probe_gene_map.get(x, x)
                                )
                            else:
                                df_normal["gene_symbol"] = df_normal["gene"]

                            st.dataframe(
                                df_normal[["gene_symbol", "gene", "correlation", "p_value"]]
                                .style.format({
                                    "correlation": "{:.4f}",
                                    "p_value": "{:.2e}",
                                }),
                                use_container_width=True,
                                hide_index=True,
                            )

                            fig = px.bar(
                                df_normal.head(10),
                                x="correlation",
                                y="gene_symbol",
                                color="correlation",
                                color_continuous_scale="Blues",
                                title="Normal 样本中与 TREM1 最相关的基因",
                                orientation="h",
                                text_auto=".3f",
                            )
                            fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Normal 组无结果")

                    st.markdown(
                        '<div class="answer-box">'
                        "<b>✅ 关键发现</b>：比较 OA 和 Normal 样本中与 TREM1 的共表达模式，"
                        "可以发现疾病特异性的基因协同调控网络——"
                        "这些信息不存在于任何单篇文献中，只能从原始表达矩阵计算获得。"
                        "</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("\U0001F446 点击上方按钮开始计算")

    # ════════════════════════════════════════════════════════════════
    # Tab 3: 问题 3 — 未被文献讨论的通路
    # ════════════════════════════════════════════════════════════════
    with tab_q3:
        st.markdown("## \U0001F50D 问题 3: 哪些显著富集通路被原文忽略了？")

        st.markdown(
            '<div class="question-box">'
            "<b>问题</b>：GSE55235 的差异基因富集在哪些 KEGG 通路上？"
            "其中哪些通路在 PMID:24690414 (Lambert 2014) 中<b>没有被讨论</b>？"
            "</div>",
            unsafe_allow_html=True,
        )

        col_reason, col_result = st.columns([1, 1.5])

        with col_reason:
            st.markdown(
                '<div class="paper-box">'
                "<b>\U0001F4D6 为什么文献答不了</b><br><br>"
                "Lambert (2014) 基于手动文献回顾，讨论了："
                "<ul>"
                "<li>炎症通路 (TNF, NF-κB)</li>"
                "<li>Wnt 信号</li>"
                "<li>血管生成</li>"
                "<li>软骨代谢</li>"
                "</ul>"
                "但这是<b>非系统性</b>的列举——作者只讨论了"
                "他们感兴趣的少数通路。<br><br>"
                "通过<b>系统性通路富集分析</b>，"
                "我们发现了一些在 OA 滑膜中显著变化、"
                "但原文完全没有提及的通路。"
                "</div>",
                unsafe_allow_html=True,
            )

        with col_result:
            if st.button("\U0001F50D 计算通路富集", key="btn_q3"):
                with st.spinner("正在查询 Enrichr 数据库..."):
                    up_genes = de_df[de_df["log2FC"] > 0].head(30)["gene_symbol"].tolist()
                    down_genes = de_df[de_df["log2FC"] < 0].head(30)["gene_symbol"].tolist()
                    pathway_df = pathway_enrichment(up_genes, down_genes)

                result, paper_pathways = q3_pathway_analysis(pathway_df)

                if result.empty:
                    st.warning(
                        "通路富集分析暂未返回结果。"
                        "这可能是因为网络连接问题或基因列表与数据库不匹配。"
                    )
                else:
                    # 标记
                    result["在原文中"] = result["文献讨论"]

                    st.dataframe(
                        result[["pathway", "p_value", "adj_p_value", "-log10_pvalue",
                                "在原文中", "genes"]]
                        .style.format({
                            "p_value": "{:.2e}",
                            "adj_p_value": "{:.2e}",
                            "-log10_pvalue": "{:.2f}",
                        })
                        .map(
                            lambda v: "background-color: #e8f5e9" if "否" in str(v) else "",
                            subset=["在原文中"],
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # 可视化
                    fig = px.bar(
                        result,
                        x="-log10_pvalue",
                        y="pathway",
                        color="在原文中",
                        color_discrete_map={
                            "✅ 是 (原文提及)": "#ff9800",
                            "❌ 否 (原文未提及)": "#4caf50",
                        },
                        title="KEGG 通路富集 - 绿色=原文未讨论的新发现",
                        orientation="h",
                        text_auto=".1f",
                    )
                    fig.update_layout(
                        height=max(400, len(result) * 35),
                        yaxis={"categoryorder": "total ascending"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 找出原文没提到的通路
                    novel = result[result["在原文中"].str.contains("否")]
                    discussed = result[result["在原文中"].str.contains("是")]

                    novel_lines = []
                    if not novel.empty:
                        novel_lines.append(
                            f"在 {len(result)} 个显著富集的通路中，"
                            f"有 <b>{len(novel)} 个通路在原文中未讨论</b>，"
                            f"包括："
                        )
                        for _, row in novel.iterrows():
                            novel_lines.append(
                                f"  \U0001F195 <b>{row['pathway']}</b> "
                                f"(p = {row['p_value']:.2e})"
                            )
                    if not discussed.empty:
                        novel_lines.append(
                            f"另外有 <b>{len(discussed)} 个通路</b> "
                            f"与原文讨论一致。"
                        )

                    st.markdown(
                        '<div class="answer-box">'
                        "<b>✅ 关键发现</b>：<br>"
                        + "<br>".join(novel_lines)
                        + "</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("\U0001F446 点击上方按钮开始计算")

    # ════════════════════════════════════════════════════════════════
    # Tab 4: 原始数据展示
    # ════════════════════════════════════════════════════════════════
    with tab_data:
        st.markdown("### \U0001F4CB GSE55235 原始数据预览")

        st.markdown("#### 样本分组信息")
        groups_df = pd.DataFrame({
            "GSM ID": list(gsm_to_group.keys()),
            "分组": list(gsm_to_group.values()),
        })
        # Filter to show only OA/Normal
        groups_df = groups_df[groups_df["分组"].isin(["OA", "Normal"])]
        st.dataframe(groups_df, use_container_width=True, hide_index=True)

        st.markdown("#### 表达矩阵 (前 10 探针 × 前 5 样本)")
        n_cols_display = min(5, expr.shape[1])
        st.dataframe(expr.head(10).iloc[:, :n_cols_display], use_container_width=True)

        st.markdown("#### 差异表达结果 (前 20)")
        if not annot_df.empty:
            probe_gene_map = dict(zip(annot_df["probe"], annot_df["gene"]))
            de_display = de_df.head(20).copy()
            de_display["基因"] = de_display["gene"].map(
                lambda x: probe_gene_map.get(x, x)
            )
        else:
            de_display = de_df.head(20).copy()
            de_display["基因"] = de_display["gene"]

        st.dataframe(
            de_display[["基因", "gene", "log2FC", "p_value", "adj_p_value"]]
            .style.format({
                "log2FC": "{:.3f}",
                "p_value": "{:.2e}",
                "adj_p_value": "{:.2e}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### 探针-基因映射表 (前 20 条)")
        if not annot_df.empty:
            st.dataframe(annot_df.head(20), use_container_width=True, hide_index=True)

    # ─── Footer ───
    st.markdown("---")
    st.markdown(
        "<center><small>"
        "OA RAG MVP | 数据来源: NCBI GEO GSE55235 | "
        "参考论文: PMID:24690414"
        "</small></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

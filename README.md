  # OA RAG MVP

  骨关节炎（OA）基因表达多数据集荟萃分析（Meta-analysis） — Streamlit 应用

  ## 概述

  本项目的核心论点：仅靠文献检索无法得到的结果，通过分析原始表达数据就能得到。

  它分析了 GEO 数据库中 OA（骨关节炎）滑膜组织的基因表达数据，通过跨数据集的 meta 分析（Fisher 合并 p
  值）寻找在多个独立队列中一致差异表达的基因。

  ## 数据集

  | 数据集 | 平台 | OA 样本数 | Normal 样本数 |
  |--------|------|-----------|--------------|
  | GSE55235 | GPL96 | 10 | 10 |
  | GSE12021 | GPL96 | 10 | 9 |
  | GSE55457 | GPL96 | 10 | 10 |
  | GSE55584 | GPL96 | 6 | 0（仅 OA/RA） |

  ## 功能

  - 单数据集差异表达分析（Welch t-test）
  - 多数据集 Fisher 法 meta 分析
  - 基因-基因互作网络可视化（pyvis）
  - 火山图、表达箱线图等交互式图表
  - RAG 检索增强问答（调用外部 LLM API 解释结果）

  ## 本地运行

  pip install -r requirements.txt
  streamlit run app.py

  ## 项目结构

  - app.py — 主应用（多数据集版）
  - app_single.py — 单数据集版
  - prepare_data.py — 数据预处理脚本
  - requirements.txt — Python 依赖
  - data/
    - de_GSE55235.csv — GSE55235 差异表达结果
    - de_GSE12021.csv — GSE12021 差异表达结果
    - de_GSE55457.csv — GSE55457 差异表达结果
    - meta_analysis.csv — Meta 分析结果
    - gpl96_annotation.csv — 探针注释
    - dataset_info.json — 数据集基本信息

  ## 技术栈

  - 前端: Streamlit
  - 分析: pandas, numpy, scipy, statsmodels
  - 可视化: plotly, pyvis, networkx

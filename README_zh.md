# GlucoSL-KAN 数值复现

本仓库公开模型代码、数据处理规则、最终配置和汇总结果。复现入口只包括数据准备、模型训练和结果评价，不要求重画论文图。

## 安装

建议使用 Python 3.12。基础模型与 XGBoost 使用 requirements.txt；完整对照另安装 requirements-tabpfn.txt。原始 macOS 环境版本保存在 requirements.lock.txt，不建议在其他系统上直接整份安装。

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-tabpfn.txt
    python scripts/reproduce.py check --models all

## 完整数值流程

    python scripts/reproduce.py all --download --models all --jobs 4 --run-dir reproduction/full

此命令从官方来源获取原始数据，查询并缓存食物成分，重建餐次与按患者划分的数据集，训练完整模型组，进行2,000次患者自助重抽样评价，再将点估计与仓库提供的参考结果核对。

数据不直接打包进仓库。中国食物成分平台的数值由使用者在本地取得，不能把仓库中的名称匹配规则当作完整食物成分数据库。平台数值会更新，来源更新导致的差异会在核对时报告。详细来源与使用条款见 DATA_SOURCES.md。

## 分步执行

    python scripts/reproduce.py prepare --download
    python scripts/reproduce.py fit --models all --jobs 4 --run-dir reproduction/full
    python scripts/reproduce.py evaluate --models all --jobs 4 --run-dir reproduction/full
    python scripts/reproduce.py verify --run-dir reproduction/full

已有原始数据和食物成分查询缓存时，可离线重建：

    python scripts/reproduce.py prepare --raw-dir /path/to/data/raw --cn-cache-dir /path/to/cncdc_query_cache --offline

原始数据目录布局及预期样本数见英文 README。生成后应得到上海训练集67人1,952餐、验证集33人670餐，美国队列44人1,147餐，其中HbA1c至少6.5%的子集14人353餐。

## 较快的核心检查

    python scripts/reproduce.py fit --models core --jobs 4 --run-dir reproduction/core
    python scripts/reproduce.py evaluate --models core --bootstrap-draws 0 --run-dir reproduction/core
    python scripts/reproduce.py verify --run-dir reproduction/core

核心组包含 GlucoSL-KAN、XGBoost、常数模型及美国训练和主食熟重变体。--bootstrap-draws 0 只核对点估计，不复现置信区间；论文的不确定性流程使用2,000次抽样。

## 输出与验证

新模型、个体预测和评价表写入指定的 reproduction 目录，不覆盖仓库中的论文参考结果。拟合与评价无需原团队的内部开发材料。环境改变时不保证序列化文件逐字节一致，应核对模型参数或预测结果。

    python -m unittest discover -s tests -v
    python scripts/summarize_results.py

REPRODUCTION_CHECK.md 说明本次发布实际执行了哪些检查。完整对照中的 TabPFN 使用原研究的 v2 权重，不使用当前库的默认新模型。额外 GAM 事后分析需要单独的 pyGAM 兼容环境，命令见英文 README。

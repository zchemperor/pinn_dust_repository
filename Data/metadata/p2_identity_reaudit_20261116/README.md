# 317行输入的来源候选复核

日期：2026-11-16。以 record_id=0…316 为唯一计数键，保留全部317行。

- `record_candidate_status_317.csv`：逐输入行状态；原始数据行号为从1起算的数据行，不含表头。
- `all_source_candidates.csv`：全部输入行×来源行的候选对，含来源表名及数据行号。站名、文献、年代、标记和粒径比例全部属于候选来源行。
- `Lambert2015_TableS1_from_docx.csv`：从转换件重新提取的62条原始表项；额外保留Word表内行号。
- `source_table_inventory.csv`：各原表行数和可解析的LGM数值数。
- `historical_marine_age_rows.csv` / `historical_marine_age_combinations.csv`：历史v2海洋表全部102行和94条完整年代的29种组合。
- `legacy_309_to_record_id.csv`：旧309行与317行之间的舍入区间对应，仅用于诊断旧表。
- `summary.json` / `manifest.json`：口径、计数、源文件及保护文件哈希、脚本和输出哈希。

原始值不舍入、不合并、不删行。v2探索坐标容差为每轴0.05°，SI为每轴0.1°，均为矩形窗口而非球面距离；另标记精确坐标相等。
通量一致性为 abs(input/source_LGM-1) <=0.001，并另列0.002敏感性结果。
近邻容差为核对先前材料的探索选择，不是经过验证的身份判别阈值。
PM10比例乘积只作另列诊断，不用于提高匹配等级。
缺失LGM不算数值一致；各表重复出现不算独立来源。

身份状态全部为 UNRESOLVED_NO_VERIFIED_ROW_LINEAGE。
数值一致提供来源候选证据；它没有补齐原始记录键、版本变换或同址独立性。
不同来源覆盖数按record_id取并集，不能相加连接行数。不得把候选档案/年代标签直接用于当前记录的分层统计。

再运行：`E:/python3.9.13/python.exe -B Code/audit_p2_identity_materials.py`。
脚本会重建本目录的派生表及四份说明文档。科学输入、源材料、论文和既有训练结果不改动。

"""Audit historical source candidates without assigning current record identities.

Run with python -B Code/audit_p2_identity_materials.py.
No training, network access, scientific-input edits, or source-table overwrites.
"""
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Data/metadata/p2_identity_reaudit_20261116'
V2 = ROOT / 'tmp/dirtmap_search/dirtmap_v2'
S1 = ROOT / 'Data/metadata/lambert2015_si'
MAP = ROOT / 'Data/model_results/v5_pmip4_2d/p2_local_lineage_audit_20261115/record_metadata_status_317.csv'
RAW = ROOT / 'Data/original_data/MAR_LGM.csv'
TABLES = ['table1_ice_core.txt', 'table2_marine_sediment.txt',
          'table4_non_carb_marine_sed.txt', 'table5_terrestrial_sediment.txt',
          'table6_all_data.txt', 'table7_flagged_data.txt']
UNRESOLVED = 'UNRESOLVED_NO_VERIFIED_ROW_LINEAGE'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def number(s):
    try:
        return float(str(s).strip().rstrip('*'))
    except ValueError:
        return float('nan')


def lon180(v):
    return v - 360 if v > 180 else v + 360 if v <= -180 else v


def source_rows():
    frames, inventory = [], []
    for name in TABLES:
        table = pd.read_csv(V2 / name, sep='\t', encoding='cp1252',
                            dtype=str, keep_default_na=False)
        inventory.append(dict(source_table=name, rows=len(table),
                              numeric_LGM=sum(math.isfinite(number(s)) for s in table['LGM g/m2/yr'])))
        for idx, r in table.iterrows():
            frames.append(dict(
                source_family='DIRTMAP_v2', source_table=name,
                source_data_row_1based=idx + 1, candidate_core=r['Core ID'],
                candidate_lon=lon180(number(r['Lon'])), candidate_lat=number(r['Lat']),
                candidate_LGM_raw=r['LGM g/m2/yr'], candidate_LGM=number(r['LGM g/m2/yr']),
                candidate_HOL=number(r.get('Modern g/m2/yr', '')),
                candidate_reference=r.get('Reference 1', ''), candidate_reference_2=r.get('Reference 2', ''),
                candidate_sample_type_code=r.get('Sample Type Code', ''),
                candidate_data_type_code=r.get('Data Type Code', r.get('DataType Code', r.get('Data Type', ''))),
                candidate_comments=r.get('Comments', ''), candidate_flag=r.get('Flag', ''),
                candidate_age_begin_ka=number(r.get('LGM Begin Age', '')),
                candidate_age_end_ka=number(r.get('LGM End Age', '')),
                candidate_age_model=r.get('Age Model Comment', r.get('Age Model Type', '')),
                candidate_PM10_fraction=float('nan'), candidate_starred_LGM=False))
    with zipfile.ZipFile(S1 / 'tmp_conv/supinfo.docx') as z:
        root = ET.fromstring(z.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    rows = root.findall('.//w:tbl/w:tr', ns)
    assert len(rows) == 64
    extracted = []
    for idx, row in enumerate(rows[2:]):
        c = [''.join(t.text or '' for t in cell.findall('.//w:t', ns))
             for cell in row.findall('w:tc', ns)]
        assert len(c) == 7
        extracted.append(dict(source_data_row_1based=idx + 1, source_word_table_row_1based=idx + 3,
                              site_location=c[0], longitude_raw=c[1], latitude_raw=c[2],
                              HOL_raw=c[3], LGM_raw=c[4], PM10_fraction_raw=c[5], reference=c[6]))
        frames.append(dict(
            source_family='Lambert2015_S1', source_table='Lambert2015_TableS1',
            source_data_row_1based=idx + 1, candidate_core=c[0],
            candidate_lon=lon180(number(c[1])), candidate_lat=number(c[2]),
            candidate_LGM_raw=c[4], candidate_LGM=number(c[4]), candidate_HOL=number(c[3]),
            candidate_reference=c[6], candidate_reference_2='', candidate_sample_type_code='',
            candidate_data_type_code='', candidate_comments='', candidate_flag='',
            candidate_age_begin_ka=float('nan'), candidate_age_end_ka=float('nan'),
            candidate_age_model='', candidate_PM10_fraction=number(c[5]),
            candidate_starred_LGM=c[4].endswith('*')))
    return pd.DataFrame(frames), pd.DataFrame(inventory), pd.DataFrame(extracted)


def write_reports(stats):
    """Write only the derived tables and reports kept in this directory."""
    base = 'Data/metadata/p2_identity_reaudit_20261116/'
    readme = '''# 317行输入的来源候选复核

以 record_id=0…316 为唯一计数键，保留全部317行。

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

再运行：`python -B Code/audit_p2_identity_materials.py`。
脚本会重建本目录的派生表及四份说明文档。科学输入、源材料、论文和既有训练结果不改动。
'''
    (OUT / 'README.md').write_text(readme, encoding='utf-8')
    a = stats['candidate_summaries']
    rows = []
    for label, key in [('v2六张相关原表', 'v2_all_tables'), ('其中table6', 'v2_table6'),
                       ('Lambert SI Table S1', 's1'), ('两类来源并集', 'combined')]:
        z = a[key]
        rows.append('| %s | %s | %s | %s | %s |' % (label,
            z['coordinate_candidate_rows']['input_records'], z['exact_coordinate_rows']['input_records'],
            z['coordinate_value_rows_0p1percent']['input_records'],
            z['exact_coordinate_value_rows_0p1percent']['input_records']))
    coverage = ('| 来源 | 容差内坐标候选 | 精确坐标候选 | 容差内坐标且值一致 | 精确坐标且值一致 |\n'
                '|---|---:|---:|---:|---:|\n' + '\n'.join(rows))
    common = '''所有计数均按当前317个record_id去重计数，保留数值重复的输入行。
“值一致”指相对误差≤0.1%；改为0.2%时本轮覆盖计数不变。
表内数值是候选证据覆盖，不是原始身份确认率。所有317行的原始来源链仍未确证。
'''
    reports = {}
    reports['材料清单总表_20261116.md'] = f'''# 317行输入溯源材料总表（复核修正版）

替代此前“146条已确认、覆盖率47.2%”等口径。
当前权威派生结果：`{base}summary.json`、`record_candidate_status_317.csv`及`all_source_candidates.csv`。

## 一、输入分母与证据覆盖

上游过滤规则保留317行，310条不同(坐标,MAR)记录、301个坐标，完全重复的额外行7条。
旧309行表把坐标舍入至两位、通量舍入至四位并去重，合并了Vostok两条不同数值（约0.01295和0.013）。它不是可用的317行身份索引。

{coverage}

{common}
并集98/317（30.9%）具有容差内坐标及LGM值一致的候选，93/317（29.3%）还满足精确坐标。
170/317（53.6%）具有容差内坐标候选。22条输入在v2候选中有多个不同站名。
这些数值不能表述为“98条身份确认”或“其余147条均为v3增补”。

## 二、五批材料及适用范围

| 批次 | 材料及位置 | 能支持什么 | 不能支持什么 |
|---|---|---|---|
| A | `Data/model_results/v5_pmip4_2d/p2_source_followup_20261115/` | 既有坐标候选与检索过程；候选不自动升级身份 | 当前317行的来源链确认 |
| B | `Data/metadata/dirtmap_source_search_20261116/sources/` | 文献文字、数据库载体线索、特定端点访问失败 | 原始表从未公开、永久不可取或全网不存在 |
| C | `Data/metadata/lambert2015_si/`原始doc及转换docx | Table S1确有62条数据，包含站名、文献、部分粒径比例 | SI全部表项都是LGM观测；连接行数就是输入覆盖数 |
| D | `tmp/dirtmap_search/dirtmap_v2/`六张相关表及readme | 历史逐记录通量、文献、档案代码、部分年代与标记 | 当前版本继承了相同年代、单位协调和粒径处理 |
| E | `tmp/dirtmap_search/albani2015/`、`preprocess.ipynb` | Holocene站点旁证；上游过滤式与365天单位换算代码 | 用Holocene通量验证LGM；证明原始测量单位协调 |

v2 table6为314行，其中152个可解析LGM数值；另10个非空单元格是“－”，不能算成数值。
table5为74行，先前汇总的15行不正确。
Table S1以从docx重提取的表为本轮依据，原始Word97文件不改动；原件与转换件均留存哈希。

## 三、可写入后续分析的有限结论

1. v2海洋表102行中94条有完整年代边界，存在29种组合，没有19–26 ka。它证明历史汇编时段异质，不能给当前317行直接赋年代。
2. v2 readme允许15–30 ka或其子区间，黄土采用MIS1/MIS2广平均。上游19–26 ka定义与当前逐记录年龄核实必须区分。
3. Lambert来源说明黄土PM10、冰芯/海洋总尘的混合口径。通量比值可辅助提出处理假说，不能证明每条的粒径阈值。
4. 当前输入在ODP 1090附近同时有约5与11，SI对应位置为11。不能再写“11被改成5”。
5. SI的Chile Coast重复来源行只对应当前两条输入，其LGM缺失，不能证明输入前向填充或这两条是重复训练记录。

## 四、仍未闭合

- 当前317行与来源版本之间的逐记录键及处理历史，尤其311→317的实际变更清单。
- 各输入的原始测量单位、实际平均时段、粒径协调和档案响应；不可把历史表字段直接移植。
- 同址条目及完全重复条目的测量独立性。重复敏感性评估不等于独立性证明。
- 当前约5通量记录的具体出处；不能根据附近SI值断言改值。

## 五、检索失败的表述边界

403/404、后端mysql_pconnect报错及存档查询未命中仅描述所查URL和查询范围。
透明评审文字“includes all empirical data ... available”说的是数据覆盖，不是“逐记录表从未公开”。
未取得v3库不能推出库从未存档，也不能把所有未匹配输入归为Maher新增。
现稿已区分上游19–26 ka与逐记录核实，并已披露PM10/总尘异质；本轮不修改论文。
'''
    reports['DIRTMAPv2_逐记录表与编制说明_20261116.md'] = f'''# DIRTMAP v2逐记录材料复核（修正版）

本版撤回“146条身份确认、47.2%覆盖率、年代/粒径/单位全部闭合”结论。
旧表`Data/metadata/dirtmap_v2_identity/input_to_DirtmapV2_identity.csv`仅作历史诊断；不再用于身份赋值。
替代结果在`{base}`，保留全部317条record_id、原始数据行号和未舍入值。

## 原表与匹配口径

纳入table1、2、4、5、6、7的全部来源行，分别保留来源表及行号，不把同站跨表出现视为独立证据。
table3为现代捕集器表，本轮未纳入LGM身份匹配。readme与原表不改动。
每轴0.05°以内列出所有候选，另标精确相等；通量误差≤0.1%，另列0.2%敏感性。
这是探索性候选规则，不能直接判定站点或测量身份。

{coverage}

{common}
v2六表中22条输入对应多个不同候选站名，禁止取第一条后作为已识别身份。
精确坐标且值一致的80条与既有历史候选审计口径一致；本轮不是首次发现v2资料。

## 年代

海洋table2共102行，94条同时有起止边界，29种组合。15–27为37条、15–25为12条、17–19为11条、15–30为9条；无19–26组合。
这是全部历史海洋表的分布，包含不能验证当前通量的条目，不是当前317行的年代分布。
容差内有历史年代候选的当前输入为55条，仍不移植为当前年代。
readme允许15–30 ka或其子区间；黄土为MIS1(0–12 ka)/MIS2(12–24 ka)。

## 粒径与数值差异

旧309行对照表中60条有可计算比值的黄土候选，input/v2中位约0.230、范围约0.130–0.662；日本五条为0.400。
这些是旧选择与舍入口径下的描述，不能作为317行的无歧义归属或粒径换算表。
旧表的`v2_ratio`实际存储历史LGM/Modern，不能当作input/v2使用；新表另用`input_over_candidate_LGM`。
旧69条“coord”包含一条没有v2 LGM数值，只有68条能算比值。海洋或冰芯接近1也不能证明逐条“未折算”。
固定因子模式与Lambert的PM10说明可以共同提出处理假说，但不验证逐条阈值或跨档案可比性。

## 计数与未解问题

table6的314行只有152个可解析LGM数值；162是包含10个“－”的非空数。
table5为74行而非15。旧309由舍入去重形成，不能替代317行；完整不同数值记录为310条。
未匹配记录不自动属于v3/Maher增补。身份、311→317版本对应、原始单位协调、当前年代和同址独立性仍需证据。
'''
    reports['Lambert2015_SI_TableS1解析_20261116.md'] = f'''# Lambert 2015 SI Table S1复核（修正版）

本版替代此前“18条身份确认”“ODP 11→5”“Chile重复照抄”“星号几乎确定推算”等表述。
原件：`Data/metadata/lambert2015_si/grl53218-sup-0010-supinfo.doc`；解析依据为原件的既有docx转换件。
Table S1重新提取至`{base}Lambert2015_TableS1_from_docx.csv`，共62条数据，保留原始字符串与来源行号。
原表标题说明它列出DIRTMAP3之外用于制图的站点，并非完整317行身份表。

## 当前输入与SI的候选对应

原始317行不舍入。每轴0.1°探索窗口内有20个候选对，覆盖18个输入record_id。
精确坐标匹配有18个候选对，但只覆盖16个输入record_id。
精确坐标且LGM通量误差≤0.1%的输入为13条；0.2%口径计数相同。
这13条包括位于(8.9,−42.91)、通量约11的当前输入。它被旧18行对照漏掉。
18条坐标候选的组成是：13条坐标和值一致、2条精确坐标但LGM不能验证的Chile输入、1条Carpathian差值、2条仅近邻候选（约5的ODP邻近输入和1.2的S Pacific邻近输入）。
所有候选仍缺完整原始来源链确认，不能称身份覆盖率。

## 三处实质纠正

1. 当前在(8.9733,−42.873)有约5，在(8.9,−42.91)另有约11。SI为后者提供数值一致候选，前者仅为近邻。没有证据证明11被改为5。
2. SI把Chile Coast两处各列两遍；当前这两处各仅一条输入。源表的重复造成连接行膨胀，不能说输入照抄成四条或解释训练表的重复。源表LGM缺失，与HOL值接近也不能证明前向填充。
3. Carpathian当前/源表LGM约0.2297，源表PM10比例为0.23。两者接近，不足以断言输入使用了未取整比例；新候选表保留乘积差值，不按该假说确认匹配。

## 星号与粒径

62条中14条有可解析LGM数值，6条带星，其LGM/HOL为5或2。
整数倍本身不能解释星号；未带星记录也有整数比值。当前可解析文本中未定位到明确图例，保留含义未知，不能判定这6条不是独立测量。
PM10列仅少数条目有值，不能据此恢复全体317行的粒径处理。

## 输出与边界

`{base}all_source_candidates.csv`保留来源表行号、全部候选、原始LGM字符串、数值差值及PM10诊断。
旧`traceable_to_TableS1.csv`保留作为已废止的历史派生表，不再作为覆盖率或身份事实底稿。
不根据本表推算311→317变化、不替换当前目标、不删除重复行、不修改论文。
'''
    reports['DIRTMAP_source_search_20261116.md'] = '''# DIRTMAP公开材料检索：证据边界修正版

本版撤回超出检索证据的排他性推断。

## 已获得什么

- 已归档Molina 2024透明评审文件、Supplementary Discussion、论文网页、INQUA页、后端响应及EPIC记录页。
- INQUA材料提供DIRTMAP3 Access载体线索；特定后端响应为“Call to undefined function mysql_pconnect()”。这只确认该端点当时无法正常查询。
- Lambert SI现已取得，含62条Table S1表项；v2原表也已取得。最新候选结果见`Data/metadata/p2_identity_reaudit_20261116/`及`材料清单总表_20261116.md`。
- 上游notebook规定了过滤式和365天单位换算。公开文本分别报告311与317，但没有取得两版的逐记录变更清单。

## 不能从这些材料推出什么

1. 作者“includes all empirical data ... available”是数据覆盖表述，不是“逐记录表从未公开”的官方自述。
2. 指定URL返回403/404不证明整篇论文没有补充材料。Lambert SI由用户后来成功取得，本身就是应保留的反例。
3. Wayback所查域名/前缀未命中，不证明Internet Archive从未存档任何别名、历史URL或镜像。
4. PHP错误不能证明永久失效，也不能证明身份信息只存在于某一个载体。
5. 311与317以及375与397的计数差，不能判定谁筛选或谁新增。SI也不是完整版本差分。
6. “缺一致粒径信息”的文献描述不能证明任一条的原始粒径资料永远无法恢复。
7. 未匹配v2不能给记录赋Maher/v3来源。不得将约145条称为已识别的v3新增部分。

## 如何记录负面结果

保留具体检索URL、时间、响应状态与查询范围，使用“本轮未取得/所查路径未命中”。不使用“从未公开”“全网不存在”“不可再找”等结论。
后续若取得新材料，应回到317个原始record_id核对全部候选，不按近邻或数值比值直接赋身份。

## 对现稿的关系

现稿已说明上游19–26 ka不是逐记录核实的年龄范围，且已说明黄土PM10与冰芯/海洋总尘的混合口径。
新材料增强历史背景与候选证据，并未使逐记录年代、单位协调、粒径与独立性全部闭合。
本轮仅修复资料与报告，不修改论文，也不联系作者或进行训练。
'''
    for name, content in reports.items():
        (ROOT / 'docs/04_数据说明' / name).write_text(content, encoding='utf-8')


def main():
    source_paths = [V2 / n for n in TABLES] + [V2 / 'readme_dirtmap_v2.txt',
        S1 / 'tmp_conv/supinfo.docx', S1 / 'grl53218-sup-0010-supinfo.doc', MAP, RAW,
        ROOT / 'Data/metadata/dirtmap_v2_identity/input_to_DirtmapV2_identity.csv']
    protected = source_paths + [ROOT / 'Data/processed_data/df_empirical_LGM.csv',
        ROOT / 'manuscript/main.tex', ROOT / 'manuscript/cas-refs.bib']
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected if p.exists()}
    old = pd.read_csv(MAP)
    raw = pd.read_csv(RAW)
    retained = raw.query('-90 < Latitude < 90 and -180 < Longitude <= 180').copy()
    assert len(old) == len(retained) == 317
    assert old.record_id.tolist() == list(range(317))
    assert old.raw_data_row_1based.tolist() == (retained.index + 1).tolist()
    assert np.array_equal(old.lon, retained.Longitude)
    assert np.array_equal(old.lat, retained.Latitude)
    assert np.array_equal(old.raw_MAR, retained.MAR)
    old['converted_g_m2_yr'] = old.raw_MAR * 31536000000
    src, inventory, s1 = source_rows()
    candidates = []
    for r in old.itertuples(index=False):
        dx = abs(src.candidate_lon - r.lon)
        dx = np.minimum(dx, 360 - dx)
        dy = abs(src.candidate_lat - r.lat)
        limit = np.where(src.source_family == 'DIRTMAP_v2', .05, .1)
        for idx in src.index[(dx <= limit + 1e-12) & (dy <= limit + 1e-12)]:
            s = src.loc[idx].to_dict()
            value = s['candidate_LGM']
            error = abs(r.converted_g_m2_yr / value - 1) if value > 0 else float('nan')
            pm = s['candidate_PM10_fraction']
            scaled_error = abs(r.converted_g_m2_yr / (value * pm) - 1) if value > 0 and pm > 0 else float('nan')
            exact = bool(dx[idx] == 0 and dy[idx] == 0)
            match = bool(error <= .001)
            s.update(record_id=r.record_id, raw_data_row_1based=r.raw_data_row_1based,
                     input_lon=r.lon, input_lat=r.lat, input_raw_MAR=r.raw_MAR,
                     input_g_m2_yr=r.converted_g_m2_yr,
                     delta_lon_deg=float(dx[idx]), delta_lat_deg=float(dy[idx]),
                     exact_coordinate=exact, relative_LGM_difference=error,
                     within_0p1percent=match, within_0p2percent=bool(error <= .002),
                     input_over_candidate_LGM=r.converted_g_m2_yr / value if value > 0 else float('nan'),
                     relative_LGM_times_PM10_difference=scaled_error,
                     numerical_evidence='COORDINATE_AND_LGM_CONSISTENT' if match else
                         'COORDINATE_ONLY_LGM_DIFFERS' if math.isfinite(value) else 'COORDINATE_ONLY_LGM_MISSING',
                     identity_status=UNRESOLVED)
            candidates.append(s)
    c = pd.DataFrame(candidates)
    statuses = old[['record_id', 'raw_data_row_1based', 'lon', 'lat', 'raw_MAR',
                    'converted_g_m2_yr', 'coordinate_multiplicity', 'exact_value_multiplicity',
                    'independence_status']].copy()
    masks = {
        'v2_all_tables': c.source_family.eq('DIRTMAP_v2'),
        'v2_table6': c.source_table.eq('table6_all_data.txt'),
        's1': c.source_family.eq('Lambert2015_S1'),
        'combined': pd.Series(True, index=c.index)}
    summaries = {}
    for label, mask in masks.items():
        z = c[mask]
        conditions = {'coordinate_candidate_rows': pd.Series(True, index=z.index),
                      'exact_coordinate_rows': z.exact_coordinate,
                      'coordinate_value_rows_0p1percent': z.within_0p1percent,
                      'exact_coordinate_value_rows_0p1percent': z.exact_coordinate & z.within_0p1percent,
                      'coordinate_value_rows_0p2percent': z.within_0p2percent,
                      'historical_age_candidate_rows': z.candidate_age_begin_ka.notna() & z.candidate_age_end_ka.notna()}
        counts = {}
        for suffix, condition in conditions.items():
            n = z[condition].groupby('record_id').size()
            statuses[label + '_' + suffix] = statuses.record_id.map(n).fillna(0).astype(int)
            counts[suffix] = {'candidate_pairs': int(condition.sum()), 'input_records': int(len(n))}
        # Multiple source rows may repeat a site across tables. Do not call them independent identities.
        site_counts = z.groupby('record_id').candidate_core.nunique()
        statuses[label + '_distinct_candidate_names'] = statuses.record_id.map(site_counts).fillna(0).astype(int)
        counts['records_with_multiple_candidate_names'] = int((site_counts > 1).sum())
        summaries[label] = counts
    statuses['numerical_evidence'] = np.select([
        statuses.combined_coordinate_value_rows_0p1percent.gt(0),
        statuses.combined_coordinate_candidate_rows.gt(0)],
        ['COORDINATE_AND_LGM_CONSISTENT', 'COORDINATE_CANDIDATES_ONLY'], default='NO_CANDIDATE_WITHIN_RULE')
    statuses['identity_status'] = UNRESOLVED
    # All metadata remains on candidate rows, never transferred as current identities.
    assert statuses.record_id.is_unique and len(statuses) == 317
    assert statuses.identity_status.eq(UNRESOLVED).all()
    assert not c.duplicated(['record_id', 'source_table', 'source_data_row_1based']).any()
    ages = src[src.source_table.eq('table2_marine_sediment.txt')].copy()
    ages['has_both_bounds'] = ages.candidate_age_begin_ka.notna() & ages.candidate_age_end_ka.notna()
    age_counts = ages[ages.has_both_bounds].groupby(['candidate_age_begin_ka', 'candidate_age_end_ka']).size().reset_index(name='historical_source_rows')
    # Diagnose the legacy 309 rows without using their rounded numbers for matching.
    legacy = pd.read_csv(ROOT / 'Data/metadata/dirtmap_v2_identity/input_to_DirtmapV2_identity.csv')
    legacy_links = []
    for idx, r in legacy.iterrows():
        hit = old[(abs(old.lon - r.lon) <= .005000001) & (abs(old.lat - r.lat) <= .005000001)
                  & (abs(old.converted_g_m2_yr - r.value_g_m2_yr) <= .000050001)]
        for _, a in hit.iterrows():
            legacy_links.append(dict(legacy_data_row_1based=idx+1, record_id=int(a.record_id),
                legacy_lon=r.lon, legacy_lat=r.lat, legacy_value=r.value_g_m2_yr,
                original_lon=a.lon, original_lat=a.lat, original_value=a.converted_g_m2_yr,
                relation='ROUNDING_BIN_ONLY_NOT_IDENTITY'))
    links = pd.DataFrame(legacy_links)
    assert set(links.record_id) == set(old.record_id)
    stats = dict(
        input_records=317, distinct_coordinate_value_rows=len(retained.drop_duplicates(['Longitude','Latitude','MAR'])),
        distinct_coordinates=len(retained.drop_duplicates(['Longitude','Latitude'])),
        exact_repeat_excess=317-len(retained.drop_duplicates(['Longitude','Latitude','MAR'])),
        formal_identity_confirmations=0, denominator=317, candidate_summaries=summaries,
        legacy_rows=len(legacy), legacy_rounding_covers_input_records=int(links.record_id.nunique()),
        legacy_rounding_bins_merging_distinct_values=int(sum(
            len(g[['original_lon','original_lat','original_value']].drop_duplicates()) > 1
            for _, g in links.groupby('legacy_data_row_1based'))),
        s1_source_rows=len(s1),
        s1_candidate_input_records=int(c[c.source_family.eq('Lambert2015_S1')].record_id.nunique()),
        historical_v2_marine_rows=len(ages), historical_v2_marine_rows_with_age_bounds=int(ages.has_both_bounds.sum()),
        historical_age_combinations=len(age_counts),
        historical_rows_with_19_26=int(((ages.candidate_age_begin_ka==19)&(ages.candidate_age_end_ka==26)).sum()),
        rules=dict(coordinates='No rounding; exact equality flagged separately. Exploratory rectangular tolerances: v2 <=0.05 deg per axis; S1 <=0.1 deg per axis; wrap longitude.',
                   values='abs(input/source_LGM - 1) <=0.001; sensitivity <=0.002; missing LGM never matches; PM10 product diagnostic only.',
                   identities='No original row IDs or documented inter-version transformations recovered; all lineage remains unresolved.',
                   counts='Coverage counts input record_id, not join rows, unique names, or sums across overlapping sources.',
                   source_rows='Every source table row preserved; repeated entries across or within tables are not independent evidence.'),
        protected_sha256=before)
    after = {str(p.relative_to(ROOT)): sha(p) for p in protected if p.exists()}
    assert before == after
    stats['protected_files_unchanged'] = True
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in [('record_candidate_status_317.csv', statuses), ('all_source_candidates.csv', c),
                        ('source_table_inventory.csv', inventory), ('Lambert2015_TableS1_from_docx.csv', s1),
                        ('historical_marine_age_rows.csv', ages), ('historical_marine_age_combinations.csv', age_counts),
                        ('legacy_309_to_record_id.csv', links)]:
        frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')
    (OUT / 'summary.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    write_reports(stats)
    (OUT / 'manifest.json').write_text(json.dumps(dict(
        script_sha256=sha(Path(__file__)), inputs_sha256=before,
        reports_sha256={p.name:sha(p) for p in (ROOT / 'docs/04_数据说明').glob('*20261116.md')
                        if p.name in ['材料清单总表_20261116.md', 'DIRTMAPv2_逐记录表与编制说明_20261116.md',
                                      'Lambert2015_SI_TableS1解析_20261116.md', 'DIRTMAP_source_search_20261116.md']},
        outputs_sha256={p.name:sha(p) for p in OUT.glob('*') if p.is_file() and p.name != 'manifest.json'}),
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in stats.items() if k not in ['protected_sha256', 'rules']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

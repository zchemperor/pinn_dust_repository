# 来源候选子集描述性诊断（2026-11-16）

采用确切record_id和当前复核规则：精确坐标+通量候选93条、容差坐标+通量98条。候选不是身份确认，也不是清洁/统一档案数据。冻结原模型及训练常数，无重训或数据改动。

## 子集分布

| split                              | definition       | selection   |   n |   n_coordinates |   fraction |   log_min |   log_max |   log_mean |   log_sd |
|:-----------------------------------|:-----------------|:------------|----:|----------------:|-----------:|----------:|----------:|-----------:|---------:|
| outer_north_america                | exact_flux93     | candidate   |   2 |               2 |     0.0312 |   -1.0969 |    0.3424 |    -0.3772 |   1.0178 |
| outer_north_america                | exact_flux93     | other       |  62 |              59 |     0.9688 |    0.8921 |    3.1796 |     2.0944 |   0.5223 |
| outer_north_america                | tolerance_flux98 | candidate   |   2 |               2 |     0.0312 |   -1.0969 |    0.3424 |    -0.3772 |   1.0178 |
| outer_north_america                | tolerance_flux98 | other       |  62 |              59 |     0.9688 |    0.8921 |    3.1796 |     2.0944 |   0.5223 |
| outer_east_asia                    | exact_flux93     | candidate   |  10 |              10 |     0.0943 |    0.2833 |    1.1559 |     0.8680 |   0.2958 |
| outer_east_asia                    | exact_flux93     | other       |  96 |              87 |     0.9057 |    0.7059 |    3.0809 |     1.7911 |   0.4452 |
| outer_east_asia                    | tolerance_flux98 | candidate   |  11 |              11 |     0.1038 |    0.2833 |    1.1559 |     0.8533 |   0.2849 |
| outer_east_asia                    | tolerance_flux98 | other       |  95 |              86 |     0.8962 |    0.7782 |    3.0809 |     1.8026 |   0.4331 |
| outer_tropical_atlantic_africa     | exact_flux93     | candidate   |  40 |              39 |     0.9524 |    0.0792 |    2.6021 |     1.1401 |   0.6450 |
| outer_tropical_atlantic_africa     | exact_flux93     | other       |   2 |               2 |     0.0476 |    1.7482 |    2.7226 |     2.2354 |   0.6890 |
| outer_tropical_atlantic_africa     | tolerance_flux98 | candidate   |  40 |              39 |     0.9524 |    0.0792 |    2.6021 |     1.1401 |   0.6450 |
| outer_tropical_atlantic_africa     | tolerance_flux98 | other       |   2 |               2 |     0.0476 |    1.7482 |    2.7226 |     2.2354 |   0.6890 |
| stress_leave_south                 | exact_flux93     | candidate   |  19 |              18 |     0.5278 |   -2.0706 |    2.4393 |     0.2492 |   1.2994 |
| stress_leave_south                 | exact_flux93     | other       |  17 |              16 |     0.4722 |   -1.9208 |    1.5587 |     0.3838 |   0.9878 |
| stress_leave_south                 | tolerance_flux98 | candidate   |  21 |              20 |     0.5833 |   -2.0706 |    2.4393 |     0.2576 |   1.2410 |
| stress_leave_south                 | tolerance_flux98 | other       |  15 |              14 |     0.4167 |   -1.9208 |    1.5587 |     0.3899 |   1.0424 |
| stress_leave_tropics               | exact_flux93     | candidate   |  57 |              56 |     0.8143 |   -0.4437 |    2.6021 |     1.0615 |   0.6631 |
| stress_leave_tropics               | exact_flux93     | other       |  13 |              13 |     0.1857 |   -0.9737 |    2.7226 |    -0.0056 |   1.0778 |
| stress_leave_tropics               | tolerance_flux98 | candidate   |  59 |              58 |     0.8429 |   -0.4437 |    2.6021 |     1.0170 |   0.6948 |
| stress_leave_tropics               | tolerance_flux98 | other       |  11 |              11 |     0.1571 |   -0.9737 |    2.7226 |     0.0395 |   1.1730 |
| stress_leave_north_excluding_p2dev | exact_flux93     | candidate   |  17 |              17 |     0.0971 |   -1.0969 |    1.1559 |     0.3466 |   0.7855 |
| stress_leave_north_excluding_p2dev | exact_flux93     | other       | 158 |             146 |     0.9029 |    0.7059 |    3.1796 |     1.9101 |   0.4980 |
| stress_leave_north_excluding_p2dev | tolerance_flux98 | candidate   |  18 |              18 |     0.1029 |   -1.0969 |    1.1559 |     0.3666 |   0.7667 |
| stress_leave_north_excluding_p2dev | tolerance_flux98 | other       | 157 |             145 |     0.8971 |    0.7782 |    3.1796 |     1.9178 |   0.4901 |

精确候选在六区分别为2、10、40、19、57、17条，占3.1%、9.4%、95.2%、52.8%、81.4%、9.7%。明显不均衡；北美仅2条，热带区域留出的非候选也只有2条。全部仅作描述，不做区间或显著性推断。未发生空集或方差为零，R²虽可算，极小样本R²不能承载泛化结论。

## 精确候选的固定Wind集成指标

| split                              |   n |     rmse |      mae |        r2 |     bias |
|:-----------------------------------|----:|---------:|---------:|----------:|---------:|
| outer_north_america                |   2 | 1.756981 | 1.586670 | -4.960346 | 1.586670 |
| outer_east_asia                    |  10 | 0.315137 | 0.254315 | -0.261101 | 0.122433 |
| outer_tropical_atlantic_africa     |  40 | 0.677927 | 0.588769 | -0.133179 | 0.272936 |
| stress_leave_south                 |  19 | 1.690513 | 1.208030 | -0.786647 | 1.003659 |
| stress_leave_tropics               |  57 | 0.756157 | 0.656373 | -0.323595 | 0.439549 |
| stress_leave_north_excluding_p2dev |  17 | 1.005496 | 0.714047 | -0.741007 | 0.552056 |

## 子集点估计变化

精确候选中各区最低集成RMSE如下（不声称显著优于）：
| split                              | model          |   n |     rmse |
|:-----------------------------------|:---------------|----:|---------:|
| outer_east_asia                    | PDE-MAP        |  10 | 0.289928 |
| outer_north_america                | Diffusion-PINN |   2 | 0.634055 |
| outer_tropical_atlantic_africa     | Diffusion-PINN |  40 | 0.527236 |
| stress_leave_north_excluding_p2dev | No-PDE MLP     |  17 | 0.778060 |
| stress_leave_south                 | Diffusion-PINN |  19 | 1.118302 |
| stress_leave_tropics               | Diffusion-PINN |  57 | 0.689196 |

北美最低RMSE从全表的训练常数变为零风（仅2点）；北部从PDE-MAP变为No-PDE（17点）。Wind对零风的RMSE点估计在北美、东亚、热带外推和北部会出现不同于全表的方向或口径图样；热带区域留出的双口径反转保留。这说明候选子集不能被当成代表全317记录的随机样本。

候选与其他记录的目标范围/均值也不同，例如北美log目标均值−0.377与2.094、北部0.347与1.910。因此不能将误差变化归因于身份已恢复、数据质量更高或某种档案类型。精确候选下Wind集成R²在六区仍全部负。98条容差集合的结果全部保留在CSV中，不依表现选择报告集合。

## 局限与产物

不使用候选档案/年代作分层，不按子集重拟合或更换训练常数，不替换正式基准，不将重叠的93/98集合当独立证据。metrics.csv保存两个口径、候选/其他及两种规则；wind_contrasts.csv保存所有可用比较对象，subset_memberships.csv保留逐行归属。输入/模型对齐及前后哈希验证通过。

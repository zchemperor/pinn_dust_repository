# 3 km近邻组冻结预测敏感性（2026-11-16）

后设分析；无训练、无删行。球面R=6371 km，3 km连通分量，全部单点保留。共10个非单点组28行；3 km是探索阈值，不是独立性尺度。

## 几何与划分

| split                              |   n_test |   n_coordinate_groups |   n_nearby_groups |   n_cross_split_components |   n_cross_split_pairs_3km |   nearest_cross_split_km |
|:-----------------------------------|---------:|----------------------:|------------------:|---------------------------:|--------------------------:|-------------------------:|
| outer_north_america                |       64 |                    61 |                60 |                          0 |                         0 |                  540.518 |
| outer_east_asia                    |      106 |                    97 |                96 |                          0 |                         0 |                  859.317 |
| outer_tropical_atlantic_africa     |       42 |                    41 |                41 |                          0 |                         0 |                  202.578 |
| stress_leave_south                 |       36 |                    33 |                33 |                          0 |                         0 |                   88.396 |
| stress_leave_tropics               |       70 |                    69 |                69 |                          0 |                         0 |                   88.396 |
| stress_leave_north_excluding_p2dev |      175 |                   163 |               161 |                          0 |                         0 |                  859.317 |

六区没有近邻连通组或直接近邻边跨训练/测试，不能据此推断更大空间尺度的独立性。

## Wind减零风：双口径

| split                              | scheme         | estimand         |     delta |   ci95_low |   ci95_high | classification   |
|:-----------------------------------|:---------------|:-----------------|----------:|-----------:|------------:|:-----------------|
| outer_north_america                | nearby_cluster | ensemble         | -0.950625 |  -1.034760 |   -0.834128 | A_lower_error    |
| outer_north_america                | nearby_cluster | mean_seed_metric | -0.953164 |  -1.036627 |   -0.837425 | A_lower_error    |
| outer_north_america                | nearby_equal   | ensemble         | -0.960437 |  -1.038688 |   -0.841046 | A_lower_error    |
| outer_north_america                | nearby_equal   | mean_seed_metric | -0.962862 |  -1.040369 |   -0.844234 | A_lower_error    |
| outer_east_asia                    | nearby_cluster | ensemble         |  0.005205 |  -0.080475 |    0.086680 | unresolved       |
| outer_east_asia                    | nearby_cluster | mean_seed_metric | -0.052329 |  -0.131995 |    0.022355 | unresolved       |
| outer_east_asia                    | nearby_equal   | ensemble         |  0.001856 |  -0.087292 |    0.087068 | unresolved       |
| outer_east_asia                    | nearby_equal   | mean_seed_metric | -0.055542 |  -0.137710 |    0.022091 | unresolved       |
| outer_tropical_atlantic_africa     | nearby_cluster | ensemble         |  0.153645 |   0.100288 |    0.203051 | A_higher_error   |
| outer_tropical_atlantic_africa     | nearby_cluster | mean_seed_metric | -0.175629 |  -0.251913 |   -0.100070 | A_lower_error    |
| outer_tropical_atlantic_africa     | nearby_equal   | ensemble         |  0.157470 |   0.104943 |    0.205506 | A_higher_error   |
| outer_tropical_atlantic_africa     | nearby_equal   | mean_seed_metric | -0.175015 |  -0.253622 |   -0.097072 | A_lower_error    |
| stress_leave_south                 | nearby_cluster | ensemble         |  0.581832 |   0.266435 |    0.821073 | A_higher_error   |
| stress_leave_south                 | nearby_cluster | mean_seed_metric |  0.541801 |   0.216528 |    0.792094 | A_higher_error   |
| stress_leave_south                 | nearby_equal   | ensemble         |  0.557903 |   0.265529 |    0.786229 | A_higher_error   |
| stress_leave_south                 | nearby_equal   | mean_seed_metric |  0.514359 |   0.213093 |    0.754124 | A_higher_error   |
| stress_leave_tropics               | nearby_cluster | ensemble         |  0.303781 |   0.102119 |    0.505501 | A_higher_error   |
| stress_leave_tropics               | nearby_cluster | mean_seed_metric |  0.017805 |  -0.173993 |    0.212429 | unresolved       |
| stress_leave_tropics               | nearby_equal   | ensemble         |  0.302540 |   0.094295 |    0.508572 | A_higher_error   |
| stress_leave_tropics               | nearby_equal   | mean_seed_metric |  0.017541 |  -0.179059 |    0.217640 | unresolved       |
| stress_leave_north_excluding_p2dev | nearby_cluster | ensemble         | -0.531748 |  -0.599436 |   -0.462602 | A_lower_error    |
| stress_leave_north_excluding_p2dev | nearby_cluster | mean_seed_metric | -0.536995 |  -0.604172 |   -0.468136 | A_lower_error    |
| stress_leave_north_excluding_p2dev | nearby_equal   | ensemble         | -0.541046 |  -0.607631 |   -0.472052 | A_lower_error    |
| stress_leave_north_excluding_p2dev | nearby_equal   | mean_seed_metric | -0.546151 |  -0.612184 |   -0.477530 | A_lower_error    |

## 敏感性变化

相对同一次计算的精确坐标聚类方案，两个近邻方案共864行全模型配对分类，其中264行Wind比较全部保持分类。所有六区最低集成RMSE预测器未改变，Wind集成R²仍全部负。热带区域留出集成零风较好、逐种子指标均值Wind较好的反转保留。

4行分类变化实际是两个确定性比较在双口径中重复出现，不是四份独立证据：
| split           | scheme       | estimand         | A       | B                   | metric   |     delta |   ci95_low |   ci95_high | classification_coordinate   | classification   |
|:----------------|:-------------|:-----------------|:--------|:--------------------|:---------|----------:|-----------:|------------:|:----------------------------|:-----------------|
| outer_east_asia | nearby_equal | ensemble         | PDE-MAP | Train-mean constant | rmse     | -0.024951 |  -0.050580 |   -0.000366 | unresolved                  | A_lower_error    |
| outer_east_asia | nearby_equal | mean_seed_metric | PDE-MAP | Train-mean constant | rmse     | -0.024951 |  -0.050580 |   -0.000366 | unresolved                  | A_lower_error    |
| outer_east_asia | nearby_equal | ensemble         | GP/RBF  | Train-mean constant | mae      |  0.041334 |  -0.008704 |    0.090200 | A_higher_error              | unresolved       |
| outer_east_asia | nearby_equal | mean_seed_metric | GP/RBF  | Train-mean constant | mae      |  0.041334 |  -0.008704 |    0.090200 | A_higher_error              | unresolved       |

东亚PDE-MAP对训练常数在组等权下略低，GP的MAE对训练常数转为未分辨；与既有精确坐标等权的边界变化性质相同，不推广为稳定的新优势。只有聚类、不改记录权重时全比较分类均无变化。

## 计算验证与限制

每方案20,000次配对组bootstrap，种子20261117；百分位名义95%区间、未多重校正、固定五训练种子。检查输入哈希、原记录等权指标、sklearn加权指标、每方案首次抽样独立展开通过；本次精确坐标区间分类与原P3全部一致。详细产物manifest.json、metrics.csv、paired_contrasts.csv及changes_vs_coordinate.csv。

组等权是先平均组内损失再平均组间损失，不平均目标或预测。训练权重与原训练常数不变，不能纠正训练重复、确证测量身份、恢复元数据或建立全面空间独立。

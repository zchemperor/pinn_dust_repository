"""Summarize completed independent residual diagnostics; leave manuscript untouched."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Data/model_results/v5_pmip4_2d/coefficient_residual_audit_20261117'
NOTICE='independent evaluation on reconstructed point sets; not a recovery of the original training losses'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(frame,name):
    frame=frame.copy();frame.insert(0,'evaluation_notice',NOTICE)
    frame.to_csv(OUT/name,index=False,float_format='%.17g',encoding='utf-8-sig')
def fmt(x):return f'{x:.6g}'
def interval(series,percent=False):
    f=100 if percent else 1
    return f'{fmt(series.min()*f)} / {fmt(series.median()*f)} / {fmt(series.max()*f)}'

def main():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='PASS' and manifest['runs']==78 and not manifest['smoke_test']
    for rel,h in manifest['input_sha256'].items():assert sha(ROOT/rel)==h,rel
    for rel,h in manifest['output_sha256'].items():assert sha(OUT/rel)==h,rel
    d=pd.read_csv(OUT/'residual_diagnostics.csv',float_precision='round_trip')
    c=pd.read_csv(OUT/'coefficients.csv',float_precision='round_trip')
    l=pd.read_csv(OUT/'historical_loss_components.csv',float_precision='round_trip')
    checks=pd.read_csv(OUT/'numerical_checks.csv',float_precision='round_trip')
    sensitivity=pd.read_csv(OUT/'point_set_sensitivity.csv',float_precision='round_trip')
    assert len(d)==156 and len(c)==78 and len(l)==78*151 and len(checks)==156
    assert d.groupby('run').size().eq(2).all()
    zero=d.model=='Diffusion-PINN';wind=d.model=='Wind-PINN'
    assert len(c[c.model=='Diffusion-PINN'])==48 and c[c.model=='Diffusion-PINN'].near_lower.all()
    assert not c[c.model=='Wind-PINN'].near_lower.any()
    assert (d.loc[zero,'q_squared_mean']>0).all() and (d.loc[zero,'conditional_gradient']>0).all()
    assert (d.loc[zero,['a_squared_mean','a_q_product_mean','signed_cross_term']]==0).all().all()
    np.testing.assert_allclose(d.residual_squared_mean,d.a_squared_mean+d.diffusion_squared_mean+d.signed_cross_term,rtol=1e-10,atol=1e-9)
    # Interior conditional minimizers have a negative cross term, except the flat/zero limit.
    wi=d[wind & (d.conditional_optimum_location=='interior')]
    np.testing.assert_allclose(2*wi.conditional_unbounded_optimum*wi.a_q_product_mean,
                              -2*wi.conditional_unbounded_optimum**2*wi.q_squared_mean,rtol=1e-12)
    assert (2*wi.conditional_unbounded_optimum*wi.a_q_product_mean<0).all()
    summary=[]
    keys=['family','model','pde_weight','split','point_set']
    measures=['d_nd','q_squared_mean','a_squared_mean','a_q_product_mean','signed_cross_term',
              'diffusion_squared_mean','residual_squared_mean','pde_fraction','conditional_gradient',
              'conditional_unbounded_optimum','cancellation_fraction']
    for key,g in d.groupby(keys,sort=False):
        r=dict(zip(keys,key));r['runs']=len(g)
        for m in measures:
            for stat in ['min','median','max']:r[m+'_'+stat]=getattr(g[m],stat)()
        summary.append(r)
    write(pd.DataFrame(summary),'summary_by_split.csv')
    logsummary=[]
    for key,g in l.groupby(['family','model','pde_weight','step'],sort=False):
        r=dict(zip(['family','model','pde_weight','step'],key));r['runs']=len(g)
        for field in ['logged_pde_fraction','logged_weighted_pde','logged_total']:
            for stat in ['min','median','max']:r[field+'_'+stat]=getattr(g[field],stat)()
        logsummary.append(r)
    write(pd.DataFrame(logsummary),'historical_fraction_summary.csv')
    final=l[l.step==150000].set_index('run')
    rebuilt=d[d.point_set=='reconstructed'].set_index('run')
    difference=pd.DataFrame({'run':rebuilt.index,
        'independent_pde_minus_logged_final':(rebuilt.weighted_residual-final.logged_weighted_pde).to_numpy(),
        'independent_composite_minus_logged_final':(rebuilt.independent_composite_objective-final.logged_total).to_numpy()})
    write(difference,'independent_vs_logged_final.csv')
    # Check all old Data CSVs and all six outward files, not just sampled sources.
    old=json.loads((ROOT/'tmp/supplement_delivery_20261117/backup.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==h for p,h in old['csv_hashes'].items())
    package=json.loads((ROOT/'tmp/supplement_delivery_20261117/package_validation.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/'manuscript/submission'/n)==h for n,h in package['hashes'].items())
    stats=dict(evaluation_notice=NOTICE,status='PASS',runs=78,evaluations=156,logged_rows=len(l),
        zero_near_lower_count=48,wind_near_lower_count=0,
        zero_d_range=[c[c.model=='Diffusion-PINN'].d_nd.min(),c[c.model=='Diffusion-PINN'].d_nd.max()],
        cross_negative_at_actual_coefficient=int((d.loc[wind,'signed_cross_term']<0).sum()),
        wind_conditional_optimum_locations=d[wind].groupby('point_set').conditional_optimum_location.value_counts().to_dict().__str__(),
        max_prediction_error=float(checks.prediction_max_abs_error.max()),
        max_decomposition_error=float(checks.decomposition_max_abs_error.max()),
        max_independent_laplacian_error=float(checks.independent_laplacian_max_abs_error.max()),
        max_tf_coefficient_gradient_error=float(checks.tf_gradient_abs_error.max()),
        max_tf_coefficient_curvature_error=float(checks.tf_curvature_abs_error.max()),
        max_reconstructed_vs_logged_pde_difference=float(difference.independent_pde_minus_logged_final.abs().max()),
        max_reconstructed_vs_logged_composite_difference=float(difference.independent_composite_minus_logged_final.abs().max()),
        original_data_csv_count=len(old['csv_hashes']),original_data_csv_unchanged=True,
        manuscript_and_submission_package_unchanged=True,no_training=True)
    (OUT/'validation.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')

    text=NOTICE+'''

# 系数优化结构与残差量级诊断结果

78次常系数最终拟合、两套点集的156次评价及历史日志核验完成。未训练、未改动任何既有材料。协议先于新增结果固定；Relative-D不套用本轮标量公式。

## 一、解析结论的准确范围

固定网络场c，q为单位D_nd的无量纲球面扩散项，A为带符号无量纲平流项，R=A+D_nd q。因此

L_R=<A²>+2D_nd<Aq>+D_nd²<q²>；∂L_R/∂D_nd=2(<Aq>+D_nd<q²>)，二阶导=2<q²>。

零风A=0，q²均值非零时，正系数方向严格递增，系数趋向下界可降低固定场残差；但有限训练、sigmoid参数化与近零曲率不保证所有训练必然近界。q恒零时系数方向平坦。有风无约束条件最优值为-<Aq>/<q²>，只有落在(0.02,2)内才是内部条件最优；边界对应开区间极限，不能声称sigmoid恰好到界。内部条件最优时交叉项为-2D*²<q²><0，实际最终拟合不自动满足该驻点条件。

这些是固定场条件结论，不证明网络与系数的联合可识别或收敛。移除风项引起条件目标结构改变，是干预的下游后果；不破坏匹配对照。风—零风比较仍估计既定训练程序中加入风项的总预测效果，而非固定最终平滑强度下的效果。D的形式量纲不赋予标准化对数态大气扩散意义。边界重训仍有预测敏感性价值，本轮未做。

## 二、实际系数原始精度

以下是跨种子的最小/最大值；完整未舍入数值、raw_parameter及距下界见coefficients.csv（17位有效数字导出）。

|实验族|模型|PDE权重|拟合数|D_nd最小|D_nd最大|近下界数|
|---|---|---:|---:|---:|---:|---:|
'''
    for key,g in c.groupby(['family','model','pde_weight'],sort=False):
        text+=f'|{key[0]}|{key[1]}|{key[2]:g}|{len(g)}|{g.d_nd.min():.17g}|{g.d_nd.max():.17g}|{g.near_lower.sum()}|\n'
    text+='''
近界判据为(D_nd−0.02)/1.98≤10⁻⁴；这里使用“近界”而非“到界”。零风48次全部近界，与固定场下降方向一致，不能当成数据确定某个扩散系数的证据。

## 三、残差量级和加权贡献

表内为同组拟合的最小/中位/最大值；主实验跨六划分，权重实验仅三个外推划分，每权重9次。不同划分的c用各自训练标准化，合并仅描述范围，不作共同物理量或统计重复解释。

|实验族/模型|权重|点集|<q²>|L_R|加权PDE占组合目标(%)|
|---|---:|---|---|---|---|
'''
    for key,g in d.groupby(['family','model','pde_weight','point_set'],sort=False):
        text+=f'|{key[0]}/{key[1]}|{key[2]:g}|{key[3]}|{interval(g.q_squared_mean)}|{interval(g.residual_squared_mean)}|{interval(g.pde_fraction,True)}|\n'
    text+='''
占比分母为加权PDE加上观测、两项接缝和南北边界加权损失。规则网格评价只替换PDE点，非PDE项保持同一重建训练/边界点，因而是独立组合目标。损失数值占比不是网络参数梯度占比，也不能单独识别训练路径或预测增益来源。较小系数不能替代<q²>；较小最终贡献也不能排除早期训练影响。

## 四、带符号交叉项与条件最优

|点集|Wind次数|实际交叉项<0次数|交叉项最小/中位/最大|抵消比例最小/中位/最大|条件最优位置计数|
|---|---:|---:|---|---|---|
'''
    for name,g in d[wind].groupby('point_set',sort=False):
        text+=f'|{name}|{len(g)}|{(g.signed_cross_term<0).sum()}|{interval(g.signed_cross_term)}|{interval(g.cancellation_fraction)}|{g.conditional_optimum_location.value_counts().to_dict()}|\n'
    text+='''
抵消比例定义为−交叉项/(<A²>+D_nd²<q²>)，保留符号；接近1表示该点集上两项平方和被交叉项大幅抵消，不是独立物理机制验证。每次实际系数与条件最优的差和条件导数均已输出；不能因为系数在内部就认定达到驻点。两点集也可能给出不同条件最优值。Relative-D未进入这项推断。

## 五、点集敏感性

同一run配对计算规则网格/重建点阵比值，不把两点集当独立拟合：

|实验族/模型|权重|<q²>比值最小/中位/最大|L_R比值最小/中位/最大|
|---|---:|---|---|
'''
    paired=sensitivity.merge(c[['run','family','model','pde_weight']],on='run',validate='one_to_one')
    for key,g in paired.groupby(['family','model','pde_weight'],sort=False):
        text+=f'|{key[0]}/{key[1]}|{key[2]:g}|{interval(g.q_squared_mean_grid_over_reconstructed)}|{interval(g.residual_squared_mean_grid_over_reconstructed)}|\n'
    text+='''
重建点阵按当前DeepXDE得到2618内点+218边界点+训练锚点（含原重复）；规则网格为同域120×60单元中心，不含锚点与边界。两者无面积加权、非全球物理积分；差异包含分辨率、分布以及锚点/边界构成差异，不能归因为单一因素。实际数组、点数和哈希已保存于point_sets/及point_set_inventory.csv。没有历史collocation数组，因此不声称恢复原训练点。

## 六、历史日志与独立评价分开

historical_loss_components.csv记录78×151个既有时点，损失原已加权。以下是历史最终PDE占比，不是重建结果：

|实验族/模型|权重|最终历史PDE占比最小/中位/最大(%)|
|---|---:|---|
'''
    for key,g in l[l.step==150000].groupby(['family','model','pde_weight'],sort=False):
        text+=f'|{key[0]}/{key[1]}|{key[2]:g}|{interval(g.logged_pde_fraction,True)}|\n'
    text+=f'''
重建点阵评价与原日志最终加权PDE的最大绝对差为{stats['max_reconstructed_vs_logged_pde_difference']:.6g}，组合目标最大差为{stats['max_reconstructed_vs_logged_composite_difference']:.6g}。数值相近可作一致性检查，但不能填补历史点阵及环境归档缺失，不升级为原训练损失恢复。151个时点的历史占比汇总在historical_fraction_summary.csv，不从这些标量恢复此前场曲率。

## 七、验收、交付与下一步边界

- 78次预测最大绝对归一化误差{stats['max_prediction_error']:.6g}；残差分解最大误差{stats['max_decomposition_error']:.6g}；独立常系数拉普拉斯形式最大误差{stats['max_independent_laplacian_error']:.6g}。TensorFlow系数一阶/二阶导与解析式最大误差分别{stats['max_tf_coefficient_gradient_error']:.6g}/{stats['max_tf_coefficient_curvature_error']:.6g}。全部满足预设数值容差。
- 原1306份Data CSV及全部追踪输入哈希复核通过。没有训练、写checkpoint、修改模型实现或对外发布。
- 入口：Code/analyze_coefficient_residual_diagnostic.py；汇总：Code/summarize_coefficient_residual_diagnostic.py。产物：Data/model_results/v5_pmip4_2d/coefficient_residual_audit_20261117/。主明细residual_diagnostics.csv含两点集每次拟合的完整分解和加权项，summary_by_split.csv保留划分层面的范围。
- 解析核验与量级检查现已完成，可以据此简洁说明“系数优化结构与解释边界”。不写可识别性不对称、不把近界当作经验物理发现，不以最终损失占比断言预测收益机制。若在论文中引用，应同步交付其必要方法和表格。
'''
    report=OUT/'diagnostic_report.md'
    report.write_text(text,encoding='utf-8')
    readme=NOTICE+'''

# Fixed-coefficient residual diagnostics

Scope: 30 primary Wind-PINN +30 primary Diffusion-PINN +18 zero-wind residual-weight fits. No Relative-D scalar-optimum inference. No training, optimization, input edits or manuscript edits.

## Files and definitions

- coefficients.csv: 78 exact checkpoint scalar transforms, raw parameter, D_nd−0.02, and near-bound fraction; 17 significant digits. Coefficients are dimensionless numerical parameters, not measured diffusivities.
- residual_diagnostics.csv: 156 independent evaluations. A is signed dimensionless advection; q is dimensionless diffusion with D_nd=1 and fixed network. a_squared_mean=<A²>; q_squared_mean=<q²>; a_q_product_mean=<Aq> is an uncentred product mean. diffusion_squared_mean=D_nd²<q²>; signed_cross_term=2D_nd<Aq>; residual_squared_mean=<R²>. Never take the absolute value of the cross term. All means are arithmetic point means.
- conditional_gradient/curvature: derivative of unweighted residual MSE with respect to an independent D_nd probe, holding the fitted field fixed; not network gradient or joint identifiability. conditional_unbounded_optimum is −<Aq>/<q²> when q²>0. Conditional location denotes interior or the limiting lower/upper bound of the sigmoid interval; it is not a claim about attained optimization. cancellation_fraction=−cross/(A²+diffusion²), retaining sign.
- observation/seam/north/south *_mse: unweighted independent losses at reconstructed observation and BC points. Non-PDE weights are 10,0.5,0.5,1,1. weighted_residual=pde_weight*residual_squared_mean; independent_composite_objective adds weighted_nonpde; pde_fraction divides the two. The regular-grid variant replaces PDE points only. These are not recovered historical objectives, nor gradient contribution estimates.
- point_sets/*.npz and point_set_inventory.csv: actual reconstructed PDE/observation/BC arrays and independent grid arrays, normalized coordinates lon/90 and lat/90. Reconstruction: 2618 interior +218 boundary +training anchors, duplicates retained. Independent grid: 120×60 cell centres on [-2,2]×[-0.89,0.89]. Both are unweighted means, not sphere-area integrals. No historical collocation arrays were recovered.
- historical_loss_components.csv: original already-weighted logged losses, not independent evaluations or recovered historical fields; 151 logged steps per fit. The evaluation_notice describes the diagnostic's scope, while data_origin distinguishes historical log extraction. historical_fraction_summary.csv summarizes these recorded values by family/model/weight/step.
- point_set_sensitivity.csv: paired grid/reconstructed ratios by run; these point sets are not independent runs. independent_vs_logged_final.csv compares numerical values without claiming historical point recovery.
- numerical_checks.csv: final prediction restoration against saved normalized predictions; algebraic decomposition, independent constant-coefficient Laplacian and autodiff probe checks. summary_by_split.csv retains split-level min/median/max. validation.json checks original inputs and current submission package are unchanged. manifest.json records source/output hashes and versions.

Missing/undefined conditional values are blank, not zero. Split and run labels retain project bookkeeping for reproducibility. Reports make no significance, convergence, atmospheric identification, or causal mechanism claim. Small final PDE loss cannot exclude earlier optimization-path effects. No new files have been added to the submission package or published remotely.
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    # Extend manifest to include postprocessing outputs, keeping evaluation input hashes intact.
    manifest['summary_script_sha256']=sha(Path(__file__))
    manifest['output_sha256']={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

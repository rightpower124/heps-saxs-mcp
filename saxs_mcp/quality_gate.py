# -*- coding: utf-8 -*-
"""L3 质量门（M2-T2）：每次拟合自动执行的可靠性检查模块，与模型无关。
由 saxs_fit 在每次拟合后自动调用。分级 SOP 见设计文档 §3.2：
  条件值 = 撞界参数（不得单独进入结论句）
  归一量 = 以模型内参照归一的参数（A 比例因子 / B 背景 / 相对电子密度等，
           跨样品、跨体标不可直接比较；由模型插件声明 normalized_params）
  稳健   = 其余参数（撞界优先级高于归一量）
chi2_flag 阈值：chi2/dof > 3.0 记为"需关注"（拟合与误差棒不自洽，可能是模型
失配或 sigma 低估），只标记不判失败，交由 Agent/人工解读。
perr 爆表规则（2026-10-10 挂账，2026-10-09 fuzzy_sphere_extended R perr~2700×
误标"稳健"实测触发）：perr/|value| > 1.0（>100%）→ 条件值；value=0 时 perr>0
即爆表。仅降级不拦截。分级优先级：撞界 > perr 爆表 > 归一量 > 稳健。
嵌套 F 检验用于模型比较（如含 PEG vs 无 PEG 对照），共享同一数据才有效。
"""
import numpy as np
from scipy.stats import f as f_dist

from fitting import at_bound

CHI2_FLAG = 3.0        # chi2/dof 关注阈值
PERR_REL_FLAG = 1.0    # perr 相对误差（perr/|value|）爆表阈值：>1.0 → 条件值
GRADE_BOUND = '条件值'
GRADE_NORM = '归一量'
GRADE_ROBUST = '稳健'


def grade_params(names, at, normalized_params=None, values=None, perr=None):
    """统一可靠性分级（quality_gate / tools_report 共用单一实现，杜绝双实现
    漂移——2026-09-24 桥接模型 A/B 误标"稳健"即双实现漂移所致）。
    优先级：撞界 > perr 爆表（>PERR_REL_FLAG）> 归一量 > 稳健。
    """
    norm = set(normalized_params or [])
    at = set(at or [])
    out = {}
    for n in names:
        if n in at:
            out[n] = GRADE_BOUND
            continue
        v = (values or {}).get(n)
        e = (perr or {}).get(n)
        if v is not None and e is not None:
            rel = abs(e) / abs(v) if v != 0 else (float('inf') if e != 0 else 0.0)
            if rel > PERR_REL_FLAG:
                out[n] = GRADE_BOUND
                continue
        out[n] = GRADE_NORM if n in norm else GRADE_ROBUST
    return out


def run_gate(q, I, model_vals, chi2, dof, names, at, normalized_params=None,
             perr=None, values=None):
    """单次拟合的质量门：相对残差统计 + 撞界 + 可靠性分级 + chi2 关注标记。
    q/I: 数据；model_vals: 最优参数下的模型值；at: 已算好的撞界列表（复用，不重算）；
    values/perr: 参数值与协方差误差 dict（perr 爆表降级用，缺省不启用该规则）。
    """
    rel = (model_vals - I) / I
    grade = grade_params(names, at, normalized_params, values, perr)
    return {
        'chi2_dof': float(chi2), 'dof': int(dof),
        'residual': {'max_abs_rel': float(np.max(np.abs(rel))),
                     'rms_rel': float(np.sqrt(np.mean(rel ** 2)))},
        'at_bound': list(at),
        'grade': grade,
        'chi2_flag': bool(chi2 > CHI2_FLAG),
    }


def f_test(ssr_full, dof_full, ssr_reduced, dof_reduced):
    """嵌套模型 F 检验：全模型相对简化模型的改善是否显著。
    ssr_* = 加权残差平方和，dof_* = 各自自由度（同一数据、嵌套关系）。
    """
    df1 = dof_reduced - dof_full
    if df1 <= 0:
        raise ValueError('简化模型自由度必须大于全模型（嵌套关系不成立）')
    F = ((ssr_reduced - ssr_full) / df1) / (ssr_full / dof_full)
    p = float(f_dist.sf(F, df1, dof_full))
    return {'F': float(F), 'p_value': p, 'df1': int(df1), 'df2': int(dof_full),
            'conclusion': '显著（α=0.05）' if p < 0.05 else '不显著（α=0.05）'}

# -*- coding: utf-8 -*-
"""fuzzy_sphere_extended 适配层（模型库 5 号，Neubert 2026 桥接复现）：
把 model.py 物理函数接入通用引擎 saxs_fit。本文件只声明"参数表 + 引用方式 +
报告文本钩子"；与 manifest.json 的元数据（labels / refs / modes）一起构成完整模型插件。

方法学来源：Rathgeber et al., J. Chem. Phys. 117 (2002) 4047-4062（blob 散射），
应用锚 Neubert et al., Macromol. Chem. Phys. 2026, 227, e00493（Figure 3；
作者亲笔 SASview 插件源码随数据包归档，逐式转写）。
"""
import numpy as np

_M = None    # register 时绑定的物理模块（model_loader 以独立模块名加载，顶层不可 import）

# ---- 布局声明：两联图（拟合曲线 log 图 + 相对残差） ----
_LAYOUT_DOUBLE = {
    'figsize': (10, 8),
    'grid': (2, 1),
    'panels': [
        {'kind': 'fit', 'at': (0, 0), 'xscale': 'log', 'yscale': 'log',
         # Y 轴范围由渲染器数据锚定缺省接管（tools_report 2026-10-09 起）：
         # 手写常量 (1e-2, 1e1) 曾错锚量级截断低 q 数据（lessons E 类 10-09）
         'xlabel': 'q (nm$^{-1}$)', 'ylabel': 'I (a.u.)',
         'title': '{plot_title}, {title}', 'model_label': '{fit_label}'},
        {'kind': 'residual', 'at': (1, 0), 'xscale': 'log',
         'xlabel': 'q (nm$^{-1}$)', 'ylabel': '(fit - data)/data',
         'title': '{residual_title}'},
    ],
}


def _dictify(p, names):
    return p if isinstance(p, dict) else dict(zip(names, p))


# ---- 分量线 calc（q, p, fixed 顺序；p 为 dict） ----
def _comp_sphere(q, p, fixed):
    pd = _dictify(p, _M.NAMES)
    amp = _M.sph_amp(q, pd['R'], pd['sigma_fuzzy'])
    return pd['A'] * amp * amp


def _comp_blob(q, p, fixed):
    pd = _dictify(p, _M.NAMES)
    return pd['A'] * _M.pb_term(q, pd['rel_amp'], pd['rg'],
                                pd['flory_nu'], pd['xi'])


def _comp_bg(q, p, fixed):
    pd = _dictify(p, _M.NAMES)
    return np.full_like(np.asarray(q, dtype=float), float(pd['B']))


_COMPONENTS = [
    {'label': 'sphere term A*amp^2', 'style': 'b:',
     'column': 'I_sphere', 'calc': _comp_sphere},
    {'label': 'blob term A*rel_amp*P_blob', 'style': 'g--',
     'column': 'I_blob', 'calc': _comp_blob},
    {'label': 'background B (constant)', 'style': 'k:',
     'column': 'I_background', 'calc': _comp_bg},
]


def _comp_sphere_S(q, p, fixed):
    pd = _dictify(p, _M.NAMES_SQ)
    amp = _M.sph_amp(q, pd['R'], pd['sigma_fuzzy'])
    return (pd['A'] * (amp * amp
                       + _M.pb_term(q, pd['rel_amp'], pd['rg'],
                                    pd['flory_nu'], pd['xi']))
            * _M.S_py(q, pd['r_hs'], pd['eta']))


def _comp_S_only(q, p, fixed):
    pd = _dictify(p, _M.NAMES_SQ)
    return _M.S_py(q, pd['r_hs'], pd['eta'])


_COMPONENTS_SQ = _COMPONENTS + [
    {'label': 'form factor x S_PY (no B)', 'style': 'm:',
     'column': 'I_formfactor_S', 'calc': _comp_sphere_S},
    {'label': 'S_PY(q) (right axis shape)', 'style': 'r:',
     'column': 'I_S_PY', 'calc': _comp_S_only},
]

_POROD_ANCHOR = ('μ = 1/ν − 1 为 blob 散射指数（高 q 尾 I_blob ∝ q^(−1/ν)，'
                 'ν=0.4 → q^−2.5）；质量分形维数 D_f = 1/ν（ν=0.4 → D_f=2.5，'
                 '良溶剂链溶胀网）')

_SMOKED_KEY_PARAMS = ('R = 半密度半径（SLD 降为一半处）；σ = 界面模糊宽度'
                      '（高斯阻尼 exp(−(qσ)²/2)）；rel_amp = blob 项相对幅度'
                      '（无量纲）；Rg = blob 链回转半径；ν = Flory 指数'
                      '（站内界 [1/3, 0.7]）；ξ = Ornstein-Zernike 关联长度')

# ---- 报告文本钩子（main） ----
_METHOD_MAIN = ('## 3. 分析方法说明', '',
                '**模型（fuzzy sphere extended，Rathgeber blob 形式）**：'
                'I(q) = A·[amp²(q) + rel_amp·P_blob(q)] + B，其中',
                '- amp(q) = [3j₁(qR)/(qR)]·exp(−(qσ)²/2)：模糊球振幅'
                '（R 半密度半径，σ 界面模糊宽度）',
                '- P_blob(q) = sin(μ·arctan(q*)) / [μ·q*·(1+q*²)^(μ/2)]，'
                'μ = 1/ν − 1，q* = q·ξ / [erf(q·Rg/√6)]³：Rathgeber blob 散射'
                '（OZ 关联涨落项；q* 结构与 Beaucage 统一模型同族）',
                '- 实现来源：作者亲笔 SASview 插件 fuzzy_sphere_extended.py/.c '
                '逐式转写（V-scaling 形式，非 Stieger fuzzy_sphere_fluct 的 '
                'V² + 1/cm i_fluc）', '',
                '**参数语义**（8 自由参数）：A 幅度（a.u. 归一量，吸收 SASview '
                '管线的 scale·Δρ²·V·1e-4 全链）；' + _SMOKED_KEY_PARAMS +
                '；B 常数背景', '',
                '**拟合协议**：',
                '- 引擎：scipy `least_squares`（trf，x_scale=jac）；σ 列存在时加权拟合',
                '- 撞界判据：双侧界 |v−bound| < 10⁻³·量程；跨量级界（量程>10³·|v|）'
                '改用 5% 参数尺度相对距离；单侧界用 10⁻³·|v|；撞界参数自动降级为**条件值**', '')


def _results_main(der, chi2_dof, dof, at, nopeg_compare):
    return [
        '**派生量**（误差为拟合误差数值传播）：',
        '- 模糊球几何：R = **%.2f ± %.2f nm**，σ = **%.2f ± %.2f nm**，'
        '2σ = **%.2f ± %.2f nm**，外半径 R_SANS = R+2σ = %.2f ± %.2f nm'
        % (der['R_nm'], der['R_nm_err'], der['two_sigma_nm'] / 2.0,
           der['two_sigma_nm_err'] / 2.0, der['two_sigma_nm'],
           der['two_sigma_nm_err'], der['R_SANS_nm'], der['R_SANS_nm_err']),
        '- blob 网络：rel_amp = %.4g ± %.4g（无量纲），Rg = %.2f ± %.2f nm，'
        'ν = %.3f ± %.3f，μ = %.3f，D_f = 1/ν = %.2f ± %.2f'
        % (der['rel_amp'], der['rel_amp_err'], der['rg_nm'], der['rg_nm_err'],
           der['flory_nu'], der['flory_nu_err'], der['mu'], der['D_f'],
           der['D_f_err']),
        '- OZ 关联长度 ξ = **%.2f ± %.2f nm**；球形式因子第一极小 '
        'q_min1 = 4.4934/R ≈ %.3f nm⁻¹'
        % (der['xi_nm'], der['xi_nm_err'], der['q_first_min_nm1']),
        '', '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
        % (chi2_dof, dof, '、'.join(at) if at else '无'), '']


def _conclusions_main(params, perr, chi2_dof, dof, at, der, nopeg_compare,
                      sample_note):
    _at = at or []
    struct_at = [x for x in _at if x in ('R', 'sigma_fuzzy', 'rg', 'xi',
                                         'flory_nu')]
    if struct_at:
        c1 = ('1. **结构尺度**：模糊球参数存在撞界（%s），撞界项仅作**条件值**'
              '呈现，不作尺度结论（χ²/dof=%.3f；数值见参数表）。'
              % ('、'.join(struct_at), chi2_dof))
    else:
        c1 = ('1. **结构尺度**（χ²/dof=%.3f）：核半径 R = %.1f nm、界面模糊 '
              'σ = %.1f nm（2σ = %.1f nm）、OZ 关联长度 ξ = %.1f nm（误差见参数表）。'
              % (chi2_dof, der['R_nm'], der['two_sigma_nm'] / 2.0,
                 der['two_sigma_nm'], der['xi_nm']))
    if 'flory_nu' in _at or 'rel_amp' in _at:
        c2 = ('2. **形态判读**：blob 项参数触界/简并（%s），分形维数判读降级为'
              '**条件值**，仅以拟合曲线+残差佐证。'
              % ('、'.join([x for x in ('flory_nu', 'rel_amp') if x in _at])))
    else:
        c2 = ('2. **形态判读**：ν = %.2f → 质量分形维数 D_f = 1/ν = %.2f；'
              'blob 散射高 q 尾 I ∝ q^(−1/ν)（%s）。'
              % (der['flory_nu'], der['D_f'], _POROD_ANCHOR))
    return [
        c1, c2,
        '3. **可靠性分级**：A/B 为归一量（a.u. 标度）不作跨样品/跨标度直接比较；'
        'R/σ/Rg/ξ/ν 为标度无关结构参数，物理可引用性以 multistart/诊断证据链为准；'
        'rel_amp 为无量纲幅度比（标度内可比）；撞界参数（%s）仅作条件值呈现。'
        % ('、'.join(_at) if _at else '无'),
        '4. **方法学披露**：Rathgeber V-scaling 形式（作者源码逐式转写）；'
        'polydispersity（作者 radius.width=0.4 nm，PD≈2.2%）未纳入，单分散近似；'
        'flory_nu 站内下界 1/3（作者 .py 为 0.25——μ>2 时 blob 高 q 尾产生负强度，'
        'ν≥1/3 保证恒正）；幅度链 A 吸收 SASview scale·Δρ²·V·1e-4 全链'
        '（a.u. 标度，跨数据不可比）。', '']


# ---- 报告文本钩子（sq） ----
_METHOD_SQ = ('## 3. 分析方法说明', '',
              '**模型（fuzzy sphere extended × PY 硬球 S(Q)）**：'
              'I(q) = A·[amp²(q) + rel_amp·P_blob(q)]·S_PY(q; r_hs, η) + B，其中',
              '- 形式因子部分与 main 模式同款（Rathgeber blob 形式，作者 SASview '
              '插件逐式转写）',
              '- S_PY：**Percus-Yevick 硬球结构因子**（Kotlarchyk & Chen 1983 '
              '闭合式，公式逐式移植自 sasmodels 1.1.0 hardsphere.c，含小 X 泰勒分支）',
              '- 适用场景：低 q 粒子间硬球关联（作者 SI Chapter 6：20 °C 曲线 '
              'S(Q) 影响微弱但纳入后拟合质量改善；**Hayter MSA 不适用**——作者原文'
              '结论，zeta 电位佐证无静电贡献）', '',
              '**参数语义**（10 自由参数）：main 全部（A/R/σ/rel_amp/Rg/ν/ξ/B）+ '
              'r_hs = 硬球半径（作者口径 radius.effective）、η = 硬球体积分数；'
              'S(Q) 以乘积近似 P·S 作用于形式因子', '',
              '**拟合协议**：',
              '- 引擎：scipy `least_squares`（trf，x_scale=jac）；σ 列存在时加权拟合',
              '- 撞界判据：双侧界 |v−bound| < 10⁻³·量程；跨量级界（量程>10³·|v|）'
              '改用 5% 参数尺度相对距离；单侧界用 10⁻³·|v|；撞界参数自动降级为**条件值**'
              '；η 上界 0.50（PY 闭合精度，披露）', '')


def _results_sq(der, chi2_dof, dof, at, nopeg_compare):
    lines = _results_main(der, chi2_dof, dof, at, nopeg_compare)
    _at = at or []
    tag = ('（%s 撞界 → 同为**条件值**）'
           % '、'.join([k for k in ('r_hs', 'eta') if k in _at])
           ) if any(k in _at for k in ('r_hs', 'eta')) else ''
    lines.insert(-2, '- 硬球关联：r_hs = **%.2f ± %.2f nm**，η = **%.3f ± %.3f**%s；'
                     'S(0) 外推 = %.3f，PY 结构峰位（网格）q_peak ≈ %.2f nm⁻¹'
                     % (der['r_hs_nm'], der['r_hs_nm_err'], der['eta'],
                        der['eta_err'], tag, der['S0'], der['q_peak_nm1']))
    return lines


def _conclusions_sq(params, perr, chi2_dof, dof, at, der, nopeg_compare,
                    sample_note):
    _at = at or []
    out = list(_conclusions_main(params, perr, chi2_dof, dof, at, der,
                                 nopeg_compare, sample_note))
    if 'r_hs' in _at or 'eta' in _at:
        csq = ('5. **粒子间关联**：硬球参数 r_hs/eta 触界（%s）→ 结构因子不可辨识，'
               '结构峰结论以诊断证据链为准（见样品备注）。'
               % ('、'.join([k for k in ('r_hs', 'eta') if k in _at])))
    else:
        csq = ('5. **粒子间关联**：硬球半径 r_hs = %.1f nm、η = %.3f'
               '（SI 锚 32.81 nm / 0.1），PY 结构峰位 q_peak ≈ %.2f nm⁻¹'
               '（其稳健性以诊断证据链为准；S(Q) 影响弱时 η 半稳健）。'
               % (der['r_hs_nm'], der['eta'], der['q_peak_nm1']))
    out.insert(-1, csq)
    return out


def _cfg(main_or_sq):
    is_sq = main_or_sq == 'sq'
    names = _M.NAMES_SQ if is_sq else _M.NAMES
    return {
        'names': names,
        'bounds': _M.BOUNDS_SQ if is_sq else _M.BOUNDS,
        'init': _M.INIT_SQ if is_sq else _M.INIT,
        'init_qmax': 0.09,                # A0 估计的低 q 窗上界（nm⁻¹；低 q Guinier 平台区，
                                          # 须小于数据膝位 √3/R≈0.10；数据 q_min=0.026 < 0.06）
                                          # 修订史：0.06→0.09（2026-10-22，相分离 1.dat 数据
                                          # q_min=0.0756 > 0.06 估计带倒空报错；0.09 仍 < 锚膝位
                                          # √3/R=0.097，锚回归对拍验证见当日 runs 记录）
        'fixed_keys': [],
        'normalized_params': ['A', 'B'],
        'calc': (lambda p, q, fixed: _M.model_sq_vec(p, q) if is_sq
                 else _M.model_vec(p, q)),
        'derived': (lambda p, perr, fixed: _M.derived_sq(p, perr) if is_sq
                    else _M.derived_main(p, perr)),
        'plot_layout': _LAYOUT_DOUBLE,
        'components': _COMPONENTS_SQ if is_sq else _COMPONENTS,
        'report': {
            'method': (lambda fixed: list(_METHOD_SQ) if is_sq
                       else list(_METHOD_MAIN)),
            'results_md': _results_sq if is_sq else _results_main,
            'conclusions': _conclusions_sq if is_sq else _conclusions_main,
        },
    }


def register(M):
    global _M
    _M = M
    return {'main': _cfg('main'), 'sq': _cfg('sq')}

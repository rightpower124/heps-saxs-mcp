# -*- coding: utf-8 -*-
"""hardcarbon_closedpore 适配层（M3-261025 立项）：把 model.py 物理函数接入通用引擎 saxs_fit。
方法学源头：Stevens & Dahn, J. Electrochem. Soc. 147 (12) 4428-4431 (2000) Eq.1/2；
应用文献：Zheng et al., Adv. Energy Mater. 2024, 14, 2303064 SI Fig.S10（main 模式公式）。

模式谱系：main = AEM SI 简化式（5 参数）；full = Stevens Eq.2 双 Debye 完整式（7 参数）；
main ≡ full 固定 Bmp=0 的嵌套特例（跨模式可用嵌套 F 检验，ssr=chi2_dof×dof 换算纪律）。
"""
import numpy as np

# ---- 布局声明：两联图（拟合曲线 log 图 + 相对残差） ----
_LAYOUT_DOUBLE = {
    'figsize': (10, 8),
    'grid': (2, 1),
    'panels': [
        {'kind': 'fit', 'at': (0, 0), 'yscale': 'log',
         'xlabel': 'q (nm$^{-1}$)', 'ylabel': 'I (a.u.)',
         'title': '{plot_title}, {title}', 'model_label': '{fit_label}'},
        {'kind': 'residual', 'at': (1, 0),
         'xlabel': 'q (nm$^{-1}$)', 'ylabel': '(fit - data)/data',
         'title': '{residual_title}'},
    ],
}


def _fmt_at(names_at):
    return '、'.join(names_at) if names_at else '无'


def _f(x):
    """Debye-Bueche 形状函数 x⁴/(1+x²)²（components 纯项声明用，与 model._f 同式）"""
    x2 = x * x
    return x2 * x2 / (1.0 + x2) ** 2


def _conclusions(params, perr, chi2_dof, dof, at, der, nopeg_compare, sample_note):
    """结论文本钩子：撞界参数不得单独进入结论句（质量门纪律）"""
    _at = at or []
    if 'a1' in _at:
        c1 = ('1. **闭孔尺度**：D_pore = %.2f nm、Rg = %.2f nm、q_knee = %.2f nm⁻¹'
              '（χ²/dof=%.3f）；a1 撞界 → 全部闭孔尺度导出量为**条件值**，不作物理结论。'
              % (der['D_pore_nm'], der['Rg_nm'], der['q_knee_nm'], chi2_dof))
    else:
        c1 = ('1. **闭孔尺度**：等效孔径 D_pore = %.2f ± %.2f nm（直径口径，= 2·a1·√10）、'
              '回转半径 Rg = %.2f ± %.2f nm、膝位 q_knee = %.2f ± %.2f nm⁻¹'
              '（χ²/dof=%.3f）。'
              % (der['D_pore_nm'], der['D_pore_nm_err'], der['Rg_nm'], der['Rg_nm_err'],
                 der['q_knee_nm'], der['q_knee_nm_err'], chi2_dof))
    return [
        c1,
        '2. **膝位落窗核查（必读）**：闭孔尺度量的物理可引用性条件于 q_knee 落在拟合窗内'
        '（q_knee ∈ [qmin×1.5, qmax/1.5]）；膝在窗外时 a1 沿幅度组合简并谷漂移，'
        '仅幂律指数 a 稳健——以诊断工具（multistart/param_scan）证据链为准。',
        '3. **可靠性分级**：A/Bp/B 为幅度量（归一量：同批数据强度标度内可比、'
        '跨标度不可直接比较）；a1 及全部导出尺度量为形状参数，可引用性以证据链为准；'
        '撞界参数（%s）仅作条件值呈现。' % _fmt_at(_at),
        '4. **方法学披露**：幂律指数 a 自由拟合（Stevens 2000 原文的 3.3 为狭缝准直'
        'Smearing 钉值，本站针孔准直不适用）；D_pore 为**直径**口径（AEM SI 记号 "R"'
        '实为直径，跨文献比对注意 2 倍口径差）；a 为自由指数时 A 与 Bp 在 a→4 处存在'
        '共线风险，以 F 检验与诊断工具裁决。', '']


def _method_main(fixed):
    """main 模式方法学（五段式；全部公式集中于模型段 LaTeX 完整展示，逐条一行分条展示）"""
    return [
        '## 3. 分析方法说明（模型与算法）', '',
        '**模型段**（本模型全部公式集中于此，LaTeX 完整展示）：', '',
        'main 模式模型式（AEM SI Fig.S10 简化式，5 参数；幂律项 + Debye-Bueche 闭孔项 + 常数背景）：',
        '$$I(q) = A\\,q^{-a} + B_p\\,q^{-4}\\,\\frac{(a_1 q)^4}{\\left[1+(a_1 q)^2\\right]^2} + B$$',
        '',
        '同族 full 模式完整式（Stevens 2000 Eq.2，双 Debye 项，7 参数；main ≡ full 固定 $B_{mp}=0$ 的嵌套特例）：',
        '$$I(q) = A\\,q^{-a} + B_p\\,q^{-4}\\,\\frac{(a_1 q)^4}{\\left[1+(a_1 q)^2\\right]^2} + B_{mp}\\,q^{-4}\\,\\frac{(a_2 q)^4}{\\left[1+(a_2 q)^2\\right]^2} + B$$',
        '',
        '与文献字面式的参数映射（Porod 口径改写，逐项恒等）：',
        '$$B_p \\equiv B_{\\mathrm{SI}}\\;(\\mathrm{AEM\\ SI}),\\qquad B_p = \\Delta\\rho\\,B\\,a_1^2,\\qquad B_{mp} = C\\,a_2^2\\;(\\mathrm{Stevens\\ Eq.2})$$',
        '',
        '派生量公式（闭孔尺度，误差为 $a_1$ 误差解析传播）：',
        '$$R_g = a_1\\sqrt{6}$$',
        '',
        '$$R_{\\mathrm{sphere}} = a_1\\sqrt{10}$$',
        '',
        '$$D_{\\mathrm{pore}} = 2 a_1 \\sqrt{10}$$',
        '',
        '$$q_{\\mathrm{knee}} = 1/a_1$$',
        '',
        '幂律指数说明：Stevens 2000 原文钉值 3.3 为狭缝准直 Smearing（无孔石墨标样定标）；本站针孔准直无 Smearing，$a$ 自由拟合。',
        '',
        '**各项物理语义段**：',
        '',
        '| 参数 | 语义 | 单位 |',
        '|---|---|---|',
        '| $A$ | 幂律项幅度（颗粒/大结构散射） | a.u. |',
        '| $a$ | 幂律指数（自由；Porod 极限 4） | — |',
        '| $B_p$ | 闭孔项 Porod 口径幅度（高 q 渐近 $B_p q^{-4}$） | a.u. |',
        '| $a_1$ | Debye 特征长度（电子密度涨落关联长度） | nm |',
        '| $B$ | 常数背景 | a.u. |',
        '| $B_{mp}$ | （full 谱系）中孔项 Porod 口径幅度 | a.u. |',
        '| $a_2$ | （full 谱系）中孔 Debye 特征长度，$a_2 > a_1$ | nm |',
        '',
        '**派生量段**（物理语义与口径说明；公式见模型段）：',
        '- $R_g$：闭孔 Debye 关联长度的回转半径口径（Porod 定义，Stevens 2000 引 Glatter & Kratky 1982）。',
        '- $R_{\\mathrm{sphere}}$：等效球形孔半径（由 $R_g = R\\sqrt{0.6}$ 反解）。',
        '- $D_{\\mathrm{pore}}$：等效球形孔**直径**；AEM SI 公式 "$R = 2 a_1 \\sqrt{10}$" 的记号 "R" 实为直径，跨文献比对注意 2 倍口径差。',
        '- $q_{\\mathrm{knee}}$：Debye 膝特征波数；落窗核查（可辨识性前提）见结论。',
        '',
        '**拟合协议段**：',
        '- 引擎：scipy `least_squares`（trf，x_scale=jac）；σ 列存在时加权拟合。',
        '- 初值：A0 由低 q 估计带自动标定（init_qmax=1.2 nm⁻¹）；B0 取高 q 末端均值。',
        '- a1 初值 0.70 nm（文献锚 HCMP 0.604 / HCMP-CO2 0.769 nm 折中）。',
        '- 撞界判据：双侧界 |v−bound| < 10⁻³·量程；跨量级界改用 5% 参数尺度相对距离；撞界参数自动降级为**条件值**。',
        '- 诊断工具：multistart（A/Bp 简并排查）、param_scan（a1 网格，膝位辨识性取证）、嵌套 F 检验（full vs main）。',
        '',
        '**备注段**：',
        '- 方法学源头：Stevens & Dahn, J. Electrochem. Soc. 147 (12) 4428-4431 (2000)；Debye, Anderson & Brumberger, J. Appl. Phys. 28 (1957) 679。',
        '- 应用文献：Zheng et al., Adv. Energy Mater. 2024, 14, 2303064 SI Fig.S10（main 模式公式与其一致，Porod 口径改写逐项恒等）。',
        '- 空孔对比度固定（无 SLD 自由参数），无 SLD 族简并。',
        '- 对比度随充钠变化的原位 Δρ 分离（Stevens 2000 招牌能力）需原位数据集，本批静态数据不适用。',
        '- 模型假设见报告 §2（refs.assumptions）。', '']


def _results_md(der, chi2_dof, dof, at, nopeg_compare):
    return [
        '**派生量**（闭孔尺度；误差为 a1 误差解析传播）：',
        '- D_pore = **%.3f ± %.3f nm**（直径口径）；R_sphere = %.3f ± %.3f nm；'
        'Rg = %.3f ± %.3f nm'
        % (der['D_pore_nm'], der['D_pore_nm_err'],
           der['R_sphere_nm'], der['R_sphere_nm_err'],
           der['Rg_nm'], der['Rg_nm_err']),
        '- q_knee = **%.3f ± %.3f nm⁻¹**（膝位；落窗核查见结论）'
        % (der['q_knee_nm'], der['q_knee_nm_err']),
        '', '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
        % (chi2_dof, dof, _fmt_at(at or [])), '']


def _method_full(fixed):
    """full 模式方法学（独立五段式：完整式主公式一次展示，main 作嵌套特例文字指向）"""
    return [
        '## 3. 分析方法说明（模型与算法）', '',
        '**模型段**：full 模式 = 幂律项 + 双 Debye-Bueche 孔项（纳米孔 + 中孔）+ 常数背景'
        '（Stevens & Dahn 2000 Eq.2 完整式）：',
        '$$I(q) = A\\,q^{-a} + B_p\\,q^{-4}\\,\\frac{(a_1 q)^4}{\\left[1+(a_1 q)^2\\right]^2}'
        ' + B_{mp}\\,q^{-4}\\,\\frac{(a_2 q)^4}{\\left[1+(a_2 q)^2\\right]^2} + B$$',
        'main 模式（AEM SI 简化式，5 参数）= 本式固定 $B_{mp}=0$ 的嵌套特例，其公式与'
        '文献映射详见该模式报告；与 Stevens 字面式的参数映射：',
        '$$B_p = \\Delta\\rho\\,B\\,a_1^2,\\qquad B_{mp} = C\\,a_2^2$$',
        '原文幂律指数 3.3 为狭缝准直 Smearing 钉值（无孔石墨标样定标）；本站针孔准直，'
        '$a$ 自由拟合。Stevens 的 $a_2$（中孔）$> a_1$（纳米孔），中孔膝位 $1/a_2$ 在更低 q。', '',
        '**各项物理语义**：',
        '',
        '| 参数 | 语义 | 单位 |',
        '|---|---|---|',
        '| $A$ | 幂律项幅度（颗粒/大结构散射） | a.u. |',
        '| $a$ | 幂律指数（自由；Porod 极限 4） | — |',
        '| $B_p$ | 纳米孔项 Porod 口径幅度（高 q 渐近 $B_p q^{-4}$） | a.u. |',
        '| $a_1$ | 纳米孔 Debye 特征长度 | nm |',
        '| $B_{mp}$ | 中孔项 Porod 口径幅度 | a.u. |',
        '| $a_2$ | 中孔 Debye 特征长度（$a_2 > a_1$） | nm |',
        '| $B$ | 常数背景 | a.u. |', '',
        '**派生量段**（双孔尺度，误差为 $a_1/a_2$ 误差解析传播）：',
        '$$R_g = a_i\\sqrt{6},\\qquad R_{\\mathrm{sphere}} = a_i\\sqrt{10},\\qquad'
        ' D_{\\mathrm{pore}} = 2 a_i \\sqrt{10},\\qquad q_{\\mathrm{knee}} = 1/a_i'
        '\\quad (i=1,2)$$',
        '$D_{\\mathrm{pore}}$ 为**直径**口径（Stevens/AEM SI 记号 "R = 2a√10" 实为直径）。', '',
        '**拟合协议段**：',
        '- 引擎：scipy `least_squares`（trf，x_scale=jac）；σ 列存在时加权拟合',
        '- 初值：A0 低 q 自动标定（init_qmax=1.2 nm⁻¹）；B0 高 q 末端均值；'
        'a1=0.70 nm（文献锚折中）、a2=1.0 nm（Stevens 钉值 10 Å）',
        '- 撞界判据：双侧界 |v−bound| < 10⁻³·量程；跨量级界改用 5% 参数尺度相对距离；'
        '撞界参数自动降级为**条件值**',
        '- 诊断工具：multistart（双幅度简并排查）、param_scan（a1/a2 网格）、'
        '嵌套 F 检验（full vs main，ssr=chi2_dof×dof 换算）', '',
        '**备注段**：',
        '- 方法学源头：Stevens & Dahn, J. Electrochem. Soc. 147 (12) 4428-4431 (2000)；'
        'Debye, Anderson & Brumberger, J. Appl. Phys. 28 (1957) 679',
        '- **两膝间距前提**：双 Debye 项可辨识性条件于两膝位间距 ≥2.5 倍；间距不足时'
        '按简并纪律回退 main 模式（本批 GXL 数据预期不满足，full 供未来数据集）',
        '- Δρ（电子密度对比度）按 Porod 口径吸收进 Bp，静态数据不可分离；'
        '原位对比度实验（Stevens 2000 招牌能力）需原位数据集',
        '- 模型假设见报告 §2（refs.assumptions）', '']


def _conclusions_full(params, perr, chi2_dof, dof, at, der, nopeg_compare, sample_note):
    _at = at or []
    lines = [
        '1. **双孔尺度**：纳米孔 D_pore = %.2f ± %.2f nm（q_knee = %.2f nm⁻¹）、'
        '中孔 D_pore = %.2f ± %.2f nm（q_knee = %.2f nm⁻¹）（χ²/dof=%.3f）；'
        '撞界参数（%s）仅作条件值。'
        % (der['D_pore_nano_nm'], der['D_pore_nano_nm_err'], der['q_knee_nano_nm'],
           der['D_pore_meso_nm'], der['D_pore_meso_nm_err'], der['q_knee_meso_nm'],
           chi2_dof, _fmt_at(_at)),
        '2. **两膝间距核查**：a1/a2 双 Debye 项可辨识性条件于两膝位间距 ≥2.5 倍'
        '（q_knee_meso / q_knee_nano）；间距不足时两幅度/长度沿简并谷漂移，'
        '按 lessons 10-06 纪律回退 main 模式。',
        '3. **幅度量分级**：A/Bp/Bmp/B 为归一量（同批标度内可比）；Δρ（电子密度对比度）'
        '已按 Porod 口径吸收进 Bp（Bp = Δρ·B·a1²），静态数据不可分离——原位对比度'
        '实验（Stevens 2000 招牌能力）需原位数据集。', '']
    return lines


def register(M):
    main = {
        'names': M.NAMES,
        'bounds': M.BOUNDS,
        'init': M.INIT,
        'init_qmax': M.INIT_QMAX,
        'fixed_keys': [],
        'normalized_params': ['A', 'Bp', 'B'],
        'calc': lambda p, q, fixed: M.model_vec(p, q),
        'derived': lambda p, perr, fixed: M.derived_main(p, perr),
        'plot_layout': _LAYOUT_DOUBLE,
        # components 纯项契约（2026-10-27 用户裁决）：子曲线不含常数背景 B，
        # B 由拟合曲线.csv 的 I_background 独立列承载；
        # components_sum_to_fit=True 接受生成器加和硬门（I_fit ≡ Σ分量 + B）
        'components_sum_to_fit': True,
        'components': [
            {'label': 'power law: A*q^-a', 'style': 'b:',
             'column': 'I_power',
             'calc': lambda q, p, fixed: p['A'] * q ** (-p['a'])},
            {'label': 'closed pores: Bp*q^-4*F(a1*q)', 'style': 'g--',
             'column': 'I_closedpore',
             'calc': lambda q, p, fixed: p['Bp'] * q ** (-4.0) * _f(p['a1'] * q)},
        ],
        'report': {
            'method': _method_main,
            'results_md': _results_md,
            'conclusions': _conclusions,
        },
    }
    full = {
        'names': M.FULL_NAMES,
        'bounds': M.FULL_BOUNDS,
        'init': M.FULL_INIT,
        'init_qmax': M.INIT_QMAX,
        'fixed_keys': [],
        'normalized_params': ['A', 'Bp', 'Bmp', 'B'],
        'calc': lambda p, q, fixed: M.model_vec_full(p, q),
        'derived': lambda p, perr, fixed: M.derived_full(p, perr),
        'plot_layout': _LAYOUT_DOUBLE,
        # components 纯项契约（2026-10-27 用户裁决，同 main 模式）
        'components_sum_to_fit': True,
        'components': [
            {'label': 'power law: A*q^-a', 'style': 'b:',
             'column': 'I_power',
             'calc': lambda q, p, fixed: p['A'] * q ** (-p['a'])},
            {'label': 'nanopores: Bp*q^-4*F(a1*q)', 'style': 'g--',
             'column': 'I_nanopore',
             'calc': lambda q, p, fixed: p['Bp'] * q ** (-4.0) * _f(p['a1'] * q)},
            {'label': 'mesopores: Bmp*q^-4*F(a2*q)', 'style': 'm-.',
             'column': 'I_mesopore',
             'calc': lambda q, p, fixed: p['Bmp'] * q ** (-4.0) * _f(p['a2'] * q)},
        ],
        'report': {
            'method': _method_full,
            'results_md': lambda der, chi2_dof, dof, at, nopeg_compare: [
                '**派生量**（双孔尺度；误差为 a1/a2 误差解析传播）：',
                '- 纳米孔：D_pore = **%.3f ± %.3f nm**；q_knee = %.3f ± %.3f nm⁻¹'
                % (der['D_pore_nano_nm'], der['D_pore_nano_nm_err'],
                   der['q_knee_nano_nm'], der['q_knee_nano_nm_err']),
                '- 中孔：D_pore = **%.3f ± %.3f nm**；q_knee = %.3f ± %.3f nm⁻¹'
                % (der['D_pore_meso_nm'], der['D_pore_meso_nm_err'],
                   der['q_knee_meso_nm'], der['q_knee_meso_nm_err']),
                '', '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
                % (chi2_dof, dof, _fmt_at(at or [])), ''],
            'conclusions': _conclusions_full,
        },
    }
    return {'main': main, 'full': full}

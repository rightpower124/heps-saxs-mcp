# -*- coding: utf-8 -*-
"""gel_network 适配层（M3-261002）：把 model.py 物理函数接入通用引擎 saxs_fit。
model.py 物理代码独立成文件；本文件只声明"参数表 + 引用方式 + 报告文本钩子"，
与 manifest.json 的元数据（labels / refs / modes）一起构成完整模型插件。

方法学来源：Wisotzki et al., Phys. Chem. Chem. Phys. 19 (2017) 12064-12074，
mode main = 文献 Eq.(1) correlation length model；mode porod = 文献 Eq.(3) 幂律。
样品体系中性措辞（用户未指定 pam/cncs 成分），物理语义按文献锚点披露。
"""
import numpy as np

# ---- 布局声明：两联图（拟合曲线 log 图 + 相对残差；无实空间剖面钩子） ----
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

_M_ANCHOR = ('m 锚点（文献 Wisotzki 2017）：1.67 良溶剂溶胀线团 / 2 theta 溶剂（OZ 极限）/'
             '2~3 支化化学凝胶（辐照系列实测 1.69→2.55）/ 趋向 1 半柔性棒状结构')
_DM_ANCHOR = ('dm 锚点（文献 Porod plot，中段 0.3–2 nm⁻¹）：1 线性棒状 / 1.67 良溶剂线团 / '
              '2~3 支化化学凝胶（辐照系列实测 1.43→2.31，物理凝胶系列 1.35→1.1）')


def _conclusions(params, perr, chi2_dof, dof, at, der, nopeg_compare, sample_note):
    """main 模式结论文本钩子：撞界参数不得单独进入结论句（质量门纪律）"""
    _at = at or []
    if 'xi' in _at:
        c1 = ('1. **网络尺度**：相关长度 ξ = %.3g nm（χ²/dof=%.3f）为**撞界条件值**，'
              'mesh 派生量同不作物理结论（可辨识性诊断见样品备注）。'
              % (float(params['xi']), chi2_dof))
    else:
        c1 = ('1. **网络尺度**：相关长度 ξ = %.2f ± %.2f nm，双幂律 crossover 位置 '
              'qc = 1/ξ = %.3f ± %.3f nm⁻¹，**网目尺寸 mesh = 2π/qc = %.1f ± %.1f nm**（χ²/dof=%.3f）。'
              % (float(params['xi']), float(perr['xi']), der['qc_nm1'], der['qc_nm1_err'],
                 der['mesh_nm'], der['mesh_nm_err'], chi2_dof))
    if 'm' in _at:
        c2 = ('2. **局部链形态**：Porod 指数 m = %.3g 为**撞界条件值**，不作形态结论'
              '（%s）。' % (float(params['m']), _M_ANCHOR))
    else:
        c2 = ('2. **局部链形态**：Porod 指数 m = %.2f ± %.2f（%s）。'
              % (float(params['m']), float(perr['m']), _M_ANCHOR))
    return [
        c1, c2,
        '3. **可靠性分级**：A/B 为归一量（a.u. 标度）不作跨样品/跨标度直接比较；'
        'ξ/m 为标度无关结构参数，其物理可引用性以可辨识性诊断为准（见样品备注）；'
        '撞界参数（%s）仅作条件值呈现。' % ('、'.join(_at) if _at else '无'),
        '4. **方法学披露**：eqn(1) 为经验模型，不含结构因子项（文献对 10 wt% 低 q misfit '
        '同款归因）；**数据低 q 无平台时 A 与 ξ^m 强共线**（仅组合 A/ξ^m 可辨），'
        'A 的单独值降级条件值、形态结论以拟合曲线+残差为准；B 为站内背景扩展项；'
        'mesh = 2πξ 口径经文献数值链验证（ξ=3.0→18.8 nm vs 文献 17.5±0.5）。', '']


def _conclusions_porod(params, perr, chi2_dof, dof, at, der, nopeg_compare, sample_note):
    """porod 模式结论文本钩子"""
    _at = at or []
    if 'dm' in _at:
        c1 = ('1. **质量分形维数**：dm = %.3g（χ²/dof=%.3f）为**撞界条件值**，不作形态结论'
              '（%s）。' % (float(params['dm']), chi2_dof, _DM_ANCHOR))
    else:
        c1 = ('1. **质量分形维数**：dm = %.2f ± %.2f（%s；χ²/dof=%.3f）。'
              % (float(params['dm']), float(perr['dm']), _DM_ANCHOR, chi2_dof))
    return [
        c1,
        '2. **可靠性分级**：A 为归一量（a.u. 标度）；dm 为标度无关形状参数，'
        '其窗口位置依赖性以诊断为准；撞界参数（%s）仅作条件值呈现。'
        % ('、'.join(_at) if _at else '无'),
        '3. **方法学披露**：porod 模式 = 文献 Eq.(3) 局部幂律（推荐中段窗 0.3–2 nm⁻¹），'
        '描述中段自相似区间的质量分形维数，与 main 模式全 q 区 Porod 指数 m 互补'
        '（m 为含转折的全局等效指数，dm 为窗口内局部指数，两者数值不必相等）；'
        '纯幂律无背景项（文献同款），窗口内应远离噪声底。', '']


def register(M):
    main = {
        'names': M.NAMES,
        'bounds': M.BOUNDS,
        'init': M.INIT,
        'init_qmax': 0.15,                # A0 估计的低 q 窗口上界（nm⁻¹）
        'fixed_keys': [],                 # 无跨样品固定参数
        'normalized_params': ['A', 'B'],  # 归一量：q=0 外推强度/背景（a.u. 标度）
        'calc': lambda p, q, fixed: M.model_vec(p, q),
        'derived': lambda p, perr, fixed: M.derived_main(p, perr),
        'plot_layout': _LAYOUT_DOUBLE,    # 两联图
        'components': [                   # 拟合曲线.csv 的附加分量线
            {'label': 'background B (constant)', 'style': 'b:',
             'column': 'I_background',
             'calc': lambda q, p, fixed: np.full_like(np.asarray(q, dtype=float), float(p['B']))},
            {'label': 'correlation term A/(1+(xi*q)^m) (no B)', 'style': 'g--',
             'column': 'I_correlation_term',
             'calc': lambda q, p, fixed: p['A'] / (1.0 + (p['xi'] * q) ** p['m'])},
        ],
        'report': {
            'method': lambda fixed: [
                '## 3. 分析方法说明', '',
                '**模型（correlation length model，Wisotzki 2017 Eq.1）**：'
                'I(q) = I(0) / (1 + (ξ·q)^m) + B，其中',
                '- I(0)：q→0 外推散射强度（参数名 A；q << 1/ξ 时为平台高度）',
                '- ξ：**相关长度**（correlation length）——网络中散射中心的平均关联距离',
                '- m：**Porod 指数**——' + _M_ANCHOR,
                '- B：常数背景（站内扩展：a.u. 强度数据以 B 吸收噪声平台；文献数据已扣背景）', '',
                '**派生量（mesh size，文献 "Mesh size" 节口径）**：',
                '- 双幂律 crossover / 肩部位置 qc = 1/ξ（I = I(0)/2 处）',
                '- **网目尺寸 mesh = 2π/qc = 2πξ**；口径经文献数值链验证：'
                'ξ=3.0 nm → mesh 18.8 nm ≈ 文献 17.5±0.5 nm；ξ=4.0 → 25.1 ≈ 26.0±0.5 nm', '',
                '**拟合协议**：',
                '- 引擎：scipy `least_squares`（trf，x_scale=jac）；无 σ 列时 unweighted',
                '- 撞界判据：|v−bound| < 10⁻³·量程；撞界参数自动降级为**条件值**', ''],
            'results_md': lambda der, chi2_dof, dof, at, nopeg_compare: [
                '**派生量**（网络尺度）：',
                '- 相关长度 ξ 见参数表；qc = 1/ξ = **%.3f ± %.3f nm⁻¹**'
                % (der['qc_nm1'], der['qc_nm1_err']),
                '- **网目尺寸 mesh = 2π/qc = %.1f ± %.1f nm**'
                % (der['mesh_nm'], der['mesh_nm_err']),
                '', '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
                % (chi2_dof, dof, '、'.join(at) if at else '无'), ''],
            'conclusions': _conclusions,
        },
    }
    porod = {
        'names': M.NAMES_POROD,
        'bounds': M.BOUNDS_POROD,
        'init': M.INIT_POROD,
        'init_qmax': 2.0,                 # A0 估计窗口上界（与推荐拟合窗同域）
        'fixed_keys': [],
        'normalized_params': ['A'],
        'calc': lambda p, q, fixed: M.model_porod_vec(p, q),
        'derived': lambda p, perr, fixed: M.derived_porod(p, perr),
        'plot_layout': _LAYOUT_DOUBLE,
        'components': [],                 # 纯幂律两参数，无附加分量线
        'report': {
            'method': lambda fixed: [
                '## 3. 分析方法说明', '',
                '**模型（Porod plot / 质量分形幂律，Wisotzki 2017 Eq.3）**：I(q) = A·q^(−dm)，其中',
                '- dm：**质量分形维数**（mass fractal dimension）——' + _DM_ANCHOR,
                '- A：幂律幅度（缩放因子，a.u. 标度归一量）',
                '- 适用：log–log 图上的局部线性区间（文献窗口 0.3–2 nm⁻¹）；'
                '与 main 模式（全 q 区 correlation length model 的等效指数 m）互补', '',
                '**拟合协议**：',
                '- 引擎：scipy `least_squares`（trf，x_scale=jac）；无 σ 列时 unweighted',
                '- 撞界判据：|v−bound| < 10⁻³·量程；撞界参数自动降级为**条件值**', ''],
            'results_md': lambda der, chi2_dof, dof, at, nopeg_compare: [
                '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
                % (chi2_dof, dof, '、'.join(at) if at else '无'),
                '', 'dm 的物理解释见结论；窗口位置依赖性建议用 saxs_diag_param_scan/多窗对照取证。', ''],
            'conclusions': _conclusions_porod,
        },
    }
    return {'main': main, 'porod': porod}

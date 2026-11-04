# -*- coding: utf-8 -*-
"""unified_power_Rg 适配层（M4-261006 复现锚定）：Beaucage 1995 统一四 level 级联模型，
忠实复现 Chaney 2025 Figure 7a/7b 的 "unified model with four levels"（小角区）。
本文件只声明"参数表 + 引用方式 + 报告文本钩子"；物理代码在 model.py。
单模式 main = 四 level 全参数统一拟合（13 自由参数 G/Rg/P×4 + B，无 A/scale——
幅度由 Gᵢ 承载，Kᵢ = Gᵢ·Pᵢ/Rgᵢ^Pᵢ·Γ(Pᵢ/2) 固定关系派生）。
"""
import numpy as np

# ---- 布局声明：两联图（拟合曲线 log-log 图 + 相对残差） ----
# 2026-10-06 渲染修订：统一模型跨窗两个量级（0.025–2.0 nm^-1），fit 面板必须
# log-x（线性 x 把低 q 全部膝区压成竖线）；ylim 下界锚定背景量级——分量线
# 全程延拓段不再把 log-y autoscale 撑到 10^-281（首版图目检教训：数值正确的
# 拟合在坏轴范围下无法目检，图像验收必须看轴范围）
# 2026-10-09 渲染修订：ylim 上界 300 → 5e4——Henderson Figure-6 数据
# I_max≈1e4，原 300 把 q<0.4 的数据点整段裁出画面（PIL 裁剪放大取证确认，
# 教训：图像验收必须核查数据点可见性，不能只看曲线存在）；5e4 对 Chaney
# （I_max≈16、Level 4 平台 ~112、背景 3.3e-5）仅多顶部留白，两数据集共存可用
_LAYOUT_DOUBLE = {
    'figsize': (10, 8),
    'grid': (2, 1),
    'panels': [
        {'kind': 'fit', 'at': (0, 0), 'xscale': 'log', 'yscale': 'log',
         'ylim': [3e-5, 5e4],
         'xlabel': 'q (nm$^{-1}$)', 'ylabel': 'I (a.u.)',
         'title': '{plot_title}, {title}', 'model_label': '{fit_label}'},
        {'kind': 'residual', 'at': (1, 0), 'xscale': 'log',
         'xlabel': 'q (nm$^{-1}$)', 'ylabel': '(fit - data)/data',
         'title': '{residual_title}'},
    ],
}

_POROD_ANCHOR = ('P 形态锚点：≈4 光滑致密界面 Porod 极限；≈3 致密三维对象表面散射'
                 '（Chaney 2025 level3 归因"非球形聚集体"）；<3 质量分形；'
                 '≈1 棒状/刚性链截面尺度对象')

_PAPER_TABLE = ('Level 4: Rg>1000 Å, P=2.9；Level 3: Rg=360 Å, P=3.8；'
                'Level 2: Rg=130 Å, P=3.1；Level 1: Rg=20 Å, P≈1'
                '（论文 Figure 7b 表值，读图级精度；正文近似表述 Rg≈380 Å 与表值 360 Å 同指 level3）')


_M = None    # register 时绑定的物理模块（model_loader 以独立模块名加载，顶层不可 import）


def _comp_level1(q, p, fixed):
    return _M.level_curve(q, p, 1)


def _comp_level2(q, p, fixed):
    return _M.level_curve(q, p, 2)


def _comp_level3(q, p, fixed):
    return _M.level_curve(q, p, 3)


def _comp_level4(q, p, fixed):
    return _M.level_curve(q, p, 4)


# 背底 component 已删（2026-10-27 契约对齐）：tools_report 对声明 'B' 参数的模型
# 自动追加 I_background 独立列，插件分量列须为**纯项**（不含 B）——本插件原在
# _COMPONENTS/_components_for 再声明一列 I_background 导致 CSV 双背底列
#（用户实测指出），删除后由引擎独立列唯一承载。
_COMPONENTS = [
    {'label': 'Level 4 (largest scale)', 'style': 'b:',
     'column': 'I_level4', 'calc': _comp_level4},
    {'label': 'Level 3', 'style': 'c--',
     'column': 'I_level3', 'calc': _comp_level3},
    {'label': 'Level 2', 'style': 'g--',
     'column': 'I_level2', 'calc': _comp_level2},
    {'label': 'Level 1 (smallest scale)', 'style': 'm:',
     'column': 'I_level1', 'calc': _comp_level1},
]


def _comp_peak_b(q, p, fixed):
    return _M.peak_curve(q, p, 'b')


def _comp_peak_n(q, p, fixed):
    return _M.peak_curve(q, p, 'n')


def _comp_peak_0(q, p, fixed):
    return _M.peak_curve(q, p, '0')


_PEAK_COMPONENTS = [
    {'label': '(100) broad', 'style': 'r--',
     'column': 'I_peak100b', 'calc': _comp_peak_b},
    {'label': '(100) narrow', 'style': 'y--',
     'column': 'I_peak100n', 'calc': _comp_peak_n},
    {'label': '(012)', 'style': 'r:',
     'column': 'I_peak012', 'calc': _comp_peak_0},
]


def _make_method():
    def method(fixed):
        return [
            '## 3. 分析方法说明', '',
            '**模型（Beaucage 1995 统一四 level 级联，与 sasmodels `unified_power_Rg` '
            'level=4 逐式一致）**——模型完整方程（LaTeX）：',
            '',
            '$$',
            r'I(q) \;=\; B \;+\; \sum_{i=1}^{4}\Bigl[\,G_i\,\exp\!\Bigl(-\frac{(q\,R_{g,i})^2}{3}\Bigr) \;+\; K_i\,\exp\!\Bigl(-\frac{(q\,R_{g,i-1})^2}{3}\Bigr)\,\bigl(q_i^{*}\bigr)^{-P_i}\,\Bigr]',
            '$$',
            '',
            '$$',
            r'q_i^{*} \;=\; \frac{q}{\Bigl[\operatorname{erf}\!\Bigl(\dfrac{q\,R_{g,i}}{\sqrt{6}}\Bigr)\Bigr]^{3}},\qquad\qquad K_i \;=\; \frac{G_i\,P_i}{R_{g,i}^{\,P_i}}\;\Gamma\!\Bigl(\frac{P_i}{2}\Bigr)',
            '$$',
            '',
            '- $q_i^{*}$：幂律尾开启因子（在本级 Guinier 膝之上平滑开启）；'
            '$K_i$：**固定关系内部派生，非自由参数**——Beaucage '
            '"unified" 本意（每级 Guinier/Porod 幅度由同一对象的尺寸+维度统一锁定；'
            'sasmodels 文档 Debye 例子同款关系）',
            '- $i=1$（最小尺度级）尾项无压制因子（$\exp(-(qR_{g,0})^2/3)\\equiv 1$）；'
            '$i=2,3,4$ 的压制因子取下一级（更小尺度）Guinier 因子，防止级间双计',
            '- 级联方向：Level 4 = 最大尺度（最低 q 平台）→ Level 1 = 最小尺度（最高 q '
            '幂律尾）', '',
            '**参数语义**（13 自由参数）：',
            '- Gᵢ：每级 Guinier 幅度（a.u. 归一量）；Kᵢ：每级 Porod 尾幅度（≡ sasmodels '
            '参数 B[i]，固定关系派生，避免与本模型常数背景 B 混淆而改名）',
            '- Rgᵢ：每级回转半径（nm；报告同时给出 Å 值直接对标文献 Figure 7b）',
            '- Pᵢ：' + _POROD_ANCHOR,
            '- scale 等效固定 1（幅度由 Gᵢ 直接承载，消除 scale-G 简并）；B：常数背景', '',
            '**复现锚点（Chaney et al., Chem. Mater. 37 (2025) 7882–7893, Figure 7a/7b）**：'
            '作者原文 "A unified model with four levels, denoted by colored shading, was '
            'used to fit the small-angle scattering region. Lorentz peaks were used to fit '
            'features in the wide-angle scattering region."（ref 65 = Beaucage 1995）；'
            '论文 Figure 7b 锚点：' + _PAPER_TABLE + '。', '',
            '**复现范围披露**：本模式只复现小角区统一模型四 level（数据窗上界 2 nm⁻¹ '
            '避开 WAXS 区）；论文的三个 Lorentz 峰（(100) narrow/broad 0.27/0.29 Å⁻¹、'
            '(012) 0.66 Å⁻¹）拟合超出本工具模型库能力范围，未纳入复现。', '',
            '**拟合协议**：',
            '- 引擎：scipy `least_squares`（trf，x_scale=jac）；σ 列存在时加权拟合',
            '- 目标函数（σ 加权残差）与优度（LaTeX）：',
            '',
            '$$',
            r'\frac{\chi^2}{\mathrm{dof}} \;=\; \frac{1}{N-m}\sum_{j=1}^{N}\left(\frac{I_{\mathrm{obs}}(q_j)-I_{\mathrm{calc}}(q_j)}{\sigma_j}\right)^{2}',
            '$$',
            '',
            '- 13 自由参数全局拟合，INIT 锚定论文 Figure 7b 值；建议 multistart 取证全局性',
            '- 撞界判据：双侧界 |v−bound| < 10⁻³·量程；跨量级界（量程>10³·|v|）改用 '
            '5% 参数尺度相对距离；单侧界用 10⁻³·|v|；撞界参数自动降级为**条件值**', '']
    return method


def _results_md(der, chi2_dof, dof, at, nopeg_compare):
    lines = ['**派生量**（四级 Rg，误差为拟合误差数值传播；Å 值直接对标文献 Figure 7b）：']
    for i in (4, 3, 2, 1):
        lines.append('- Level %d：Rg = **%.1f ± %.1f Å**（%.2f ± %.2f nm），'
                     'Guinier 特征弯折 q_knee = %.3g nm⁻¹'
                     % (i, der['rg%d_A' % i], der['rg%d_A_err' % i],
                        der['rg%d_nm' % i], der['rg%d_nm_err' % i],
                        der['q_knee%d_nm1' % i]))
    lines += ['', '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
              % (chi2_dof, dof, '、'.join(at) if at else '无'), '']
    return lines


def _conclusions(params, perr, chi2_dof, dof, at, der, nopeg_compare, sample_note):
    _at = at or []
    rg_at = [x for x in _at if x.startswith('Rg')]
    p_at = [x for x in _at if x.startswith('P')]
    if rg_at:
        c1 = ('1. **结构尺度**：Rg 存在撞界参数（%s）——撞界级仅作**条件值**呈现，'
              '不作尺度结论；未撞界级数值见参数表（χ²/dof=%.3f）。' % ('、'.join(rg_at), chi2_dof))
    else:
        c1 = ('1. **结构尺度**（χ²/dof=%.3f）：Level 4 Rg = %.3g Å、Level 3 Rg = '
              '%.3g Å、Level 2 Rg = %.3g Å、Level 1 Rg = %.3g Å（误差见参数表）。'
              % (chi2_dof, der['rg4_A'], der['rg3_A'], der['rg2_A'], der['rg1_A']))
    if p_at:
        c2 = ('2. **形态判读**：P 存在撞界参数（%s），形态判读降级为**条件值**，'
              '仅以拟合曲线+残差佐证。' % ('、'.join(p_at)))
    else:
        c2 = ('2. **形态判读**：P4 = %.2f、P3 = %.2f、P2 = %.2f、P1 = %.2f（%s）。'
              % (float(params['P4']), float(params['P3']), float(params['P2']),
                 float(params['P1']), _POROD_ANCHOR))
    return [
        c1, c2,
        '3. **可靠性分级**：G/B 为归一量（a.u. 标度）不作跨样品/跨标度直接比较；'
        'Rg/P 为标度无关结构参数，物理可引用性以 multistart/诊断证据链为准；'
        '撞界参数（%s）仅作条件值呈现。' % ('、'.join(_at) if _at else '无'),
        '4. **方法学披露**：本拟合 = 论文同款四 level 统一模型（Beaucage 1995 级联形式，'
        'sasmodels `unified_power_Rg` level=4 同式），INIT 锚定 Figure 7b 表值；'
        '数据为作者交付的溶剂/空白扣除后曲线（sol561−buf555−emp544，a.u. 强度）；'
        'WAXS 区三个 Lorentz 峰未复现（超出模型库能力范围，窗上界 2 nm⁻¹ 避开）。', '']


_PAPER_PEAKS = ('(100) broad：中心 2.7 nm⁻¹、FWHM 0.8 nm⁻¹；'
                '(100) narrow：中心 2.9 nm⁻¹、FWHM 0.2 nm⁻¹；'
                '(012)：中心 6.6 nm⁻¹、FWHM 0.4 nm⁻¹'
                '（论文 Figure 7b 表值，读图级精度）')


def _make_method_full():
    def method(fixed):
        base = _make_method()(fixed)
        peak = [
            '**WAXS 峰（Lorentz 形，论文 Figure 7a 同款）**——峰完整方程（LaTeX）：',
            '',
            '$$',
            r'L(q) \;=\; \frac{A}{1 + 4\,(q-q_0)^2/\mathrm{FWHM}^2}',
            '$$',
            '',
            '- 每峰 3 参数（幅度 $A$ 归一量、中心 $q_0$、FWHM），共 3 峰 9 参数',
            '- 峰锚（Chaney 2025）：' + _PAPER_PEAKS,
            '- full 模式 = 统一四 level（小角区）+ 三 Lorentz 峰（WAXS 区）+ '
            '共享常数背景 B，共 22 自由参数，全谱一次拟合（论文同款做法）', '']
        # 插在"复现范围披露"之前：base 中该行以 '**复现范围披露**' 开头；
        # 该行文案是 main 模式的"峰未复现"披露，full 模式替换为同款位置的
        # 全谱披露，避免方法说明自相矛盾
        out, inserted = [], False
        for ln in base:
            # main 版协议/语义标题的"13 自由参数"口径，full 模式替换为 22 参数口径
            if ln.startswith('- 13 自由参数'):
                out.append('- 22 自由参数（13 统一 + 9 峰）全局拟合，INIT 锚定'
                           '论文 Figure 7b 值；22 参数强相关，multistart 取证'
                           '全局性为必跑项')
                continue
            if ln.startswith('**参数语义**（13 自由参数）'):
                out.append('**统一段参数语义**（13 自由参数；峰参数语义见上）：')
                continue
            if not inserted and ln.startswith('**复现范围披露**'):
                out += peak
                out.append('**复现范围披露（full）**：本模式复现论文 Figure 7a '
                           '全谱 = 小角区统一四 level + WAXS 区三个 Lorentz 峰，'
                           '数据窗 [0.025, 9.98] nm⁻¹ 全谱一次拟合。', )
                inserted = True
                continue
            if not inserted and ln.startswith('**复现锚点'):
                out.append(ln)
                continue
            out.append(ln)
        return out
    return method


def _results_md_full(der, chi2_dof, dof, at, nopeg_compare):
    lines = _results_md(der, chi2_dof, dof, at, nopeg_compare)
    lines += ['', '**WAXS 峰拟合量**（中心/FWHM，nm⁻¹，直接对标论文 Figure 7b 峰锚；'
              '误差为拟合误差数值传播）：']
    for key, name in (('100b', '(100) broad'), ('100n', '(100) narrow'),
                      ('012', '(012)')):
        lines.append('- %s：中心 = **%.3f ± %.3f nm⁻¹**，FWHM = '
                     '**%.2f ± %.2f nm⁻¹**'
                     % (name, der['peak%s_center' % key],
                        der['peak%s_center_err' % key],
                        der['peak%s_fwhm' % key], der['peak%s_fwhm_err' % key]))
    return lines


def _conclusions_full(params, perr, chi2_dof, dof, at, der, nopeg_compare,
                      sample_note):
    base = _conclusions(params, perr, chi2_dof, dof, at, der, nopeg_compare,
                        sample_note)
    peak_c = ('4. **WAXS 峰形态**（χ²/dof=%.3f）：(100) 双峰中心 %.2f/%.2f nm⁻¹、'
              '(012) 中心 %.2f nm⁻¹（FWHM 见参数表），与论文峰锚'
              '（2.7/2.9、6.6 nm⁻¹）直接对照；峰幅度为归一量不作跨标度比较。'
              % (chi2_dof, der['peak100b_center'], der['peak100n_center'],
                 der['peak012_center']))
    # base[3] = main 版方法学披露（含"WAXS 峰未复现"），full 模式整体替换为
    # 全谱口径（防自相矛盾），编号顺延为第 5 条
    meth_c = ('5. **方法学披露**：本拟合 = 论文同款全谱复现——小角区四 level '
              '统一模型（Beaucage 1995 级联形式，sasmodels `unified_power_Rg` '
              'level=4 同式）+ WAXS 区三 Lorentz 峰（论文 Figure 7a 同款），'
              'INIT 锚定 Figure 7b 表值；数据为作者交付的溶剂/空白扣除后曲线'
              '（sol561−buf555−emp544，a.u. 强度）。')
    return base[:3] + [peak_c, meth_c]


# ---------------------------------------------------------------------------
# 降阶比较模式 lv1/lv2/lv3（2026-10-09 增）：level 数 1-3 的级联截断，
# 与主模式（4 level，13 参数）构成嵌套模型族，服务模型阶数比较（奥卡姆剃刀）。
# 阶数取舍判据：嵌套 F 检验（saxs_diag_f_test，param_diff=3/级）+ 参数可辨识性
# （perr 量级 + multistart 多谷取证）；χ²/dof 单一指标不作降阶依据。
# ---------------------------------------------------------------------------
_LV_STYLES = {1: 'm:', 2: 'g--', 3: 'c--', 4: 'b:'}   # 级号→线型（与 main 分量线一致）


def _components_for(k):
    """k 级降阶模式的分量线声明（Level k→1 降序；calc 闭包默认参绑定
    防晚绑定——lessons C 类经典坑）。背底 component 已删（2026-10-27 契约
    对齐）：I_background 由 tools_report 独立列唯一承载，避免 CSV 双背底列"""
    comps = []
    for i in range(k, 0, -1):
        def _f(q, p, fixed, _i=i):
            return _M.level_curve(q, p, _i)
        if i == k:
            lab = 'Level %d (largest scale)' % i
        elif i == 1:
            lab = 'Level %d (smallest scale)' % i
        else:
            lab = 'Level %d' % i
        comps.append({'label': lab, 'style': _LV_STYLES[i],
                      'column': 'I_level%d' % i, 'calc': _f})
    return comps


def _make_method_lv(k):
    def method(fixed):
        return [
            '## 3. 分析方法说明', '',
            '**模型（Beaucage 1995 统一 %d level 级联截断，与 sasmodels '
            '`unified_power_Rg` level=%d 同式）**——模型完整方程（LaTeX）：' % (k, k),
            '',
            '$$',
            r'I(q) \;=\; B \;+\; \sum_{i=1}^{%d}\Bigl[\,G_i\,\exp\!\Bigl(-\frac{(q\,R_{g,i})^2}{3}\Bigr) \;+\; K_i\,\exp\!\Bigl(-\frac{(q\,R_{g,i-1})^2}{3}\Bigr)\,\bigl(q_i^{*}\bigr)^{-P_i}\,\Bigr]' % k,
            '$$',
            '',
            '$$',
            r'q_i^{*} \;=\; \frac{q}{\Bigl[\operatorname{erf}\!\Bigl(\dfrac{q\,R_{g,i}}{\sqrt{6}}\Bigr)\Bigr]^{3}},\qquad\qquad K_i \;=\; \frac{G_i\,P_i}{R_{g,i}^{\,P_i}}\;\Gamma\!\Bigl(\frac{P_i}{2}\Bigr)',
            '$$',
            '',
            '- $q_i^{*}$：幂律尾开启因子（在本级 Guinier 膝之上平滑开启）；'
            '$K_i$：**固定关系内部派生，非自由参数**（与主模式同款）',
            '- 级联方向：Level %d = 最大尺度（最低 q）→ Level 1 = 最小尺度'
            '（最高 q 幂律尾，$i=1$ 尾项无压制因子，$\exp$ 因子恒为 1）' % k, '',
            '**参数语义**（%d 自由参数）：%s；B：常数背景；scale 等效固定 1'
            % (3 * k + 1, '、'.join('G%d/Rg%d/P%d' % (i, i, i)
                                    for i in range(k, 0, -1))),
            '- Rgᵢ：每级回转半径（nm；报告同时给出 Å 值）；Pᵢ：' + _POROD_ANCHOR, '',
            '**模式定位（降阶比较）**：本模式为 %d level 截断版，与主模式'
            '（4 level，13 参数）构成嵌套模型族，服务模型阶数比较（奥卡姆剃刀）。'
            '阶数取舍判据：①嵌套 F 检验（saxs_diag_f_test，param_diff=3/级）'
            '②参数可辨识性（perr 量级 + multistart 多谷取证）。'
            'χ²/dof 单一指标不作为降阶依据。' % k, '',
            '**INIT 披露**：降阶模式无独立文献锚，INIT 按目标数据形态分配'
            '（低 q 斜率→最大级、高 q 渐近→最小级）；换数据集应核对 INIT 适配性。', '',
            '**拟合协议**：',
            '- 引擎：scipy `least_squares`（trf，x_scale=jac）；σ 列存在时加权拟合',
            '- %d 自由参数全局拟合；建议 multistart 取证全局性' % (3 * k + 1),
            '- 撞界判据：双侧界 |v−bound| < 10⁻³·量程；跨量级界（量程>10³·|v|）改用 '
            '5% 参数尺度相对距离；单侧界用 10⁻³·|v|；撞界参数自动降级为**条件值**', '']
    return method


def _results_md_lv(k):
    def results_md(der, chi2_dof, dof, at, nopeg_compare):
        lines = ['**派生量**（%d 级 Rg，误差为拟合误差数值传播；Å 值对标文献口径）：' % k]
        for i in range(k, 0, -1):
            lines.append('- Level %d：Rg = **%.1f ± %.1f Å**（%.2f ± %.2f nm），'
                         'Guinier 特征弯折 q_knee = %.3g nm⁻¹'
                         % (i, der['rg%d_A' % i], der['rg%d_A_err' % i],
                            der['rg%d_nm' % i], der['rg%d_nm_err' % i],
                            der['q_knee%d_nm1' % i]))
        lines += ['', '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
                  % (chi2_dof, dof, '、'.join(at) if at else '无'), '']
        return lines
    return results_md


def _conclusions_lv(k):
    def conclusions(params, perr, chi2_dof, dof, at, der, nopeg_compare,
                    sample_note):
        _at = at or []
        rg_at = [x for x in _at if x.startswith('Rg')]
        p_at = [x for x in _at if x.startswith('P')]
        if rg_at:
            c1 = ('1. **结构尺度**：Rg 存在撞界参数（%s）——撞界级仅作**条件值**呈现，'
                  '不作尺度结论；未撞界级数值见参数表（χ²/dof=%.3f）。'
                  % ('、'.join(rg_at), chi2_dof))
        else:
            c1 = ('1. **结构尺度**（χ²/dof=%.3f）：%s（误差见参数表）。'
                  % (chi2_dof, '、'.join('Level %d Rg = %.3g Å' % (i, der['rg%d_A' % i])
                                         for i in range(k, 0, -1))))
        if p_at:
            c2 = ('2. **形态判读**：P 存在撞界参数（%s），形态判读降级为**条件值**，'
                  '仅以拟合曲线+残差佐证。' % ('、'.join(p_at)))
        else:
            c2 = ('2. **形态判读**：%s（%s）。'
                  % ('、'.join('P%d = %.2f' % (i, float(params['P%d' % i]))
                               for i in range(k, 0, -1)), _POROD_ANCHOR))
        return [
            c1, c2,
            '3. **可靠性分级**：G/B 为归一量（a.u. 标度）不作跨样品/跨标度直接比较；'
            'Rg/P 为标度无关结构参数，物理可引用性以 multistart/诊断证据链为准；'
            '撞界参数（%s）仅作条件值呈现。' % ('、'.join(_at) if _at else '无'),
            '4. **方法学披露**：本拟合 = %d level 统一级联截断（Beaucage 1995，'
            'sasmodels `unified_power_Rg` level=%d 同式），属模型阶数比较降阶模式'
            '（与主模式 4 level 构成嵌套族）；阶数取舍以 F 检验 + 可辨识性为准；'
            '无独立文献锚，INIT 按目标数据形态分配。' % (k, k), '']
    return conclusions


def _cfg_lv(k):
    """k 级降阶模式配置（与 main 同款两联布局/质量门/归一量分级；经 _M 取
    物理模块——register 时绑定，本函数不可在 register 前调用）"""
    def _calc(p, q, fixed, _k=k):
        return _M.model_n(_apply_fixed(p, fixed, _M.NAMES_N[_k]), q, _k)

    def _derived(p, perr, fixed, _k=k):
        return _M.derived_n(p, perr, _k)
    return {
        'names': _M.NAMES_N[k],
        'bounds': _M.BOUNDS_LV[k],
        'init': _M.INIT_LV[k],
        'init_qmax': _M.INIT_QMAX,
        'fixed_keys': [],                 # 无跨样品固定参数
        'normalized_params': ['G%d' % i for i in range(1, k + 1)] + ['B'],
        'selftest_ab': (1.0, 0.0),        # 冒烟合成数据幅度量级（B=0 恢复检查自动跳过）
        'smoke_q': np.concatenate([       # 低 q 补点覆盖最大级膝区（含窗外缘）
            np.geomspace(0.02, 0.3, 25),
            np.linspace(0.3, 4.4, 60)]),
        'calc': _calc,
        'derived': _derived,
        'plot_layout': _LAYOUT_DOUBLE,
        'components': _components_for(k),
        'report': {
            'method': _make_method_lv(k),
            'results_md': _results_md_lv(k),
            'conclusions': _conclusions_lv(k),
        },
    }


def _apply_fixed(p, fixed, names):
    """fixed 参数锁定覆盖（2026-10-27 增，liposome peg 同款语义）：调用方可经
    saxs_fit / make_plots / saxs_report 的 fixed= 传入任意参数定值（如背底
    B=1e-3 定值拟合），锁定值在 calc 层强制覆盖自由参数——贯穿初值/拟合/绘图/
    报告全链路；锁定参数的 perr 无物理意义，由调用方在报告披露。
    p 两形态兼容：引擎路径传参数向量（param_array，按 names 序）、诊断/绘图
    路径传参数 dict——分别按位覆盖/按键覆盖。"""
    if not fixed:
        return p
    if isinstance(p, dict):
        p = dict(p)
        for kk, vv in fixed.items():
            if kk in names:
                p[kk] = float(vv)
        return p
    p = np.asarray(p, dtype=float).copy()
    for kk, vv in fixed.items():
        if kk in names:
            p[list(names).index(kk)] = float(vv)
    return p


def register(M):
    global _M
    _M = M
    cfg = {
        'names': M.NAMES,
        'bounds': M.BOUNDS,
        'init': M.INIT,
        'init_qmax': M.INIT_QMAX,
        'fixed_keys': [],                 # 无必填跨样品固定参数；fixed 可选锁定（_apply_fixed）
        'normalized_params': ['G1', 'G2', 'G3', 'G4', 'B'],
        'selftest_ab': (1.0, 0.0),        # 冒烟合成数据幅度量级（B=0 恢复检查自动跳过）
        'smoke_q': np.concatenate([       # 专属冒烟网格（2026-10-27，与 lv/full 同款）：
            np.geomspace(0.02, 0.3, 25),  #   INIT_QMAX=0.4 后 init_qmax/2=0.2 仍会触发
            np.linspace(0.3, 4.4, 60)]),  #   默认低 q 补点口径漂移；改为显式声明自含
                                          #   低 q 补点（Level4 膝区），网格不再随
                                          #   INIT_QMAX 漂移，恢复检查与 lv 同基线
        'calc': lambda p, q, fixed: M.model_vec(_apply_fixed(p, fixed, M.NAMES), q),
        'derived': lambda p, perr, fixed: M.derived_main(p, perr),
        'plot_layout': _LAYOUT_DOUBLE,
        'components': _COMPONENTS,
        'report': {
            'method': _make_method(),
            'results_md': _results_md,
            'conclusions': _conclusions,
        },
    }
    cfg_full = {
        'names': M.NAMES_FULL,
        'bounds': {**M.BOUNDS, **M.BOUNDS_PEAK},
        'init': {**M.INIT, **M.INIT_PEAK},
        'init_qmax': M.INIT_QMAX,
        'fixed_keys': [],                 # 无必填跨样品固定参数；fixed 可选锁定（_apply_fixed）
        'normalized_params': ['G1', 'G2', 'G3', 'G4', 'B', 'A_b', 'A_n', 'A_0'],
        'selftest_ab': (1.0, 0.0),        # 冒烟合成数据幅度量级（B=0 恢复检查自动跳过）
        'smoke_q': np.concatenate([       # 专属冒烟网格：默认两段逐位保留 + (012) 峰区
            np.geomspace(0.025, 0.3, 20),  #   （c_0=6.6 nm^-1 在默认网格 max 4.4 窗外，
            np.linspace(0.3, 4.4, 60),     #    峰参数窗内不可辨识致冒烟假报 issue）
            np.linspace(5.8, 7.4, 33)]),
        'calc': lambda p, q, fixed: M.model_full(_apply_fixed(p, fixed, M.NAMES_FULL), q),
        'derived': lambda p, perr, fixed: M.derived_full(p, perr),
        'plot_layout': _LAYOUT_DOUBLE,    # 与 main 同款两联布局（轴范围已声明）
        'components': _COMPONENTS + _PEAK_COMPONENTS,
        'report': {
            'method': _make_method_full(),
            'results_md': _results_md_full,
            'conclusions': _conclusions_full,
        },
    }
    # 降阶比较模式（2026-10-09 增）：lv1/lv2/lv3 与 main 构成嵌套模型族
    return {'main': cfg, 'full': cfg_full,
            'lv1': _cfg_lv(1), 'lv2': _cfg_lv(2), 'lv3': _cfg_lv(3)}

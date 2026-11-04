# -*- coding: utf-8 -*-
"""sasmodels 桥接通道（M2.5-4）：官方模型库基础形状接入通用引擎，开箱即用。

设计（设计文档 §10.1 M2.5-4）：
- 直调通道：不在 model_library 建目录；model_loader.list_models / load_model
  在本地库未命中时到此查找，工具面（fit/plots/report/selftest）零改动复用；
- scale/background 映射为引擎 A/B 约定：init_p0 初值策略、L3 质量门、selftest
  冒烟拟合全部零改动复用（A=scale 乘性因子，B=background 常数背景，均为归一量）；
- 单位：引擎数据 q 统一 nm⁻¹，sasmodels 内核原生 Å⁻¹——calc 包装器内部 ×0.1
  换算，模型几何参数保持 sasmodels 原生单位（长度 Å），labels 从 info 带出；
- 图布局：两联（拟合曲线 log + 相对残差）——M2.5-2 布局 DSL 的非三联实例；
- 状态标注 external：官方实现，未经本站 L1/L2 验证门，报告不生成模型专属结论。
"""
import numpy as np


def _install_utf8_open_guard():
    """进程级 UTF-8 缺省编码保护（M2.5-4 实测坑）：MCP 服务器进程无 PYTHONUTF8=1 时
    Windows 默认 GBK，sasmodels 读 ASCII 的 C 源码竟报 UnicodeDecodeError('gbk')。
    UTF-8 模式只能在解释器启动时设定，进程内无法开启，故在文本模式且调用方未显式
    指定 encoding 时补 utf-8。本项目自身文件 I/O 均显式编码（curve_io 探测链、
    model_loader json、tools_report csv），不受影响；与 run_server.bat 的
    PYTHONUTF8=1 双保险对齐。"""
    import builtins
    if getattr(builtins, '_saxs_utf8_guard', False):
        return
    _orig = builtins.open

    def _open(file, mode='r', *a, **kw):
        # 位置参数 a: (buffering, encoding, errors, newline)——len(a)>=2 表示 encoding 已按位置给出
        if 'b' not in mode and 'encoding' not in kw and len(a) < 2:
            kw['encoding'] = 'utf-8'
        return _orig(file, mode, *a, **kw)

    builtins.open = _open
    builtins._saxs_utf8_guard = True


_install_utf8_open_guard()

Q_NM2A = 0.1                      # nm⁻¹ → Å⁻¹
BRIDGE_ZONE = 'sasmodels_bridge'
# 2026-10-21 扩容：+ hollow_cylinder / core_shell_cylinder（中空管与核壳棒，
# 实验站老师常用体系需求）；hollow_cylinder 语义注意：radius=内腔半径、
# thickness=壁厚，scale=管壁材料体积分数（core_shell_cylinder 则按总体积归一）
# 2026-10-22 扩容：+ teubner_strey / broad_peak（双连续相分离/界面层族需求，
# 相分离 1.dat 数据两轮已验证模型结构性失配后立项；teubner_strey = TS 1987
# 正源模型（c1<0 出峰，导出周期 d 与相关长度 xi），broad_peak = 幂律底+广延峰）
WHITELIST = ['sphere', 'core_shell_sphere', 'cylinder', 'ellipsoid',
             'hollow_cylinder', 'core_shell_cylinder',
             'teubner_strey', 'broad_peak']

_K = {}                           # name -> {'model','kernel','orig','full'}
_unavailable = None               # None=已可用；str=不可用原因
_VER = ''


def _make_eval(m):
    """sasmodels 1.1.0 直评闭包（实测确认的调用形态）：
    DirectModel(empty_data1D(q), model)(**pars)；call_kernel 在此版本收 build 后
    的 kernel 而非 Model，不可用。DirectModel.__call__(**pars) 返回 I(q)。"""
    from sasmodels.data import empty_data1D
    from sasmodels.direct_model import DirectModel

    def _eval(q, kw):
        calc = DirectModel(empty_data1D(np.asarray(q, dtype=float)), m)
        return np.asarray(calc(**kw), dtype=float)

    return _eval


def _ensure():
    """惰性导入 sasmodels 并构建内核缓存；任一形状失败则整体不可用"""
    global _unavailable, _VER
    if _unavailable is not None or _K:
        return
    try:
        import sasmodels as _sm
        from sasmodels.core import load_model as _load
    except Exception as e:
        _unavailable = 'sasmodels 不可导入: %r' % e
        return
    _VER = getattr(_sm, '__version__', '')
    for name in WHITELIST:
        try:
            m = _load(name)
            ps = m.info.parameters
            orig = [p.name for p in ps.kernel_parameters]
            d = {p.name: p for p in ps.kernel_parameters}
            full = _build_full(name, m, orig, d)
            full['_kernel_slot']['k'] = _make_eval(m)
            _K[name] = {'model': m, 'orig': orig, 'full': full}
        except Exception as e:
            _unavailable = '模型 %s 加载失败: %r' % (name, e)
            _K.clear()
            return


def unavailable_reason():
    _ensure()
    return _unavailable


def _build_full(name, m, orig, d):
    """构建 load_model 同构条目 {'manifest','modes','zone','dir','physics'}"""
    desc = getattr(m.info, 'description', None) or name
    names = list(orig) + ['A', 'B']
    bounds = {n: (float(d[n].limits[0]), float(d[n].limits[1])) for n in orig}
    bounds['A'] = (0.0, float('inf'))
    bounds['B'] = (0.0, float('inf'))
    init = {n: float(d[n].default) for n in orig}
    init['A'] = 1.0                 # 占位：init_p0 会按数据重估 A/B
    init['B'] = 0.001
    kernel_holder = {}

    def calc(p, q, fixed):
        pd = p if isinstance(p, dict) else dict(zip(names, p))
        kw = {n: float(pd[n]) for n in orig}
        kw['scale'] = float(pd['A'])
        kw['background'] = float(pd['B'])
        return kernel_holder['k'](np.asarray(q, dtype=float) * Q_NM2A, kw)

    labels = {n: (n, (d[n].units or '')) for n in orig}
    labels['A'] = ('scale（sasmodels 缩放因子）', '相对')
    labels['B'] = ('background（sasmodels 常数背景）', 'sasmodels 原生')
    man = {
        'name': name,
        'version': 'sasmodels %s' % (_VER or '?'),
        'zone': BRIDGE_ZONE,
        'status': 'external',
        'entry': None,                    # 无本地物理文件（必需键占位，桥接直调）
        'plugin': None,
        'description': 'sasmodels 官方模型（桥接通道）：' + desc,
        # modes[mode] = 渲染/报告元数据（与站内模型同契约：make_plots 标题模板、
        # result.json 的 sample、CSV 拟合列名都从这里取）；注意 cfg 不进 manifest
        'modes': {'main': {
            'sample': '（桥接模型未声明样品名）',
            'fit_column': 'I_fit_%s' % name,
            'plot_title': '%s (sasmodels bridge)' % name,
            'fit_label': 'fit: %s' % name,
            'residual_title': 'rel. residual',
        }},
        'labels': labels,
        'refs': {'paper': 'SASview/sasmodels 官方文档（模型 %s）' % name,
                 'si': 'https://www.sasview.org/docs/',
                 'assumptions': ['官方库实现，未经本站 L1/L2 验证门',
                                 '单分散近似（未启用 polydispersity）']},
        'recommended_workflow': {
            'purpose': 'M2.5-4 桥接通道：官方基础形状开箱即用；通用纪律见 agent_prompts/saxs_workflow.md',
            'q_window_nm': [0.3, 4.4],
            'single_sample': [
                'saxs_load_curve（体检，确认点数/量程/丢点）',
                "saxs_fit(model='%s', mode='main', path, q_unit=与用户确认值)" % name,
                'make_plots（两联图：拟合曲线 + 相对残差）',
                'saxs_report（四件套交付）',
            ],
            'multi_sample': None,
            'diagnostics': [
                {'tool': 'saxs_diag_multistart', 'when': '质疑最优解全局性时'},
                {'tool': 'saxs_diag_relax_bound', 'when': '参数撞界 / 疑似简并时'},
            ],
        },
    }
    cfg = {
        'names': names,
        'bounds': bounds,
        'init': init,
        'init_qmax': 1.0,
        # 冒烟真值覆盖（2026-10-21）：core_shell_cylinder 的 sasmodels 默认
        # sld_core=sld_shell=4（核壳零反差，均匀圆柱极限）→ radius/thickness
        # 仅剩外径约束强简并，通用冒烟 χ²=4.18 + R/T 恢复偏差>20% 假报；
        # 修复分两轮：sld_shell=2 解决反差简并（χ² 4.18→0.677）但仍 >0.5 阈值
        # ——三 SLD 共同平移是精确物理简并（I 只依赖差值），扰动起点沿零曲率
        # 方向游走与 length/A 耦合停在次优收敛点（离线取证：抛光二级拟合
        # 无效、真值起点直拟合 χ²=0）；第二轮 sld_core=6 加强核反差陡化耦合
        # 谷（χ² 0.677→0.167 PASS）。sld 族在 smoke_skip_recovery 跳过单参数
        # 恢复检查，此覆盖仅影响合成数据形态
        'smoke_truth': ({'sld_shell': 2.0, 'sld_core': 6.0}
                        if name == 'core_shell_cylinder' else {}),
        'fixed_keys': [],
        'normalized_params': ['A', 'B'],
        'selftest_ab': (1.0, 0.001),      # 冒烟合成数据的 scale / background 量级
        # 冒烟恢复检查跳过项：①SLD 族与 scale 仅反差组合可辨识
        # （I ∝ A·(sld−sld_solvent)²）；②取向角在 1D 取向平均下完全不可辨识。
        # 两者均为物理简并非缺陷，跳过依据 sasmodels 参数类型声明。
        # 2026-10-22 +teubner_strey: volfraction_a 与 (sld_a−sld_b)² 幅度耦合
        # （I ∝ volfrac·Δρ²，与 core_shell_cylinder 的 scale 管壁体积分数同类
        # 反差组合简并），恢复检查跳过
        'smoke_skip_recovery': (['A'] + [n for n in orig if 'sld' in n.lower()]
                                + [n for n in orig if getattr(d[n], 'type', '') == 'orientation']
                                + (['volfraction_a'] if name == 'teubner_strey' else [])),
        'calc': calc,
        'derived': lambda p, perr, fixed: {},
        'plot_layout': {
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
        },
        'report': {
            'method': lambda fixed: [
                '## 3. 分析方法说明', '',
                '**模型**：`%s`（sasmodels %s 官方实现，经桥接通道接入；'
                'scale/background 映射为引擎 A/B）' % (name, _VER or '?'),
                '', '**拟合协议**：',
                '- 数据 q 单位 nm⁻¹，内核计算时换算为 sasmodels 原生 Å⁻¹（×0.1）；'
                '模型几何参数保持原生单位（长度 Å，见参数表单位列）',
                '- 窗口与权重协议与站内模型一致（1/σ²，scipy least_squares trf）',
                '- **状态：external**——官方实现，未经本站 L1/L2 验证门；'
                '本报告未含模型专属物理解读', ''],
            'results_md': lambda der, chi2_dof, dof, at, nopeg_compare: [
                '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
                % (chi2_dof, dof, '、'.join(at) if at else '无'), ''],
            'conclusions': lambda params, perr, chi2_dof, dof, at, der,
                           nopeg_compare, sample_note: [
                '1. 拟合参数与可靠性分级见上表与参数表.csv（A/B 为归一量）。',
                '2. 本报告由 sasmodels 桥接通道（external 状态）生成，未含模型专属物理判据；'
                '定量结论请结合样品背景，并优先采用已过验证门的站内模型。', ''],
        },
    }
    kernel_holder['k'] = None          # 占位防误用；真正内核由 _ensure 写入
    return {'manifest': man, 'modes': {'main': cfg}, 'zone': BRIDGE_ZONE,
            'dir': None, 'physics': None, '_kernel_slot': kernel_holder}


def available():
    """可用桥接模型摘要（与 list_models 条目同构）；不可用时返回空列表"""
    _ensure()
    out = []
    for name, rec in _K.items():
        man = rec['full']['manifest']
        out.append({'name': name, 'version': man['version'], 'zone': BRIDGE_ZONE,
                    'status': 'external', 'description': man['description'],
                    'modes': ['main'],
                    'recommended_workflow': man['recommended_workflow'],
                    'dir': None})
    return out


def lookup(name):
    """model_loader 未命中时的查找口：命中返回 load_model 同构 dict，否则 None"""
    _ensure()
    rec = _K.get(name)
    return None if rec is None else rec['full']

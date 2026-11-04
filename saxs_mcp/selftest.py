# -*- coding: utf-8 -*-
"""服务器自检（M2-T4）：saxs_selftest 工具的实现模块。
目的：改代码后无需"重启试错"，一次调用即可在服务器进程内验证：
  0. 整备层   —— gap_fill 引擎合成冒烟：detect 深谷/孤立点可命中（M3pre-2）
  0b. 取向层  —— ruland 引擎合成冒烟：B_eq 全链路恢复偏差 <10%（260930）
  1. 引擎层   —— fitting 共享函数、quality_gate.f_test 可调用且结果合理
  2. 模型库层 —— manifest 键完整、插件契约（必需键 / bounds 覆盖 / 初值覆盖 /
                 fixed_bounds ⊇ fixed_keys / compare 一致性）
  3. 冒烟拟合 —— 每个模式用自身模型生成无噪合成数据，扰动初值重拟合：
                 χ²/dof 应≈0、参数恢复偏差 <20%；对照块只验证可运行
                 （2026-09-24 np 导入遗漏事故正是该 bug 类，冒烟可捕获）
背景扣除约定：合成数据的 A/B 典型值可由插件 selftest_ab 覆盖，默认 (1e-4, 6e-3)。
"""
import math
import os

import numpy as np

import model_loader

# MCP 宿主 cwd 不可控，冒烟产物与 gap_fill 同规则锚定工作区根
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import quality_gate
from fitting import do_fit, chi2_dof, at_bound, init_p0, param_array, bound_arrays

_REQUIRED_MODE_KEYS = ('names', 'bounds', 'init', 'init_qmax', 'calc', 'derived')
_REQUIRED_MANIFEST_KEYS = ('name', 'version', 'entry', 'modes', 'labels', 'refs', 'description')


def _check_mode(model_name, mode_name, cfg, issues, warnings):
    """模型库层：插件契约完整性检查"""
    tag = '%s/%s' % (model_name, mode_name)
    for k in _REQUIRED_MODE_KEYS:
        if k not in cfg:
            issues.append('%s 缺少必需键 %r' % (tag, k))
            return                       # 后续检查依赖这些键
    names = list(cfg['names'])
    for n in names:
        if n not in cfg['bounds']:
            issues.append('%s bounds 缺少参数 %r' % (tag, n))
    # 初值覆盖检查（2026-10-22 通用化）：幅度键经 cfg['init_amp_key'] 声明
    # （默认 'A'，与 fitting.init_p0 同源）；B 键恒在检查集
    _amp_key = cfg.get('init_amp_key', 'A')
    cover = (set(cfg['init']) | set(cfg.get('init_defaults', {}))
             | {_amp_key, 'B'})
    for n in names:
        if n not in cover:
            warnings.append('%s 初值未覆盖 %r（A/B 兜底可能不适用）' % (tag, n))
    fk = cfg.get('fixed_keys', [])
    fb = cfg.get('fixed_bounds') or {}
    for k in fk:
        if k not in fb:
            warnings.append('%s fixed_keys %r 未声明 fixed_bounds（saxs_diag_release 不可用）'
                            % (tag, k))
    ccfg = cfg.get('compare')
    if ccfg:
        for k in ('names', 'calc', 'p0', 'zero_param'):
            if k not in ccfg:
                issues.append('%s compare 缺少键 %r' % (tag, k))
        for n in (ccfg.get('names') or []):
            if n not in cfg['bounds']:
                issues.append('%s compare 参数 %r 不在模式 bounds 中' % (tag, n))
        zp = ccfg.get('zero_param')
        if zp and zp not in names:
            issues.append('%s compare.zero_param %r 不在模式参数表中' % (tag, zp))
    # M2.5-2 渲染器契约：布局必须声明且 panel 合法（kind / at 在 grid 内）
    lay = cfg.get('plot_layout')
    if lay is None:
        issues.append('%s 未声明 plot_layout（M2.5-2 渲染器契约：make_plots 无法绘图）' % tag)
    else:
        if 'grid' not in lay or 'panels' not in lay:
            issues.append('%s plot_layout 缺少 grid/panels 键' % tag)
        else:
            gr, gc = lay['grid']
            for i, p in enumerate(lay['panels']):
                if p.get('kind') not in ('fit', 'residual', 'profile', 'curve'):
                    issues.append('%s plot_layout panel#%d kind 非法: %r'
                                  % (tag, i, p.get('kind')))
                at = p.get('at', (0, 0))
                if not (0 <= at[0] < gr and 0 <= at[1] < gc):
                    issues.append('%s plot_layout panel#%d at %r 超出 grid %r'
                                  % (tag, i, at, lay['grid']))


def _smoke_fit(model_name, mode_name, cfg, issues):
    """冒烟拟合：合成数据生成 → 扰动初值重拟合 → 参数恢复；对照块只验证可运行"""
    tag = '%s/%s' % (model_name, mode_name)
    names = list(cfg['names'])
    fb = cfg.get('fixed_bounds') or {}
    fixed = {k: (fb[k][0] + fb[k][1]) / 2.0
             for k in cfg.get('fixed_keys', []) if k in fb}
    a_def, b_def = cfg.get('selftest_ab', (1e-4, 6e-3))
    p_true = dict(cfg.get('init_defaults', {}))
    p_true.update(cfg['init'])
    # 声明性冒烟真值覆盖（2026-10-21）：默认参数本身可能处于物理简并态
    # （如 core_shell_cylinder 默认 sld_core=sld_shell → 核壳零反差，
    # radius/thickness 仅剩外径约束），smoke_truth 覆盖真值使特征可辨
    p_true.update(cfg.get('smoke_truth', {}))
    # A/B 冒烟真值注入（2026-10-22 通用化）：幅度参数名经 cfg['init_amp_key']
    # 声明（默认 'A'）；仅当该键在参数表中才注入（历史模型零变化；
    # phase_separation_ruland A_c KeyError 实测——无条件 p_true['A'] 对
    # 幅度参数不叫 A 的模型注入幽灵键无害但 'B' 注入同理需条件化）
    _amp_key = cfg.get('init_amp_key', 'A')
    if _amp_key in names:
        p_true[_amp_key] = a_def
    if 'B' in names:
        p_true['B'] = b_def
    # 冒烟网格（2026-10-02 二次修订）：q>=0.3 段保持历史 linspace 网格逐位不变——
    # 桥接模型（init_qmax=1.0）的 B 背景恢复对该段网格密度敏感，首轮 geomspace
    # 全域替换曾致 cylinder 冒烟 B 恢复偏差 >20% 误报（信号在高 q 段的加权占比
    # 变小）；仅当插件 init_qmax<0.3 时在低 q 端补点覆盖 A0 初值估计窗
    # （alcu/gel main 的 init_qmax=0.15，估计窗倒空则 init_p0 空窗报错）。
    q = np.linspace(0.3, 4.4, 60)
    qlo = min(0.3, cfg['init_qmax'] / 2.0)
    if qlo < 0.3:
        q = np.unique(np.concatenate([np.geomspace(qlo, 0.3, 20), q]))
    # 声明性专属冒烟网格（2026-10-06）：全谱模型的特征（如高 q 衍射峰）可能落在
    # 默认网格 max q=4.4 之外，特征参数在窗内结构性不可辨识 → 无噪冒烟下沿平坦
    # 方向漂移假报 issue；插件 cfg 声明 smoke_q（须自含低 q 补点）时整体覆盖
    q_override = cfg.get('smoke_q')
    if q_override is not None:
        q = np.asarray(q_override, dtype=float)
    try:
        I = cfg['calc'](param_array(p_true, names), q, fixed)
    except Exception as e:
        issues.append('%s 合成数据生成失败: %r' % (tag, e))
        return 'FAIL'
    sig = 0.05 * np.abs(I) + 1e-8
    lo, hi = bound_arrays(cfg['bounds'], names)
    p0 = param_array(p_true, names).copy()
    span = np.where(np.isfinite(hi - lo), hi - lo, np.abs(p0) + 1.0)
    pert = np.where(np.arange(len(names)) % 2 == 0, 1.05, 0.95)
    p0 = np.clip(p0 * pert, lo + 1e-6 * span, hi - 1e-6 * span)
    try:
        p0i = init_p0(cfg, names, fixed, q, I)          # 引擎初值策略可调用
        res = do_fit(lambda p: (cfg['calc'](p, q, fixed) - I) / sig, p0, lo, hi)
        chi2, dof, perr = chi2_dof(res, len(names))
        at_bound(res.x, lo, hi, names)                  # 撞界标记可调用
    except Exception as e:
        issues.append('%s 合成拟合失败: %r' % (tag, e))
        return 'FAIL'
    if not (chi2 < 0.5):
        issues.append('%s 冒烟拟合 chi2/dof=%.3g（无噪合成数据应≈0）' % (tag, chi2))
    for n, x in zip(names, res.x):
        if n in cfg.get('smoke_skip_recovery', []):
            continue                      # 声明性简并参数（如 sld 反差族）跳过单参数恢复
        t = p_true[n]
        if abs(t) > 0 and abs(x - t) / abs(t) > 0.2:
            issues.append('%s 参数 %r 恢复偏差 >20%%（%.4g vs %.4g）' % (tag, n, x, t))
    ccfg = cfg.get('compare')
    if ccfg:
        try:
            cnames = list(ccfg['names'])
            clo, chi_hi = bound_arrays(cfg['bounds'], cnames)
            cp0 = np.asarray(ccfg['p0'](p_true['A'], p_true['B'], p_true), dtype=float)
            do_fit(lambda p: (ccfg['calc'](p, q, fixed) - I) / sig, cp0, clo, chi_hi)
        except Exception as e:
            issues.append('%s 对照拟合失败: %r' % (tag, e))
            return 'FAIL'
    return 'PASS'


def _check_engine(issues):
    """引擎层：quality_gate.f_test 数值合理性"""
    try:
        r = quality_gate.f_test(100.0, 10, 120.0, 12)   # F = (20/2)/(100/10) = 1.0
        if not (r['F'] > 0 and 0.0 <= r['p_value'] <= 1.0):
            issues.append('quality_gate.f_test 结果异常: %r' % r)
    except Exception as e:
        issues.append('quality_gate.f_test 不可调用: %r' % e)


def _check_conditioning(issues):
    """整备链路冒烟（M3pre-2）：合成曲线 detect 全链路——引擎可导入可运行，
    深谷核心（code 1）与孤立异常点（code 3）可命中。副作用：runs/_selftest_gapdetect/
    下写出标记图与 gap 表（用户清 runs 时一并清理即可）"""
    try:
        import gap_fill
    except Exception as e:
        issues.append('gap_fill 引擎导入失败: %r' % e)
        return 'FAIL'
    q = np.exp(np.linspace(np.log(0.05), np.log(5.0), 200))
    I = 1e4 * q ** -2.5 + 50.0
    iA0 = int(np.argmin(np.abs(q - 2.0)))
    I[iA0:iA0 + 10] *= 1e-4                       # 深谷核心（期望 code 1）
    iC = int(np.argmin(np.abs(q - 0.9)))
    I[iC] *= 0.01                                 # 孤立异常点（期望 code 3）
    out_dir = os.path.join(_ROOT, 'runs', '_selftest_gapdetect')
    src = os.path.join(out_dir, '_smoke.dat')
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(src, 'w', encoding='utf-8') as f:
            for v_q, v_i in zip(q, I):
                f.write('%.6f %.6f\n' % (v_q, v_i))
        card = gap_fill.detect_card(src, x_semantic='q', x_unit='nm^-1',
                                    col=1, out_dir=out_dir)
    except Exception as e:
        issues.append('整备冒烟 detect 失败: %r' % e)
        return 'FAIL'
    codes = {int(k) for k in card['segment_counts'] if card['segment_counts'][k]}
    if 1 not in codes or 3 not in codes:
        issues.append('整备冒烟段码不全（codes=%r，应同时命中深谷 1 与孤立点 3）'
                      % card['segment_counts'])
        return 'FAIL'
    return 'PASS'


def _check_orientation(issues):
    """取向链路冒烟（260930 Ruland）：内存合成 I(φ,q) 小矩阵（单峰无噪声、
    φ₀=37° 故意非 90° 防硬编码）→ choose_phi_ref → bpi2_series → ruland_beq
    全链路可运行，B_eq 恢复偏差 <10%（冒烟级阈值；正式判据 =
    test_data/ruland_l1 L1 套件）。纯内存链路无写盘副作用。"""
    try:
        import ruland as rl
    except Exception as e:
        issues.append('ruland 引擎导入失败: %r' % e)
        return 'FAIL'
    try:
        L_true, Beq_true, phi0 = 35.0, 0.10, 37.0
        phi = np.arange(0.0, 360.0, 1.0)                     # φ 轴 deg，1° 步长
        q = np.linspace(0.5, 2.0, 8)                          # q_c nm^-1
        Btot = np.sqrt(1.0 / (q * L_true) ** 2 + Beq_true ** 2)   # 式 2 正向
        sig_deg = Btot / np.sqrt(2.0 * np.pi) * 180.0 / np.pi
        d = (phi - phi0 + 180.0) % 360.0 - 180.0              # 圆周距离防 0/360 截断
        Y = np.empty((phi.size, q.size))
        for j in range(q.size):
            Y[:, j] = 10.0 + 1000.0 * (0.5 / q[j]) ** 2 \
                * np.exp(-0.5 * (d / sig_deg[j]) ** 2)
        sel = rl.choose_phi_ref(phi, Y, list(range(q.size)))
        ser = rl.bpi2_series(phi, Y, list(range(q.size)), sel['phi_ref'],
                             q_values=q, q_unit='nm^-1',
                             per_col_peaks=sel['per_col_peaks'])
        fit = rl.ruland_beq([r['q'] for r in ser['rows']],
                            [r['B_rad_corr'] for r in ser['rows']])
    except Exception as e:
        issues.append('取向冒烟链路失败: %r' % e)
        return 'FAIL'
    dev = abs(fit['B_eq_rad'] - Beq_true) / Beq_true
    if dev >= 0.10:
        issues.append('取向冒烟 B_eq 恢复偏差 %.1f%% ≥10%%（%.4g vs %.4g rad）'
                      % (100 * dev, fit['B_eq_rad'], Beq_true))
        return 'FAIL'
    # 提速三件套 T-A：扫描自动采纳冒烟——断言机制自洽而非"必命中"。
    # 合成数据无噪声 → OLS 残差≈0 → 曲率 F 检验分母病态、p 值数值随机，
    # 命中/adopt_manual 兜底两分支均合法；真实噪声数据上 p 的有效性由
    # 000064 回归基准承载（lessons.md：F 检验三态语义，无 flag≠通过）。
    try:
        scan = rl.ruland_beq_scan([r['q'] for r in ser['rows']],
                                  [r['B_rad_corr'] for r in ser['rows']],
                                  [float(q[2]), float(q[3]), float(q[4])])
    except Exception as e:
        issues.append('取向冒烟扫描链路失败: %r' % e)
        return 'FAIL'
    if len(scan['windows']) + len(scan['window_errors']) != 3:
        issues.append('取向冒烟扫描窗口数不自洽: %d 窗 + %d 错 ≠ 3'
                      % (len(scan['windows']), len(scan['window_errors'])))
        return 'FAIL'
    if scan['adopted_qmin'] is None:
        if not scan['flags'] or not scan['flags'][0].startswith('adopt_manual'):
            issues.append('取向冒烟扫描未采纳且缺 adopt_manual flag: %s'
                          % scan['flags'])
            return 'FAIL'
    else:
        hit = [r for r in scan['windows'] if r['qmin'] == scan['adopted_qmin']]
        if not hit or not (math.isfinite(hit[0]['curvature_F_p'])
                           and hit[0]['curvature_F_p'] >= 0.05):
            issues.append('取向冒烟扫描采纳窗 F 检验不自洽: %r'
                          % (hit[0] if hit else None))
            return 'FAIL'
        for r in scan['windows']:
            if r['qmin'] >= scan['adopted_qmin']:
                break
            if math.isfinite(r['curvature_F_p']) and r['curvature_F_p'] >= 0.05:
                issues.append('取向冒烟扫描 first-pass 语义违反: qmin=%g 亦通过'
                              % r['qmin'])
                return 'FAIL'
    return 'PASS'


def run_selftest(model=None):
    """执行自检，返回 {ok, models, engine}"""
    engine_issues = []
    _check_engine(engine_issues)
    cond_status = _check_conditioning(engine_issues)   # M3pre-2 整备链路冒烟
    orient_status = _check_orientation(engine_issues)  # 260930 取向链路冒烟
    models = {}
    try:                              # M2.5-4：桥接不可用时给出原因，避免静默缺席难排查
        import sasmodels_bridge
        reason = sasmodels_bridge.unavailable_reason()
        if reason:
            engine_issues.append('sasmodels 桥接不可用: %s' % reason)
    except Exception as e:
        engine_issues.append('sasmodels 桥接导入失败: %r' % e)
    for m in model_loader.list_models():
        if model and m['name'] != model:
            continue
        mi = {'issues': [], 'warnings': [], 'modes': {}}
        try:
            info = model_loader.load_model(m['name'])
        except Exception as e:
            mi['issues'].append('模型加载失败: %r' % e)
            models[m['name']] = mi
            continue
        for k in _REQUIRED_MANIFEST_KEYS:
            if k not in info['manifest']:
                mi['issues'].append('manifest 缺少键 %r' % k)
        if 'recommended_workflow' not in info['manifest']:
            mi['warnings'].append('manifest 未声明 recommended_workflow（M2.5-3 能力自发现）：'
                                  'Agent 无法自动编排该模型的分析流程')
        for mode_name, cfg in info['modes'].items():
            _check_mode(m['name'], mode_name, cfg, mi['issues'], mi['warnings'])
            if m.get('zone') == 'inbox':
                # 261104：inbox 收件区模型未过验证门——只做契约静态检查，
                # 不跑冒烟拟合（更不进入正式分析，fit/diag 已拒收）
                mi['modes'][mode_name] = {'smoke': 'SKIP_INBOX'}
                mi['warnings'].append('inbox 待入库模型：仅契约检查，未过验证门，'
                                      '禁止正式分析；入库流程见 workflow §7')
            else:
                mi['modes'][mode_name] = {'smoke': _smoke_fit(m['name'], mode_name, cfg, mi['issues'])}
        models[m['name']] = mi
    ok = (not engine_issues) and all(not mi['issues'] for mi in models.values())
    return {'ok': ok, 'models': models,
            'engine': {'issues': engine_issues,
                       'status': 'OK' if not engine_issues else 'ISSUES',
                       'conditioning_smoke': cond_status,
                       'orientation_smoke': orient_status}}


def register(mcp):
    @mcp.tool()
    def saxs_selftest(model: str = None) -> dict:
        """服务器自检：验证整备引擎（gap detect 合成冒烟，M3pre-2）、取向引擎
        （Ruland 全链路合成冒烟，B_eq 恢复偏差 <10%，260930）、拟合引擎
        函数可调用、模型库可加载、各模式插件契约完整，并对每个模式做合成数据
        冒烟拟合（含对照块可运行性）。
        model: 只检查指定模型；省略时检查模型库全部模型。
        返回 {ok, models, engine}；ok=False 时逐条看 issues。改代码重启后建议先调用本工具。"""
        return run_selftest(model)

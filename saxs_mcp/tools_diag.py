# -*- coding: utf-8 -*-
"""诊断工具（M2-T3）：baseline/liposome/diag_*.py 三个诊断脚本的能力通用化。
与模型无关、按需调用（区别于 quality_gate 的每次自动执行）：
  saxs_diag_multistart  多起点重拟合 —— 检验最优解是否全局（diag_peg_check 诊断1）
  saxs_diag_relax_bound 放宽某参数边界重拟合 —— 撞界敏感性 + 参数相关（诊断2）
  saxs_diag_param_scan  固定某参数于网格值逐点重拟合 —— 简并度/位置依赖（diag_peg_scan）
  saxs_diag_release     放开部分跨样品固定参数重拟合 —— 方案对比（diag_membrane_thickness）
  saxs_diag_f_test      任意两个嵌套拟合的 F 检验（复用 quality_gate.f_test）
"""
import os
import numpy as np

import model_loader
import quality_gate
from curve_io import load_curve
from fitting import do_fit, chi2_dof, at_bound, init_p0, param_array, bound_arrays


def _resolve(path):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(os.getcwd(), path))


def _prep(model, mode, path, q_unit, qmin, qmax, fixed, fmt=None):
    """公共准备：加载模型 + 校验模式与 fixed + 读曲线"""
    info = model_loader.load_model(model)
    if info.get('zone') == 'inbox':      # 261104：inbox 收件区禁入正式分析
        raise ValueError(
            '模型 %s 位于 model_inbox 待入库区（未过验证门），禁止诊断拟合；'
            '请走 workflow §7 入库流程' % model)
    if mode not in info['modes']:
        raise ValueError('模型 %s 无模式 %r，可用: %s' % (model, mode, list(info['modes'])))
    cfg = info['modes'][mode]
    need = cfg.get('fixed_keys', [])
    missing = [k for k in need if not (fixed and k in fixed)]
    if missing:
        raise ValueError("模式 '%s' 需要固定参数 %s（fixed 中缺失: %s）" % (mode, need, missing))
    q, I, sig, _ = load_curve(_resolve(path), q_unit, qmin, qmax, fmt)
    return cfg, q, I, sig


def _run(cfg, calc, names, p0, lo, hi, fixed, q, I, sig, derived_fn=None, with_gate=True):
    """单次诊断拟合的标准输出（params/perr/chi2/dof/at_bound/derived/quality）"""
    res = do_fit(lambda p: (calc(p, q, fixed) - I) / sig, p0, lo, hi)
    chi2, dof, perr = chi2_dof(res, len(names))
    pd = {n: float(v) for n, v in zip(names, res.x)}
    pe = {n: float(v) for n, v in zip(names, perr)}
    at = at_bound(res.x, lo, hi, names)
    out = {'names': list(names), 'params': pd, 'perr': pe,
           'chi2_dof': chi2, 'dof': int(dof), 'at_bound': at}
    if derived_fn is not None:
        out['derived'] = derived_fn(pd, pe, fixed)
    if with_gate:
        out['quality'] = quality_gate.run_gate(
            q, I, calc(res.x, q, fixed), chi2, dof, names, at,
            cfg.get('normalized_params'), perr=pe, values=pd)
    return out, res


def _corr_of(res, chi2, names, param):
    """参数相关系数（协方差按 chi2/dof 缩放后归一化），返回 {其他参数: corr}"""
    pcov = np.linalg.pinv(res.jac.T @ res.jac) * chi2
    perr = np.sqrt(np.diag(pcov))
    corr = pcov / np.outer(perr, perr)
    k = names.index(param)
    return {n: float(corr[k, i]) for i, n in enumerate(names) if n != param}


def register(mcp):
    @mcp.tool()
    def saxs_diag_multistart(model: str, mode: str, path: str, starts: list,
                             target: str = 'mode', fixed: dict = None,
                             q_unit: str = 'A^-1', qmin: float = 0.3,
                             qmax: float = 4.4, fmt: dict = None) -> dict:
        """多起点重拟合：检验最优解是否全局最优（diag_peg_check 诊断1）。
        starts: 初值覆盖列表，每项为 {参数名: 值}（覆盖在标准初值之上，可只给部分参数）；
        target: 'mode'（主模式）或 'compare'（模式声明的对照模型，如 peg 的无PEG对照）；
        fmt: 数据格式显式声明（saxs_probe_curve 确认卡，None = 常规三列路径）。
        返回各起点 chi2/dof 与参数、全局最优。"""
        cfg, q, I, sig = _prep(model, mode, path, q_unit, qmin, qmax, fixed, fmt)
        if target == 'compare':
            ccfg = cfg.get('compare')
            if not ccfg:
                raise ValueError('模式 %s 未声明对照模型（compare）' % mode)
            names = list(ccfg['names'])
            calc = ccfg['calc']
            lo, hi = bound_arrays(cfg['bounds'], names)
            p0_mode = init_p0(cfg, cfg['names'], fixed, q, I)
            p0 = list(ccfg['p0'](p0_mode['A'], p0_mode['B'], p0_mode))
        elif target == 'mode':
            names = list(cfg['names'])
            calc = cfg['calc']
            lo, hi = bound_arrays(cfg['bounds'], names)
            p0 = list(param_array(init_p0(cfg, names, fixed, q, I), names))
        else:
            raise ValueError("target 只支持 'mode' 或 'compare'")
        rows = []
        for s in starts:
            p0s = np.array(p0, dtype=float)
            for k, v in s.items():
                if k not in names:
                    raise ValueError('起点参数 %r 不在 %s 参数表中 %s' % (k, target, names))
                p0s[names.index(k)] = float(v)
            out, _ = _run(cfg, calc, names, p0s, lo, hi, fixed, q, I, sig,
                          derived_fn=cfg['derived'] if target == 'mode' else None,
                          with_gate=False)
            rows.append({'start': s, 'chi2_dof': out['chi2_dof'], 'params': out['params']})
        best = min(rows, key=lambda r: r['chi2_dof'])
        return {'target': target, 'n_starts': len(rows), 'rows': rows,
                'best_chi2_dof': best['chi2_dof'], 'best_params': best['params']}

    @mcp.tool()
    def saxs_diag_relax_bound(model: str, mode: str, path: str, param: str,
                              lo: float = None, hi: float = None, params: dict = None,
                              fixed: dict = None, q_unit: str = 'A^-1',
                              qmin: float = 0.3, qmax: float = 4.4,
                              fmt: dict = None) -> dict:
        """放宽/修改某自由参数的边界后重拟合（diag_peg_check 诊断2：
        如 tau_peg 下界 0.80→0.30，检验撞界参数的敏感性与简并）。
        lo/hi 只传需要修改的一侧；params 可传原拟合结果作暖启动；fmt: 数据格式
        显式声明（saxs_probe_curve 确认卡，None = 常规三列路径）。
        返回重拟合结果 + 该参数与其余参数的相关系数。"""
        cfg, q, I, sig = _prep(model, mode, path, q_unit, qmin, qmax, fixed, fmt)
        names = list(cfg['names'])
        if param not in names:
            raise ValueError('参数 %r 不在模式 %s 参数表中 %s' % (param, mode, names))
        bounds = {n: tuple(b) for n, b in cfg['bounds'].items()}
        old = bounds[param]
        bounds[param] = (old[0] if lo is None else float(lo),
                         old[1] if hi is None else float(hi))
        lo_a, hi_a = bound_arrays(bounds, names)
        p0d = params if params else init_p0(cfg, names, fixed, q, I)
        out, res = _run(cfg, cfg['calc'], names, param_array(p0d, names), lo_a, hi_a,
                        fixed, q, I, sig, derived_fn=cfg['derived'])
        out['param'] = param
        out['bounds'] = {'old': list(old), 'new': list(bounds[param])}
        out['corr'] = _corr_of(res, out['chi2_dof'], names, param)
        return out

    @mcp.tool()
    def saxs_diag_param_scan(model: str, mode: str, path: str, param: str, values: list,
                             params: dict = None, fixed: dict = None,
                             q_unit: str = 'A^-1', qmin: float = 0.3,
                             qmax: float = 4.4, fmt: dict = None) -> dict:
        """固定某自由参数于网格值、其余参数逐点重拟合（diag_peg_scan 诊断4/5：
        tau_peg 扫描量化宽度-幅度简并；z_peg 扫描检验 PEG 高斯的位置偏好）。
        values: 网格值列表；params: 暖启动（建议传原拟合 params），逐点 warm start；
        fmt: 数据格式显式声明（saxs_probe_curve 确认卡，None = 常规三列路径）。
        返回逐行 {value, chi2_dof, at_bound, params, derived}。"""
        cfg, q, I, sig = _prep(model, mode, path, q_unit, qmin, qmax, fixed, fmt)
        names = list(cfg['names'])
        if param not in names:
            raise ValueError('参数 %r 不在模式 %s 参数表中 %s' % (param, mode, names))
        k = names.index(param)
        free = [n for n in names if n != param]
        lo_f, hi_f = bound_arrays(cfg['bounds'], free)
        warm = params if params else init_p0(cfg, names, fixed, q, I)
        rows = []
        for v in values:
            v = float(v)

            def resid_free(pf, v=v):
                p = np.empty(len(names))
                p[k] = v
                j = 0
                for i, n in enumerate(names):
                    if i != k:
                        p[i] = pf[j]
                        j += 1
                return (cfg['calc'](p, q, fixed) - I) / sig

            p0f = [float(warm[n]) for n in free]
            res = do_fit(resid_free, p0f, lo_f, hi_f)
            chi2, dof, perr = chi2_dof(res, len(free))
            pd = {n: float(x) for n, x in zip(free, res.x)}
            pd[param] = v
            pe = {n: float(e) for n, e in zip(free, perr)}
            pe[param] = 0.0                       # 被固定的参数无误差
            at = at_bound(res.x, lo_f, hi_f, free)
            rows.append({'value': v, 'chi2_dof': chi2, 'dof': int(dof),
                         'at_bound': at, 'params': pd,
                         'derived': cfg['derived'](pd, pe, fixed)})
            warm = pd
        return {'param': param, 'n': len(rows), 'rows': rows}

    @mcp.tool()
    def saxs_diag_release(model: str, mode: str, path: str, release: list, fixed: dict,
                          params: dict = None, q_unit: str = 'A^-1',
                          qmin: float = 0.3, qmax: float = 4.4,
                          fmt: dict = None) -> dict:
        """放开部分跨样品固定参数重拟合（diag_membrane_thickness 方案对比：
        检验"膜参数固定自另一样品"的隐含假设，如 d_hh = 2·zh 是否漂移）。
        release ⊆ 模式声明的 fixed_keys（可为部分列表）；放开参数的边界取插件
        fixed_bounds，初值取 fixed 中的原值；params 可传原拟合结果作基础参数暖启动；
        fmt: 数据格式显式声明（saxs_probe_curve 确认卡，None = 常规三列路径）。
        返回放开后的拟合结果与派生量（派生量用放开后的值计算）。"""
        cfg, q, I, sig = _prep(model, mode, path, q_unit, qmin, qmax, fixed, fmt)
        names = list(cfg['names'])
        fb = cfg.get('fixed_bounds')
        bad = [k for k in release if k not in cfg.get('fixed_keys', [])]
        if bad:
            raise ValueError('release 中的参数不是该模式声明的固定参数: %s（可用: %s）'
                             % (bad, cfg.get('fixed_keys', [])))
        if not release:
            raise ValueError('release 不能为空')
        nofb = [k for k in release if not (fb and k in fb)]
        if nofb:
            raise ValueError('插件未声明放开参数的边界（fixed_bounds）: %s' % nofb)
        names_ext = names + list(release)
        n_base = len(names)
        base_calc = cfg['calc']

        def calc_ext(p, q_, fx):
            f2 = dict(fx or {})
            for i, k in enumerate(release):
                f2[k] = float(p[n_base + i])
            return base_calc(p[:n_base], q_, f2)

        def derived_ext(pd, pe):
            f2 = dict(fixed)
            f2.update({k: pd[k] for k in release})
            return cfg['derived'](pd, pe, f2)

        p0d = params if params else init_p0(cfg, names, fixed, q, I)
        p0 = [float(p0d[n]) for n in names] + [float(fixed[k]) for k in release]
        lo = [cfg['bounds'][n][0] for n in names] + [fb[k][0] for k in release]
        hi = [cfg['bounds'][n][1] for n in names] + [fb[k][1] for k in release]
        res = do_fit(lambda p: (calc_ext(p, q, fixed) - I) / sig,
                     np.array(p0, dtype=float), np.array(lo, dtype=float),
                     np.array(hi, dtype=float))
        chi2, dof, perr = chi2_dof(res, len(names_ext))
        pd = {n: float(v) for n, v in zip(names_ext, res.x)}
        pe = {n: float(v) for n, v in zip(names_ext, perr)}
        at = at_bound(res.x, np.array(lo, dtype=float), np.array(hi, dtype=float), names_ext)
        return {'names': names_ext, 'released': list(release),
                'params': pd, 'perr': pe, 'chi2_dof': chi2, 'dof': int(dof),
                'at_bound': at, 'derived': derived_ext(pd, pe),
                'quality': quality_gate.run_gate(
                    q, I, calc_ext(res.x, q, fixed), chi2, dof, names_ext, at,
                    cfg.get('normalized_params'), perr=pe, values=pd)}

    @mcp.tool()
    def saxs_diag_f_test(ssr_full: float, dof_full: int,
                         ssr_reduced: float, dof_reduced: int) -> dict:
        """任意两个嵌套拟合的 F 检验（全模型 vs 简化模型，须同数据）。
        ssr = 加权残差平方和 = chi2_dof × dof。返回 F / p 值 / 显著性结论。"""
        return quality_gate.f_test(float(ssr_full), int(dof_full),
                                   float(ssr_reduced), int(dof_reduced))

# -*- coding: utf-8 -*-
"""拟合工具（M2 通用化）：saxs_fit(model, data) 通用引擎
模型经 model_loader 从 model_library 加载（manifest.json 定位 + plugin.py 适配），
引擎本身不含任何模型知识。

初值构造 = 通用策略，与原脚本逐位一致（回归基准
baseline/liposome/regression_baseline.json）：
  B0 = 高 q 末端 100 点均值
  A0 = median((I - B0) / I_shape)，I_shape 为 A=1、B=0 的纯形状（低 q 窗口内）
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


def _fit_generic(model, mode, path, q_unit, qmin, qmax, fixed, fmt=None):
    info = model_loader.load_model(model)
    if info.get('zone') == 'inbox':      # 261104：inbox 收件区禁入正式分析
        raise ValueError(
            '模型 %s 位于 model_inbox 待入库区（未过验证门），禁止正式拟合；'
            '请走 workflow §7 入库流程（L1 合成自检 + L2 文献对照 + 人工审核签字），'
            '验证通过后移入 model_library/validated' % model)
    man = info['manifest']
    if mode not in info['modes']:
        raise ValueError('模型 %s 无模式 %r，可用: %s' % (model, mode, list(info['modes'])))
    cfg = info['modes'][mode]

    need = cfg.get('fixed_keys', [])
    missing = [k for k in need if not (fixed and k in fixed)]
    if missing:
        raise ValueError("模式 '%s' 需要固定参数 %s（fixed 中缺失: %s）" % (mode, need, missing))

    path = _resolve(path)
    q, I, sig, _ = load_curve(path, q_unit, qmin, qmax, fmt)
    names = cfg['names']

    # ---- 初值构造（通用策略，数值与原脚本逐位一致；T3 提取到 fitting.init_p0 共享）----
    p0 = init_p0(cfg, names, fixed, q, I)

    lo, hi = bound_arrays(cfg['bounds'], names)

    def resid(p):
        return (cfg['calc'](p, q, fixed) - I) / sig

    res = do_fit(resid, param_array(p0, names), lo, hi)
    chi2, dof, perr = chi2_dof(res, len(names))

    p_dict = {n: float(v) for n, v in zip(names, res.x)}
    perr_dict = {n: float(v) for n, v in zip(names, perr)}
    at = at_bound(res.x, lo, hi, names)
    der = cfg['derived'](p_dict, perr_dict, fixed)

    # ---- L3 质量门（每次拟合自动执行）：相对残差/撞界/分级/chi2 关注标记 ----
    model_vals = cfg['calc'](res.x, q, fixed)
    gate = quality_gate.run_gate(q, I, model_vals, chi2, dof, names, at,
                                 cfg.get('normalized_params'),
                                 perr=perr_dict, values=p_dict)

    out = {'model': model, 'model_version': '%s %s' % (man['name'], man.get('version', '')),
           'mode': mode,
           'params': p_dict, 'perr': perr_dict, 'chi2_dof': chi2, 'dof': int(dof),
           'at_bound': at, 'derived': der, 'quality': gate,
           'data_summary': {'n': int(len(q)), 'q_min': float(q[0]), 'q_max': float(q[-1])}}

    # ---- 对照拟合（模式声明了 compare 时执行，如 peg 的无 PEG 对照）----
    ccfg = cfg.get('compare')
    if ccfg:
        cnames = ccfg['names']
        clo, chi_b = bound_arrays(cfg['bounds'], cnames)

        def resid_c(p):
            return (ccfg['calc'](p, q, fixed) - I) / sig

        res_c = do_fit(resid_c, ccfg['p0'](p0['A'], p0['B'], p0), clo, chi_b)
        chi0, dof0, perr_c = chi2_dof(res_c, len(cnames))
        I_full = cfg['calc'](res.x, q, fixed)
        p_zero = res.x.copy()
        p_zero[names.index(ccfg['zero_param'])] = 0.0
        eff = np.abs((I_full - cfg['calc'](p_zero, q, fixed)) / I_full)
        out['nopeg_compare'] = {'chi2_dof': chi0, 'dof': int(dof0),
                                'max_rel_effect_pct': float(100 * np.max(eff)),
                                # T5：对照模型参数（供 make_plots 画对照线 / 报告对照数据列）
                                'params': {n: float(v) for n, v in zip(cnames, res_c.x)},
                                'perr': {n: float(v) for n, v in zip(cnames, perr_c)}}
        # 嵌套 F 检验：全模型（mode）相对对照模型（compare）的改善显著性
        out['nopeg_compare']['f_test'] = quality_gate.f_test(
            float((res.fun ** 2).sum()), int(dof),
            float((res_c.fun ** 2).sum()), int(dof0))
    return out


def register(mcp):
    @mcp.tool()
    def saxs_list_models() -> dict:
        """列出模型库全部模型（261104 起含三区）：validated（验证通过，正式分析
        用）/ candidates（候选区，需用户明示同意）/ inbox（根目录 model_inbox 待
        入库收件区，未过验证门，saxs_fit 与 saxs_diag_* 拒绝使用）+ sasmodels
        桥接 external 通道。返回名称/版本/zone/可用拟合模式/描述/推荐分析流程
        recommended_workflow（q 窗口 / 单样品 / 多样品参数传递 / 诊断建议）。
        拟合前先用本工具选模型，并按 recommended_workflow 编排流程。"""
        return {'models': model_loader.list_models()}

    @mcp.tool()
    def saxs_fit(model: str, mode: str, path: str, q_unit: str = 'A^-1',
                 qmin: float = 0.3, qmax: float = 4.4, fixed: dict = None,
                 fmt: dict = None) -> dict:
        """通用拟合引擎：fit(model, data)。
        model: 模型名（saxs_list_models 可查；inbox 区模型被拒收）；mode: 该模型的拟合模式；
        path: .dat 数据文件路径；q_unit: 'A^-1' 或 'nm^-1'；qmin/qmax: 拟合窗口 (nm^-1)，
        建议经 saxs_suggest_window 建议卡 + 用户确认后显式传入（261104：不再对所有
        曲线一刀切用同一默认窗）；
        fixed: 跨样品固定参数 dict（如 liposome_membrane 'peg' 模式需传无PEG拟合的 params）；
        fmt: 数据格式显式声明（M2.5-1 probe_format 确认卡，如 2 列 txt；
        None = 常规三列路径）。建议先用 saxs_probe_curve 生成。
        返回 params / perr / chi2_dof / dof / at_bound / derived / quality（L3 质量门：
        相对残差统计 + 可靠性分级 + chi2 关注标记）（+ nopeg_compare：chi2/效应/f_test/
        对照模型 params+perr，若模式支持对照）。
        """
        return _fit_generic(model, mode, path, q_unit, qmin, qmax, fixed, fmt)

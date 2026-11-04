# -*- coding: utf-8 -*-
"""least_squares 封装（M1 最小实现）
移植自两个 fit 脚本的拟合段：trf + x_scale='jac'，chi2/dof、协方差误差、撞界标记。
"""
import numpy as np
from scipy.optimize import least_squares


def do_fit(resid, p0, lo, hi):
    """与原脚本完全一致的拟合调用"""
    return least_squares(resid, p0, bounds=(lo, hi), method='trf', x_scale='jac', max_nfev=20000)


def chi2_dof(res, npar):
    """返回 (chi2/dof, dof, perr)；perr = 协方差对角线开方（res.fun² 缩放）"""
    dof = len(res.fun) - npar
    chi2 = float((res.fun**2).sum() / dof)
    pcov = np.linalg.pinv(res.jac.T @ res.jac) * (res.fun**2).sum() / dof
    perr = np.sqrt(np.diag(pcov))
    return chi2, dof, perr


def at_bound(x, lo, hi, names, rel=1e-3):
    """撞界标记：|v-bound| < 1e-3·range 视为撞界。
    双侧有限边界用 range 定容差；单侧边界用参数自身尺度 |v| 定容差，
    避免 A=1e-4 这类远离 0 边界的小量被误标（重启服务器后生效）。
    跨量级双侧边界防误标（2026-10-02 hardcarbon_ts 实测）：range 由远端界
    主导（rng > 1e3·|v|）时，rel·range 容差会把离边界多个数量级的正常收敛
    值误标为撞界（C2 界 [1e-8,1e4] 收敛 0.0165 被标、C1=0.031 界 [1e-6,50]
    被标）；此时改用 5% 参数尺度相对距离判据（贴界=参数被推到与边界同
    数量级），非跨量级路径与单侧路径数值行为不变。"""
    hits = []
    for v, l, h, n in zip(x, lo, hi, names):
        if np.isfinite(l) and np.isfinite(h):
            rng = h - l
            scale = max(abs(v), 1e-12)
            if rng > 1e3 * scale:
                if abs(v - l) <= 0.05 * scale or abs(h - v) <= 0.05 * scale:
                    hits.append(n)
            else:
                tol = rel * rng
                if abs(v - l) <= tol or abs(h - v) <= tol:
                    hits.append(n)
        else:
            tol = rel * max(abs(v), 1e-12)
            if (np.isfinite(l) and abs(v - l) <= tol) or (np.isfinite(h) and abs(h - v) <= tol):
                hits.append(n)
    return hits


# ---- 以下为 M2 通用引擎共享的壳逻辑（T3 从 tools_fit 提取，供 fit/diag 共用）----

def param_array(d, names):
    """参数 dict -> (n,) 向量（顺序按 names）"""
    return np.array([d[n] for n in names], dtype=float)


def bound_arrays(bounds, names):
    """bounds 字典 {name: (lo, hi)} -> (lo, hi) 两个 (n,) 数组"""
    lo = np.array([bounds[n][0] for n in names], dtype=float)
    hi = np.array([bounds[n][1] for n in names], dtype=float)
    return lo, hi


def init_p0(cfg, names, fixed, q, I):
    """通用初值策略（数值与原脚本逐位一致）：
    B0 = 高 q 末端 100 点均值；A0 = median((I-B0)/I_shape)，
    I_shape 为 A=1、B=0 的纯形状（低 q 窗口内）。
    cfg: 模型插件模式配置（init / init_defaults / init_qmax / calc）。
    通用化（2026-10-22，phase_separation_ruland A_c KeyError 实测）：幅度参数名
    可经 cfg['init_amp_key'] 声明（默认 'A'，历史模型零变化）；比例幅度键
    cfg['init_amp_scales']（如 Ruland 模型的 A_int 与 A_c 同乘 A0）随主幅度
    等比缩放。A0 标定语义 = 用低 q 窗数据标定模型整体幅度。
    """
    amp_key = cfg.get('init_amp_key', 'A')
    amp_scales = [k for k in cfg.get('init_amp_scales', []) if k in names]
    B0 = float(I[-100:].mean())
    if not np.isfinite(B0) or B0 < 0.0:
        # 背景初值兜底（2026-10-09，Neubert 20 °C 已扣背景数据实测）：数据高 q 端
        # I≈0 振荡含负值时末端均值 <0，违反 B≥0 下界 → least_squares
        # "Initial guess is outside of provided bounds"（与 10-01 估计带倒空 /
        # 10-02 A0≤0 同族第三变体）。背景物理非负，兜底取 0（恰在界上，
        # trf 的 make_strictly_feasible 自动内推）；B0≥0 正常路径逐位不变。
        B0 = 0.0
    p0 = dict(cfg.get('init_defaults', {}))
    p0.update(cfg['init'])
    p0.update({k: float(v) for k, v in (fixed or {}).items() if k in names})
    if amp_key in names:
        p0.setdefault(amp_key, 1.0)
    if 'B' in names:
        p0.setdefault('B', B0)
    w = q <= cfg['init_qmax']
    if amp_key not in names and not amp_scales:
        # 幅度键不进参数表（如 unified_power_Rg：NAMES 无 'A'，A0 标定对 x0
        # 零影响）时，估计带仅为防空窗护栏——数据 q_min 超过 init_qmax 的数据集
        # （2026-10-27，闭孔硬碳 A1/A2 q_min=0.3094 > 0.09 实测）直接跳过估计，
        # x0 与有窗路径逐位一致；估计带语义见上方 docstring。
        return p0
    if not w.any():
        # 空窗显式报错（2026-10-02，selftest 冒烟网格与 init_qmax=0.15 不相交实测）：
        # 旧行为 median(空)=NaN 静默传播 → least_squares "x0 is infeasible" 难排查；
        # 按"宁可报错不要猜错"纪律在此给出明确诊断。
        raise ValueError('init_qmax=%g 与数据 q 范围不相交（q_min=%.4g），无法估计 A0'
                         % (cfg['init_qmax'], float(q.min())))
    p_shape = dict(p0)
    if amp_key in names:
        p_shape[amp_key] = 1.0
    for k in amp_scales:
        p_shape[k] = 0.0        # 纯形状：主幅度=1，比例幅度键清零（幅度标定只测主键）
    if 'B' in names:
        p_shape['B'] = 0.0                               # 纯形状（原脚本 Iraw0 的构造方式）
    I0 = cfg['calc'](param_array(p_shape, names), q, fixed)
    A0 = float(np.median((I[w] - B0) / I0[w]))
    if not np.isfinite(A0) or A0 <= 0.0:
        # 窄窗兜底（2026-10-02，gel_network porod 窄窗对照实测）：窗口末端 I 逼近 B0 时
        # I-B0 中位可≤0，原逻辑直接产出违反 A≥0 界的初值（least_squares 报
        # "Initial guess is outside of provided bounds"）。兜底取窗内最大 I（幂律形态下
        # ≈A·q_min^-dm，偏大但界内合理）；median>0 的正常路径数值逐位不变。
        A0 = max(float(np.max(I[w])), 1e-12)
    if amp_key in names:
        p0[amp_key] = A0
    for k in amp_scales:
        # 比例幅度键：按主幅度等比缩放（cfg init 值视为"幅度=1 时"的形状比例）
        p0[k] = float(cfg['init'].get(k, p0.get(k, 0.0))) * A0
    if 'B' in names:
        p0['B'] = B0
    return p0

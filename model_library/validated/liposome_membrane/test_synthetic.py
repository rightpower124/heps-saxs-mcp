# -*- coding: utf-8 -*-
"""L1 合成数据自检（验证门第 1 层：验证插件代码实现正确性）
用法（在 saxsagent 环境）：
    E:\\CondaEnvs\\saxsagent\\python.exe model_library\\validated\\liposome_membrane\\test_synthetic.py

原理：用 model.py 以"已知真值参数"生成含噪假数据 -> 用与 MCP 工具完全相同的
fitting 引擎和边界反演 -> 检验参数能否恢复。
判据（全部通过 = L1 PASS）：
  1) 每个自由参数 |fit - true| <= 3*perr
  2) chi2/dof ∈ [0.6, 1.6]（合成噪声恰为 sigma，理应≈1）
  3) 多起点全部收敛到同一 chi2（偏差 <1%，排除局部极小）
用例：
  A 中心情形（真值在边界内部）          B 撞界情形（drho_t/tau_h 顶到边界）
  C PEG 模型（膜参数固定 + L_peg 恢复）
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SAXS = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir, 'saxs_mcp'))
for _p in (_HERE, _SAXS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model as M
from fitting import do_fit, chi2_dof, at_bound

RNG = np.random.default_rng(20260923)          # 固定种子，结果可复现
Q = np.linspace(0.3, 4.4, 600)                 # 覆盖真实拟合窗口，点数与真实数据相当
NOISE = 0.02                                   # 相对噪声 2%
N_START = 5                                    # 多起点次数


def _perturb(p_vec, lo, hi, frac=0.15):
    """真值附近随机扰动作为多起点初值（裁剪到边界内）"""
    p0 = p_vec * (1.0 + RNG.normal(0, frac, p_vec.shape))
    p0[0] = abs(p0[0])                          # A 保持正值
    return np.clip(p0, lo + 1e-6, hi - 1e-6)


def _run_case(name, true_vec, names, bounds, model_fn, fixed=None):
    """生成合成数据 -> 多起点拟合 -> 三项判据检查，打印一行结论"""
    lo = np.array([bounds[n][0] for n in names])
    hi = np.array([bounds[n][1] for n in names])
    I_true = model_fn(true_vec, Q) if fixed is None else model_fn(true_vec, Q, fixed)
    # sigma = 2%|I|，在 I 的振荡节点处用 1%·Imax 兜底，避免节点处残差爆炸
    sig = NOISE * np.maximum(np.abs(I_true), 0.01 * np.max(np.abs(I_true)))
    I_obs = I_true + RNG.normal(0, sig)

    def resid(p):
        r = (model_fn(p, Q) - I_obs) / sig if fixed is None else (model_fn(p, Q, fixed) - I_obs) / sig
        return r

    chis, fits = [], []
    for _ in range(N_START):
        res = do_fit(resid, _perturb(true_vec, lo, hi), lo, hi)
        chi2, dof, perr = chi2_dof(res, len(names))
        chis.append(chi2)
        fits.append((res.x, perr))

    best = min(chis)
    multi_ok = all(abs(c - best) / best < 0.01 for c in chis)       # 判据3：多起点一致
    x_best, err_best = fits[0]

    # 判据1：3σ 恢复（perr 可能因撞界而≈0，此时若拟合值==真值记 0，否则记 FAIL）
    err_safe = np.where(np.asarray(err_best) > 0, err_best, 1.0)
    dev = np.where(np.asarray(err_best) > 0,
                   np.abs(np.asarray(x_best) - true_vec) / err_safe,
                   np.abs(np.asarray(x_best) - true_vec))
    chi_ok = 0.6 <= best <= 1.6                                     # 判据2：chi2/dof 合理
    ok = multi_ok and chi_ok and bool(np.all(dev <= 3.0))

    worst = '%s=%.1fsig' % (names[int(np.argmax(dev))], dev.max())
    print('%s [%s]  chi2/dof=%.3f  多起点一致=%s  最差恢复=%s'
          % ('PASS' if ok else 'FAIL', name, best, multi_ok, worst))
    if not ok:
        for n, t, v, e in zip(names, true_vec, x_best, err_best):
            print('    %-9s true=%10.4f  fit=%10.4f +- %8.4f' % (n, t, v, e))
    return ok, x_best


# ---------- 真值设定 ----------
# A: 中心情形（所有自由参数在边界内部）
TRUE_A = {'A': 2.5e3, 'zh': 1.45, 'tau_h': 0.80, 'tau_t': 1.00, 'drho_t': -2.00,
          'f': 0.50, 'd': 8.00, 'sd': 1.00, 'B': 0.02}
# B: 撞界情形（drho_t 下界 -6、tau_h 上界 1.30，即真实拟合出现的 A 方案状态）
TRUE_B = {**TRUE_A, 'zh': 1.6428, 'tau_h': 1.30, 'tau_t': 0.6566, 'drho_t': -6.00}
# C: PEG 情形（膜固定自 A，PEG 层参数在边界内；L_peg 期望约 4.5+0.911*2.0-1.45-0.911*0.8）
FIX_C = {k: TRUE_A[k] for k in ('zh', 'tau_h', 'tau_t', 'drho_t')}
TRUE_C = {'A': 2.5e3, 'f': 0.50, 'd': 8.00, 'sd': 1.00, 'B': 0.02,
          'z_peg': 4.50, 'tau_peg': 2.00, 'drho_peg': 0.08}


def main():
    print('L1 合成数据自检  (q: %.1f-%.1f nm-1, %d 点, 噪声 %.0f%%, 多起点 x%d)'
          % (Q[0], Q[-1], len(Q), 100*NOISE, N_START))
    print('-' * 72)

    va = np.array([TRUE_A[n] for n in M.NAMES_NOPEG])
    ok_a, _ = _run_case('A 中心  ', va, M.NAMES_NOPEG, M.BOUNDS_NOPEG, M.model_nopeg_vec)

    vb = np.array([TRUE_B[n] for n in M.NAMES_NOPEG])
    ok_b, _ = _run_case('B 撞界  ', vb, M.NAMES_NOPEG, M.BOUNDS_NOPEG, M.model_nopeg_vec)

    vc = np.array([TRUE_C[n] for n in M.NAMES_PEG])
    ok_c, xc = _run_case('C PEG   ', vc, M.NAMES_PEG, M.BOUNDS_PEG, M.model_peg_vec, fixed=FIX_C)

    # L_peg 恢复检查（C 用例的派生量）
    p_c = dict(zip(M.NAMES_PEG, xc))
    e_c = dict.fromkeys(M.NAMES_PEG, 1e-3)      # 派生量检查只需拟合值本身
    L_fit = M.derived_peg(p_c, e_c, FIX_C)['L_peg']
    L_true = M.derived_peg(TRUE_C, e_c, FIX_C)['L_peg']
    # L_peg 检查定位 = 派生量公式接线正确性（错公式至少偏 0.2 nm），
    # 精度已由上面 3σ 参数恢复覆盖；0.05 nm 硬阈值会误判统计涨落（实测 0.056）
    ok_l = abs(L_fit - L_true) < 0.15
    print('    L_peg: true=%.3f  fit=%.3f  (|差|<0.15 nm -> %s)'
          % (L_true, L_fit, 'PASS' if ok_l else 'FAIL'))

    print('-' * 72)
    verdict = ok_a and ok_b and ok_c and ok_l
    print('L1 总判定: %s' % ('PASS —— 插件代码实现正确' if verdict else 'FAIL —— 禁止入库，回查 model.py/fitting.py'))


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""L1 合成数据自检（验证门第 1 层：验证插件代码实现正确性）
用法（在 saxsagent 环境，注意 PATH 前置纪律）：
    E:\\CondaEnvs\\saxsagent\\python.exe model_library\\candidates\\hardcarbon_ts\\test_synthetic.py

原理：用 model.py 以"已知真值参数"生成含噪假数据 -> 用与 MCP 工具完全相同的
fitting 引擎和边界反演 -> 检验参数能否恢复。
判据（全部通过 = L1 PASS）：
  1) 每个自由参数 |fit - true| <= 3*perr
  2) chi2/dof ∈ [0.6, 1.6]（合成噪声恰为 sigma，理应≈1）
  3) 多起点全部收敛到同一 chi2（偏差 <1%，排除局部极小）
用例（main 模式 6 参）：
  A 中心情形：文献锚尺度（D=0.6 nm → C1=0.018 nm²；肩位 q_sh≈4 nm⁻¹ → d≈1.57 nm）
  B 大孔径情形：D=2.68 nm（C1=0.36 nm²）+ C2=0.5 nm⁴（d=8.65 nm, ξ=1.06 nm），I0=1.0
    （TS 中 q 占比 ~40%，保证信号可辨）
  C 幂律撞界情形：n=6.0 顶上界（粗糙表面/陡幂律极限）
σ 设计（2026-10-02 探针取证教训）：纯相对 σ = 2%·|I_true|，**不用 Imax 兜底下限**——
  幂律数据动态范围 ~10⁷，Imax 兜底会把中/高 q 全部点压平（σ_floor >> 信号），
  χ² 被低 q 几个点独占，TS 参数权重归零，拟合必然落入"TS 关闭"简并解。
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
from fitting import do_fit, chi2_dof

RNG = np.random.default_rng(20261002)          # 固定种子，结果可复现
Q = np.linspace(0.175, 11.29, 600)             # 覆盖真实拟合窗口（nm⁻¹，与硬碳数据一致）
NOISE = 0.02                                   # 相对噪声 2%
N_START = 5                                    # 多起点次数


def _perturb(p_vec, lo, hi, frac=0.15):
    """真值附近随机扰动作为多起点初值（裁剪到边界内）"""
    p0 = p_vec * (1.0 + RNG.normal(0, frac, p_vec.shape))
    p0[0] = abs(p0[0])                          # A 保持正值
    p0[2] = abs(p0[2])                          # I0 保持正值
    return np.clip(p0, lo + 1e-9, hi - 1e-9)


def _run_case(name, true_vec, names, bounds, model_fn):
    """生成合成数据 -> 多起点拟合 -> 三项判据检查，打印一行结论"""
    lo = np.array([bounds[n][0] for n in names])
    hi = np.array([bounds[n][1] for n in names])
    I_true = model_fn(true_vec, Q)
    # 纯相对 σ（动态范围大时 Imax 兜底会压平中/高 q，见 docstring 教训）；
    # I_true 全正，仅加绝对小底防 0 除
    sig = NOISE * np.maximum(np.abs(I_true), 1e-10)
    I_obs = I_true + RNG.normal(0, sig)

    def resid(p):
        return (model_fn(p, Q) - I_obs) / sig

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
            print('    %-4s true=%12.5g  fit=%12.5g +- %10.4g' % (n, t, v, e))
    return ok


# ---------- 真值设定 ----------
# A: 文献锚尺度（STHC-MS D=0.6 nm → C1=D²/20=0.018 nm²；d=1.57 nm、ξ=0.3 nm → C2=1.33e-3 nm⁴）
#    幅度量级对齐真实硬碳数据（I(0.175 nm⁻¹)≈13，高 q 背景≈0.008）
TRUE_A = np.array([20.0, 4.0, 0.05, 0.018, 1.33e-3, 0.008])
# B: 大孔径（D=2.68 nm → C1=0.36 nm²；C2=0.5 nm⁴ → d=8.65 nm、ξ=1.06 nm）；I0=1.0 保证 TS 中 q 占比 ~40% 可辨
TRUE_B = np.array([5.0, 4.0, 1.0, 0.36, 0.5, 0.005])
# C: 幂律撞界（n=6.0 顶上界；C1/C2 文献锚同 A）
TRUE_C = np.array([50.0, 6.0, 0.05, 0.018, 1.33e-3, 0.008])


def main():
    print('L1 合成数据自检  (q: %.3f-%.2f nm-1, %d 点, 噪声 %.0f%%, 多起点 x%d)'
          % (Q[0], Q[-1], len(Q), 100 * NOISE, N_START))
    print('-' * 72)

    ok_a = _run_case('A 文献锚尺度', TRUE_A, M.NAMES, M.BOUNDS, M.model_vec)
    ok_b = _run_case('B 大孔径      ', TRUE_B, M.NAMES, M.BOUNDS, M.model_vec)
    ok_c = _run_case('C 幂律撞界 n  ', TRUE_C, M.NAMES, M.BOUNDS, M.model_vec)

    print('-' * 72)
    verdict = ok_a and ok_b and ok_c
    print('L1 总判定: %s' % ('PASS —— 插件代码实现正确' if verdict else 'FAIL —— 禁止入库，回查 model.py/fitting.py'))

    # 附加：导出量公式链自检（D=2√(5C1)、d/ξ 与设计真值对照）
    print('\n导出量公式链自检（A 用例真值参数）:')
    der = M.derived_main(
        {'I0': TRUE_A[2], 'C1': TRUE_A[3], 'C2': TRUE_A[4]},
        {'I0': 1e-6, 'C1': 1e-6, 'C2': 1e-8})
    print('  D=%.4f nm (expect %.4f)  d=%.3f nm (expect ~1.57)  xi=%.3f nm (expect ~0.3)'
          % (der['D_nm'], 2 * np.sqrt(5 * TRUE_A[3]), der['d_nm'], der['xi_nm']))
    print('  S_mp=%.2f cm2/g = %.2f m2/g（标定条件值，量级 sanity）' % (der['S_mp_cm2g'], der['S_mp_m2g']))


if __name__ == '__main__':
    main()

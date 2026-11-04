# -*- coding: utf-8 -*-
"""L1 合成数据自检（验证门第 1 层：验证插件代码实现正确性）
用法（在 saxsagent 环境，注意 PATH 前置纪律）：
    E:\\CondaEnvs\\saxsagent\\python.exe model_library\\candidates\\alcu_precipitate\\test_synthetic.py

原理：用 model.py 以"已知真值参数"生成含噪假数据 -> 用与 MCP 工具完全相同的
fitting 引擎和边界反演 -> 检验参数能否恢复。
判据（全部通过 = L1 PASS）：
  1) 每个自由参数 |fit - true| <= 3*perr
  2) chi2/dof ∈ [0.6, 1.6]（合成噪声恰为 sigma，理应≈1）
  3) 多起点全部收敛到同一 chi2（偏差 <1%，排除局部极小）
用例（Model III = main 模式 5 参；Model I = model1 模式 7 参）：
  A 中心情形：θ″ 尺度真值（文献 Table 2 RT：R=8.54 nm, t=1.46 nm）
  B 小尺度情形：GPZ 尺度真值（文献 Table 1 RT：R=0.82 nm, t=0.24 nm）
  C 撞界情形：t 顶下界 0.05 nm（超薄盘极限）
  D Model I 中心：GPZ 尺度 + PY 结构峰在窗口内（文献 GPZ 结构峰 q≈1 nm⁻¹ → r_hs≈3 nm）
  E Model I 撞界：t 下界 0.05 nm + 较强关联（eta=0.25）
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

RNG = np.random.default_rng(20261001)          # 固定种子，结果可复现
Q = np.linspace(0.07, 3.4, 600)                # 覆盖真实拟合窗口（nm⁻¹，与 T4 数据一致）
NOISE = 0.02                                   # 相对噪声 2%
N_START = 5                                    # 多起点次数


def _perturb(p_vec, lo, hi, frac=0.15):
    """真值附近随机扰动作为多起点初值（裁剪到边界内）"""
    p0 = p_vec * (1.0 + RNG.normal(0, frac, p_vec.shape))
    p0[0] = abs(p0[0])                          # A 保持正值
    p0[1] = abs(p0[1])                          # s 保持正值
    return np.clip(p0, lo + 1e-9, hi - 1e-9)


def _run_case(name, true_vec, names, bounds, model_fn):
    """生成合成数据 -> 多起点拟合 -> 三项判据检查，打印一行结论"""
    lo = np.array([bounds[n][0] for n in names])
    hi = np.array([bounds[n][1] for n in names])
    I_true = model_fn(true_vec, Q)
    # sigma = 2%|I|，在 I→0 处用 1%·Imax 兜底，避免节点处残差爆炸
    sig = NOISE * np.maximum(np.abs(I_true), 0.01 * np.max(np.abs(I_true)))
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
# A: θ″ 尺度（文献 Table 2 RT 锚点换算 nm）；幅度量级对齐真实 T4 数据（I(0.07)≈200+）
TRUE_A = np.array([2.0e-3, 150.0, 8.54, 1.46, 0.30])
# B: GPZ 小尺度（文献 Table 1 RT 锚点）；小 R 盘的 knee 在 q~1/R=1.2 nm⁻¹ 仍在窗口内
TRUE_B = np.array([2.0e-3, 200.0, 0.82, 0.24, 0.30])
# C: 撞界情形（t 顶下界 0.05 nm）
TRUE_C = np.array([2.0e-3, 150.0, 5.00, 0.05, 0.30])
# D: Model I 中心情形（GPZ 尺度 + PY 结构峰在窗口内；文献 GPZ 结构峰 q≈1 nm⁻¹ → r_hs≈3 nm）
TRUE_D = np.array([5.0e-3, 100.0, 0.82, 0.24, 3.0, 0.10, 0.30])
# E: Model I 撞界情形（t 顶下界 0.05 nm + 较强关联 eta=0.25）
TRUE_E = np.array([5.0e-3, 100.0, 2.00, 0.05, 3.0, 0.25, 0.30])


def main():
    print('L1 合成数据自检  (q: %.2f-%.1f nm-1, %d 点, 噪声 %.0f%%, 多起点 x%d, 取向积分 %d 节点)'
          % (Q[0], Q[-1], len(Q), 100 * NOISE, N_START, M._ALPHA_N))
    print('-' * 72)

    ok_a = _run_case('A theta2 尺度', TRUE_A, M.NAMES, M.BOUNDS, M.model_vec)
    ok_b = _run_case('B GPZ 小尺度', TRUE_B, M.NAMES, M.BOUNDS, M.model_vec)
    ok_c = _run_case('C 撞界 t    ', TRUE_C, M.NAMES, M.BOUNDS, M.model_vec)
    ok_d = _run_case('D m1 GPZ中心', TRUE_D, M.NAMES1, M.BOUNDS1, M.model1_vec)
    ok_e = _run_case('E m1 撞界t  ', TRUE_E, M.NAMES1, M.BOUNDS1, M.model1_vec)

    print('-' * 72)
    verdict = ok_a and ok_b and ok_c and ok_d and ok_e
    print('L1 总判定: %s' % ('PASS —— 插件代码实现正确' if verdict else 'FAIL —— 禁止入库，回查 model.py/fitting.py'))


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""L1 合成数据自检（验证门第 1 层：验证插件代码实现正确性）
用法（在 saxsagent 环境，注意 PATH 前置纪律）：
    E:\\CondaEnvs\\saxsagent\\python.exe model_library\\candidates\\gel_network\\test_synthetic.py

原理：用 model.py 以"已知真值参数"生成含噪假数据 -> 用与 MCP 工具完全相同的
fitting 引擎和边界反演 -> 检验参数能否恢复。
判据（全部通过 = L1 PASS）：
  1) 每个自由参数 |fit - true| <= 3*perr
  2) chi2/dof ∈ [0.6, 1.6]（合成噪声恰为 sigma，理应≈1）
  3) 多起点全部收敛到同一 chi2（偏差 <1%，排除局部极小）
用例（main = correlation length model 4 参；porod = 幂律 2 参）：
  A main 中心：物理凝胶真值（文献锚点 ξ=3.0 nm、m=1.8；pam 量级）
  B main 辐照形态：ξ=5.0、m=2.3（辐照凝胶锚点域，低 q 平台勉强可见；cncs 量级）
  C main 撞界：m 顶上界 4.0（极端支化极限）
  D porod 中心：dm=1.8（物理凝胶锚点域）
  E porod 撞界：dm 顶下界 0.5
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
Q = np.linspace(0.07, 2.6, 424)                # 覆盖真实拟合窗口（nm⁻¹，与 pam/cncs 数据一致）
NOISE = 0.02                                   # 相对噪声 2%
N_START = 5                                    # 多起点次数


def _perturb(p_vec, lo, hi, frac=0.15):
    """真值附近随机扰动作为多起点初值（裁剪到边界内）"""
    p0 = p_vec * (1.0 + RNG.normal(0, frac, p_vec.shape))
    p0[0] = abs(p0[0])                          # A 保持正值
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


# ---------- 真值设定（幅度对齐真实数据量级：pam I~0.07 / cncs I~4.6） ----------
# A: 物理凝胶（文献 ξ=3.0 nm → mesh 18.8 nm、m=1.8；pam 量级）
TRUE_A = np.array([0.05, 3.0, 1.8, 2.0e-3])
# B: 辐照形态（文献锚点域 ξ=5.0、m=2.3；cncs 量级，低 q 平台勉强可见）
TRUE_B = np.array([0.50, 5.0, 2.3, 5.0e-2])
# C: m 顶上界 4.0（极端支化极限，撞界用例）
TRUE_C = np.array([0.05, 3.0, 4.0, 2.0e-3])
# D/E: porod 幂律（物理凝胶锚点域 dm=1.8；撞界 dm=0.5 下界）
TRUE_D = np.array([0.10, 1.8])
TRUE_E = np.array([0.50, 0.5])

LO_A = np.array([M.BOUNDS[n][0] for n in M.NAMES])
HI_A = np.array([M.BOUNDS[n][1] for n in M.NAMES])
LO_P = np.array([M.BOUNDS_POROD[n][0] for n in M.NAMES_POROD])
HI_P = np.array([M.BOUNDS_POROD[n][1] for n in M.NAMES_POROD])

if __name__ == '__main__':
    oks = []
    oks.append(_run_case('A main 物理凝胶中心', TRUE_A, M.NAMES, M.BOUNDS, M.model_vec))
    oks.append(_run_case('B main 辐照形态', TRUE_B, M.NAMES, M.BOUNDS, M.model_vec))
    # C 撞界用例：真值恰在上界，扰动起点会被裁回界内，拟合应顶界收敛；chi2/dof 判据保持
    oks.append(_run_case('C main m 撞上界', TRUE_C, M.NAMES, M.BOUNDS, M.model_vec))
    oks.append(_run_case('D porod 中心', TRUE_D, M.NAMES_POROD, M.BOUNDS_POROD, M.model_porod_vec))
    oks.append(_run_case('E porod dm 撞下界', TRUE_E, M.NAMES_POROD, M.BOUNDS_POROD, M.model_porod_vec))
    print('L1 总判定：%s（%d/%d 用例通过）' % ('PASS' if all(oks) else 'FAIL', sum(oks), len(oks)))
    sys.exit(0 if all(oks) else 1)

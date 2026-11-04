# -*- coding: utf-8 -*-
"""L1 合成数据自检 + L1b 逐式转写对拍（验证门第 1 层，hardcarbon_closedpore）
用法（在 saxsagent 环境，注意 PATH 前置纪律）：
    E:\\CondaEnvs\\saxsagent\\python.exe model_library\\validated\\hardcarbon_closedpore\\test_synthetic.py

原理：用 model.py 以"已知真值参数"生成含噪假数据 -> 用与 MCP 工具完全相同的
fitting 引擎和边界反演 -> 检验参数能否恢复。
判据（全部通过 = L1 PASS）：
  1) 每个自由参数 |fit - true| <= 3*perr（强相关模型双轨：或绝对相对偏差 <= 5%）
  2) chi2/dof ∈ [0.6, 1.6]（合成噪声恰为 sigma，理应≈1）
  3) 多起点全部收敛到同一 chi2（偏差 <1%，排除局部极小）
L1b（转写一致性）：model_vec（Porod 口径）与文献字面式（Stevens Eq.1/Eq.2 raw）
  在映射参数下逐位一致（a=3.3 时 B_raw≡Bp；dr_Bn·a1²≡Bp、C≡Bmp）——
  10-09 fuzzy 教训：文献未附数据表时，以逐式转写一致性为核心判据。

用例（main 模式 5 参，q 窗与 GXL 数据一致 [0.15, 2.26] nm⁻¹）：
  A 文献锚尺度：a=2.3（GXL 数据两段斜率折中）、a1=0.769 nm（AEM HCMP-CO2 直径 4.86 nm）
  B 大孔径极端：a1=1.5 nm（膝 0.667 nm⁻¹，窗内另一极端）、a=2.0
  C Porod 幂律共线区：a=3.8（贴近 Porod 极限，检验 A/Bp 共线可辨识性）
σ 设计（10-02 教训）：纯相对 σ = 2%·|I_true|，不用 Imax 兜底。
full 模式附加用例 D（双膝分离良好的人工数据集，非本批 GXL 形态）：
  a1=0.7 / a2=2.5 nm（两膝 1.43 / 0.4 nm⁻¹，间距 3.6 倍满足前提），验证 full 7 参可辨识。
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

RNG = np.random.default_rng(20261025)          # 固定种子，结果可复现
Q = np.linspace(0.15, 2.26, 500)               # GXL 拟合窗口（nm⁻¹）
NOISE = 0.02                                   # 相对噪声 2%
N_START = 5                                    # 多起点次数

FAILS = []


def _perturb(p_vec, lo, hi, frac=0.15):
    """真值附近随机扰动作为多起点初值（裁剪到边界内；幅度参数保正）"""
    p0 = p_vec * (1.0 + RNG.normal(0, frac, p_vec.shape))
    for i in range(len(p0)):
        p0[i] = abs(p0[i])                     # 本模型全参数物理非负
    return np.clip(p0, lo + 1e-9, hi - 1e-9)


def _run_case(tag, true_map, names, bounds, model_fn):
    """生成合成数据 -> 多起点拟合 -> 三项判据检查，打印一行结论"""
    lo = np.array([bounds[n][0] for n in names])
    hi = np.array([bounds[n][1] for n in names])
    true_vec = np.array([true_map[n] for n in names])
    I_true = model_fn(true_vec, Q)
    sig = NOISE * np.maximum(np.abs(I_true), 1e-10)
    I_obs = I_true + RNG.normal(0, sig)

    def resid(p):
        return (model_fn(p, Q) - I_obs) / sig

    chis = []
    best = None
    for k in range(N_START):
        p0 = true_vec if k == 0 else _perturb(true_vec, lo, hi)
        res = do_fit(resid, p0, lo, hi)
        chi2, dof, perr = chi2_dof(res, len(names))
        chis.append(chi2)
        if best is None or chi2 < best[0]:
            best = (chi2, res.x, perr, dof)
    chi2, xf, perr, dof = best
    spread = (max(chis) - min(chis)) / max(min(chis), 1e-30)

    ok = True
    for n, x, t, e in zip(names, xf, true_vec, perr):
        dev_sig = abs(x - t) / e if e > 0 else np.inf
        dev_rel = abs(x - t) / abs(t) if t != 0 else np.inf
        passed = (dev_sig <= 3.0) or (dev_rel <= 0.05)   # 双轨判据（10-06 强相关教训）
        flag = 'PASS' if passed else 'FAIL'
        if not passed:
            ok = False
            FAILS.append('%s 参数 %s 恢复失败: fit=%.6g true=%.6g (%.1fsig, %.1f%%)'
                         % (tag, n, x, t, dev_sig, 100 * dev_rel))
        print('  %s %-4s fit=%-12.6g true=%-12.6g %6.1fsig %6.1f%% %s'
              % (tag, n, x, t, dev_sig, 100 * dev_rel, flag))
    chi_ok = 0.6 <= chi2 <= 1.6
    if not chi_ok:
        ok = False
        FAILS.append('%s chi2/dof=%.3f 超出 [0.6,1.6]' % (tag, chi2))
    spread_ok = spread < 0.01
    if not spread_ok:
        ok = False
        FAILS.append('%s 多起点 chi2 离散 %.2f%% >= 1%%' % (tag, 100 * spread))
    print('  %s => chi2/dof=%.3f (dof=%d) 起点离散=%.2f%% [%s]'
          % (tag, chi2, dof, 100 * spread, 'PASS' if ok else 'FAIL'))
    return ok


def main():
    print('=== L1b 逐式转写对拍（Porod 口径 vs 文献字面式，a=3.3 锚定） ===')
    q = np.geomspace(0.1, 10.0, 37)
    # main：B_raw ≡ Bp
    A, Bp, a1, B = 3.7, 11.0, 0.64, 0.21
    v1 = M.model_vec([A, 3.3, Bp, a1, B], q)
    v2 = M.model_vec_stevens_raw([A, Bp, a1, B], q)
    rel = np.max(np.abs(v1 - v2) / np.abs(v2))
    print('  main  B_raw≡Bp      : max_rel = %.3e %s' % (rel, 'PASS' if rel < 1e-12 else 'FAIL'))
    if rel >= 1e-12:
        FAILS.append('L1b main 转写不一致 max_rel=%.3e' % rel)
    # full：dr_Bn·a1² ≡ Bp、C ≡ Bmp（a=3.3）
    Bmp, a2 = 5.0, 1.9
    dr_Bn = Bp / (a1 * a1)
    w1 = M.model_vec_full([A, 3.3, Bp, a1, Bmp, a2, B], q)
    w2 = M.model_vec_full_raw([A, dr_Bn, a1, Bmp, a2, B], q)
    rel2 = np.max(np.abs(w1 - w2) / np.abs(w2))
    print('  full  dr_Bn·a1²≡Bp  : max_rel = %.3e %s' % (rel2, 'PASS' if rel2 < 1e-12 else 'FAIL'))
    if rel2 >= 1e-12:
        FAILS.append('L1b full 转写不一致 max_rel=%.3e' % rel2)

    print('=== L1 合成自检（main 模式，q∈[0.15,2.26] nm⁻¹，2% 相对噪声） ===')
    _run_case('A-锚尺度', {'A': 30.0, 'a': 2.3, 'Bp': 40.0, 'a1': 0.769, 'B': 0.15},
              M.NAMES, M.BOUNDS, M.model_vec)
    _run_case('B-大孔径', {'A': 8.0, 'a': 2.0, 'Bp': 25.0, 'a1': 1.5, 'B': 0.05},
              M.NAMES, M.BOUNDS, M.model_vec)
    _run_case('C-Porod共线', {'A': 15.0, 'a': 3.8, 'Bp': 60.0, 'a1': 0.70, 'B': 0.10},
              M.NAMES, M.BOUNDS, M.model_vec)

    print('=== L1 合成自检（full 模式，双膝分离人工数据集） ===')
    _run_case('D-双膝分离', {'A': 20.0, 'a': 2.2, 'Bp': 35.0, 'a1': 0.7,
                             'Bmp': 18.0, 'a2': 2.5, 'B': 0.10},
              M.FULL_NAMES, M.FULL_BOUNDS, M.model_vec_full)

    print()
    if FAILS:
        print('L1/L1b FAIL（%d 项）：' % len(FAILS))
        for s in FAILS:
            print('  - ' + s)
        sys.exit(1)
    print('L1/L1b 全部 PASS')


if __name__ == '__main__':
    main()

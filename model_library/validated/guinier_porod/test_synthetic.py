# -*- coding: utf-8 -*-
"""L1 合成数据自检（guinier_porod 插件，验证门第 1 层：验证插件代码实现正确性）
用法（在 saxsagent 环境，注意 PATH 前置纪律）：
    E:\\CondaEnvs\\saxsagent\\python.exe model_library\\candidates\\guinier_porod\\test_synthetic.py

原理：用 model.py 以"已知真值参数"生成含噪假数据 -> 用与 MCP 工具完全相同的
fitting 引擎和边界反演 -> 检验参数能否恢复。
判据（全部通过 = L1 PASS）：
  1) 每个自由参数 |fit - true| <= 3*perr（perr≈0 且拟合值==真值时偏差为绝对量，仍须 <3）
  2) chi2/dof ∈ [0.6, 1.6]（合成噪声恰为 sigma，理应≈1）
  3) 多起点全部收敛到同一 chi2（偏差 <1%，排除局部极小）
用例（三形态覆盖 tiered 三窗 + 维度参数 s 的三个取值；膝区均在各自窗内）：
  A level3 型球状：Rg=38 nm, s=0, P=3, B=0，窗 geomspace(0.025, 0.09, 40)，膝 0.056
  B level1 型棒状：Rg=2 nm, s=1, P=4, B=0.01，窗 geomspace(0.4, 2.0, 60)，膝 0.87
  C 片状（膝居中最严苛）：Rg=5.8 nm, s=2, P=3.5, B=0.5，窗 geomspace(0.06, 0.5, 60)，膝 0.149
    （2026-10-06 设计修订：初版窗 [0.1,0.5] 膝区距低 q 端仅 1.5 倍因子，q^-2 纯 Guinier 段
    点数不足 → A/Rg/s 沿简并谷滑动（χ²=0.763 反优于真值、多起点一致落入同一浅谷、
    A 偏差 7.4σ）——Guinier 族经典简并的合成复现；下探低 q 端至 q1/2.5，让 q^-2 斜率段
    有 ~26 点压住 s 后简并解除。教训：GP 类模型 L1 用例必须保证膝下纯 Guinier 段 ≥
    一个十进位的一半，否则 3σ 判据必挂）
σ 设计（lessons E 类 10-02）：纯相对 σ = 2%·|I_true|，不用 Imax 兜底（GP 动态范围
10²–10² 量级在此窗内不大，但纪律统一：纯相对 σ 保证全 q 点权重均衡）。
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

RNG = np.random.default_rng(20261006)          # 固定种子，结果可复现
NOISE = 0.02                                   # 相对噪声 2%
N_START = 5                                    # 多起点次数


def _perturb(p_vec, lo, hi, frac=0.15):
    """真值附近随机扰动作为多起点初值（裁剪到边界内）"""
    p0 = p_vec * (1.0 + RNG.normal(0, frac, p_vec.shape))
    return np.clip(p0, lo + 1e-9, hi - 1e-9)


def _run_case(name, true_vec, Q):
    """生成合成数据 -> 多起点拟合 -> 三项判据检查，打印一行结论"""
    lo = np.array([M.BOUNDS[n][0] for n in M.NAMES])
    hi = np.array([M.BOUNDS[n][1] for n in M.NAMES])
    true_vec = np.asarray(true_vec, dtype=float)
    I_true = M.model_vec(true_vec, Q)
    sig = NOISE * np.maximum(np.abs(I_true), 1e-10)
    I_obs = I_true + RNG.normal(0, sig)

    def resid(p):
        return (M.model_vec(p, Q) - I_obs) / sig

    chis, fits = [], []
    for _ in range(N_START):
        res = do_fit(resid, _perturb(true_vec, lo, hi), lo, hi)
        chi2, dof, perr = chi2_dof(res, len(M.NAMES))
        chis.append(chi2)
        fits.append((res.x, perr))

    best = min(chis)
    multi_ok = all(abs(c - best) / best < 0.01 for c in chis)       # 判据3：多起点一致
    x_best, err_best = fits[int(np.argmin(chis))]

    # 判据1：3σ 恢复（perr 可能因撞界而≈0，此时记绝对偏差，仍须 <3）
    err_safe = np.where(np.asarray(err_best) > 0, err_best, 1.0)
    dev = np.where(np.asarray(err_best) > 0,
                   np.abs(np.asarray(x_best) - true_vec) / err_safe,
                   np.abs(np.asarray(x_best) - true_vec))
    chi_ok = 0.6 <= best <= 1.6                                     # 判据2：chi2/dof 合理
    ok = multi_ok and chi_ok and bool(np.all(dev <= 3.0))

    worst = '%s=%.1fsig' % (M.NAMES[int(np.argmax(dev))], dev.max())
    print('%s [%s]  chi2/dof=%.3f  多起点一致=%s  最差恢复=%s'
          % ('PASS' if ok else 'FAIL', name, best, multi_ok, worst))
    if not ok:
        for n, t, v, e in zip(M.NAMES, true_vec, x_best, err_best):
            print('    %-3s true=%12.5g  fit=%12.5g +- %10.4g' % (n, t, v, e))
    return ok


# ---------- 真值设定（NAMES 顺序：A, Rg, s, P, B） ----------
# A: level3 型（文献锚尺度：Rg=38 nm=380 Å、球形 s=0、P=3 非球形聚集体；窗与 tiered level3 同域）
TRUE_A = np.array([100.0, 38.0, 0.0, 3.0, 0.0])
# B: level1 型棒状（s=1 中间 q 区 q^-1；窗与 tiered level1 同域；B 非零背景）
TRUE_B = np.array([1.0, 2.0, 1.0, 4.0, 0.01])
# C: 片状（s=2、膝 q1=0.149 居窗中最严苛；B 较大检验背景分离）
TRUE_C = np.array([50.0, 5.8, 2.0, 3.5, 0.5])

QA = np.geomspace(0.025, 0.09, 40)
QB = np.geomspace(0.4, 2.0, 60)
QC = np.geomspace(0.06, 0.5, 60)


def main():
    print('L1 合成数据自检  (噪声 %.0f%%, 多起点 x%d, 5 自由参数)'
          % (100 * NOISE, N_START))
    print('-' * 72)

    ok_a = _run_case('A level3 型球状', TRUE_A, QA)
    ok_b = _run_case('B level1 型棒状', TRUE_B, QB)
    ok_c = _run_case('C 片状膝居中  ', TRUE_C, QC)

    print('-' * 72)
    verdict = ok_a and ok_b and ok_c
    print('L1 总判定: %s' % ('PASS —— 插件代码实现正确' if verdict else 'FAIL —— 禁止入库，回查 model.py/fitting.py'))

    # 附加：导出量公式链自检（q1 = sqrt((3-s)(P-s)/2)/Rg 与无量纲积 q1·Rg = sqrt(n·ms)）
    print('\n导出量公式链自检（C 用例真值参数）:')
    der = M.derived_main({'Rg': 5.8, 's': 2.0, 'P': 3.5},
                         {'Rg': 1e-6, 's': 1e-6, 'P': 1e-6})
    print('  q1=%.6f nm^-1 (expect %.6f)   q1*Rg=%.6f (expect sqrt(0.75)=%.6f)'
          % (der['q1_nm1'], np.sqrt(0.75) / 5.8, der['q1_rg'], np.sqrt(0.75)))


if __name__ == '__main__':
    main()

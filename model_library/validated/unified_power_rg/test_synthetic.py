# -*- coding: utf-8 -*-
"""L1 合成数据自检（unified_power_Rg 插件，验证门第 1 层：验证插件代码实现正确性）
用法（在 saxsagent 环境，注意 PATH 前置纪律）：
    E:\\CondaEnvs\\saxsagent\\python.exe model_library\\candidates\\unified_power_rg\\test_synthetic.py

原理：用 model.py 以"已知真值参数"生成含噪假数据 -> 用与 MCP 工具完全相同的
fitting 引擎和边界反演 -> 检验参数能否恢复。
判据（全部通过 = L1 PASS）：
  1) 每个自由参数 |fit - true| <= 3*perr（perr≈0 且拟合值==真值时偏差为绝对量，仍须 <3）
  2) chi2/dof ∈ [0.6, 1.6]（合成噪声恰为 sigma，理应≈1）
  3) 多起点全部收敛到同一 chi2（偏差 <1%，排除局部极小）
用例（13 自由参数 G/Rg/P×4 + B；K_i 由固定关系 K=G·P/Rg^P·Γ(P/2) 在模型内部派生。
2026-10-06 设计修订：初版 17 参数（K_i 自由）存在结构性简并谷——相邻级被压制幂律尾
K·q^-P 与 K'·q^-P' 在有限窗内互相补偿，A 用例 chi2=0.931 完美但 P3 撞上界、
K3 漂 3 个量级、多起点不一致；固定关系后 13 参数简并解除）：
  A 论文锚形态：INIT 全参数（Level4/3/2/1 Rg=150/36/13/2 nm、P=2.9/3.8/3.1/1.0），
    B=0，窗 geomspace(0.005, 2.0, 250)——四膝 0.0115/0.048/0.13/0.87 全覆盖；
    兼作真实拟合 INIT 锚值的合法性预演
  B 小尺度主导型：G 比例反转（G1..G4=0.02/0.2/1.0/6.0）、B=0.01 非零背景——
    检验低 q 弱信号级（Level4）与背景分离
  C 非单调幅度型：G3>G2 反常分配、P 交替 2/4——检验公式普适性（不依赖
    "幅度逐级递减"的先验形态）。
    （2026-10-06 设计修订：初版 G1=0.1/B=0.5 使 Level 1 结构贡献在窗内低于背景
    14 倍（q=2 处 Level1≈0.035 vs B=0.5）→ P1/G1/Rg1 结构性不可辨识，P1 撞上界、
    χ² 却完美（1.062）——信号淹没型不可辨识的合成复现；改 G1=0.5/B=0.05 后
    Level 1 高 q 端贡献占比 ~78%，可辨识性恢复）
σ 设计（lessons E 类 10-02）：纯相对 σ = 2%·|I_true|，全 q 点权重均衡。
级联方向自检：A 用例附加检查拟合曲线形态——低 q 平台≈ΣG、最高 q 端斜率≈P1
（若级联方向写反，中间级尾被压死、高 q 端由 K4·q^-P4 主导，形态与恢复双双失败）。
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


def _vec(g1, rg1, p1, g2, rg2, p2, g3, rg3, p3, g4, rg4, p4, b):
    """按 NAMES 顺序组装 13 长度真值向量（K_i 由固定关系在模型内部派生）"""
    return np.array([g1, rg1, p1, g2, rg2, p2, g3, rg3, p3, g4, rg4, p4, b],
                    dtype=float)


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
            print('    %-4s true=%12.5g  fit=%12.5g +- %10.4g' % (n, t, v, e))
    return ok, x_best


# ---------- 真值设定（NAMES 顺序：G1,Rg1,P1,G2,Rg2,P2,G3,Rg3,P3,G4,Rg4,P4,B） ----------
# A: 论文锚形态（INIT 全参数 + B=0）
TRUE_A = _vec(M.INIT['G1'], M.INIT['Rg1'], M.INIT['P1'],
              M.INIT['G2'], M.INIT['Rg2'], M.INIT['P2'],
              M.INIT['G3'], M.INIT['Rg3'], M.INIT['P3'],
              M.INIT['G4'], M.INIT['Rg4'], M.INIT['P4'], 0.0)
# B: 小尺度主导型（G 比例反转 + 非零背景）
TRUE_B = _vec(0.02, 3.0, 1.5, 0.2, 20.0, 2.0, 1.0, 60.0, 3.5, 6.0, 200.0, 2.5, 0.01)
# C: 非单调幅度型（G3>G2、P 交替；G1=0.5/B=0.05 保证 Level 1 高 q 端贡献占比 ~78%
#    可辨识——初版 G1=0.1/B=0.5 信号淹没于背景致 P1 撞界假 FAIL）
TRUE_C = _vec(0.5, 1.0, 2.0, 1.0, 5.0, 4.0, 0.8, 25.0, 2.0, 30.0, 120.0, 3.5, 0.05)

QA = np.geomspace(0.005, 2.0, 250)
QB = np.geomspace(0.005, 2.0, 250)
QC = np.geomspace(0.005, 2.0, 250)

# ---------- D: full 模式（13 统一 + 9 峰 = 22 自由参数，2026-10-06 增） ----------
# 网格：低 q 下探 0.005（Level 4 膝 q_knee=√3/150=0.0115 之下，平台必须完整——
# 初版 0.02 起步平台缺半截，G4/Rg4/P4 与峰参数互锁补偿 χ²=1.02 却全漂，
# 与 10-06 GP 用例"膝下纯段不足"教训同族）+ WAXS 峰区 linspace 加密
# （narrow FWHM 0.2 nm^-1 内 ≥8 点）。
# 真值形态=峰凸显型（P1=3.5 快衰减尾、峰高按真实数据量级）——初版直接拼
# TRUE_A（P1=1 慢尾）外推到 WAXS 区尾部达 2.7e-3、峰净高仅占 1.5%，峰淹没
# 于尾 → 拟合用"峰标签互换+(012) 撞下界"等效解 χ²=1.017 假 FAIL（信号淹没
# 第 3 变体：真值形态外推致峰淹没，与 10-02/10-06 用例设计教训同族）。
# L1 只验代码正确性（各组分可辨条件下反演无偏）；真实数据的可辨识性由
# L2 multistart 取证，不属于本用例职责。
TRUE_D = {'G1': 0.02, 'Rg1': 2.0, 'P1': 3.5,
          'G2': 0.5, 'Rg2': 13.0, 'P2': 3.1,
          'G3': 8.0, 'Rg3': 36.0, 'P3': 3.8,
          'G4': 100.0, 'Rg4': 150.0, 'P4': 2.9,
          'B': 0.0,
          'A_b': 3.0e-4, 'c_b': 2.7, 'w_b': 0.8,
          'A_n': 3.0e-4, 'c_n': 2.9, 'w_n': 0.2,
          'A_0': 1.0e-4, 'c_0': 6.6, 'w_0': 0.4}
QD = np.concatenate([np.geomspace(0.005, 2.0, 200), np.linspace(2.0, 10.0, 350)])


def _run_case_full(name, true_d, Q):
    """full 版逐用例：22 参数反演。判据针对 22 参数强相关景观修订（probe 取证
    2026-10-06，Temp/sasmodels_survey/full_l1_probe.py：10 起点 9 个收敛同一谷
    χ²=0.96559 且参数实际恢复全部 ≤1%——代码正确性成立；但 ①强相关参数的
    Hessian 病态使单参数 perr 系统性低估，0.6% 的正确恢复被 perr 归一虚标
    8.4σ，②偶发起点落入"窄峰未起"差谷 χ²=2.38）：
      判据1 双轨：perr 归一 ≤3σ **或**绝对相对偏差 ≤5%（perr 低估防护）
      判据3 多数一致：≥80% 起点收敛 best χ² 的 1% 内（不要求 5/5 全一致，
        但任何"同 χ² 异参数"解仍会被判据 1 的 rel_dev 拦截）"""
    names = M.NAMES_FULL
    # 注意：dict.get(k, default) 的 default 急切求值，不能写
    # BOUNDS.get(n, BOUNDS_PEAK[n])（对统一参数会先 KeyError）
    _b = [M.BOUNDS[n] if n in M.BOUNDS else M.BOUNDS_PEAK[n] for n in names]
    lo = np.array([b[0] for b in _b])
    hi = np.array([b[1] for b in _b])
    true_vec = np.array([true_d[n] for n in names], dtype=float)
    I_true = M.model_full(true_vec, Q)
    sig = NOISE * np.maximum(np.abs(I_true), 1e-10)
    I_obs = I_true + RNG.normal(0, sig)

    def resid(p):
        return (M.model_full(p, Q) - I_obs) / sig

    chis, fits = [], []
    for _ in range(N_START):
        res = do_fit(resid, _perturb(true_vec, lo, hi), lo, hi)
        chi2, dof, perr = chi2_dof(res, len(names))
        chis.append(chi2)
        fits.append((res.x, perr))

    best = min(chis)
    n_same = sum(1 for c in chis if abs(c - best) / best < 0.01)
    multi_ok = n_same / len(chis) >= 0.8
    x_best, err_best = fits[int(np.argmin(chis))]
    err_safe = np.where(np.asarray(err_best) > 0, err_best, 1.0)
    dev = np.where(np.asarray(err_best) > 0,
                   np.abs(np.asarray(x_best) - true_vec) / err_safe,
                   np.abs(np.asarray(x_best) - true_vec))
    rel_dev = (np.abs(np.asarray(x_best) - true_vec)
               / np.where(np.abs(true_vec) > 0, np.abs(true_vec), 1.0))
    chi_ok = 0.6 <= best <= 1.6
    dev_ok = bool(np.all((dev <= 3.0) | (rel_dev <= 0.05)))
    ok = multi_ok and chi_ok and dev_ok
    worst = '%s=%.1fsig(%.1f%%)' % (names[int(np.argmax(rel_dev))],
                                    dev[int(np.argmax(rel_dev))],
                                    100 * rel_dev[int(np.argmax(rel_dev))])
    print('%s [%s]  chi2/dof=%.3f  多起点一致=%d/%d  最差恢复=%s'
          % ('PASS' if ok else 'FAIL', name, best, n_same, len(chis), worst))
    if not ok:
        for n, t, v, e, rd in zip(names, true_vec, x_best, err_best, rel_dev):
            print('    %-4s true=%12.5g  fit=%12.5g +- %10.4g (rel %.1e)'
                  % (n, t, v, e, rd))
    return ok, x_best


def main():
    print('L1 合成数据自检  (噪声 %.0f%%, 多起点 x%d, %d 自由参数)'
          % (100 * NOISE, N_START, len(M.NAMES)))
    print('-' * 72)

    ok_a, xa = _run_case('A 论文锚形态  ', TRUE_A, QA)
    ok_b, _ = _run_case('B 小尺度主导型', TRUE_B, QB)
    ok_c, _ = _run_case('C 非单调幅度型', TRUE_C, QC)
    ok_d, _ = _run_case_full('D 全谱统一+峰', TRUE_D, QD)

    print('-' * 72)
    verdict = ok_a and ok_b and ok_c and ok_d
    print('L1 总判定: %s' % ('PASS —— 插件代码实现正确' if verdict else 'FAIL —— 禁止入库，回查 model.py/fitting.py'))

    # 附加：级联方向形态自检（A 用例真值曲线）——
    # 低 q 平台 ≈ ΣG + B；高 q 端（q≫全部膝）斜率 ≈ -P1（K1 无压制主导）
    print('\n级联方向形态自检（A 用例真值）:')
    q_lo = np.array([1e-4])
    q_hi = np.array([50.0])
    plat = float(M.model_vec(TRUE_A, q_lo)[0])
    tail = float(M.model_vec(TRUE_A, q_hi)[0])
    k1_true = M.k_of(M.INIT['G1'], M.INIT['Rg1'], M.INIT['P1'])
    print('  I(q→0)=%.4g (ΣG+B=%.4g)  I(q=50)=%.4g  K1·q^P1@50=%.4g (K1 true=%.4g)'
          % (plat, sum(M.INIT['G%d' % i] for i in range(1, 5)), tail,
             tail * 50.0 ** float(M.INIT['P1']), k1_true))

    # 附加：导出量公式链自检（rg_A = rg×10；k_i 派生量非自检）
    der = M.derived_main(dict(zip(M.NAMES, TRUE_A)), dict((n, 1e-6) for n in M.NAMES))
    print('  derived rg3_A=%.2f (expect %.2f)' % (der['rg3_A'], M.INIT['Rg3'] * 10.0))


if __name__ == '__main__':
    main()

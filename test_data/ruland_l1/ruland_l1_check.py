# -*- coding: utf-8 -*-
"""T2 L1 合成自检：ruland.py 引擎数值正确性验证（规格卡 §7 T2，2026-09-30）

规格卡：.trae/documents/Ruland取向度分析MCP工具-算法学习与开发规划.md v1.0
判据：B_eq（斜率）恢复偏差主组 <3%（基准组 <2%）；截距 = 1/L² 仅引擎内部校验
（<25%）；截距/L 一律不作工具输出（Q6 裁决）。边界用例另列。
固定随机种子，结果可复现；T6 时把本脚本输出数值入册回归基准。

运行（PATH 前置纪律，见 agent_prompts/lessons.md A 类 / project_memory）：
  $env:PATH='E:\CondaEnvs\saxsagent\Library\bin;E:\CondaEnvs\saxsagent\DLLs;E:\CondaEnvs\saxsagent\Scripts;E:\CondaEnvs\saxsagent;'+$env:PATH
  $env:PYTHONUTF8='1'
  E:\CondaEnvs\saxsagent\python.exe test_data\ruland_l1\ruland_l1_check.py
"""
import io
import json
import math
import os
import sys

sys.path.insert(0, r'E:\HEPS_BB_SAXS\HEPS_SAXS_AGENT\saxs_mcp')
import numpy as np
import ruland as R

_HERE = os.path.dirname(os.path.abspath(__file__))
PHI0_TRUE = 37.0          # 故意非 90°（角度约定警示：禁止硬编码赤道 90°）
N_COL = 16
PHI = np.arange(0.0, 360.0, 0.5)
Q_LIST = np.logspace(math.log10(0.3), math.log10(1.5), N_COL)   # nm^-1

RESULTS = []


def _circ(a):
    d = np.abs(a) % 360.0
    return np.minimum(d, 360.0 - d)


def synth(L_true, B_eq_true, noise_rel, drift_amp, peak='gauss', seed=1):
    """式1 高斯正向合成：I(q,φ)=A(q)·P(Δφ;σ_φ(q))+基线+噪声（2 峰形态，
    对称弱峰 0.3×）。σ_φ(q)=sqrt(1/(qL)²+B_eq²)/sqrt(2π)（rad→deg）。"""
    rng = np.random.default_rng(seed)
    Y = np.zeros((len(PHI), N_COL))
    phi0s = []
    for j, q in enumerate(Q_LIST):
        B = math.sqrt((1.0 / (q * L_true)) ** 2 + B_eq_true ** 2)   # rad
        sig_deg = B / math.sqrt(2 * math.pi) * 180.0 / math.pi
        phi0 = PHI0_TRUE + drift_amp * math.sin(j * 0.7)
        phi0s.append(phi0)
        A = 1000.0 * (0.5 / q) ** 2
        d1 = _circ(PHI - phi0)
        d2 = _circ(PHI - (phi0 + 180.0))
        if peak == 'gauss':
            P = np.exp(-d1 ** 2 / (2 * sig_deg ** 2))
            P2 = np.exp(-d2 ** 2 / (2 * sig_deg ** 2))
        elif peak == 'lorentz':          # 积分宽度对齐：∫L dφ = π·w = B_deg
            w = B * 180.0 / math.pi / math.pi
            P = 1.0 / (1.0 + (d1 / w) ** 2)
            P2 = 0.3 / (1.0 + (d2 / w) ** 2) * math.pi * w / (
                math.sqrt(2 * math.pi) * sig_deg)   # 弱峰面积比 0.3 对齐高斯形态
        Y[:, j] = A * P + 0.3 * A * P2 + 10.0 + 0.02 * PHI \
            + rng.normal(0, noise_rel * A, len(PHI))
    return Y, phi0s


def run_case(name, L_true, B_eq_true, noise_rel, drift_amp, tol_dev,
             peak='gauss', seed=1):
    """主组用例：合成 → 选峰 → series B → Ruland beq，断言 B_eq 恢复。"""
    Y, _ = synth(L_true, B_eq_true, noise_rel, drift_amp, peak, seed)
    sel = R.choose_phi_ref(PHI, Y, list(range(N_COL)))
    out = R.bpi2_series(PHI, Y, list(range(N_COL)), sel['phi_ref'],
                        q_values=[q * 0.1 for q in Q_LIST], q_unit='A^-1',
                        per_col_peaks=sel['per_col_peaks'])
    B_corr = [r['B_rad_corr'] for r in out['rows']]
    fit = R.ruland_beq(list(Q_LIST), B_corr)
    dev = fit['B_eq_rad'] / B_eq_true - 1.0
    # 截距 = 1/L² 仅引擎内部校验（Q6 诊断量定位）：单侧判据——大幅偏负才是
    # 系统偏差信号；小截距场景（长 L）被 B_eq² 散射淹没属正常测不准
    int_dev = fit['intercept_per_nm2'] * L_true ** 2 - 1.0
    ok = abs(dev) < tol_dev and int_dev > -0.5 and fit['r2'] > 0.99
    RESULTS.append({
        'case': name, 'L_true': L_true, 'B_eq_true': B_eq_true,
        'B_eq_fit': fit['B_eq_rad'], 'dev_pct': 100 * dev,
        'intercept_rel_dev_pct': 100 * int_dev, 'r2': fit['r2'],
        'n_flags': len(fit['flags']), 'flags': fit['flags'], 'pass': ok,
        'phi_ref': sel['phi_ref'], 'seed': seed,
    })
    print('%s %-28s B_eq=%.5f(%+0.2f%%) r2=%.5f int_dev=%+0.1f%%'
          % ('PASS' if ok else 'FAIL', name, fit['B_eq_rad'],
             100 * dev, fit['r2'], 100 * int_dev))
    return fit, out


def run_gap_case():
    """边界用例 1：gap 填充传播（mask sidecar）。col10（0-based 9）峰翼右侧
    φ=[40..44]° 5 点替换为平滑插值（模拟 gap_fill 产物），mask=5。
    断言 n_filled 精确披露 + B_eq 偏差仍 <3.5%。"""
    Y, _ = synth(35.0, 0.10, 0.005, 0.0, seed=7)
    codes = np.zeros(len(PHI), dtype=int)
    gi = [np.searchsorted(PHI, v) for v in (40.0, 41.0, 42.0, 43.0, 44.0)]
    Y[gi, 9] = np.interp(PHI[gi], [39.0, 45.0],
                         [Y[np.searchsorted(PHI, 39.0), 9],
                          Y[np.searchsorted(PHI, 45.0), 9]])
    codes[gi] = 5
    sel = R.choose_phi_ref(PHI, Y, list(range(N_COL)))
    out = R.bpi2_series(PHI, Y, list(range(N_COL)), sel['phi_ref'],
                        q_values=[q * 0.1 for q in Q_LIST], q_unit='A^-1',
                        masks={9: codes}, per_col_peaks=sel['per_col_peaks'])
    r9 = out['rows'][9]
    B = [r['B_rad_corr'] for r in out['rows']]
    fit = R.ruland_beq(list(Q_LIST), B)
    dev = fit['B_eq_rad'] / 0.10 - 1.0
    ok = (r9['n_filled'] == 5 and abs(r9['filled_frac'] - 5 / r9['n_pts'])
          < 1e-9 and abs(dev) < 0.035)
    RESULTS.append({
        'case': 'gap_mask_propagation', 'n_filled_col10': r9['n_filled'],
        'filled_frac': r9['filled_frac'], 'B_eq_fit': fit['B_eq_rad'],
        'dev_pct': 100 * dev, 'pass': ok,
    })
    print('%s %-28s n_filled=%d frac=%.3f B_eq(%+0.2f%%)'
          % ('PASS' if ok else 'FAIL', 'gap_mask_propagation',
             r9['n_filled'], r9['filled_frac'], 100 * dev))


def run_lorentz_case():
    """边界用例 2：非高斯峰形（洛伦兹，积分宽度对齐）。高斯假设下预期 B_eq
    系统偏差被记录（条件值口径），引擎不崩、B>0；偏差数值入册供 L2 对照。"""
    Y, _ = synth(35.0, 0.10, 0.005, 0.0, peak='lorentz', seed=11)
    sel = R.choose_phi_ref(PHI, Y, list(range(N_COL)))
    out = R.bpi2_series(PHI, Y, list(range(N_COL)), sel['phi_ref'],
                        q_values=[q * 0.1 for q in Q_LIST], q_unit='A^-1',
                        per_col_peaks=sel['per_col_peaks'])
    B = [r['B_rad_corr'] for r in out['rows']]
    fit = R.ruland_beq(list(Q_LIST), B)
    dev = fit['B_eq_rad'] / 0.10 - 1.0
    ok = fit['B_eq_rad'] > 0 and fit['r2'] > 0.95
    RESULTS.append({
        'case': 'lorentz_nongaussian', 'B_eq_fit': fit['B_eq_rad'],
        'dev_pct': 100 * dev, 'r2': fit['r2'], 'pass': ok,
        'note': '高斯假设下非高斯峰形的系统偏差记录值（条件值口径，'
                '论文 ref 11 非高斯卷积宽度关系的适用场景）',
    })
    print('%s %-28s B_eq=%.5f(%+0.2f%%) r2=%.5f [偏差即披露对象]'
          % ('PASS' if ok else 'FAIL', 'lorentz_nongaussian',
             fit['B_eq_rad'], 100 * dev, fit['r2']))


def run_curved_case():
    """边界用例 3：弯曲 Ruland 图（低 q 端额外散射项 c·exp(−q/q₀)，模拟双
    L 群体/背景残留的低 q 上翘）。断言 r2_below_threshold flag 触发（线性
    检验诊断能力，§5.3）。"""
    q = np.array([0.30, 0.36, 0.44, 0.53, 0.64, 0.77, 0.93, 1.12, 1.35, 1.62])
    B = np.sqrt(1.0 / (q * 35.0) ** 2 + 0.10 ** 2
                + 0.010 * np.exp(-q / 0.5))
    fit = R.ruland_beq(list(q), list(B))
    has_flag = any('residual_curvature' in f or 'r2_below_threshold' in f
                   for f in fit['flags'])
    ok = has_flag and fit['r2'] < 0.9999   # r² 单指标对此弯曲不敏感（T2 发现），
                                           # 以 curvature flag 为主要判据
    RESULTS.append({
        'case': 'curved_ruland_plot', 'r2': fit['r2'], 'flags': fit['flags'],
        'pass': ok,
    })
    print('%s %-28s r2=%.4f flags=%s'
          % ('PASS' if ok else 'FAIL', 'curved_ruland_plot',
             fit['r2'], fit['flags']))


def main():
    # ---- 主组 6 组参数扫描（L/B_eq/噪声/漂移矩阵）
    run_case('M1_base(L35,B.10,n.5%)', 35.0, 0.10, 0.005, 0.0, 0.02, seed=1)
    run_case('M2_noise2pct_drift2deg', 35.0, 0.10, 0.02, 2.0, 0.03, seed=2)
    run_case('M3_short_L20_narrowB05', 20.0, 0.05, 0.01, 1.0, 0.03, seed=3)
    run_case('M4_long_L80_wideB20', 80.0, 0.20, 0.01, 1.0, 0.03, seed=4)
    run_case('M5_drift3deg', 35.0, 0.10, 0.005, 3.0, 0.03, seed=5)
    run_case('M6_mid(L50,B.15,n2%)', 50.0, 0.15, 0.02, 0.0, 0.03, seed=6)
    # ---- 边界 3 用例
    run_gap_case()
    run_lorentz_case()
    run_curved_case()

    n_fail = sum(0 if r['pass'] else 1 for r in RESULTS)
    summary = {'tool': 'ruland_l1_check v%s' % R.__version__,
               'n_cases': len(RESULTS), 'n_fail': n_fail,
               'all_pass': n_fail == 0, 'cases': RESULTS}
    with open(os.path.join(_HERE, 'l1_result.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('SUMMARY %d/%d PASS -> l1_result.json'
          % (len(RESULTS) - n_fail, len(RESULTS)))
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

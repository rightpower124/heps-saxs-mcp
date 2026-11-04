# -*- coding: utf-8 -*-
"""拟合窗口建议引擎（部署优化，2026-11-04）

背景：旧缺省窗 q∈[0.3, 4.4] nm^-1 对所有曲线一刀切，与"每条曲线按自身数据
定窗"的要求不符。用户裁决（2026-11-04）：初始默认 = 用户数据的完整范围，
再按低 q 直通光、高 q 噪声/震荡证据适当扣除——采纳方式为"建议 + 确认"：
本引擎只产生建议窗 + 证据 flag，由 Agent 呈卡、用户确认后作为显式 qmin/qmax
传给拟合工具；不做静默自动采纳（与"不做猜测"纪律一致）。

判据（证据驱动、保守，全部在 log-log 域，阈值为绝对判据不依赖基线自比）：
  低 q 直通光 —— 从低 q 端起滑动窗口局部斜率：
    平台型直通光翼：|s| < 0.3（强度不随 q 衰减的束流泄漏翼）
    陡于 Porod 上翘：s < -4.5（Porod 极限 4 + 余量，源自 lessons 261006 幂律上界纪律）
    命中则建议 qmin = 首个斜率回到物理区间 [-4.5, -0.5] 的窗口起点；未命中不裁剪。
  高 q 噪声地板 —— 双判据：
    ① σ/I 相对误差 > 0.3 的尾部区（加权数据；unweighted 数据自动跳过）
    ② 高频震荡：逐点 zigzag 变号占比（相邻一阶差分符号翻转）> 0.6 的连续尾区
       ——平滑曲线 ≈0、噪声主导 ≈0.5+；不用二阶差分幅度（261104 探针取证：
       它把"曲线趋平的真实曲率"误判为震荡）。震荡区不吃掉多半数据才建议内收。
    命中则建议 qmax = 最后一个干净点（双判据同时命中取更内者并分别披露）。
  无任何 flag 时建议窗 = 完整数据范围。

本引擎为启发式证据呈现，不替代用户判断；窗口的物理合理性（如避开结构峰、
保留模型特征区）仍由用户确认卡裁决。
"""
import numpy as np

_SLOPE_LO, _SLOPE_HI = -4.5, -0.5   # 物理斜率区间（Porod 极限 4 + 余量）
_PLATEAU = 0.3                      # 平台型直通光翼判据 |s| < 0.3
_SIG_REL = 0.3                      # σ/I 噪声地板判据
_ZIGZAG_TH = 0.6                    # 高频震荡判据：窗口内相邻一阶差分变号占比
_WIN_FRAC = 20                      # 局部斜率/震荡窗口 = n // 20（下限 5/8 点）


def _slope_at(lx, ly, i0, win):
    """窗口 [i0, i0+win) 的 log-log 斜率；点不足或含非有限值返回 None"""
    j1 = min(i0 + win, len(lx))
    if j1 - i0 < 3:
        return None
    x, y = lx[i0:j1], ly[i0:j1]
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None
    return float(np.polyfit(x, y, 1)[0])


def _low_q_evidence(q, I):
    """低 q 直通光证据 → (flag, detail, cut_q or None)"""
    pos = I > 0
    if pos.sum() < 8:
        return None, '正强度点不足，跳过低 q 判据', None
    lx, ly = np.log(q[pos]), np.log(I[pos])
    qpos = q[pos]
    win = max(5, len(lx) // _WIN_FRAC)
    s0 = _slope_at(lx, ly, 0, win)
    if s0 is None:
        return None, '低 q 端含非有限值，跳过低 q 判据', None
    if abs(s0) < _PLATEAU:
        flag = 'direct_beam_plateau'
    elif s0 < _SLOPE_LO:
        flag = 'steeper_than_porod'
    else:
        return None, '低 q 端斜率 %.2f 在物理区间内，无直通光证据' % s0, None
    # 从低 q 端向内找首个斜率回物理区的窗口起点
    cut = None
    for i0 in range(0, max(1, len(lx) - 3)):
        s = _slope_at(lx, ly, i0, win)
        if s is not None and _SLOPE_LO <= s <= _SLOPE_HI:
            cut = float(qpos[i0])
            break
    detail = ('低 q 端 log-log 斜率 %.2f（%s），首个物理区间窗口起点 q=%.4g'
              % (s0, '平台翼' if flag == 'direct_beam_plateau' else '陡于 Porod', cut))
    return flag, detail, cut


def _high_q_evidence(q, I, sig, unweighted):
    """高 q 噪声地板证据 → (flag, detail, cut_q or None)。
    双判据绝对阈：σ/I 与 zigzag 变号占比；命中区间不吃掉多半数据才建议内收。"""
    pos = I > 0
    if pos.sum() < 12:
        return None, '正强度点不足，跳过高 q 判据', None
    qpos, Ipos = q[pos], I[pos]
    n = len(qpos)
    flags, cut = [], None
    # ① σ/I 噪声地板（加权数据；unweighted 的 σ≡1 无意义，跳过）
    if not unweighted and sig is not None:
        sr = np.asarray(sig, float)[pos] / Ipos
        ok = sr <= _SIG_REL
        if n >= 4 and not ok[-(n // 4):].all():   # 尾部 1/4 存在超阈
            flags.append('sigma_inflated(σ/I>%.1f)' % _SIG_REL)
            if ok.any():
                cut = float(qpos[int(np.max(np.nonzero(ok)[0]))])
    # ② 高频震荡：zigzag 变号占比（平滑曲线≈0，噪声主导≈0.5+）。
    #    从高 q 端向内找连续超阈尾区；不吃掉多半数据才建议内收。
    y = np.log(Ipos)
    dy = np.diff(y)
    zig = (dy[:-1] * dy[1:]) < 0.0
    W = max(8, n // _WIN_FRAC)
    if len(zig) > W:
        i = len(zig)
        while i - W >= 0 and float(zig[i - W:i].mean()) > _ZIGZAG_TH:
            i -= 1
        if i < len(zig):
            flags.append('high_freq_oscillation(尾区zigzag变号占比>%.1f)' % _ZIGZAG_TH)
            c = float(qpos[min(i, n - 1)])
            if c > qpos[n // 2]:
                cut = c if cut is None else min(cut, c)
    if not flags:
        return None, '高 q 端 σ/I 与高频震荡均在阈下，无噪声地板证据', None
    detail = '；'.join(flags) + ('，建议内收至 q=%.4g' % cut if cut is not None
                                 else '（命中区未成建议内收的连续尾区，仅披露 flag）')
    return '; '.join(flags), detail, cut


def suggest_window(q, I, sig, unweighted=False):
    """全范围数据 → 窗口建议卡（建议+确认纪律：只建议不采纳）。
    返回 {q_full_range, n_points, suggested_window, evidence, note}。"""
    q, I = np.asarray(q, float), np.asarray(I, float)
    sig = np.asarray(sig, float) if sig is not None else None
    card = {'n_points': int(len(q)),
            'q_full_range': [float(q[0]), float(q[-1])],
            'suggested_window': [float(q[0]), float(q[-1])],
            'evidence': {}, 'note': ''}
    f_low, d_low, cut_lo = _low_q_evidence(q, I)
    f_high, d_high, cut_hi = _high_q_evidence(q, I, sig, unweighted)
    card['evidence']['low_q'] = {'flag': f_low, 'detail': d_low, 'cut_to': cut_lo}
    card['evidence']['high_q'] = {'flag': f_high, 'detail': d_high, 'cut_to': cut_hi}
    lo = cut_lo if cut_lo is not None else float(q[0])
    hi = cut_hi if cut_hi is not None else float(q[-1])
    if lo >= hi:
        lo, hi = float(q[0]), float(q[-1])
        card['note'] = '证据窗口自洽性检查失败，回退完整范围（请人工判读证据）'
    else:
        card['suggested_window'] = [lo, hi]
        if f_low is None and f_high is None:
            card['note'] = '无裁剪证据，建议窗 = 完整数据范围'
        else:
            card['note'] = ('建议窗为证据扣除结果（初值引擎 A0/B0 依赖窗口结构），'
                            '请确认后作为显式 qmin/qmax 传入拟合工具')
    card['note'] += '。本卡仅为建议（建议+确认纪律），最终窗口由用户裁决。'
    return card

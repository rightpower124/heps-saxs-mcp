# -*- coding: utf-8 -*-
"""Ruland 取向度分析引擎（orientation v0.1.0，T1 纯函数内核）

规格卡：.trae/documents/Ruland取向度分析MCP工具-算法学习与开发规划.md v1.0（2026-09-30 冻结）
论文：Thünemann & Ruland, Macromolecules 2000, 33, 1848–1852, DOI 10.1021/ma991427x
（论文记号 s → 本项目 q；论文 φ 从纤维轴量起、赤道=π/2；实验站 φ 轴零点取决于试样
安置，本引擎一律以实际峰中心 φ₀ 为准，绝不硬编码 90°——规格卡 §2.1 角度约定警示。）

职责（T1，规格卡 §7）：series 级统一选峰（Q11）→ 逐列触达式峰积分（Q12/Q4/Q5）→
B_π/2 表 → Ruland 变换 + 无权重 OLS（§5.3）→ B_eq。
不 import mcp、不注册工具；工具薄壳 saxs_ruland_bpi2 / saxs_ruland_beq 在 T3 落地
tools_ruland.py（注册 + 写盘 + QC 图 + 确认卡）。

裁决映射（规格卡 §6）：
  Q2=A 两工具链 | Q3 触达式边界+全系列统一选峰+gap 前置 | Q4=B 线性基线 |
  Q5=B 峰区高斯拟合峰值 | Q6=L 不产出、截距仅诊断量 | Q7=A 只出 B_eq |
  Q8=A v1 无 σ_B | Q11=C 自动聚类+可覆盖 | Q12=C 主判 f×H + 下限 n×σ + m 点防抖 |
  Q13=A 缺峰硬报错+披露对称侧候选

硬纪律：
  - 填充传播披露（§1.3/§5.2 步5）：mask sidecar 可选传入（gap_fill 各列 _mask.csv 的
    code 数组，5=filled）；积分区内 filled 计数逐列披露；未提供 mask 时 filled=None
    并披露"未统计"——条件值口径不因缺席 mask 而豁免
  - 高斯尾校正（§5.2 步2）：f 触达 ≈ 截断于 k=√(2·ln(1/f)) σ；截断高斯积分宽度
    （面积/峰高）= 全宽 × erf(k/√2) = 全宽 × erf(√(ln(1/f)))，故
    B_corr = B_meas / erf(√(ln(1/f)))；f=0.02 → k=2.797、×1.005（≈0.5% 尾部；
    2026-09-30 Q12 修订：缺省 f 由 0.05 改 0.02——YR8 取证真实峰尾略重于高斯）
  - 线性性成立本身即高斯近似有效性的证据（论文原文 §4）；截距 < 0 = 系统偏差信号
    （Q6 裁决：背景残留 / 线性区不当 / B 口径不一致），触发复核提示；
    截距与 L_app 不进任何结论输出，L 一律不产出（L=2π/q 原则）
  - 单位：B 内部统一 rad（输出 deg/rad 双单位）；q 内部统一 nm⁻¹（A^-1 → ×10 由
    工具壳完成，引擎不猜单位——宁可报错不要猜错）
  - 360° 周期性：峰位聚类 / 容差窗 / 峰心距离一律走圆周距离 min(|Δ|, 360−|Δ|)
  - 缺峰硬报错（Q13）：φ_ref 容差窗内无有效峰 → 报错并列出对称侧候选（±180° 附近），
    绝不静默跳列、绝不自动换侧
"""
import json
import math

import numpy as np

from gap_fill import _read_columns, _sliding_median  # 复用既有解码/读盘/平滑链

__version__ = '0.2.0'

TOOL_TAG = 'saxs_ruland v%s' % __version__

_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi

# 取向度 f_v 系数（Thünemann & Ruland 2000 公式 (3)：f_v = 1 − 2√(ln2/π³)·B_eq，
# B_eq 须代入弧度制；f_v=0 各向同性随机，f_v=1 完全取向。2026-09-30 用户定调
# 固化为 beq 正式产物）
_FV_COEF = 2.0 * math.sqrt(math.log(2.0) / math.pi ** 3)

_DEFAULTS = {
    # ---- 触达判据（Q12 裁决 C：A 主 + B 下限 + 防抖），全系列统一并披露
    'f_touch': 0.02,     # 主判：I−bl ≤ f×(I_peak−bl)；f=0.02 → 截断于 2.797σ
                         # （2026-09-30 Q12 修订：0.05→0.02，YR8 真实数据 f 敏感性
                         # 取证峰尾略重于高斯，高强度列 B +1~2.4%，见规格卡修订记录）
    'n_noise': 3.0,      # 下限：I−bl ≤ n×σ_local（噪声地板，防弱峰边界过宽）
    'm_hold': 3,         # 防抖：触达后须保持 m 点（或至数据端点）才算真触达
    # ---- series 级选峰（Q11 裁决 C：自动聚类，φ_ref 可由参数覆盖）
    'cluster_tol_deg': 5.0,   # 跨列聚类容差（圆周 deg）
    'peak_tol_deg': 5.0,      # 逐列峰位容差窗（φ_ref ± 此值，圆周）
    'prominence_k': 5.0,      # 峰显著性：峰高−粗基线 ≥ k×σ_local
    'smooth_w': 7,            # 峰检测平滑窗（滑动中值，点，自动取奇）
    'min_sep_deg': 3.0,       # 同列相邻候选峰最小间隔（deg）
    'mirror_halfwidth_deg': 15.0,  # 对称侧候选披露搜索半宽（φ_ref±180°±此值）
    # ---- 基线与噪声估计
    'base_frac': 0.25,   # 粗基线滑动中值窗 = n×base_frac（≥51 点，取奇）
    'sigma_w': 9,        # σ_local 局部窗（点，取奇）：窗内二阶差分 MAD
    'm_anchor': 7,       # 高斯拟合窗向触达边界外扩展的锚区点数上限（连续 ≤T；
                         # 基线由锚区约束、峰尾由高斯分量吸收，防基线被峰尾抬高）
    # ---- 高斯尾校正（§5.2 步2；Q4 线性基线下截断面积/峰高的解析修正）
    'tail_correction': True,
    # ---- 第三步线性拟合（§5.3）
    'loo_warn_rel': 0.10,    # leave-one-out B_eq 最大相对偏差超此值 → 条件值 flag
    'r2_min': 0.95,          # R² 低于此值 → 条件值 flag（不拦截，分级披露）
    'q_span_min': 3.0,       # q_max/q_min 低于此值 → q 覆盖提示（不设硬门槛）
}

_CODE_LABEL = {0: 'kept', 1: 'deep_gap_core', 2: 'shoulder', 3: 'isolated_outlier',
               4: 'broad_shadow', 5: 'filled'}


def _defaults_with(params):
    d = dict(_DEFAULTS)
    if params:
        unk = set(params) - set(_DEFAULTS)
        if unk:
            raise ValueError('未知参数: %s（支持: %s）'
                             % (sorted(unk), sorted(_DEFAULTS)))
        d.update(params)
    return d


def _tail_corr_factor(f):
    """触达判据 f 对应的高斯尾校正系数 1/erf(√(ln(1/f)))（推导见模块 docstring）"""
    return 1.0 / math.erf(math.sqrt(math.log(1.0 / f)))


# ---------------------------------------------------------------- 噪声与峰检测

def _sigma_local(I, w):
    """逐点局部噪声 σ：滑动窗内二阶差分的 MAD / 1.4826 / √6。
    二阶差分 d2[i]=I[i-1]-2I[i]+I[i+1] 对白噪声的方差 = 6σ²（MAD→σ 除 √6）；
    对缓变基线/宽峰近似零，天然抑制信号成分。"""
    w = max(int(w) | 1, 3)
    n = len(I)
    d2 = np.zeros(n)
    d2[1:-1] = I[:-2] - 2.0 * I[1:-1] + I[2:]
    d2[0], d2[-1] = d2[1], d2[-2]
    med = _sliding_median(d2, w)
    mad = _sliding_median(np.abs(d2 - med), w)
    return mad / 1.4826 / math.sqrt(6.0)


def _col_peaks(x, I, prm):
    """单列峰候选：平滑滑动中值取局部极大，显著性 = 峰高−粗基线 ≥ k×σ_local。
    返回 [{'idx', 'phi', 'height'}]（height = 平滑峰高−粗基线），按 phi 升序。"""
    n = len(I)
    Is = _sliding_median(I, prm['smooth_w'])
    w_base = max(int(n * prm['base_frac']) | 1, 51)
    bl = _sliding_median(I, w_base)
    sig = _sigma_local(I, prm['sigma_w'])

    cands = []
    for i in range(1, n - 1):
        if Is[i] >= Is[i - 1] and Is[i] >= Is[i + 1]:
            h = Is[i] - bl[i]
            if h > 0 and h >= prm['prominence_k'] * max(sig[i], 1e-12):
                cands.append({'idx': i, 'phi': float(x[i]), 'height': float(h)})
    if not cands:
        return []

    # min_sep 合并：圆周距离内保留最高者
    cands.sort(key=lambda c: c['phi'])
    kept = []
    for c in cands:
        dup = False
        for k in kept:
            if _circ_dist(c['phi'], k['phi']) < prm['min_sep_deg']:
                if c['height'] > k['height']:
                    kept.remove(k)
                    kept.append(c)
                dup = True
                break
        if not dup:
            kept.append(c)
    kept.sort(key=lambda c: c['phi'])
    return kept


def _circ_dist(a, b):
    """圆周距离（deg）：min(|Δ|, 360−|Δ|)"""
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def _circ_median(phis):
    """圆周中位数：以圆周均值为中心 unwrap 到 (mean−180, mean+180] 后取普通中位数"""
    ph = np.asarray(phis, dtype=float) % 360.0
    ang = ph * _DEG2RAD
    c, s = float(np.mean(np.cos(ang))), float(np.mean(np.sin(ang)))
    mean = math.atan2(s, c) / _DEG2RAD % 360.0
    unwrapped = (ph - mean + 180.0) % 360.0 - 180.0 + mean
    return float(np.median(unwrapped) % 360.0)


# ---------------------------------------------------------------- series 级选峰

def choose_phi_ref(x, Y, columns, params=None, phi_ref_override=None):
    """series 级统一选峰（规格卡 §5.2，Q11=C）。

    流程：逐列峰检测 → 跨列圆周聚类 → 簇按（覆盖列数, 峰高中位数）排序 →
    首簇圆周中位数 = φ_ref。phi_ref_override 非空时跳过自动选择（仍做逐列
    检测供确认卡与 Q13 对称侧披露）。

    返回 dict：
      phi_ref            统一选峰位置（deg，圆周）
      source             'override' | 'auto'
      per_col_peaks      {col_0based: [candidate dicts]}
      clusters           [{'phi_med', 'n_cols', 'cols', 'height_med', 'phis'}]
      mirror_candidates  φ_ref±180°±半宽 内的其它簇（beamstop 常毁一侧，
                         对称侧同强度候选披露供改选 φ_ref 重跑）
    """
    prm = _defaults_with(params)
    per_col = {}
    for c in columns:
        per_col[c] = _col_peaks(x, Y[:, c], prm)

    # 跨列圆周聚类：所有候选按 phi 排序，相邻圆周距离 ≤ tol 连边 → 连通分量
    pts = [(float(p['phi']), c, float(p['height']), i)
           for c in columns for i, p in enumerate(per_col[c])]
    pts.sort(key=lambda t: _circ_dist(t[0], 0.0))  # 相对 0° 展开成线性序
    clusters = []
    used = [False] * len(pts)
    for a in range(len(pts)):
        if used[a]:
            continue
        grp = [pts[a]]
        used[a] = True
        grew = True
        while grew:
            grew = False
            for b in range(len(pts)):
                if used[b]:
                    continue
                if any(_circ_dist(pts[b][0], g[0]) <= prm['cluster_tol_deg']
                       for g in grp):
                    grp.append(pts[b])
                    used[b] = True
                    grew = True
        phis = [g[0] for g in grp]
        clusters.append({
            'phi_med': _circ_median(phis),
            'n_cols': len({g[1] for g in grp}),
            'cols': sorted({g[1] for g in grp}),
            'height_med': float(np.median([g[2] for g in grp])),
            'phis': phis,
        })
    # 排序：覆盖列数降序 → 峰高中位数降序
    clusters.sort(key=lambda cl: (-cl['n_cols'], -cl['height_med']))

    if phi_ref_override is not None:
        phi_ref = float(phi_ref_override) % 360.0
        source = 'override'
    else:
        if not clusters:
            raise ValueError('series 级选峰失败：所有列均未检出显著峰'
                             '（prominence_k=%g）；请检查数据或降低显著性门槛'
                             % prm['prominence_k'])
        phi_ref = clusters[0]['phi_med']
        source = 'auto'

    mirror = [cl for cl in clusters
              if abs(_circ_dist(cl['phi_med'], phi_ref) - 180.0)
              <= prm['mirror_halfwidth_deg']]

    return {'phi_ref': phi_ref, 'source': source, 'per_col_peaks': per_col,
            'clusters': clusters, 'mirror_candidates': mirror,
            'params': {k: prm[k] for k in
                       ('cluster_tol_deg', 'peak_tol_deg', 'prominence_k',
                        'smooth_w', 'min_sep_deg')}}


# ---------------------------------------------------------------- 逐列触达式峰积分

def _touch_edge(Ic, bl, T, i0, step, m_hold):
    """从峰位索引 i0 沿 step(±1) 方向触达搜索（Q12：主判+下限已折算为阈值 T；
    m 点防抖）。返回 (idx, reached, to_data_end)。
    假触达（触达后回升）→ 跳过继续前进；全程未触达 → 端点兜底 to_data_end=True。"""
    n = len(Ic)
    j = i0 + step
    while 0 <= j < n:
        if Ic[j] - bl[j] <= T:
            # 防抖：从 j 起沿方向连续 ≤T 的 run
            k, r = j, 0
            while 0 <= k < n and Ic[k] - bl[k] <= T:
                r += 1
                k += step
            if r >= m_hold or k < 0 or k >= n:   # m 点保持 或 run 达数据端
                return j, True, False
            j = k                                 # 假触达，跳过回升区继续
        else:
            j += step
    return (0 if step < 0 else n - 1), False, True


def _extend_anchor(Ic, bl, T, i_edge, step, m_anchor):
    """从积分边界向外扩展基线锚区：连续 I−bl ≤ T 的点（上限 m_anchor）。
    返回扩展端索引（无扩展时 = 边界本身）。锚区被邻峰坡脚（>T 回升）或
    数据端点截断——与触达判据同口径，防锚区混入邻峰。"""
    n = len(Ic)
    j, last, cnt = i_edge + step, i_edge, 0
    while 0 <= j < n and cnt < m_anchor and Ic[j] - bl[j] <= T:
        last = j
        j += step
        cnt += 1
    return last


def _baseline_anchors(x, I, iL, iR, m, to_end_L, to_end_R):
    """线性基线锚：触达边界外侧各 m 点中值（R8 式审计友好：锚不落在峰翼上）。
    边界为数据端点（未触达）时用边界内侧单点。"""
    n = len(I)
    iLa = max(iL - m, 0) if not to_end_L else max(iL, 0)
    iRa = min(iR + 1 + m, n) if not to_end_R else min(iR + 1, n)
    vL = float(np.median(I[iLa:max(iL, iLa + 1)])) if iL > iLa or to_end_L \
        else float(I[iL])
    vR = float(np.median(I[min(iR + 1, n - 1):iRa])) if iR < iRa - 1 or to_end_R \
        else float(I[iR])
    return vL, vR


def _fit_gauss_lin(phi, y, mu0, h0, sig0):
    """峰区高斯+线性基线联合拟合（Q5=B）：
        y ≈ h·exp(−(φ−μ)²/2σ²) + c0 + c1·(φ−μ)
    返回 (h, mu, sig, c0, c1, ok, at_bound)。ok=False 时调用方回退实测峰高。"""
    from scipy.optimize import least_squares
    span = float(phi[-1] - phi[0])
    if span <= 0:
        return 0.0, mu0, sig0, 0.0, 0.0, False, False
    sig_lo = max(span / 200.0, 1e-6)
    sig_hi = span / 1.2          # σ 过宽即等效线性 → 撞界判失败
    mu_lo, mu_hi = float(phi[0]), float(phi[-1])

    def resid(p):
        h, mu, sig, c0, c1 = p
        return h * np.exp(-(phi - mu) ** 2 / (2.0 * sig ** 2)) + c0 \
            + c1 * (phi - mu) - y

    p0 = [max(h0, 1e-12), min(max(mu0, mu_lo), mu_hi),
          min(max(sig0, sig_lo), sig_hi), 0.0, 0.0]
    lo = [0.0, mu_lo, sig_lo, -np.inf, -np.inf]
    hi = [np.inf, mu_hi, sig_hi, np.inf, np.inf]
    try:
        r = least_squares(resid, p0, bounds=(lo, hi), method='trf',
                          max_nfev=20000)
    except Exception:
        return 0.0, mu0, sig0, 0.0, 0.0, False, False
    h, mu, sig, c0, c1 = r.x
    at_bound = (abs(sig - sig_lo) <= 1e-9 * max(sig_lo, 1.0)
                or abs(sig - sig_hi) <= 1e-9 * max(sig_hi, 1.0))
    ok = (r.status > 0) and (h > 0) and (not at_bound)
    return float(h), float(mu), float(sig), float(c0), float(c1), ok, at_bound


def integrate_peak(x, I, phi_ref, prm, mask_filled=None, col_peaks=None):
    """单列触达式峰积分（规格卡 §5.2 五步）。x: φ deg；I: 强度列；
    mask_filled: 与 I 等长的状态码数组（None=未提供 mask sidecar）；
    col_peaks: 该列 _col_peaks 产出（series 选峰结果透传，避免重复检测与口径漂移；
    None 时内部现算）。

    返回 dict（逐列行 + QC 字段）：
      phi_L/phi_R      触达边界（deg，phi_L < phi_R，非圆周跨 0 形态）
      phi0, sigma_g    高斯拟合峰位/峰宽（deg）
      I_peak           峰高（高斯拟合峰值；拟合失败回退实测并 flag）
      A_deg            线性基线扣除后峰面积（deg·强度）
      B_deg, B_rad, B_rad_corr   积分宽度（B=A/I_peak；corr 含高斯尾校正）
      tail_factor      校正系数（1.0 = 未启用）
      n_filled, filled_frac   积分区内 filled 单元数/占比（mask 未提供 → None）
      reached_L/R, to_end_L/R   触达状态（未触达 → 端点兜底并 flag）
      flags            披露列表
    """
    flags = []
    n = len(x)
    tol = prm['peak_tol_deg']

    # ---- 步1：φ_ref 容差窗内定本列峰位（Q13：无峰硬报错）
    # 显著性闸门直接复用 _col_peaks 口径（平滑滑动中值 + prominence_k×σ_local），
    # 与 series 选峰同一套判据——原始 I 最大点会被噪声尖峰高估，禁止另立口径
    if col_peaks is None:
        col_peaks = _col_peaks(x, I, prm)
    cands = [p for p in col_peaks if _circ_dist(p['phi'], phi_ref) <= tol]
    if not cands:
        raise ValueError('φ_ref=%.3f° 容差窗 ±%.1f° 内无有效峰候选'
                         '（显著性 = prominence_k=%g×σ_local，平滑窗 %d 点）'
                         '——Q13 硬报错：该列缺 B，请检查数据或改选 φ_ref'
                         '（对称侧候选见选峰结果披露）'
                         % (phi_ref, tol, prm['prominence_k'], prm['smooth_w']))
    i0 = max(cands, key=lambda p: p['height'])['idx']

    # ---- 步2：触达式边界（预基线 = 粗滑动中值；Q12 阈值 T=max(f×H, n×σ)）
    bl_pre = _sliding_median(I, max(int(n * prm['base_frac']) | 1, 51))
    sig = _sigma_local(I, prm['sigma_w'])
    H = float(I[i0] - bl_pre[i0])
    T = max(prm['f_touch'] * H, prm['n_noise'] * float(sig[i0]))
    iL, reached_L, to_end_L = _touch_edge(I, bl_pre, T, i0, -1, prm['m_hold'])
    iR, reached_R, to_end_R = _touch_edge(I, bl_pre, T, i0, +1, prm['m_hold'])
    if to_end_L:
        flags.append('left_edge_unreached')
    if to_end_R:
        flags.append('right_edge_unreached')
    if iR - iL < 4:
        raise ValueError('积分区过窄（%d 点）：触达边界异常，拒绝产出 B'
                         % (iR - iL + 1))

    # ---- 步3：高斯+线性基线联合拟合（Q5=B；拟合窗向触达边界外扩锚区）
    # 锚区（连续 ≤T，与触达同判据）约束基线 c0/c1，峰尾由高斯分量吸收——
    # 否则触达边界处信号≈T 被当基线扣掉，B 系统性偏低 3~6%（T1 探针取证）
    iLx = _extend_anchor(I, bl_pre, T, iL, -1, prm['m_anchor'])
    iRx = _extend_anchor(I, bl_pre, T, iR, +1, prm['m_anchor'])
    if iLx < iL:
        flags.append('anchor_ext_L%d' % (iL - iLx))
    if iRx > iR:
        flags.append('anchor_ext_R%d' % (iRx - iR))
    fit_seg = slice(iLx, iRx + 1)
    h0 = max(H, 1e-12)
    sig0 = max((x[iR] - x[iL]) / 6.0, 1e-3)
    h_fit, mu_fit, sig_fit, c0, c1, ok, at_bound = _fit_gauss_lin(
        x[fit_seg], I[fit_seg].astype(float), float(x[i0]), h0, sig0)

    # ---- 步4：基线确定与峰面积（积分窗仍为触达边界内，用户口径不变）
    seg = slice(iL, iR + 1)
    phi_seg = x[seg]
    if ok:
        bl_use = c0 + c1 * (phi_seg - mu_fit)
        I_peak, phi0, sigma_g = h_fit, mu_fit, sig_fit
    else:
        # 回退：边界外侧 m 点中值锚线性基线 + 实测峰高（条件值口径）
        vL, vR = _baseline_anchors(x, I, iL, iR, prm['m_hold'], to_end_L, to_end_R)
        bl_use = vL + (vR - vL) * (phi_seg - x[iL]) / max(x[iR] - x[iL], 1e-12)
        y_fb = I[seg] - bl_use
        I_peak = float(np.max(y_fb))
        phi0 = float(x[iL + int(np.argmax(y_fb))])
        sigma_g = sig0
        flags.append('gauss_fit_fallback_measured_peak'
                     + ('_sigma_at_bound' if at_bound else '_no_converge'))
    A_deg = float(np.trapezoid(I[seg] - bl_use, phi_seg))  # numpy 2.x（trapz 已移除）
    if A_deg <= 0:
        raise ValueError('峰面积非正（%.4g）：基线/积分区异常，拒绝产出 B' % A_deg)
    if I_peak <= 0:
        raise ValueError('峰高非正（%.4g）：积分区异常，拒绝产出 B' % I_peak)

    # ---- 步5：B 计算 + 高斯尾校正 + 填充传播披露
    B_deg = A_deg / I_peak
    B_rad = B_deg * _DEG2RAD
    if prm['tail_correction']:
        tail = _tail_corr_factor(prm['f_touch'])
        B_rad_corr = B_rad * tail
    else:
        tail = 1.0
        B_rad_corr = B_rad

    if mask_filled is not None:
        codes = np.asarray(mask_filled)[seg]
        n_filled = int(np.sum(codes == 5))
        filled_frac = n_filled / len(codes)
    else:
        n_filled, filled_frac = None, None
        flags.append('mask_not_provided_filled_not_counted')

    return {
        'phi_L': float(x[iL]), 'phi_R': float(x[iR]),
        'iL': int(iL), 'iR': int(iR),
        'phi0': phi0, 'sigma_g': sigma_g, 'I_peak': float(I_peak),
        'A_deg': A_deg, 'B_deg': float(B_deg), 'B_rad': float(B_rad),
        'B_rad_corr': float(B_rad_corr), 'tail_factor': float(tail),
        'n_pts': int(iR - iL + 1),
        'n_filled': n_filled, 'filled_frac': filled_frac,
        'reached_L': reached_L, 'reached_R': reached_R,
        'to_end_L': to_end_L, 'to_end_R': to_end_R,
        'T_threshold': float(T), 'flags': flags,
    }


def bpi2_series(x, Y, columns, phi_ref, params=None, masks=None,
                q_values=None, q_unit='nm^-1', per_col_peaks=None):
    """series 级 B_π/2 计算（工具 1 引擎主入口，T3 薄壳 saxs_ruland_bpi2 调用）。

    x: φ deg 内部轴；Y: [n, n_col]；columns: 0-based 因变量列清单；
    phi_ref: 统一选峰位置（choose_phi_ref 产出或用户覆盖值）；
    masks: {col_0based: 状态码数组} 或 None（§5.2 步5 填充披露）；
    per_col_peaks: choose_phi_ref 的 per_col_peaks 透传（Q13 闸门与选峰同口径）；
    q_values: 与 columns 等长的 q_c 清单（nm⁻¹ 或 A^-1，须给 q_unit；
              引擎统一换算 nm⁻¹——单位不做猜测，T3 工具壳负责文件头/参数兜底确认）。

    返回 {'phi_ref', 'source_note', 'rows': [...]}；
    Q13：任一列失败 → 收集全部失败列后统一抛 ValueError（硬报错，不静默跳列）。
    """
    prm = _defaults_with(params)
    columns = list(columns)
    if q_values is not None:
        if len(q_values) != len(columns):
            raise ValueError('q_values(%d) 与 columns(%d) 数量不一致'
                             % (len(q_values), len(columns)))
        if q_unit not in ('nm^-1', 'A^-1'):
            raise ValueError("q_unit 只支持 'nm^-1'/'A^-1'，不做单位猜测")
        q_nm = [float(q) * (10.0 if q_unit == 'A^-1' else 1.0)
                for q in q_values]
    else:
        q_nm = [None] * len(columns)

    rows, errors = [], []
    for c, qv in zip(columns, q_nm):
        try:
            r = integrate_peak(x, Y[:, c], phi_ref, prm,
                               mask_filled=(masks.get(c) if masks else None),
                               col_peaks=(per_col_peaks.get(c)
                                          if per_col_peaks else None))
        except ValueError as e:
            errors.append('col%d(1-based %d): %s' % (c, c + 1, e))
            continue
        r.update({'col_0based': c, 'col_1based': c + 1, 'q': qv})
        rows.append(r)
    if errors:
        raise ValueError('Q13 缺峰硬报错：%d/%d 列未产出 B：\n%s'
                         % (len(errors), len(columns), '\n'.join(errors)))
    return {'phi_ref': float(phi_ref), 'rows': rows,
            'params': {k: prm[k] for k in
                       ('f_touch', 'n_noise', 'm_hold', 'peak_tol_deg',
                        'tail_correction', 'base_frac', 'sigma_w')}}


# ---------------------------------------------------------------- 第三步：Ruland 变换 + 线性拟合

def _ols(x, y):
    """无权重 OLS。返回 (slope, intercept, r2, resid)"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm, ym = float(np.mean(x)), float(np.mean(y))
    sxx = float(np.sum((x - xm) ** 2))
    if sxx <= 0:
        raise ValueError('x 无离散（q² 全同），无法线性拟合')
    slope = float(np.sum((x - xm) * (y - ym)) / sxx)
    inter = ym - slope * xm
    resid = y - (slope * x + inter)
    sst = float(np.sum((y - ym) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / sst if sst > 0 else float('nan')
    return slope, inter, r2, resid


def _curvature_f_test(x, resid):
    """残差曲率 F 检验（T2 curved 用例发现：r² 对小幅系统性弯曲不敏感——
    x 跨度大时 SST 被主导项撑大，30% 低 q 上翘 R² 仍 0.999）。二次项对线性
    残差的 SSR 改善作 F 检验，p<0.05 = 显著曲率（线性区选取不当/双群体/
    背景残留的信号，规格卡 §5.3 残差系统性弯曲标记）。返回 (p, is_curved)。"""
    from scipy import stats
    n = len(resid)
    if n < 6:
        return float('nan'), False
    z = np.asarray(x, dtype=float)
    A = np.vstack([np.ones(n), z, z ** 2]).T
    try:
        beta, *_ = np.linalg.lstsq(A, resid, rcond=None)
    except np.linalg.LinAlgError:
        return float('nan'), False
    ssr_quad = float(np.sum((resid - A @ beta) ** 2))
    ssr_lin = float(np.sum(resid ** 2))
    if ssr_quad <= 0:
        return 0.0, True
    f_stat = (ssr_lin - ssr_quad) / (ssr_quad / (n - 3))
    p = float(1.0 - stats.f.cdf(f_stat, 1, n - 3))
    return p, p < 0.05


def ruland_beq(q_nm, B_rad, q_window=None, params=None):
    """第三步：Ruland 变换 + 无权重 OLS → B_eq（工具 2 引擎主入口，T3 薄壳调用）。

    q_nm: 各列 q_c（nm⁻¹）；B_rad: 对应 B_π/2（rad）；q_window: (qmin,qmax)
    线性区域筛选（缺省全范围 + 诊断，§5.3：不自动裁剪，人工裁决线性区）。

    Q6 裁决落地：斜率 = B_eq²（目标量）；截距 = 1/L_app² 仅作诊断量
    （<0 = 系统偏差信号 → flag 触发线性区复核），L 一律不产出。

    返回 dict：
      B_eq_rad / B_eq_deg    取向参数（双单位）
      f_v / f_v_coeff        取向度（公式 3：f_v = 1 − f_v_coeff·B_eq_rad，
                             系数 2√(ln2/π³)≈0.299032；弧度制口径，分级继承 B_eq）
      slope, intercept, r2   拟合量（x=q² nm⁻², y=q²·B² ）
      loo_*                  leave-one-out 敏感性（§5.3 质量门）
      flags                  分级披露（r2 低 / loo 超阈 / 截距<0 / q 覆盖窄 /
                             点数少 / f_v<0 非物理）
    """
    prm = _defaults_with(params)
    q = np.asarray(q_nm, dtype=float)
    B = np.asarray(B_rad, dtype=float)
    if q.shape != B.shape or q.ndim != 1:
        raise ValueError('q 与 B 须为等长一维数组（实得 %s vs %s）'
                         % (q.shape, B.shape))
    if q.size < 3:
        raise ValueError('线性拟合至少需要 3 个 (q, B) 点，实得 %d' % q.size)
    if np.any(q <= 0):
        raise ValueError('q 含非正值（%d 个）：q² 变换要求 q>0'
                         % int(np.sum(q <= 0)))
    if np.any(~np.isfinite(B)):
        raise ValueError('B 含非有限值（%d 个）' % int(np.sum(~np.isfinite(B))))
    if np.any(B <= 0):
        raise ValueError('B 含非正值（%d 个）：积分宽度物理上必须 >0'
                         '（y=q²B² 会抹掉负号，必须入口拦截）'
                         % int(np.sum(B <= 0)))

    sel = np.ones(q.size, dtype=bool)
    if q_window is not None:
        qmin, qmax = float(q_window[0]), float(q_window[1])
        if not (0 < qmin < qmax):
            raise ValueError('q_window 须 0<qmin<qmax（实得 %s）' % (q_window,))
        sel = (q >= qmin) & (q <= qmax)
        if int(sel.sum()) < 3:
            raise ValueError('q_window 内仅 %d 点（需 ≥3）' % int(sel.sum()))
    qs, Bs = q[sel], B[sel]

    x = qs ** 2
    y = qs ** 2 * Bs ** 2
    slope, inter, r2, resid = _ols(x, y)
    flags = []
    if slope <= 0:
        raise ValueError('斜率非正（%.4g）：B_eq 无意义；请检查 B 口径/线性区'
                         '（截距=%.4g）' % (slope, inter))

    B_eq_rad = math.sqrt(slope)

    # ---- 取向度 f_v（公式 3，弧度制；分级继承 B_eq 全部 flags）
    f_v = 1.0 - _FV_COEF * B_eq_rad
    if f_v < 0:
        flags.append('fv_negative(f_v=%.4g<0；B_eq=%.3f rad 已超线性式 (3) '
                     '适用域——各向同性样品应报 Q13 而非走 beq，请复核分析前提)'
                     % (f_v, B_eq_rad))

    # ---- leave-one-out 敏感性（§5.3）
    loo = []
    for i in range(qs.size):
        m = np.ones(qs.size, dtype=bool)
        m[i] = False
        try:
            sl_i, _it, _r2, _rd = _ols(x[m], y[m])
            loo.append(math.sqrt(sl_i) if sl_i > 0 else float('nan'))
        except ValueError:
            loo.append(float('nan'))
    loo = np.asarray(loo, dtype=float)
    fin = np.isfinite(loo) & (loo > 0)
    loo_max_rel = float(np.max(np.abs(loo[fin] - B_eq_rad) / B_eq_rad)) \
        if fin.any() else float('nan')

    # ---- 诊断分级（§5.3；Q6 截距仅诊断量）
    p_curv, is_curved = _curvature_f_test(x, resid)
    if is_curved:
        flags.append('residual_curvature(F_p=%.2g<0.05；线性区选取不当/'
                     '双 L 群体/背景残留——请复核 q 窗口)' % p_curv)
    if r2 < prm['r2_min']:
        flags.append('r2_below_threshold(%.4f<%.3f)' % (r2, prm['r2_min']))
    if math.isfinite(loo_max_rel) and loo_max_rel > prm['loo_warn_rel']:
        flags.append('loo_exceeds(%.1f%%>%.0f%%)'
                     % (100 * loo_max_rel, 100 * prm['loo_warn_rel']))
    if inter < 0:
        flags.append('intercept_negative_systematic_bias_signal'
                     '(复核线性区/背景残留/B 口径)')
    if qs.size < 5:
        flags.append('few_points(%d<5)' % qs.size)
    span_ratio = float(qs.max() / qs.min())
    if span_ratio < prm['q_span_min']:
        flags.append('q_span_narrow(qmax/qmin=%.2f<%.1f；q²B² 近过原点直线、'
                     '斜率外推误差增大——L=2π/q 原则提示中低 q 覆盖)'
                     % (span_ratio, prm['q_span_min']))

    return {
        'n_pts': int(qs.size),
        'q_nm': [float(v) for v in qs],
        'x_q2': [float(v) for v in x],
        'y_q2B2': [float(v) for v in y],
        'residuals': [float(v) for v in resid],
        'slope_per_nm2': slope,
        'intercept_per_nm2': inter,
        'r2': float(r2),
        'B_eq_rad': B_eq_rad,
        'B_eq_deg': B_eq_rad * _RAD2DEG,
        'f_v': f_v,
        'f_v_coeff': _FV_COEF,
        'loo_B_eq_rad': [float(v) for v in loo],
        'loo_max_rel_dev': loo_max_rel,
        'curvature_F_p': p_curv,
        'q_span_ratio': span_ratio,
        'q_window': ([float(q_window[0]), float(q_window[1])]
                     if q_window is not None else None),
        'flags': flags,
        'params': {k: prm[k] for k in ('r2_min', 'loo_warn_rel', 'q_span_min')},
    }


def ruland_beq_scan(q_nm, B_rad, qmins, qmax=None, params=None):
    """线性区扫描 + 最大无曲率窗自动采纳（提速三件套 T-A，规格卡
    《提速三件套-beq扫描内置-批量填充-链式编排-规划.md》§1）。

    逐窗调用 ruland_beq（零双实现漂移）；qmax 缺省 = 数据最大 q。

    采纳规则（2026-09-30 用户裁决：规则命中即自动产出）：qmin 升序扫描，
    第一个残差曲率 F 检验通过（p≥0.05 且非 NaN）的窗自动作为采纳窗；
    全部窗曲率显著或 F 检验不可用（NaN，三态语义——不可用≠通过）→
    adopted_qmin=None + adopt_manual flag 转人工裁决（000059 无平台场景
    兜底）。人工覆盖路径：显式传 q_window 走单窗模式。

    返回 dict：windows（逐窗证据，含 errors 窗）/ adopted_qmin / rule /
    flags（adopt_manual 等）。"""
    q = np.asarray(q_nm, dtype=float)
    qmax_eff = float(q.max()) if qmax is None else float(qmax)
    rows, errs = [], {}
    for qmin in sorted(float(v) for v in qmins):
        try:
            fit = ruland_beq(q_nm, B_rad, q_window=(qmin, qmax_eff),
                             params=params)
        except ValueError as e:
            errs['%.6g' % qmin] = str(e)
            continue
        rows.append({'qmin': qmin, 'qmax': qmax_eff, 'n_pts': fit['n_pts'],
                     'B_eq_rad': fit['B_eq_rad'], 'B_eq_deg': fit['B_eq_deg'],
                     'f_v': fit['f_v'], 'r2': fit['r2'],
                     'curvature_F_p': fit['curvature_F_p'],
                     'q_span_ratio': fit['q_span_ratio'],
                     'flags': fit['flags']})
    adopted = None
    for r in rows:
        p = r['curvature_F_p']
        if math.isfinite(p) and p >= 0.05:
            adopted = r['qmin']
            break
    flags = []
    if adopted is None and rows:
        flags.append('adopt_manual(全部窗曲率显著或 F 检验不可用——最大无曲率'
                     '窗规则无法命中，请人工裁决线性区后显式传 q_window 重跑)')
    return {'qmax_nm': qmax_eff, 'windows': rows, 'window_errors': errs,
            'adopted_qmin': adopted,
            'rule': 'first-pass 最大无曲率窗（残差曲率 F 检验 p≥0.05 且非 NaN，'
                    'qmin 升序；2026-09-30 用户裁决规则命中即自动产出）',
            'flags': flags}


# ================================================================ T3 card 层
# （读盘 / 头解析 / B 表 CSV / QC 渲染 / 报告写盘；工具薄壳在 tools_ruland.py）

import os
from datetime import datetime

from gap_fill import _check_x_monotonic   # noqa: E402  复用单调校验（P7）
import workdir                            # noqa: E402  会话级工作目录（261104）缺省产物重定向

# 工作区根锚定（lessons.md 2026-09-28：MCP 宿主 cwd 不可控，相对 out_dir 一律
# 锚定工作区根；与 gap_fill.py / selftest.py 同款定义）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_q_from_header(header_lines, n_ycol):
    """Q9 裁决 C 兜底：从文件头注释行解析 q_c 清单。唯一候选行才采用——
    候选 = 含 'q' 字样且浮点 token 数 == n_ycol 的 # 行；0 个或多个候选 →
    歧义报错列出全部头行（宁可报错不要猜错，单位由调用方 q_unit 显式给）。"""
    cands = []
    for ln in header_lines:
        toks = ln.lstrip('#').replace(':', ' ').replace('(', ' ') \
            .replace(')', ' ').replace('=', ' ').split()
        nums = []
        for t in toks:
            try:
                v = float(t)
            except ValueError:
                continue
            if math.isfinite(v):
                nums.append(v)
        if 'q' in ln.lower() and len(nums) == n_ycol:
            cands.append((ln, nums))
    if len(cands) == 1:
        return cands[0]
    detail = '\n'.join('  %s' % ln for ln in header_lines) or '  （空）'
    raise ValueError('文件头 q_c 清单解析歧义（候选 %d 个，需唯一：含 q 字样'
                     '且浮点数个数 == 因变量列数 %d）；请显式传 q_values 参数'
                     '（清单须经用户确认）。文件头内容：\n%s'
                     % (len(cands), n_ycol, detail))


def _read_mask_file(path, x_ref):
    """读 gap_fill 单列 mask sidecar（x,mask 两列 CSV）→ 对齐主文件行序的
    code 数组。x 逐行与主文件 isclose 校验（P7：不一致报错不猜）。"""
    raw = _read_columns(path)
    if raw['n_col'] != 2:
        raise ValueError('mask 文件须 2 列（x,mask）: %s' % path)
    if raw['n_rows'] != len(x_ref):
        raise ValueError('mask 行数 %d 与主文件 %d 不一致: %s'
                         % (raw['n_rows'], len(x_ref), path))
    for i in range(raw['n_rows']):
        if not math.isclose(float(raw['raw_rows'][i][0]), float(x_ref[i]),
                            rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError('mask 第 %d 行 x 与主文件不一致（%s vs %s）: %s'
                             % (i + 1, raw['raw_rows'][i][0], x_ref[i], path))
    codes = raw['Y'][:, 0].astype(int)
    unknown = set(codes.tolist()) - set(_CODE_LABEL)
    if unknown:
        raise ValueError('mask 含未知状态码 %r: %s' % (sorted(unknown), path))
    return codes


_BPI2_COLS = ('q_nm', 'B_deg', 'B_rad', 'B_rad_corr', 'n_pts', 'n_filled',
              'filled_frac', 'phi_L_deg', 'phi_R_deg', 'phi0_deg', 'I_peak',
              'col_1based', 'flags')


def _render_bpi2_qc(path, x, Y, rows, phi_ref, prm, mask_map):
    """逐列峰积分 QC 网格图（4 列 × ceil(n/4) 行）：实测曲线 + 触达预基线 +
    最终基线 + 积分区着色 + φ_ref/边界标线。mask 已提供的列以红点标 filled。"""
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    n = len(rows)
    ncol = 4
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.6 * nrow),
                             squeeze=False)
    for k, r in enumerate(rows):
        ax = axes[k // ncol][k % ncol]
        c = r['col_0based']
        I = Y[:, c]
        ax.plot(x, I, '-', lw=0.8, color='#366')
        seg = slice(r['iL'], r['iR'] + 1)
        ax.fill_between(x[seg], I[seg], float(np.min(I)),
                        alpha=0.25, color='#f90')
        ax.axvline(phi_ref, color='g', lw=0.7, ls='--')
        ax.axvline(r['phi_L'], color='r', lw=0.7)
        ax.axvline(r['phi_R'], color='r', lw=0.7)
        if mask_map and c in mask_map:
            fm = mask_map[c] == 5
            if fm.any():
                ax.plot(x[fm], I[fm], '.', ms=3, color='r')
        qtxt = ('q=%.3g' % r['q']) if r['q'] is not None else 'q=?'
        ax.set_title('col%d %s B=%.4g° f%s' % (c + 1, qtxt, r['B_deg'],
                                               r['flags'][:1] or ''),
                     fontsize=7)
        ax.tick_params(labelsize=6)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.suptitle('saxs_ruland_bpi2 QC | φ_ref=%.2f° | f=%g n=%g m=%d%s'
                 % (phi_ref, prm['f_touch'], prm['n_noise'], prm['m_hold'],
                    ' | 尾校正×%.4f' % _tail_corr_factor(prm['f_touch'])
                    if prm['tail_correction'] else ''), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_bq(path, rows):
    """B(q) 汇总图：B vs q 散点（条件值口径注记）。"""
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    qv = [r['q'] for r in rows if r['q'] is not None]
    bv = [r['B_rad'] for r in rows if r['q'] is not None]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    if qv:
        ax.plot(qv, bv, 'o-', lw=0.8, ms=4, color='#366')
    ax.set_xlabel('q (nm$^{-1}$)')
    ax.set_ylabel('B$_{π/2}$ (rad)')
    ax.set_title('B$_{π/2}$(q) — Ruland 变换前汇总（填充依赖列=条件值）',
                 fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bpi2_card(path, q_unit, q_values=None, columns=None, phi_ref=None,
              params=None, mask_files=None, out_dir=None):
    """工具 1 主入口（T3）：读 N+1 列 φ 文件 → 选峰 → 逐列 B_π/2 → 写盘。
    产出四件套：B 表 CSV（Origin 可拖）/ QC 网格图 / B(q) 汇总图 / result.json。
    q_values=None 时走文件头解析兜底（Q9，唯一候选才用，歧义报错）。"""
    if not os.path.isfile(path):
        raise ValueError('文件不存在: %s' % path)
    if q_unit not in ('nm^-1', 'A^-1'):
        raise ValueError("q_unit 只支持 'nm^-1'/'A^-1'（q_c 清单单位），"
                         "必填不猜")
    raw = _read_columns(path)
    # _read_columns 的 Y = 各行 t[1:]（纯因变量阵，不含 x 列）；x 须从 raw_rows
    # 首 token 取（260930 实测踩坑：Y[:,0] 是第 1 个因变量列，把强度当轴查单调）
    x = np.array([float(t[0]) for t in raw['raw_rows']], dtype=float)
    _check_x_monotonic(x)
    Y = raw['Y']
    n_ycol = raw['n_col'] - 1
    prm = _defaults_with(params)

    cols0 = ([c - 1 for c in columns] if columns else list(range(n_ycol)))
    if any(not (0 <= c < n_ycol) for c in cols0):
        raise ValueError('columns 越界（1-based %r，文件 %d 个因变量列）'
                         % (columns, n_ycol))

    # ---- q_c 清单（Q9：显式优先，头解析兜底；单位一律 q_unit 显式）
    if q_values is not None:
        q_src = 'explicit'
        q_list = [float(v) for v in q_values]
    else:
        q_src = 'header'
        ln, q_list = _parse_q_from_header(raw['header_lines'], n_ycol)
        if len(q_list) < len(cols0):
            raise ValueError('文件头 q 清单 %d 个 < 选用列数 %d'
                             % (len(q_list), len(cols0)))
    if len(q_list) != len(cols0):
        raise ValueError('q_values 数量 %d 与选用列数 %d 不一致'
                         % (len(q_list), len(cols0)))
    scale = 10.0 if q_unit == 'A^-1' else 1.0
    q_nm = [v * scale for v in q_list]

    # ---- mask sidecar（§5.2 步5 填充披露；显式路径列表，不猜命名）
    mask_map = {}
    if mask_files is not None:
        if len(mask_files) != len(cols0):
            raise ValueError('mask_files(%d) 与选用列数(%d) 不一致'
                             % (len(mask_files), len(cols0)))
        for c0, mf in zip(cols0, mask_files):
            if not os.path.isfile(mf):
                raise ValueError('mask 文件不存在: %s' % mf)
            mask_map[c0] = _read_mask_file(mf, x)

    # ---- 选峰（Q11：自动聚类 + 可覆盖）
    sel = choose_phi_ref(x, Y, cols0, params=prm, phi_ref_override=phi_ref)

    # ---- series B
    out = bpi2_series(x, Y, cols0, sel['phi_ref'], params=prm, masks=mask_map,
                      q_values=q_nm, q_unit='nm^-1',
                      per_col_peaks=sel['per_col_peaks'])

    # ---- 写盘（out_dir 锚定工作区根，MCP 宿主 cwd 不可控）
    stem = os.path.splitext(os.path.basename(path))[0]
    if out_dir is None:
        out_dir = workdir.default_out('%s_ruland_bpi2' % stem)
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    csv_path = os.path.join(out_dir, '%s_ruland_Bpi2.csv' % stem)
    lines = [
        '# SAXS Ruland B_pi/2 table | %s' % TOOL_TAG,
        '# source_file: %s' % os.path.basename(path),
        '# timestamp: %s | q_unit_in: %s | q_source: %s' % (ts, q_unit, q_src),
        '# phi_ref: %.4f deg (%s) | 触达判据 f=%g n=%g m=%d | 峰容差 ±%g°'
        % (sel['phi_ref'], sel['source'], prm['f_touch'], prm['n_noise'],
           prm['m_hold'], prm['peak_tol_deg']),
        '# B_rad_corr 含高斯尾校正 ×%.4f（f 触达截断，假设高斯尾）'
        % _tail_corr_factor(prm['f_touch']),
        '# 填充依赖列=条件值：filled 单元为 gap 数学插值（非信号恢复），'
        '下游披露依赖占比',
    ]
    lines.append(','.join(_BPI2_COLS))
    for r, qv in zip(out['rows'], q_list):
        lines.append(','.join([
            '%.6g' % (qv * scale), '%.6g' % r['B_deg'], '%.6g' % r['B_rad'],
            '%.6g' % r['B_rad_corr'], str(r['n_pts']),
            '' if r['n_filled'] is None else str(r['n_filled']),
            '' if r['filled_frac'] is None else '%.4f' % r['filled_frac'],
            '%.4g' % r['phi_L'], '%.4g' % r['phi_R'], '%.4g' % r['phi0'],
            '%.6g' % r['I_peak'], str(r['col_1based']),
            ';'.join(r['flags']),
        ]))
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    qc_path = os.path.join(out_dir, '%s_ruland_QC.png' % stem)
    _render_bpi2_qc(qc_path, x, Y, out['rows'], sel['phi_ref'], prm, mask_map)
    bq_path = os.path.join(out_dir, '%s_ruland_Bq.png' % stem)
    _render_bq(bq_path, out['rows'])
    rj_path = os.path.join(out_dir, '%s_ruland_bpi2_result.json' % stem)
    result = {
        'tool': TOOL_TAG, 'mode': 'bpi2', 'timestamp': ts,
        'source': path, 'out_dir': out_dir,
        'q_unit_in': q_unit, 'q_source': q_src,
        'q_nm': q_nm, 'columns_1based': [c + 1 for c in cols0],
        'phi_ref': sel['phi_ref'], 'phi_ref_source': sel['source'],
        'mirror_candidates': [cl['phi_med'] for cl in sel['mirror_candidates']],
        'clusters': [{k: cl[k] for k in ('phi_med', 'n_cols', 'height_med')}
                     for cl in sel['clusters'][:8]],
        'rows': [{k: r[k] for k in
                  ('col_1based', 'q', 'B_deg', 'B_rad', 'B_rad_corr', 'n_pts',
                   'n_filled', 'filled_frac', 'phi_L', 'phi_R', 'phi0',
                   'I_peak', 'reached_L', 'reached_R', 'flags')}
                 for r in out['rows']],
        'params': out['params'],
        'files': {'bpi2_csv': csv_path, 'qc_png': qc_path, 'bq_png': bq_path,
                  'json': rj_path},
    }
    with open(rj_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)
    return {'tool': TOOL_TAG, 'mode': 'bpi2', 'source': path,
            'out_dir': out_dir, 'phi_ref': sel['phi_ref'],
            'phi_ref_source': sel['source'],
            'mirror_candidates': [cl['phi_med']
                                  for cl in sel['mirror_candidates']],
            'n_columns': len(out['rows']),
            'q_source': q_src,
            'filled_summary': {str(r['col_1based']):
                               (None if r['filled_frac'] is None
                                else round(r['filled_frac'], 4))
                               for r in out['rows']
                               if r['filled_frac'] not in (None, 0)},
            'files': result['files']}


def _read_bpi2_csv(path):
    """读 B 表（bpi2_card 产出或同形手写表）→ q_nm + B_rad(+corr) + 元信息。
    B 表含 flags 字符串列与空 token（filled 未统计）——禁用 _read_columns
    （其全列 float 化会把含 flags 的行判 bad 丢弃，T3 自查），自写解析：
    # 块跳过 → 列名行 → 逐行数值列 float 化（flags 列保留原串）。"""
    with open(path, 'rb') as f:
        from gap_fill import _decode_raw
        text, _enc, _ev = _decode_raw(f.read())
    if text is None:
        raise ValueError('编码无法判定: %s' % path)
    lines = [ln for ln in text.split('\n')
             if ln.strip() and not ln.strip().startswith('#')]
    if not lines:
        raise ValueError('B 表无数据行: %s' % path)
    names = [t.strip() for t in lines[0].split(',')]
    idx = {nm: i for i, nm in enumerate(names)}
    for need in ('q_nm', 'B_rad', 'B_rad_corr'):
        if need not in idx:
            raise ValueError('B 表缺必需列 %r（实得列 %r）: %s'
                             % (need, names, path))
    i_q, i_br, i_bc = idx['q_nm'], idx['B_rad'], idx['B_rad_corr']
    i_ff = idx.get('filled_frac')
    q_nm, b_rad, b_corr, f_frac = [], [], [], []
    for ln in lines[1:]:
        toks = [t.strip() for t in ln.split(',')]
        if len(toks) != len(names):
            raise ValueError('B 表行列数不一致（应为 %d 列）: %s'
                             % (len(names), path))
        try:
            q_nm.append(float(toks[i_q]))
            b_rad.append(float(toks[i_br]))
            b_corr.append(float(toks[i_bc]))
            f_frac.append(float(toks[i_ff])
                          if i_ff is not None and toks[i_ff] else float('nan'))
        except ValueError:
            raise ValueError('B 表数值列含非数值 token: %s | 行: %s'
                             % (path, ln))
    return {'q_nm': np.array(q_nm), 'B_rad': np.array(b_rad),
            'B_rad_corr': np.array(b_corr), 'filled_frac': np.array(f_frac),
            'columns': names, 'n_rows': len(q_nm)}


def _render_ruland(path, fit, b_col_name):
    """Ruland 图两联：散点+拟合线（虚线外推展示截距，仅诊断不标 L）+ 残差。"""
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    x = np.array(fit['x_q2'])
    y = np.array(fit['y_q2B2'])
    rd = np.array(fit['residuals'])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True,
                                 gridspec_kw={'height_ratios': [2.4, 1]})
    a1.plot(x, y, 'o', ms=5, color='#366', label='观测')
    xs = np.linspace(0.0, x.max() * 1.05, 50)
    a1.plot(xs, fit['slope_per_nm2'] * xs + fit['intercept_per_nm2'], '-',
            lw=1.0, color='#c33', label='OLS 拟合')
    a1.plot(0.0, fit['intercept_per_nm2'], 's', ms=5, color='#c33')
    a1.annotate('截距=%.3g nm$^{-2}$（诊断量，1/L$_{app}$² 口径——L 不产出）'
                % fit['intercept_per_nm2'],
                xy=(0.02, 0.02), xycoords='axes fraction', fontsize=8)
    a1.set_ylabel('q$^2$·B$_{π/2}^2$ (rad$^2$·nm$^{-2}$)')
    a1.set_title('Ruland 图 — B_eq=%.4f rad (%.3f°)  f_v=%.4f  R²=%.5f  [%s]'
                 % (fit['B_eq_rad'], fit['B_eq_deg'], fit['f_v'],
                    fit['r2'], b_col_name),
                 fontsize=10)
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)
    a2.axhline(0.0, color='k', lw=0.6)
    a2.stem(x, rd, basefmt=' ')
    a2.set_xlabel('q$^2$ (nm$^{-2}$)')
    a2.set_ylabel('残差')
    a2.grid(alpha=0.3)
    if fit['flags']:
        fig.text(0.5, 0.005, 'flags: ' + ' | '.join(fit['flags']),
                 ha='center', fontsize=7, color='#a00')
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def beq_card(bpi2_csv, q_window=None, b_column='B_rad_corr', params=None,
             out_dir=None, scan_qmins=None):
    """工具 2 主入口（T3）：读 B 表 → Ruland 变换 + OLS → B_eq → 写盘。
    产出三件套：Ruland 图 PNG / 分析报告 md / result.json。
    b_column: 'B_rad_corr'（缺省，含尾校正）或 'B_rad'（原始）。
    scan_qmins: 线性区扫描 qmin 清单（提速三件套 T-A；与 q_window 互斥）——
    逐窗扫描后按最大无曲率窗规则自动采纳（2026-09-30 用户裁决），采纳窗
    全套交付物复用单窗交付路径写入 <out_dir>/adopted/；全部窗不通过 →
    仅出证据链 + adopt_manual 转人工裁决。人工覆盖：显式传 q_window。"""
    if not os.path.isfile(bpi2_csv):
        raise ValueError('B 表不存在: %s' % bpi2_csv)
    if b_column not in ('B_rad_corr', 'B_rad'):
        raise ValueError("b_column 只支持 'B_rad_corr'/'B_rad'")
    if scan_qmins is not None:
        if q_window is not None:
            raise ValueError('scan_qmins 与 q_window 互斥（人工覆盖采纳窗请'
                             '只传 q_window 走单窗模式）')
        return _beq_scan_card(bpi2_csv, scan_qmins, b_column=b_column,
                              params=params, out_dir=out_dir)
    tab = _read_bpi2_csv(bpi2_csv)
    b = tab['B_rad_corr'] if b_column == 'B_rad_corr' else tab['B_rad']
    fit = ruland_beq(tab['q_nm'], b, q_window=q_window, params=params)

    stem = os.path.splitext(os.path.basename(bpi2_csv))[0]
    stem = stem.replace('_ruland_Bpi2', '')
    if out_dir is None:
        out_dir = workdir.default_out('%s_ruland_beq' % stem)
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    png_path = os.path.join(out_dir, '%s_ruland_plot.png' % stem)
    _render_ruland(png_path, fit, b_column)
    rep_path = os.path.join(out_dir, '%s_ruland_report.md' % stem)
    grade = '稳健' if not fit['flags'] else '条件值（诊断 flag 触发，见下）'
    filled_rows = [i for i, v in enumerate(tab['filled_frac'])
                   if math.isfinite(v) and v > 0]
    L = ['# Ruland 取向度分析报告（B_eq）', '',
         '- 工具：%s' % TOOL_TAG,
         '- 时间：%s | B 表：`%s`（B 列 = %s）' % (ts, bpi2_csv, b_column),
         '- q 窗口：%s（缺省全范围）'
         % (fit['q_window'] if fit['q_window'] else '全部 %d 点' % fit['n_pts']),
         '',
         '## 结果（%s）' % grade,
         '',
         '| 量 | 数值 |',
         '|---|---|',
         '| **B_eq（取向参数，越小取向越好）** | **%.5f rad = %.4f°** |'
         % (fit['B_eq_rad'], fit['B_eq_deg']),
         '| **取向度 f_v（公式 3：1−2√(ln2/π³)·B_eq，弧度制）** | **%.4f** |'
         % fit['f_v'],
         '| 斜率 = B_eq² | %.6g rad²·nm² |' % fit['slope_per_nm2'],
         '| R²（线性性） | %.5f |' % fit['r2'],
         '| 截距（诊断量） | %.6g nm⁻² |' % fit['intercept_per_nm2'],
         '| leave-one-out 最大相对偏差 | %.2f%% |'
         % (100 * fit['loo_max_rel_dev']),
         '| q 覆盖 | %.3g–%.3g nm⁻¹（跨度比 %.2f）|'
         % (min(fit['q_nm']), max(fit['q_nm']), fit['q_span_ratio']),
         '',
         '## 口径与披露',
         '',
         '- 高斯近似有效性：线性性成立本身即证据（论文 §4）——'
         'R²=%.5f，曲率 F 检验 p=%.3g。'
         % (fit['r2'], fit['curvature_F_p']),
         '- **微孔长度 L 不产出**（L=2π/q 原则：长 L 须实测小 q 端数据，'
         '模型法不可采信）；截距仅作质量诊断量。' ,
         '- f_v 口径：f_v = 1 − 2√(ln2/π³)·B_eq_rad（系数 %.9f，无量纲），'
         'B_eq 必须代入弧度制；f_v=0 各向同性随机、f_v=1 完全取向，'
         '线性式适用域 B_eq≪π，f_v<0 触发 fv_negative flag；'
         'f_v 分级与诊断 flags 完全继承 B_eq。' % fit['f_v_coeff'],
         '- B_π/2 含高斯尾校正（f 触达截断，假设高斯尾）。' ,
         '- 填充依赖：B 表 %s' % (
             '第 %s 列含填充单元（条件值传播，占比见 B 表 filled_frac 列）'
             % ','.join(str(i + 1) for i in filled_rows)
             if filled_rows else '无填充单元披露（B 表未含或全为 0）'),
         '- q_span 提示：%s' % (
             'q 跨度偏窄，斜率外推误差增大（L=2π/q 原则提示中低 q 覆盖）'
             if any('q_span_narrow' in f for f in fit['flags'])
             else '正常'),
         '',
         '## 诊断 flags',
         '',
         '\n'.join('- %s' % f for f in fit['flags']) if fit['flags']
         else '- 无（全部诊断通过）',
         '',
         '## Ruland 图',
         '',
         '![Ruland plot](%s)' % os.path.basename(png_path),
         '']
    with open(rep_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    rj_path = os.path.join(out_dir, '%s_ruland_beq_result.json' % stem)
    result = dict(fit)
    result.update({'tool': TOOL_TAG, 'mode': 'beq', 'timestamp': ts,
                   'bpi2_csv': bpi2_csv, 'b_column': b_column,
                   'out_dir': out_dir, 'grade': grade,
                   'files': {'png': png_path, 'report': rep_path,
                             'json': rj_path}})
    with open(rj_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)
    return {'tool': TOOL_TAG, 'mode': 'beq', 'bpi2_csv': bpi2_csv,
            'out_dir': out_dir, 'grade': grade,
            'B_eq_rad': fit['B_eq_rad'], 'B_eq_deg': fit['B_eq_deg'],
            'f_v': fit['f_v'], 'f_v_coeff': fit['f_v_coeff'],
            'r2': fit['r2'], 'n_pts': fit['n_pts'], 'flags': fit['flags'],
            'files': result['files']}


def _beq_scan_card(bpi2_csv, scan_qmins, b_column='B_rad_corr', params=None,
                   out_dir=None):
    """扫描模式交付（提速三件套 T-A）：逐窗证据链 CSV + 扫描报告 + result.json；
    采纳窗命中时复用既有 beq_card 单窗交付路径产出全套三件套（零双实现）。"""
    tab = _read_bpi2_csv(bpi2_csv)
    b = tab['B_rad_corr'] if b_column == 'B_rad_corr' else tab['B_rad']
    scan = ruland_beq_scan(tab['q_nm'], b, scan_qmins, params=params)

    stem = os.path.splitext(os.path.basename(bpi2_csv))[0]
    stem = stem.replace('_ruland_Bpi2', '')
    if out_dir is None:
        out_dir = workdir.default_out('%s_ruland_beq_scan' % stem)
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ---- 证据链 CSV（Origin 可拖）
    csv_path = os.path.join(out_dir, '%s_ruland_scan.csv' % stem)
    with open(csv_path, 'w', encoding='utf-8-sig') as f:
        f.write('qmin_nm,qmax_nm,n_pts,B_eq_rad,B_eq_deg,f_v,r2,'
                'curvature_F_p,q_span_ratio,adopted,flags\n')
        for r in scan['windows']:
            f.write('%.6g,%.6g,%d,%.10g,%.10g,%.10g,%.6g,%.6g,%.6g,%s,"%s"\n'
                    % (r['qmin'], r['qmax'], r['n_pts'], r['B_eq_rad'],
                       r['B_eq_deg'], r['f_v'], r['r2'], r['curvature_F_p'],
                       r['q_span_ratio'],
                       'yes' if r['qmin'] == scan['adopted_qmin'] else 'no',
                       '; '.join(r['flags'])))

    # ---- 采纳窗全套交付物 = 复用既有单窗交付路径（零双实现）
    adopted_card = None
    if scan['adopted_qmin'] is not None:
        adopted_card = beq_card(
            bpi2_csv, q_window=[scan['adopted_qmin'], scan['qmax_nm']],
            b_column=b_column, params=params,
            out_dir=os.path.join(out_dir, 'adopted'))

    # ---- 扫描报告（规则披露 + 证据表）
    rep_path = os.path.join(out_dir, '%s_ruland_scan_report.md' % stem)
    L = ['# Ruland 线性区扫描报告（最大无曲率窗自动采纳）', '',
         '- 工具：%s' % TOOL_TAG,
         '- 时间：%s | B 表：`%s`（B 列 = %s）' % (ts, bpi2_csv, b_column),
         '- 扫描规则：%s' % scan['rule'],
         '- qmax = %.6g nm⁻¹（数据最大 q）；qmin 清单：%s'
         % (scan['qmax_nm'], ', '.join('%.6g' % v for v in sorted(scan_qmins))),
         '']
    if adopted_card is not None:
        L += ['## 采纳窗（规则自动命中，可人工覆盖重跑）', '',
              '- **采纳窗 q ∈ [%.6g, %.6g] nm⁻¹**'
              % (scan['adopted_qmin'], scan['qmax_nm']),
              '- B_eq = %.6g rad = %.4f° | R² = %.5f | f_v = %.4f'
              % (adopted_card['B_eq_rad'], adopted_card['B_eq_deg'],
                 adopted_card['r2'], adopted_card['f_v']),
              '- 分级：%s' % adopted_card['grade'],
              '- 完整交付物（Ruland 图/报告/result.json）：`%s`'
              % adopted_card['out_dir'],
              '- 人工覆盖路径：对 saxs_ruland_beq 显式传 q_window 重跑单窗模式。',
              '']
    else:
        L += ['## 未自动采纳（转人工裁决）', '',
              '- 全部窗曲率显著或 F 检验不可用（NaN = 检验不可用，不等于通过）。',
              '- flags：%s' % ('; '.join(scan['flags']) if scan['flags'] else '无'),
              '- 处置：人工裁决线性区后对 saxs_ruland_beq 显式传 q_window 重跑。',
              '']
    L += ['## 逐窗证据链', '',
          '| qmin | n_pts | B_eq_deg | f_v | R² | 曲率 F_p | adopted | flags |',
          '|---|---|---|---|---|---|---|---|']
    for r in scan['windows']:
        L.append('| %.6g | %d | %.4f | %.4f | %.5f | %.3g | %s | %s |'
                 % (r['qmin'], r['n_pts'], r['B_eq_deg'], r['f_v'], r['r2'],
                    r['curvature_F_p'],
                    'yes' if r['qmin'] == scan['adopted_qmin'] else 'no',
                    '; '.join(r['flags']) if r['flags'] else '—'))
    if scan['window_errors']:
        L += ['', '## 跳过的窗（ValueError）', '']
        L += ['- qmin %.6g：%s' % (float(k), v)
              for k, v in sorted(scan['window_errors'].items())]
    L += ['', '![采纳窗 Ruland plot](%s)' % (
        'adopted/' + os.path.basename(adopted_card['files']['png'])
        if adopted_card is not None else '（无采纳窗）'), '']
    with open(rep_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')

    rj_path = os.path.join(out_dir, '%s_ruland_scan_result.json' % stem)
    result = {'tool': TOOL_TAG, 'mode': 'beq_scan', 'timestamp': ts,
              'bpi2_csv': bpi2_csv, 'b_column': b_column, 'out_dir': out_dir,
              'rule': scan['rule'], 'qmax_nm': scan['qmax_nm'],
              'adopted_qmin': scan['adopted_qmin'],
              'windows': scan['windows'], 'window_errors': scan['window_errors'],
              'scan_flags': scan['flags'],
              'grade': adopted_card['grade'] if adopted_card is not None
              else '未采纳（adopt_manual，转人工裁决）',
              'files': {'csv': csv_path, 'report': rep_path, 'json': rj_path,
                        'adopted': adopted_card['files']
                        if adopted_card is not None else None}}
    with open(rj_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)

    card = {'tool': TOOL_TAG, 'mode': 'beq_scan', 'bpi2_csv': bpi2_csv,
            'out_dir': out_dir, 'rule': scan['rule'],
            'qmax_nm': scan['qmax_nm'], 'adopted_qmin': scan['adopted_qmin'],
            'n_windows': len(scan['windows']),
            'window_errors': scan['window_errors'],
            'scan_flags': scan['flags'],
            'grade': result['grade'],
            'files': {'csv': csv_path, 'report': rep_path, 'json': rj_path,
                      'adopted': adopted_card['files']
                      if adopted_card is not None else None}}
    if adopted_card is not None:
        card.update({'B_eq_rad': adopted_card['B_eq_rad'],
                     'B_eq_deg': adopted_card['B_eq_deg'],
                     'f_v': adopted_card['f_v'], 'r2': adopted_card['r2'],
                     'n_pts': adopted_card['n_pts'],
                     'flags': adopted_card['flags']})
    return card

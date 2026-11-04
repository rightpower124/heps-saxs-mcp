# -*- coding: utf-8 -*-
"""1D SAXS 曲线 gap 检测与多项式填充引擎（M3pre-2 curve_conditioning v0.1.0）

规格卡：.trae/documents/一维曲线gap剔除与填充-规划讨论.md v1.0（2026-09-28 冻结）
本模块是纯引擎（不 import mcp、不做编排）；工具薄壳在 tools_gap.py，批量由 Agent 逐列编排。

三级裁决流的引擎侧职责：
  Tier A/B/C 的区分只体现在 detect 输出的 code/confidence 字段与 fill 报告的
  decision_source 记录上——引擎给出候选与证据，裁决（Agent 目检留痕 / 用户问询）
  发生在编排层。

状态码分类学（写死，规格卡 §5.3）：
  0 = kept（实测保留）
  1 = deep_gap_core（模块 gap 深谷核心，Tier A 全自动）
  2 = shoulder（渐变肩部，Tier B Agent 主判）
  3 = isolated_outlier（孤立异常点，Tier A 全自动）
  4 = broad_shadow（宽缓阴影候选，beamstop 支撑杆，confidence=low，需裁决）
  5 = filled（填充值；一律按"条件值"对待，下游使用必须披露）

硬纪律（规格卡 §3.3 P1–P7 + 用户 R1–R10 裁决）：
  - 填充 = 插值整备，不是信号恢复（R7）；报告措辞禁用"恢复"
  - "x 照抄" = 原文件 token 级照抄，仅替换 mask 行目标列（R1）——规避 float 往返尾数漂移
  - 拟合域分治（R2）：x_semantic='q' → 固定 log I；'phi' → 动态范围 >2 个数量级 → log，
    否则线性；fit_log 参数可显式覆盖
  - 包络硬门（R3）：填充值 ∈ [窗口拟合点 min, max]；违例 → 降阶 → 扩窗 →
    仍违例 clip 到包络 + 强制披露
  - 阶数选择（R9）：1..d_max 逐阶 F 检验（p<0.05 才升阶），防高阶振荡
  - 退化报错（P7，宁可报错不要猜错）：端部 gap、相邻段窗口互叠、mask 占比 >50%、
    phi 跨周期边界疑似、x 非严格单调
  - 无 σ 列 → 两轮 IRLS（均匀初拟合 → 残差 MAD σ 估计 → 加权重拟合），非静默按 1
"""
import json
import math
import os
import sys
from datetime import datetime

import numpy as np

from curve_io import _decode_raw, check_data_file_size   # 复用既有解码链（BOM/UTF-8/GBK），不重写第二份
import workdir                     # 会话级工作目录（部署优化 261104）：缺省产物重定向

# MCP 宿主进程 cwd 不可控（2026-09-28 实测 TRAE 以用户主目录启动 server.py），
# 缺省输出目录必须锚定工作区根（saxs_mcp 的上级），禁止相对 cwd 的 'runs/...'
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

__version__ = '0.2.0'

TOOL_TAG = 'saxs_gap_fill v%s' % __version__

_DEFAULTS = {
    'D_deep': 1.0,      # 深谷判据：log I 相对基线跌落 ≥1 个数量级
    's_tol': 0.2,       # 肩部恢复判据：恢复到基线的 (1-s_tol)=80% 为止
    'D_broad': 0.25,    # 宽缓凹陷确认判据：锚点弦深 ≥25%（T2b 轮 6 锚点法：参照须取自提案区外）
    'D_ext': 0.11,      # 浅筛/回升/扩展判据：11%（提案门槛、U 型回升幅度、两翼扩展，T2b 裁决 a）
    'W_min': 5,         # 宽缓段最小连续点数
    'w_frac': 0.12,     # 基线滑动中值窗口 = n*w_frac（≥11 点，取奇；T2b 候选 1 后 base 仅供 L1 深谷/L2 肩部，L3 已回双窗差分不依赖 base）
    'd_max': 4,         # 多项式最高阶
    'p_f': 0.05,        # F 检验升阶显著性阈值
    'margin': 0.0,      # 包络门余量（相对窗口拟合点 min/max）
    'min_gap_pts': 8,   # 相邻段最小间隔（窗口互叠判据）
    'j_tol': 0.10,      # 衔接台阶阈值（拟合域 dex）：填充段两端与实测邻点的
                        # 最大跳变超此值 → 收窗重拟合，仍超 → 强制披露条件值
    'j_strict': 0.02,   # 端点过冲阈值（dex）：填充端点超出邻域上限此值且邻域
                        # 趋势不支撑 → 判逆趋势过冲（YR8 col1 fix4 事故闸）
    'j_cont': 0.03,     # 连续机制触发阈（dex）：梯级交付台阶超此值且端点正常
                        # 水平 → C0/C1 约束重拟合（用户裁决 2026-09-29 ②）
    'slope_c1': 0.01,   # C1 斜率约束触发阈：端点邻域局部斜率 |dex/x单位| ≥ 此值
    'rms_tol_k': 3.0,   # 拟合误差闸（用户裁决④，相对版）：交付拟合 rms ≤
                        # max(rms_tol_k×梯级最优 rms, rms_floor)。σ_local 二阶
                        # 差分只测高频噪声，低频纹波主导的残差比它大 5-10 倍
                        # （2026-09-29 探针取证），不能作闸参照
    'rms_floor': 0.005, # 误差闸下限（拟合域绝对量：log 域 dex / 线性域相对中位）
    'd_asy': 0.2,       # 不对称收窗重试（2026-09-29 col10 seg1 取证）：对称梯级
                        # 双闸全败且深谷紧贴陡峭真实峰（任何对称窗都含峰翼）时，
                        # 以衔接邻点为锚向外累计 |lg I − lg I_邻| ≤ 此值的连续段
                        # 为该侧窗宽上限，仅收窄剧烈侧、平缓侧保持 3× 起点
}

_CODE_LABEL = {
    0: 'kept', 1: 'deep_gap_core', 2: 'shoulder', 3: 'isolated_outlier',
    4: 'broad_shadow', 5: 'filled',
}

# ---------------------------------------------------------------- 读盘


def _read_columns(path):
    """读 2 列 / N+1 列数值表。返回 dict：
      raw_rows：原始 token 行（每行 [x_tok, y1_tok, ...]），照抄用（R1）
      Y：[n, n_col-1] 数值因变量阵（t[1:]，不含 x 列——x 从 raw_rows 首 token
      取；260930 修正 docstring：原误写 [n, n_col] 致 ruland card 层误用作 x 轴）；
      n_col；header_lines；n_bad
    硬纪律：nan/na/-- 等非有限 token 拒收计 bad 不猜（与 curve_io 同规）；
    行间列数不一致 → 报错。
    """
    check_data_file_size(path)   # 数据类文件 >1MB 拒读（261104 用户裁决，与 load_curve 同规）
    with open(path, 'rb') as f:
        text, encoding, _ev = _decode_raw(f.read())
    if text is None:
        raise ValueError('编码无法判定（非 BOM/UTF-8/GBK）: %s' % path)

    header_lines, raw_rows, n_bad = [], [], 0
    for ln in text.split('\n'):
        s = ln.strip()
        if not s or s.startswith('#'):
            if s:
                header_lines.append(s)
            continue
        toks = s.split(',') if (s.count(',') and '\t' not in s) else \
            s.split('\t') if '\t' in s else s.split()
        toks = [t.strip() for t in toks if t.strip()]
        if not toks:
            continue
        try:
            vals = [float(t) for t in toks]
        except ValueError:
            n_bad += 1
            continue
        if any(not math.isfinite(v) for v in vals):
            n_bad += 1
            continue
        raw_rows.append(toks)

    if not raw_rows:
        raise ValueError('未解析到任何数据行: %s' % path)
    n_col = len(raw_rows[0])
    if n_col < 2:
        raise ValueError('至少需要 2 列（自变量 + 因变量），实得 %d 列' % n_col)
    bad_rows = [i for i, t in enumerate(raw_rows) if len(t) != n_col]
    if bad_rows:
        raise ValueError('数据行列数不一致（首处行号 %d，文件总行 %d，应为 %d 列）；'
                         '不做猜测性裁决，请检查文件' % (bad_rows[0], len(raw_rows), n_col))

    Y = np.array([[float(v) for v in t[1:]] for t in raw_rows], dtype=float)
    return {'raw_rows': raw_rows, 'Y': Y, 'n_col': n_col,
            'header_lines': header_lines, 'n_bad': n_bad,
            'n_rows': len(raw_rows), 'encoding': encoding}


def _internal_x(raw_rows, x_semantic, x_unit):
    """原始 x token → 内部数值轴。q: A^-1→nm^-1 ×10 / nm^-1 ×1（与 curve_io 同惯例）/
    pixel 原值（探测器像素半径轴，2026-09-29 用户裁决扩展：整备为逐点形态学操作，
    横轴仅作标注，像素轴直接可用）；phi: deg 原值。语义/单位必须显式，不做猜测（R2/P4）。"""
    xv = np.array([float(t[0]) for t in raw_rows], dtype=float)
    if x_semantic == 'q':
        if x_unit not in ('A^-1', 'nm^-1', 'pixel'):
            raise ValueError("x_unit 只支持 'A^-1'/'nm^-1'/'pixel'（x_semantic='q'），不做单位猜测")
        return xv * (10.0 if x_unit == 'A^-1' else 1.0)
    if x_semantic == 'phi':
        if x_unit != 'deg':
            raise ValueError("x_semantic='phi' 时 x_unit 必须为 'deg'，不做单位猜测")
        return xv
    raise ValueError("x_semantic 只支持 'q'（径向积分）或 'phi'（方位角切割）")


def _check_x_monotonic(x):
    """x 严格单调递增校验（P7）"""
    d = np.diff(x)
    if np.any(d <= 0):
        raise ValueError('自变量非严格单调递增（%d 处 dX<=0）；不做猜测性裁决，'
                         '请先检查数据排序' % int(np.sum(d <= 0)))


# ---------------------------------------------------------------- 检测引擎

def _sliding_median(a, w):
    """nan 感知滑动中值（窗取奇；边缘 nan 填充）"""
    w = int(w) | 1
    n = len(a)
    if n <= w:
        med = np.nanmedian(a) if np.any(np.isfinite(a)) else np.nan
        return np.full(n, med)
    h = w // 2
    pad = np.concatenate([np.full(h, np.nan), a, np.full(h, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(pad, w)
    with np.errstate(all='ignore'):
        return np.nanmedian(win, axis=1)


def _robust_baseline_log(I, exclude=None, mad_rounds=None):
    """log10 域 robust 基线：滑动中值 + 2 轮 MAD 剔除重算。
    exclude：剔除区布尔掩码（T2b 裁决 α：深谷核心置 NaN 后重算，滑动中值天然
    跳过毒化区——实测 20 点深谷可把邻域基线拉低 ≈0.3 dex 令肩部 rel 归零）。
    mad_rounds：MAD 清洗轮数；None = exclude 存在时取 0，否则 2。
    【T2b 轮 2 取证：exclude 孔 × 全局 3σ 清洗 = 孔生长灾难——log 域平滑曲线
    mad≈0.002 → 3σ 阈值仅 0.0085 dex，单轮剔 130 点，基线 NaN 孔从核心区扩散
    至曲线头部（lgw NaN 241/400），深谷 rel 全非有限致检测反丢。pass-2 输入
    已剔核心区，清洗冗余且唯余害处，故 exclude 存在时跳过清洗。】
    I<=0 点以正值中值占位参与估计（P1），其异常性由深谷判据另行捕捉。"""
    n = len(I)
    lg = np.full(n, np.nan)
    pos = np.isfinite(I) & (I > 0)
    lg[pos] = np.log10(I[pos])
    fill_val = float(np.nanmedian(lg[pos])) if pos.any() else 0.0
    lgw = np.where(pos, lg, fill_val)
    if exclude is not None:
        lgw = np.where(exclude, np.nan, lgw)

    w = max(int(n * _DEFAULTS['w_frac']), 11) | 1
    base = _sliding_median(lgw, w)
    rounds = (0 if exclude is not None else 2) if mad_rounds is None else int(mad_rounds)
    for _ in range(rounds):
        r = lgw - base
        med_r = np.nanmedian(r)
        mad = np.nanmedian(np.abs(r - med_r))
        if not np.isfinite(mad) or mad <= 0:
            break
        lgw = np.where(np.abs(r - med_r) <= 3.0 * 1.4826 * mad, lgw, np.nan)
        base = _sliding_median(lgw, w)
    return base


def _runs(mask):
    """布尔数组 → 连续 True 段 [(start_idx, end_idx)（含端点）]"""
    segs, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        elif not m and s is not None:
            segs.append((s, i - 1))
            s = None
    if s is not None:
        segs.append((s, len(mask) - 1))
    return segs


def _seg_tag(seg):
    return '[%d..%d]x[%.4g..%.4g]' % (seg['start_idx'], seg['end_idx'],
                                      seg.get('x_start', float('nan')),
                                      seg.get('x_end', float('nan')))


def _phi_full_circle_hint(x):
    """phi 全周期疑似：跨度≈360°且两端贴近周期边界（仅提示，不裁决）"""
    span = float(x[-1] - x[0])
    if span < 350.0:
        return False
    def _edge_dist(v):
        return min(abs(v + 180.0), abs(v), abs(v - 360.0))
    return _edge_dist(float(x[0])) < 5.0 and _edge_dist(float(x[-1])) < 5.0


def detect_gaps(x, Y, col=0, params=None):
    """单列 gap 检测（规格卡 §5.1）。x: 内部单调轴；Y: [n, n_col]；col: 0-based。
    返回 (segments, base, notes)。
      segments 段字段：start_idx/end_idx/x_start/x_end/code/label/confidence/note
      code: 1 深谷核心 / 2 肩部 / 3 孤立异常点 / 4 宽缓阴影候选
    """
    p = dict(_DEFAULTS)
    if params:
        p.update(params)
    I = Y[:, col]
    n = len(x)
    notes = []

    # ---- Level 0：非物理值直接入候选（P1：I<=0 或非有限）
    seed_mask = ~np.isfinite(I) | (I <= 0)

    base = _robust_baseline_log(I)
    with np.errstate(all='ignore'):
        lg = np.where(I > 0, np.log10(np.where(I > 0, I, 1.0)), np.nan)
    rel1 = lg - base                   # 第一遍：毒化基线，仅作深谷初判（判据 -1 dex 远大于毒化幅度 ≈0.3 dex）

    # ---- Level 1 第一遍：深谷核心初判（含 1-2 点短深谷，T2a 修复）
    core1 = seed_mask.copy()
    for s, e in _runs(np.isfinite(rel1) & (rel1 < -p['D_deep'])):
        core1[s:e + 1] = True

    # ---- T2b 裁决 α：核心区置 NaN 重算基线，第二遍为最终参照
    base = _robust_baseline_log(I, exclude=core1)
    rel = lg - base                    # 相对基线（log10 比值）

    # ---- Level 1 终判：≥3 点深谷核心；1-2 点深谷归孤立异常点（T2a 修复）
    deep = np.zeros(n, dtype=bool)
    short_deep = np.zeros(n, dtype=bool)
    for s, e in _runs(np.isfinite(rel) & (rel < -p['D_deep'])):
        if e - s + 1 >= 3:
            deep[s:e + 1] = True
        else:
            short_deep[s:e + 1] = True
    core_mask = deep | seed_mask | short_deep   # I<=0 并入深谷核心（无光子记录）

    # ---- Level 2：肩部扩展（双参照护栏，col10 事故修复 2026-09-29）：
    # 判据① rel < log10(1-s_tol)（相对剔除后基线）+ 判据② rel1 < 0（相对第一遍
    # 正常基线不得偏高）。仅①时基线重算的 NaN 孔会让孔缘滑动中值取到峰麓
    # 中值（col10：idx138 处基线≈1200 而真基线 605），高于真实基线的完整峰麓
    # 数据（rel1≈+0.04..+0.20）被误判"低于基线 26%"→ 吃满 k_max 扩展上限，
    # 剔除区吞掉真实峰形致 rms 恶化 5–10 倍。判据②取 rel1<0 而非 <thr_sh：
    # col1 真肩部 [577..582] rel1=−0.059..−0.093 高于 thr_sh=−0.097（第二遍
    # 基线被邻侧结构抬高致 rel 更负才入选，实测取证），用 thr_sh 会砍掉钦定
    # 基准肩部；物理分界线是"高于正常水平的数据绝不是渐变翼"（col10 伪肩部
    # rel1 全为正 / col1 真肩部 rel1 全为负，逐点取证 2026-09-29）。
    # 距离上限 K（T2b 轮 3 取证）：幂律陡降段相对滞后基线 rel≈-0.10~-0.14 持续
    # 为负，无上限扩展会吞掉 ~9 个正常点（false_pos 爆表）。肩部物理宽度有限
    # （gap 边缘像素部分遮挡 ≈ 核心宽的 10-30%，规格卡 §1.2），K = max(3, core_w//4)
    thr_sh = math.log10(1.0 - p['s_tol'])
    shoulder = np.zeros(n, dtype=bool)
    for s, e in _runs(core_mask):
        k_max = max(3, (e - s + 1) // 4)
        i = s - 1
        while i >= 0 and (s - i) <= k_max and not core_mask[i] \
                and np.isfinite(rel[i]) and rel[i] < thr_sh \
                and np.isfinite(rel1[i]) and rel1[i] < 0.0:
            shoulder[i] = True
            i -= 1
        j = e + 1
        while j < n and (j - e) <= k_max and not core_mask[j] \
                and np.isfinite(rel[j]) and rel[j] < thr_sh \
                and np.isfinite(rel1[j]) and rel1[j] < 0.0:
            shoulder[j] = True
            j += 1

    # ---- Level 3：宽缓凹陷候选（锚点弦参照 + U 型回升验证；confidence=low，Tier B/C）
    # T2b 轮 5 取证：中值类参照（双窗差分/单 base）对广浅凹陷均失效——中值统计天然
    # 稀释浅凹（阴影占窄窗 84%，med1-med2 仅 -0.07~-0.12），且参照自身被阴影拖低
    # （med2 窗内占 38%，去趋势深度实测仅 -0.109）。轮 6 起改锚点法（用户裁决 3 之候选 2）：
    #   ① 提案：去趋势信号 d = lg_l3 - med2 < d_ext（浅筛）连续 ≥W_min；
    #   ② 锚点：提案区两侧窗外（避开 core/shoulder/已确认区）各取 w1 宽锚窗中值连线
    #      为局部趋势弦；锚点不足（如紧邻深谷）→ 如实拒绝留痕；
    #   ③ 确认：区内最大弦深 ≥D_broad 且两端弦深回升 ≥D_ext（U 型——排除单调陡降
    #      曲率伪迹，规格 §5.1"两侧同高"的鲁棒实现；轮 5 实测伪迹右缘无回升）；
    #   ④ 扩展：确认区两翼按弦深 ≥D_ext 收入浅翼。
    lg_l3 = np.where(core_mask, np.nan, lg)
    w1 = max(n // 16, 15) | 1
    w2 = (4 * w1) | 1
    with np.errstate(all='ignore'):
        med2 = _sliding_median(lg_l3, w2)
        d_det = lg_l3 - med2              # 去趋势信号（浅筛用；仅提案，不作深度结论）
    d_broad = -math.log10(1.0 - p['D_broad'])   # 弦深阈值（dex，正值约定）
    d_ext = -math.log10(1.0 - p['D_ext'])
    edge_guard = np.zeros(n, dtype=bool)
    edge_guard[w1 // 2:n - w1 // 2] = True   # 窄窗半宽（原 w2/2 过宽，盖掉末端真实阴影——L1 实测）
    prop = np.isfinite(d_det) & (d_det < -d_ext) & ~core_mask & ~shoulder & edge_guard
    broad = np.zeros(n, dtype=bool)
    ineligible = core_mask | shoulder         # 锚点不可用区
    for s, e in _runs(prop):
        if e - s + 1 < max(int(p['W_min']), 3):
            continue
        iL = [i for i in range(max(0, s - w1), s)
              if np.isfinite(lg_l3[i]) and not ineligible[i] and not broad[i]]
        iR = [i for i in range(e + 1, min(n, e + 1 + w1))
              if np.isfinite(lg_l3[i]) and not ineligible[i] and not broad[i]]
        if len(iL) < 5 or len(iR) < 5:
            notes.append('L3 提案 [%d..%d] 锚点不足（L%d/R%d，疑邻 gap 或数据边缘），拒绝留痕'
                         % (s, e, len(iL), len(iR)))
            continue
        xL, yL = float(np.median(iL)), float(np.nanmedian(lg_l3[iL]))
        xR, yR = float(np.median(iR)), float(np.nanmedian(lg_l3[iR]))
        idx_z = np.arange(s, e + 1)
        chord = yL + (yR - yL) * (idx_z - xL) / (xR - xL)
        depth = chord - lg_l3[s:e + 1]        # 正值 = 低于趋势弦
        dfin = depth[np.isfinite(depth)]
        if len(dfin) < max(int(p['W_min']), 3):
            continue
        dmax = float(np.max(dfin))
        # U 型回升验证（规格 §5.1"两侧同高"），回升在区外远端侧翼测量——轮 7 取证：
        # 平滑凹陷的提案区只捕获平底（区内缘深度恒 ~0.21，区内测回升恒失败）；
        # 侧翼窗取区外 [s-15,s-5)/[e+6,e+16)：真凹陷此处已回升近趋势，单调陡降
        # 伪迹此处仍在趋势上（其确认由锚点不足/深度门另行拦截）
        fL = [yL + (yR - yL) * (i - xL) / (xR - xL) - lg_l3[i]
              for i in range(max(0, s - 15), max(0, s - 5))
              if np.isfinite(lg_l3[i]) and not ineligible[i] and not broad[i]]
        fR = [yL + (yR - yL) * (i - xL) / (xR - xL) - lg_l3[i]
              for i in range(e + 6, min(n, e + 16))
              if np.isfinite(lg_l3[i]) and not ineligible[i] and not broad[i]]
        if len(fL) < 3 or len(fR) < 3:
            notes.append('L3 提案 [%d..%d] 侧翼不足，回升无法验证，拒绝留痕' % (s, e))
            continue
        recL = float(np.median(fL))
        recR = float(np.median(fR))
        if dmax >= d_broad and recL <= dmax - d_ext and recR <= dmax - d_ext:
            broad[s:e + 1] = True
            # ④ 扩展：两翼按弦深 ≥ d_ext 收入（弦延拓到区外）
            i = s - 1
            while i >= 0 and not broad[i] and not ineligible[i] and edge_guard[i] \
                    and np.isfinite(lg_l3[i]) \
                    and (yL + (yR - yL) * (i - xL) / (xR - xL) - lg_l3[i]) >= d_ext:
                broad[i] = True
                i -= 1
            j = e + 1
            while j < n and not broad[j] and not ineligible[j] and edge_guard[j] \
                    and np.isfinite(lg_l3[j]) \
                    and (yL + (yR - yL) * (j - xL) / (xR - xL) - lg_l3[j]) >= d_ext:
                broad[j] = True
                j += 1

    # ---- 组装 segments
    segments = []
    for s, e in _runs(core_mask):
        code, conf, note = 1, 'high', ''
        if e - s + 1 <= 2 and not seed_mask[s:e + 1].any():
            code, conf, note = 3, 'high', '1-2 点深谷且非 I<=0，归孤立异常点'
        segments.append({'start_idx': s, 'end_idx': e,
                         'x_start': float(x[s]), 'x_end': float(x[e]),
                         'code': code, 'label': _CODE_LABEL[code],
                         'confidence': conf, 'note': note})
    for s, e in _runs(shoulder):
        segments.append({'start_idx': s, 'end_idx': e,
                         'x_start': float(x[s]), 'x_end': float(x[e]),
                         'code': 2, 'label': _CODE_LABEL[2],
                         'confidence': 'medium',
                         'note': '肩部扩展：两侧独立，允许不对称'})
    for s, e in _runs(broad):
        segments.append({'start_idx': s, 'end_idx': e,
                         'x_start': float(x[s]), 'x_end': float(x[e]),
                         'code': 4, 'label': _CODE_LABEL[4],
                         'confidence': 'low',
                         'note': '宽缓凹陷候选：易与真实宽峰混淆，需 Agent 目检/用户裁决'})

    # ---- 退化情形备注（P7）
    if segments:
        if segments[0]['start_idx'] == 0:
            notes.append('端部 gap：首段贴曲线起点，无左侧邻域，填充将报错（P7）')
        if segments[-1]['end_idx'] == n - 1:
            notes.append('端部 gap：末段贴曲线终点，无右侧邻域，填充将报错（P7）')
        for a, b in zip(segments[:-1], segments[1:]):
            gap_pts = b['start_idx'] - a['end_idx'] - 1
            if gap_pts < p['min_gap_pts']:
                notes.append('相邻段窗口互叠风险：%s 与 %s 间隔仅 %d 点 < %d'
                             % (_seg_tag(a), _seg_tag(b), gap_pts, p['min_gap_pts']))
    if _phi_full_circle_hint(x):
        notes.append('phi 全周期疑似：段触及两端时为跨周期边界 gap，'
                     '本引擎不自动拼接，需用户确认（P7）')
    return segments, base, notes


# ---------------------------------------------------------------- 填充引擎

def _choose_fit_log(I_pos, fit_log, x_semantic):
    """拟合域分治（R2）：'q' → log；'phi' → 动态范围 >2 数量级 → log 否则线性；
    fit_log 显式传参时覆盖自动判定。I_pos: 有限正值数组。"""
    if fit_log is not None:
        return bool(fit_log)
    if x_semantic == 'q':
        return True
    if len(I_pos) < 5:
        raise ValueError('正强度点不足 5 个，无法自动判定拟合域；请显式传 fit_log')
    return float(I_pos.max() / I_pos.min()) > 100.0


def _f_test_p(ssr_small, dof_small, ssr_big, dof_big):
    """嵌套模型 F 检验：高阶是否显著改善。返回 p 值。"""
    from scipy.stats import f as f_dist
    d1 = dof_small - dof_big
    if d1 <= 0 or ssr_big <= 0 or dof_big <= 0:
        return 1.0
    F = (ssr_small - ssr_big) / d1 / (ssr_big / dof_big)
    if F <= 0:
        return 1.0
    return float(f_dist.sf(F, d1, dof_big))


def _polyfit_order(xn, y, w, d_max, p_f):
    """加权多项式 1..d_max 逐阶 F 检验选阶（R9）。返回 (order, coefs, ssr)。"""
    n_eff = len(y)
    best = (0, None, np.inf)
    for d in range(1, d_max + 1):
        try:
            c = np.polyfit(xn, y, d, w=np.sqrt(w))
        except (np.linalg.LinAlgError, ValueError):
            break
        r = y - np.polyval(c, xn)
        ssr = float(np.sum(w * r * r))
        dof = n_eff - (d + 1)
        if dof <= 1:
            if best[1] is None:
                best = (d, c, ssr)
            break
        if best[1] is not None:
            p = _f_test_p(best[2], n_eff - best[0] - 1, ssr, dof)
            if p >= p_f:
                break                     # 升阶不显著 → 停在上一阶
        best = (d, c, ssr)
    if best[1] is None:
        c = np.polyfit(xn, y, 1, w=np.sqrt(np.maximum(w, 1e-300)))
        r = y - np.polyval(c, xn)
        best = (1, c, float(np.sum(np.maximum(w, 1e-300) * r * r)))
    return best


def _fit_on_window(x_win, I_win, valid, use_log, p):
    """在给定窗口上做两轮 IRLS 加权多项式拟合。
    x_win/I_win: 窗口切片；valid: 参与拟合布尔（排除全部 mask 点）。
    返回 dict(order, coefs, rms, n_eff, x_lo, x_hi, sigma, fit_log)。"""
    xv = x_win[valid]
    yv_lin = I_win[valid]
    yv = np.log10(yv_lin) if use_log else yv_lin.copy()
    x_lo, x_hi = float(xv.min()), float(xv.max())
    span = max(x_hi - x_lo, 1e-30)
    mid = 0.5 * (x_lo + x_hi)
    xn = 2.0 * (xv - mid) / span

    w = np.ones_like(yv)                                       # 第 1 轮均匀
    d, coef, ssr = _polyfit_order(xn, yv, w, p['d_max'], p['p_f'])
    r = yv - np.polyval(coef, xn)                              # 第 2 轮 IRLS
    mad = float(np.median(np.abs(r - np.median(r))))
    sigma = 1.4826 * mad if mad > 0 else 1.0
    if math.isfinite(sigma) and sigma > 0:
        w = 1.0 / (sigma * sigma) * np.ones_like(yv)
    d, coef, ssr = _polyfit_order(xn, yv, w, p['d_max'], p['p_f'])
    return {'order': d, 'coefs': coef, 'rms': math.sqrt(ssr / len(yv)),
            'n_eff': int(len(yv)), 'x_lo': x_lo, 'x_hi': x_hi, 'span': span,
            'mid': mid, 'sigma': float(sigma), 'fit_log': bool(use_log)}


def _eval_fit(fit, x_pts):
    """拟合式求值（log 域 → 线性域；强制非负 P1）"""
    xn = 2.0 * (np.asarray(x_pts, dtype=float) - fit['mid']) / fit['span']
    y = np.polyval(fit['coefs'], xn)
    if fit['fit_log']:
        y = np.power(10.0, y)
    return np.maximum(y, 0.0)


def fill_gaps(x, Y, col, segments, x_semantic, params=None, fit_log=None):
    """单列填充（规格卡 §5.2）。x: 内部轴；Y: [n, n_col]；col: 0-based。
    segments: detect 输出或用户手写表（dict 列表；无 start_idx/end_idx 时用
    x_start/x_end 最近邻映射）。
    返回 (Y_filled, mask_codes, fill_report, issues, use_log, filled_fraction)。
    """
    p = dict(_DEFAULTS)
    if params:
        p.update(params)
    n, n_col = Y.shape
    I = Y[:, col].copy()
    issues = []

    # ---- gap_table 规格化
    segs = []
    for sg in segments:
        s = dict(sg)
        if 'start_idx' not in s:
            if 'x_start' not in s or 'x_end' not in s:
                raise ValueError('gap_table 段缺少 start_idx/end_idx 或 x_start/x_end: %s' % sg)
            si = int(np.argmin(np.abs(x - float(s['x_start']))))
            ei = int(np.argmin(np.abs(x - float(s['x_end']))))
            if si > ei:
                si, ei = ei, si
            s['start_idx'], s['end_idx'] = si, ei
        s['code'] = int(s.get('code', 1))
        s.setdefault('confidence', 'user')
        s.setdefault('note', '')
        s.setdefault('decision_source', 'AUTO')
        segs.append(s)
    segs.sort(key=lambda s: s['start_idx'])

    # ---- 合并为逐点 mask 的连续 run（同源链 core+shoulder+shadow 天然贴邻，
    # 必须先合并再跑 P7 纪律；合并 code 取最严重的 min(codes)）
    code_arr = np.zeros(n, dtype=int)
    for s in segs:
        sl = slice(s['start_idx'], s['end_idx'] + 1)
        cur = code_arr[sl]
        code_arr[sl] = np.where(cur == 0, s['code'], np.minimum(cur, s['code']))
    runs = []
    for a, b in _runs(code_arr > 0):
        codes_merged = sorted({int(c) for c in code_arr[a:b + 1] if c > 0})
        src_segs = [s for s in segs
                    if s['start_idx'] <= b and s['end_idx'] >= a]
        runs.append({
            'start_idx': a, 'end_idx': b,
            'x_start': float(x[a]), 'x_end': float(x[b]),
            'code': int(min(codes_merged)), 'codes_merged': codes_merged,
            'label': _CODE_LABEL.get(int(min(codes_merged)), '?'),
            'confidence': 'user' if any(s['confidence'] == 'user' for s in src_segs)
            else ';'.join(s['confidence'] for s in src_segs),
            'note': ('合并自 codes=%s：' % codes_merged
                     + '；'.join(s['note'] for s in src_segs if s['note']))
            if len(codes_merged) > 1
            else ';'.join(s['note'] for s in src_segs if s['note']),
            'decision_source': ('USER' if any(s['decision_source'] == 'USER'
                                              for s in src_segs)
                                else src_segs[0]['decision_source']),
        })
    segs = runs

    # ---- 退化报错（P7，宁可报错不要猜错；基于合并后的 run）
    if not segs:
        raise ValueError('gap_table 为空：没有需要填充的区段')
    if segs[0]['start_idx'] == 0 or segs[-1]['end_idx'] == n - 1:
        raise ValueError('端部 gap（区段贴曲线首/尾端，无单侧邻域）；P7 纪律：报错不填充')
    for a, b in zip(segs[:-1], segs[1:]):
        if b['start_idx'] - a['end_idx'] - 1 < p['min_gap_pts']:
            raise ValueError('相邻 mask 区段窗口互叠（%s 与 %s 间隔 < %d 点）；'
                             'P7 纪律：报错不填充'
                             % (_seg_tag(a), _seg_tag(b), p['min_gap_pts']))
    n_mask = sum(s['end_idx'] - s['start_idx'] + 1 for s in segs)
    if n_mask > 0.5 * n:
        raise ValueError('mask 占比 %.1f%% > 50%%；P7 纪律：报错不填充' % (100.0 * n_mask / n))
    if x_semantic == 'phi' and segs[0]['start_idx'] == 0 and segs[-1]['end_idx'] == n - 1:
        raise ValueError('phi 全周期曲线两端均为 gap：疑似跨周期边界，'
                         '本引擎不自动拼接；请用户确认后手动处理')

    # ---- 拟合域（R2）
    pos_all = I[np.isfinite(I) & (I > 0)]
    use_log = _choose_fit_log(pos_all, fit_log, x_semantic)

    mask_codes = np.zeros(n, dtype=int)
    fill_report = []
    Y_filled = Y.copy()

    for sg in segs:
        s_i, e_i = sg['start_idx'], sg['end_idx']
        seg_w = e_i - s_i + 1
        rec = {'start_idx': s_i, 'end_idx': e_i,
               'x_start': float(x[s_i]), 'x_end': float(x[e_i]),
               'code': sg['code'], 'label': _CODE_LABEL.get(sg['code'], '?'),
               'confidence': sg['confidence'], 'note': sg['note'],
               'decision_source': sg['decision_source'], 'disclosure': []}

        # 核验回退（共享 gap 表场景）：本列该处须有实际跌落，否则跳过该段
        chk = I[max(0, s_i - 3):min(n, e_i + 4)]
        pos_chk = chk[np.isfinite(chk) & (chk > 0)]
        if len(pos_chk) == 0:
            rec['disclosure'].append('段内无正强度点，跳过该段（shared 表核验回退）')
            issues.append({'type': 'segment_skipped', 'segment': _seg_tag(sg),
                           'reason': 'no_positive_I'})
            fill_report.append(rec)
            continue
        base_lvl = float(np.median(pos_chk))

        # ---- 选窗梯级 + 评分（2026-09-29 用户裁决①④重构）：
        # 起点 = 3× 规则窗（两侧各 1× 段宽），×0.75 逐级自动收窗至
        # floor = 2× 规则窗（两侧各 0.5× 段宽；强度变化剧烈处自然落到窄级）。
        # 逐级按"交付拟合"评分（包络处置后），选取规则：
        #   ① 拟合误差闸（用户裁决④，相对版）：交付拟合的拟合域 rms ≤
        #     max(rms_tol_k×梯级最优 rms, rms_floor)——"误差小"以本列数据可达
        #     的最优拟合为参照（引擎旧 'rms' 是残差自归一化量恒 ≈1 无判别力；
        #     σ_local 二阶差分只测高频噪声，低频纹波主导时失效，见探针取证）
        #   ② 双闸（误差+台阶）合格级中取拟合 rms 最小者（拟合误差小是填充
        #     前提）；近平局（5% 带）取更窄窗（拟合质量统计不可分时，窄窗少含
        #     峰区数据、外推风险小，与"范围过宽"裁决同向）
        #   ③ 无双闸合格级 → 误差合格级中取台阶最小者（披露台阶）；仍无 →
        #     全级取 rms 最小者（披露"拟合失败"），一律按条件值降级
        def _jstep(fit, w_lo, w_hi):
            yj = _eval_fit(fit, np.array([x[s_i], x[e_i]]))   # 线性域（同包络门口径）
            ref = np.array([I[s_i - 1], I[e_i + 1]], dtype=float)
            if fit['fit_log']:
                with np.errstate(invalid='ignore'):
                    yj = np.log10(np.where(yj > 0, yj, np.nan))
                    ref = np.log10(np.where(ref > 0, ref, np.nan))
            d = np.abs(yj - ref)
            return float(np.nanmax(d)) if np.isfinite(d).any() else float('inf')

        x_fill = x[s_i:e_i + 1]

        def _win_kp(w_lo, w_hi):
            """窗口内参与拟合/包络的未标记点掩码；拟合域非有限点（log 域
            I≤0 → log10=-inf）一并剔除——1.dat pixel 首测（2026-09-29）：
            端部零点撤案后留窗，-inf 毒化多项式拟合且 min(nan) 顺序依赖
            污染候选选取，nan 填充值落盘。所有 kp/keep/valid 掩码统一走本函数。"""
            k = np.ones(w_hi - w_lo + 1, dtype=bool)
            k[s_i - w_lo:e_i - w_lo + 1] = False
            for other in segs:
                a, b = other['start_idx'], other['end_idx']
                if b < w_lo or a > w_hi or (a == s_i and b == e_i):
                    continue
                k[max(a - w_lo, 0):min(b - w_lo, w_hi - w_lo) + 1] = False
            if use_log:
                with np.errstate(invalid='ignore', divide='ignore'):
                    k &= np.isfinite(np.log10(I[w_lo:w_hi + 1]))
            return k

        def _delivered(fit0, w_lo, w_hi):
            """包络门处置后的"交付拟合"。梯级评分必须用交付拟合——
            YR8 col1 教训：h2=39 梯级原生 4 阶 j=0.0067 全场最优，但包络违例
            降 3 阶后交付 j 恶化至 0.022（左端 11299 过冲事故）；h2=28 原生
            3 阶无违例、交付 j=0.0114 才是全局最优（用户钦定 fix 基准）。"""
            kp = _win_kp(w_lo, w_hi)      # 与拟合同规（含非有限点剔除）
            e_lo = float(I[w_lo:w_hi + 1][kp].min()) - p['margin']
            e_hi = float(I[w_lo:w_hi + 1][kp].max()) + p['margin']
            note = None
            fl = _eval_fit(fit0, x_fill)
            if ((fl < e_lo) | (fl > e_hi)).any():
                r = (np.log10(I[w_lo:w_hi + 1][kp]) if fit0['fit_log']
                     else I[w_lo:w_hi + 1][kp])
                xn1 = 2.0 * (x[w_lo:w_hi + 1][kp] - fit0['mid']) / fit0['span']
                for dd in range(int(fit0['order']) - 1, 0, -1):
                    ftd = dict(fit0)
                    ftd['order'], ftd['coefs'] = dd, np.polyfit(xn1, r, dd)
                    y2 = _eval_fit(ftd, x_fill)
                    if not ((y2 < e_lo) | (y2 > e_hi)).any():
                        fit0, note = ftd, '包络门违例 → 降为 %d 阶重拟合' % dd
                        break
            return fit0, note

        def _end_overshoot(fit):
            """端点逆趋势过冲检查（fix4 事故第二道闸）：填充端点（拟合域）
            超出邻域最近 4 个未标记点上限 j_strict dex，且邻域趋势不支撑该
            抬高（左端斜率≤+容差 / 右端斜率≥−容差）→ 过冲。返回过冲侧数。"""
            mset = set()
            for other in segs:
                if (other['start_idx'], other['end_idx']) == (s_i, e_i):
                    continue
                mset.update(range(other['start_idx'], other['end_idx'] + 1))

            def _lg(v):
                v = np.asarray(v, dtype=float)
                if fit['fit_log']:
                    with np.errstate(invalid='ignore'):
                        return np.log10(np.where(v > 0, v, np.nan))
                return v

            hits = 0
            for side, edge in (('L', s_i), ('R', e_i)):
                idxs = []
                rng = range(edge - 1, -1, -1) if side == 'L' else range(edge + 1, n)
                for i2 in rng:
                    if i2 not in mset:
                        idxs.append(i2)
                    if len(idxs) == 4:
                        break
                if len(idxs) < 2:
                    continue
                ys = _lg(I[np.array(idxs)])
                if not np.isfinite(ys).all():
                    continue
                sl = float(np.polyfit(x[np.array(idxs)], ys, 1)[0])
                ceil = float(np.nanmax(ys)) + p['j_strict']
                v_fit = float(_lg(_eval_fit(fit, np.array([x[edge]])))[0])
                if not np.isfinite(v_fit) or v_fit <= ceil:
                    continue
                if (side == 'L' and sl <= 5e-3) or (side == 'R' and sl >= -5e-3):
                    hits += 1
            return hits

        def _local_sigma(w_lo, w_hi):
            """数据内禀噪声 σ（拟合域）：二阶差分 MAD（等间距下 Var(d2)=6σ²）。
            点距非均匀时近似成立，闸阈值另有 rms_tol_k 倍裕量兜底。"""
            kp = _win_kp(w_lo, w_hi)
            v = np.log10(I[w_lo:w_hi + 1]) if use_log else I[w_lo:w_hi + 1]
            d2 = [v[i2 - 1] - 2.0 * v[i2] + v[i2 + 1]
                  for i2 in range(1, w_hi - w_lo)
                  if kp[i2 - 1] and kp[i2] and kp[i2 + 1]]
            if len(d2) < 5:
                return float('nan')
            d2 = np.array(d2)
            mad = float(np.median(np.abs(d2 - np.median(d2))))
            return mad / 1.4826 / math.sqrt(6.0) if mad > 0 else float('nan')

        floor = max(p['min_gap_pts'], seg_w // 2)
        cands, half = [], max(seg_w, p['min_gap_pts'])   # 3× 规则窗起点
        for _attempt in range(12):
            lo = max(0, s_i - half)
            hi = min(n - 1, e_i + half)
            valid = _win_kp(lo, hi)
            if int(valid.sum()) < max(8, p['d_max'] + 2):
                if half == max(seg_w, p['min_gap_pts']):
                    half = max(half + 1, int(half * 1.5))  # 起步级即不足：扩窗兜底
                    continue
                break
            fit0 = _fit_on_window(x[lo:hi + 1], I[lo:hi + 1], valid, use_log, p)
            fitD, noteD = _delivered(fit0, lo, hi)
            ovs = _end_overshoot(fitD)
            j = _jstep(fitD, lo, hi) + 1.0 * ovs
            kp = _win_kp(lo, hi)
            yw = np.log10(I[lo:hi + 1]) if fitD['fit_log'] else I[lo:hi + 1]
            rvec = yw[kp] - np.polyval(fitD['coefs'],
                                       2.0 * (x[lo:hi + 1][kp] - fitD['mid'])
                                       / fitD['span'])
            rms_d = float(math.sqrt(float(np.mean(rvec * rvec))))
            sig_l = _local_sigma(lo, hi)
            cands.append({'fit': fitD, 'lo': lo, 'hi': hi, 'j': j, 'ovs': ovs,
                          'rms': rms_d, 'sig': sig_l, 'note': noteD,
                          'half': half})
            if os.environ.get('GAP_FILL_DEBUG'):
                print('[rung] h=%d win=[%.2f,%.2f] order=%d rms=%.4f sig=%.4f '
                      'j=%.4f ovs=%d'
                      % (half, x[lo], x[hi], fitD['order'], rms_d, sig_l,
                         j, ovs), file=sys.stderr)
            if half <= floor:
                break
            half = max(floor, int(half * 0.75))
        if not cands:
            raise ValueError('区段 %s 找不到有效拟合窗口（有效点不足）；P7 纪律：报错不填充'
                             % _seg_tag(sg))
        # 相对误差闸（探针取证后修正，见 _DEFAULTS rms_tol_k 注）+
        # 触发式不对称收窗重试（col10 seg1 取证：峰紧贴 gap 边缘时任何对称
        # 窗都含陡峭峰翼，全梯级 j 超阈属原理性失效；以衔接邻点为锚仅收窄
        # 剧烈侧、平缓侧保持 3× 起点；对称达标场景（col1 等）elig 非空永不
        # 触发，钦定基准构造性不动）
        def _apply_gates():
            fin_r = [c['rms'] for c in cands if math.isfinite(c['rms'])]
            if not fin_r:                 # 全 nan：min(nan) 顺序依赖防线
                for c in cands:
                    c['thr'], c['gate'] = float('nan'), False
                return []
            rms_b = min(fin_r)
            for c in cands:
                c['thr'] = max(p['rms_tol_k'] * rms_b, p['rms_floor'])
                c['gate'] = c['rms'] <= c['thr']
            return [c for c in cands if c['gate'] and c['j'] <= p['j_tol']]

        elig = _apply_gates()
        if not elig:
            def _cap(edge, step, half0):
                """该侧不对称收窗上限：以 gap 衔接邻点为锚，向外累计与锚点
                |lg I − lg I_锚| ≤ d_asy 的连续点数（上界 half0，下限 3，
                遇其他 gap 段即断）。锚点缺失/非有限 → 返回 half0（不收窄）。"""
                j0 = edge + step
                if not (0 <= j0 < n):
                    return half0
                with np.errstate(invalid='ignore', divide='ignore'):
                    lr = float(np.log10(I[j0])) if use_log else float(I[j0])
                if not np.isfinite(lr):
                    return half0
                cnt, i2 = 0, j0 + step
                while 0 <= i2 < n and cnt < half0:
                    if i2 in mset_asy:
                        break
                    with np.errstate(invalid='ignore', divide='ignore'):
                        lv = float(np.log10(I[i2])) if use_log else float(I[i2])
                    if not np.isfinite(lv) or abs(lv - lr) > p['d_asy']:
                        break
                    cnt += 1
                    i2 += step
                return max(3, cnt)

            def _geom(cap, floor2):
                seq, h = [], cap
                while True:
                    seq.append(h)
                    if h <= floor2:
                        break
                    h2 = max(floor2, int(h * 0.75))
                    if h2 >= h:
                        break
                    h = h2
                return seq

            def _asy(hl, hr):
                lo2 = max(0, s_i - hl)
                hi2 = min(n - 1, e_i + hr)
                kp0 = _win_kp(lo2, hi2)
                if (lo2, hi2) in seen_asy or \
                        int(kp0.sum()) < max(8, p['d_max'] + 2):
                    return
                fit0 = _fit_on_window(x[lo2:hi2 + 1], I[lo2:hi2 + 1],
                                      kp0, use_log, p)
                fitD, noteD = _delivered(fit0, lo2, hi2)
                ovs = _end_overshoot(fitD)
                j2 = _jstep(fitD, lo2, hi2) + 1.0 * ovs
                yw = np.log10(I[lo2:hi2 + 1]) if fitD['fit_log'] \
                    else I[lo2:hi2 + 1]
                rvec = yw[kp0] - np.polyval(
                    fitD['coefs'],
                    2.0 * (x[lo2:hi2 + 1][kp0] - fitD['mid']) / fitD['span'])
                rms_d = float(math.sqrt(float(np.mean(rvec * rvec))))
                seen_asy.add((lo2, hi2))
                cands.append({'fit': fitD, 'lo': lo2, 'hi': hi2, 'j': j2,
                              'ovs': ovs, 'rms': rms_d,
                              'sig': _local_sigma(lo2, hi2), 'note': noteD,
                              'half': hl + hr, 'asy': (hl, hr)})
                if os.environ.get('GAP_FILL_DEBUG'):
                    print('[rung-asy] hl=%d hr=%d win=[%.2f,%.2f] order=%d '
                          'rms=%.4f sig=%.4f j=%.4f ovs=%d'
                          % (hl, hr, x[lo2], x[hi2], fitD['order'], rms_d,
                             cands[-1]['sig'], j2, ovs), file=sys.stderr)

            half0 = max(seg_w, p['min_gap_pts'])
            floor2 = max(3, p['min_gap_pts'] // 2)
            mset_asy = set()
            for other in segs:
                if (other['start_idx'], other['end_idx']) == (s_i, e_i):
                    continue
                mset_asy.update(range(other['start_idx'],
                                      other['end_idx'] + 1))
            cap_l = _cap(s_i, -1, half0)
            cap_r = _cap(e_i, +1, half0)
            seen_asy = {(c['lo'], c['hi']) for c in cands}
            n0 = len(cands)
            for hr in _geom(cap_r, floor2):          # 右族：左侧保持 3× 起点
                _asy(half0, hr)
            if cap_l < seg_w:                        # 左族：仅剧烈左侧触发
                for hl in _geom(cap_l, floor2):
                    _asy(hl, half0)
            if cap_l < seg_w and cap_r < seg_w:      # 双窄族
                for hl in _geom(cap_l, floor2):
                    for hr in _geom(cap_r, floor2):
                        _asy(hl, hr)
            if os.environ.get('GAP_FILL_DEBUG'):
                print('[rung-asy] seg=%s cap_l=%d cap_r=%d floor2=%d 新增%d 级'
                      % (_seg_tag(sg), cap_l, cap_r, floor2, len(cands) - n0),
                      file=sys.stderr)
            if len(cands) > n0:
                elig = _apply_gates()
        if elig:
            rms_min = min(c['rms'] for c in elig)
            near = [c for c in elig if c['rms'] <= rms_min * 1.05 + 1e-15]
            # 近平局取更窄窗：拟合质量统计不可分时，窄窗少含峰区数据、外推风险小
            bc = min(near, key=lambda c: c['half'])
        else:
            g = [c for c in cands if c['gate']]
            if g:
                bc = min(g, key=lambda c: c['j'])
                _jf = _jstep(bc['fit'], bc['lo'], bc['hi'])
                rec['disclosure'].append(
                    '衔接台阶 %.2f dex 超阈（j_tol=%.2f），%.2f× 规则窗内无更优级；'
                    '填充段按条件值降级使用' % (_jf, p['j_tol'], bc['half'] / seg_w))
            else:
                fin_c = [c for c in cands if math.isfinite(c['rms'])]
                if not fin_c:
                    raise ValueError(
                        '区段 %s 全部梯级拟合失败（窗口内拟合域有效点不足，'
                        'rms 均 nan）；P7 纪律：报错不填充' % _seg_tag(sg))
                bc = min(fin_c, key=lambda c: c['rms'])
                rec['disclosure'].append(
                    '拟合误差闸未过（rms=%.4g > 阈 %.4g，拟合域）：拟合本身失败，'
                    '取误差最小级填充并按条件值降级使用' % (bc['rms'], bc['thr']))
        if bc.get('asy'):
            rec['disclosure'].append(
                '对称窗未达衔接阈值，剧烈侧收窄为不对称窗（左%d/右%d 点）重拟合'
                % bc['asy'])
        best, lo, hi = bc['fit'], bc['lo'], bc['hi']
        j_best = bc['j']
        if bc['note']:
            rec['disclosure'].append(bc['note'])
        fit_rms_fin = bc['rms']

        # ---- 连续机制（用户裁决②保留，C0/C1 分档）：台阶超 j_cont 且端点处于
        # 正常水平 → 端点等式约束（C0：拟合曲线穿过衔接邻点）重拟合，陡坡侧
        # （|局部斜率| ≥ slope_c1）加斜率约束（C1）；端点深陷谷中（低于窗口
        # 未标记点拟合域中位 0.5 dex）→ 该侧拒绝连续（R7：强制连续会把填充
        # 曲线拽进深谷，等于把 gap 补成"真实信号"）
        def _constrained_refit(fit0, cons, kp_c):
            """带等式约束的多项式重拟合（KKT 精确解；权重与 _fit_on_window
            一致取均匀权重）。cons: [(基行, 目标值)]，基行为归一化坐标多项式基。"""
            xv = x[lo:hi + 1][kp_c]
            yv = (np.log10(I[lo:hi + 1]) if fit0['fit_log']
                  else I[lo:hi + 1])[kp_c]
            dd = int(fit0['order'])
            m = len(cons)
            if m > dd + 1:
                cons = cons[:dd + 1]
                m = len(cons)
            A = np.vander(2.0 * (xv - fit0['mid']) / fit0['span'], dd + 1)
            K = np.zeros((dd + 1 + m, dd + 1 + m))
            K[:dd + 1, :dd + 1] = A.T @ A
            rhs = np.concatenate([A.T @ yv, np.array([t for _, t in cons])])
            for i2, (g, _) in enumerate(cons):
                K[:dd + 1, dd + 1 + i2] = g
                K[dd + 1 + i2, :dd + 1] = g
            try:
                c = np.linalg.solve(K, rhs)[:dd + 1]
            except np.linalg.LinAlgError:
                return None
            r2 = yv - np.polyval(c, 2.0 * (xv - fit0['mid']) / fit0['span'])
            f = dict(fit0)
            f['coefs'] = c
            f['rms'] = float(math.sqrt(float(np.mean(r2 * r2))))
            f['constrained'] = True
            return f

        if j_best > p['j_cont']:
            kp_sel = _win_kp(lo, hi)
            ygl = (np.where(I > 0, np.log10(np.where(I > 0, I, 1.0)), np.nan)
                   if best['fit_log'] else I.astype(float))
            med_lvl = float(np.nanmedian(ygl[lo:hi + 1][kp_sel]))
            mset = set()
            for other in segs:
                if (other['start_idx'], other['end_idx']) != (s_i, e_i):
                    mset.update(range(other['start_idx'], other['end_idx'] + 1))
            cons0, cons1 = [], []
            for side, edge in (('L', s_i), ('R', e_i)):
                nb = edge - 1 if side == 'L' else edge + 1
                lnb = float(ygl[nb])
                if not math.isfinite(lnb) or lnb < med_lvl - 0.5:
                    continue
                u = 2.0 * (x[nb] - best['mid']) / best['span']
                cons0.append((np.array([u ** k
                                        for k in range(int(best['order']) + 1)]),
                              lnb))
                idxs = []
                rng = range(edge - 1, -1, -1) if side == 'L' else range(edge + 1, n)
                for i2 in rng:
                    if i2 not in mset:
                        idxs.append(i2)
                    if len(idxs) == 4:
                        break
                if len(idxs) >= 2 and int(best['order']) >= 3:
                    ys = ygl[np.array(idxs)]
                    if np.isfinite(ys).all():
                        sl_x = float(np.polyfit(x[np.array(idxs)], ys, 1)[0])
                        if abs(sl_x) >= p['slope_c1']:
                            gd = np.array([k * u ** (k - 1) if k >= 1 else 0.0
                                           for k in range(int(best['order']) + 1)])
                            cons1.append((gd, sl_x * best['span'] / 2.0))
            # 梯级降档（col2 探针取证：4 约束恰定 3 阶多项式零自由度，
            # rms 0.007→1.35 灾难）——C0+C1 劣化退 C0-only，仍劣化才回退
            thr_c = max(3.0 * bc['rms'], bc['rms'] + 1e-9)
            tries = []
            if cons1:
                tries.append(('C0+C1', cons0 + cons1))
            if cons0:
                tries.append(('C0', cons0))
            fC, tag_c = None, None
            for tag_c, cl in tries:
                fC = _constrained_refit(best, cl, kp_sel)
                if fC is not None and math.isfinite(fC['rms']) \
                        and fC['rms'] <= thr_c:
                    break
                fC = None
            if fC is not None:
                best = fC
                j_best = _jstep(best, lo, hi)
                fit_rms_fin = fC['rms']
                rec['disclosure'].append(
                    '连续约束重拟合（%s）：填充曲线锚定实测衔接邻点' % tag_c)
            elif tries:
                rec['disclosure'].append(
                    '连续约束重拟合均劣化（阈 rms ≤ %.4g），回退无约束拟合'
                    % thr_c)

        rec['window'] = {'x_lo': float(x[lo]), 'x_hi': float(x[hi]),
                         'n_eff': best['n_eff']}
        rec['fit_log'] = best['fit_log']
        rec['sigma_est'] = best['sigma']
        rec['fit_rms'] = float(fit_rms_fin)
        rec['sigma_local'] = (float(bc['sig']) if math.isfinite(bc['sig'])
                              else None)
        rec['fit_err_ratio'] = (float(bc['rms'] / bc['sig'])
                                if math.isfinite(bc['sig']) and bc['sig'] > 0
                                else None)
        rec['fit_err_gate'] = bool(bc['gate'])
        rec['fit_err_thr'] = (float(bc['thr']) if math.isfinite(bc['thr'])
                              else None)
        rec['window_half_seg_w'] = float(bc['half']) / seg_w

        # ---- 包络门（用户指标②；线性域；margin 缺省 0）
        keep = _win_kp(lo, hi)            # 与拟合同规（含非有限点剔除）
        env_lo = float(I[lo:hi + 1][keep].min()) - p['margin']
        env_hi = float(I[lo:hi + 1][keep].max()) + p['margin']

        filled = _eval_fit(best, x_fill)
        if ((filled < env_lo) | (filled > env_hi)).any() \
                and not best.get('constrained'):
            # 降阶重试：同窗口逐阶下降（d-1…1），取能过包络的最高阶——
            # 直接跳一阶会损失曲率致衔接台阶恶化（YR8 col1 教训）
            r = (np.log10(I[lo:hi + 1][keep]) if best['fit_log']
                 else I[lo:hi + 1][keep])
            xn1 = 2.0 * (x[lo:hi + 1][keep] - best['mid']) / best['span']
            down_ok = False
            for dd in range(int(best['order']) - 1, 0, -1):
                ftd = dict(best)
                cd = np.polyfit(xn1, r, dd)
                ftd['order'], ftd['coefs'] = dd, cd
                y2 = _eval_fit(ftd, x_fill)
                if not ((y2 < env_lo) | (y2 > env_hi)).any():
                    filled = y2
                    best = ftd
                    down_ok = True
                    rec['disclosure'].append('包络门违例 → 降为 %d 阶重拟合' % dd)
                    break
            if not down_ok:
                # 扩窗重试（一次，×2）
                half2 = half * 2
                lo2 = max(0, s_i - half2)
                hi2 = min(n - 1, e_i + half2)
                valid2 = _win_kp(lo2, hi2)   # 与拟合同规（含非有限点剔除）
                if int(valid2.sum()) >= max(8, p['d_max'] + 2):
                    fit2 = _fit_on_window(x[lo2:hi2 + 1], I[lo2:hi2 + 1],
                                          valid2, use_log, p)
                    y3 = _eval_fit(fit2, x_fill)
                    if not ((y3 < env_lo) | (y3 > env_hi)).any():
                        filled = y3
                        best, lo, hi = fit2, lo2, hi2
                        rec['window'] = {'x_lo': float(x[lo2]), 'x_hi': float(x[hi2]),
                                         'n_eff': fit2['n_eff']}
                        rec['disclosure'].append('包络门违例 → 扩窗重拟合')
        if ((filled < env_lo) | (filled > env_hi)).any():
            n_clip = int(((filled < env_lo) | (filled > env_hi)).sum())
            filled = np.clip(filled, env_lo, env_hi)
            rec['disclosure'].append('包络门违例（%d 点）→ clip 到 [%.4g, %.4g]'
                                     % (n_clip, env_lo, env_hi))
            issues.append({'type': 'envelope_clipped', 'segment': _seg_tag(sg),
                           'n_points': n_clip, 'envelope': [env_lo, env_hi]})

        # ---- 衔接台阶终检（以包络处置后的最终拟合为准）：包络降阶/扩窗会改变
        # 拟合，衔接台阶须复核——恶化超阈 → 强制披露条件值（YR8 col1 教训：
        # 收窗选中的 4 阶 j 达标，包络降为 1 阶后 j 恶化至 0.19 而原实现未查）
        j_fin = _jstep(best, lo, hi)
        if j_fin > p['j_tol'] and j_fin > j_best + 1e-9:
            rec['disclosure'].append('衔接台阶 %.2f dex 超阈（j_tol=%.2f）；'
                                     '填充段按条件值降级使用' % (j_fin, p['j_tol']))
        rec['junction_step_dex'] = j_fin
        rec['envelope'] = {'lo': env_lo, 'hi': env_hi}
        rec['order'] = int(best['order'])
        rec['rms'] = float(fit_rms_fin)   # 交付拟合的拟合域 rms（非自归一化旧量）
        rec['baseline_level'] = base_lvl

        Y_filled[s_i:e_i + 1, col] = filled
        mask_codes[s_i:e_i + 1] = 5
        fill_report.append(rec)

    filled_fraction = float(np.mean(mask_codes == 5))
    return Y_filled, mask_codes, fill_report, issues, bool(use_log), filled_fraction


# ---------------------------------------------------------------- 写盘五件套


def _fmt_num(v):
    """填充值格式化：%.6g（SAXS 动态范围匹配）"""
    return '%.6g' % float(v)


def write_outputs(out_dir, stem, raw, col, fill_report, mask_codes,
                  Y_filled, issues, source_path, x_semantic, x_unit):
    """五件套写盘（规格卡 §5.3 / R1 / R8）。col: 0-based 因变量列。
    1 <stem>_filled.dat  # 表头块 + 原始行 token 照抄（仅 mask 行目标列替换）
    2 <stem>_mask.csv    x 照抄 + 状态码
    3 <stem>_标记图.png  前后叠加 + mask 着色 + 每段放大（R8 审计界面）
    4 <stem>_整备报告.md 中文
    5 <stem>_result.json 机器可读
    返回 files 列表。
    """
    os.makedirs(out_dir, exist_ok=True)
    x_tokens = [t[0] for t in raw['raw_rows']]
    Y = raw['Y']
    n = raw['n_rows']
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ---- 1 主输出：token 级照抄，仅替换 mask 行目标列（R1）
    out_lines = [
        '# SAXS gap-filled curve | %s' % TOOL_TAG,
        '# source_file: %s' % os.path.basename(source_path),
        '# timestamp: %s' % ts,
        '# x_semantic: %s | x_unit_in: %s | x 与未替换列: 原文件 token 照抄'
        % (x_semantic, x_unit),
        '# 处理列: 因变量第 %d 列（1-based）' % (col + 1),
        '# status_codes: 0=kept 1=deep_gap_core 2=shoulder 3=isolated_outlier '
        '4=broad_shadow 5=filled（逐点状态见 _mask.csv）',
        '# n_segments: %d | filled_fraction: %.2f%%'
        % (len(fill_report), 100.0 * float(np.mean(mask_codes == 5))),
        '# decision_source: %s'
        % ','.join(sorted({r['decision_source'] for r in fill_report})),
        '# 填充值=条件值（插值整备，非信号恢复）：下游使用含填充段窗口时必须披露填充占比',
    ]
    for i in range(n):
        toks = list(raw['raw_rows'][i])
        if mask_codes[i] == 5:
            toks[1 + col] = _fmt_num(Y_filled[i, col])
        out_lines.append(' '.join(toks))
    fpath = os.path.join(out_dir, stem + '_filled.dat')
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines) + '\n')

    # ---- 2 mask.csv
    mpath = os.path.join(out_dir, stem + '_mask.csv')
    with open(mpath, 'w', encoding='utf-8') as f:
        f.write('# %s | source: %s | ts: %s\n' % (TOOL_TAG, os.path.basename(source_path), ts))
        f.write('# code: 0=kept 1=deep_gap_core 2=shoulder 3=isolated_outlier '
                '4=broad_shadow 5=filled\n')
        f.write('x,mask\n')
        for i in range(n):
            f.write('%s,%d\n' % (x_tokens[i], mask_codes[i]))

    # ---- 3 标记图（R8：剔除前后叠加 = 事后审计界面）
    png = os.path.join(out_dir, stem + '_标记图.png')
    _render_png(png, x_tokens, Y[:, col], Y_filled[:, col], mask_codes,
                fill_report, x_semantic, x_unit)

    # ---- 4 整备报告
    rep = os.path.join(out_dir, stem + '_整备报告.md')
    _write_report(rep, source_path, ts, x_semantic, x_unit, col,
                  fill_report, issues, float(np.mean(mask_codes == 5)), n)

    # ---- 5 result.json
    rj = os.path.join(out_dir, stem + '_result.json')
    result = {
        'tool': TOOL_TAG, 'timestamp': ts, 'source': source_path,
        'x_semantic': x_semantic, 'x_unit_in': x_unit, 'column_1based': col + 1,
        'n_rows': n, 'n_segments': len(fill_report),
        'filled_fraction': float(np.mean(mask_codes == 5)),
        'segments': fill_report, 'issues': issues,
        'status_codes': {str(k): v for k, v in _CODE_LABEL.items()},
        'files': {'filled': fpath, 'mask': mpath, 'png': png, 'report': rep,
                  'json': rj},
    }
    with open(rj, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return [fpath, mpath, png, rep, rj]


def _write_report(path, source, ts, x_semantic, x_unit, col, fill_report,
                  issues, frac, n):
    L = ['# 一维曲线 gap 整备报告', '',
         '- 工具：%s' % TOOL_TAG,
         '- 源文件：`%s`' % source,
         '- 时间：%s' % ts,
         '- x 语义：%s（%s）；处理列：因变量第 %d 列（1-based）'
         % (x_semantic, x_unit, col + 1),
         '- 数据点数：%d' % n,
         '- 填充占比：%.2f%%' % (100.0 * frac),
         '',
         '> **纪律声明（R7）**：填充 = 插值整备，**不是信号恢复**。被遮挡处无光子记录；'
         '填充值一律按**条件值**对待，下游使用含填充段窗口时必须披露填充占比。',
         '',
         '## 区段明细', '',
         '| # | x 范围 | 状态 | 置信 | 决策 | 窗口 | 阶数 | 拟合rms | σ_local | 误差闸 | 拟合域 | 披露 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k, r in enumerate(fill_report, 1):
        L.append('| %d | %.4g ~ %.4g | %s(%d) | %s | %s | %.4g~%.4g (%d点) | %d | %.4g | %s | %s | %s | %s |'
                 % (k, r['x_start'], r['x_end'], r['label'], r['code'],
                    r['confidence'], r['decision_source'],
                    r['window']['x_lo'], r['window']['x_hi'], r['window']['n_eff'],
                    r['order'], r.get('fit_rms', r['rms']),
                    ('%.4g' % r['sigma_local']) if r.get('sigma_local') is not None else '—',
                    '过' if r.get('fit_err_gate') else '未过',
                    'log' if r['fit_log'] else 'linear',
                    '；'.join(r['disclosure']) if r['disclosure'] else '—'))
    L += ['', '> 拟合rms/σ_local 为拟合域量（log 域 = dex）；误差闸：rms ≤ max(3×σ_local, 0.005)。'
          '闸未过 = 拟合失败，填充段已按条件值降级并披露。', '']
    L += ['', '## 强制披露项', '']
    if issues:
        for it in issues:
            L.append('- **%s**：%s' % (it['type'], json.dumps(it, ensure_ascii=False)))
    else:
        L.append('- 无（包络门零违例，无跳过段）')
    L += ['', '## 决策来源统计', '']
    src = {}
    for r in fill_report:
        src[r['decision_source']] = src.get(r['decision_source'], 0) + 1
    for k, v in sorted(src.items()):
        L.append('- %s：%d 段' % (k, v))
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


def _x_label(x_semantic, x_unit):
    """横轴标签（pixel 扩展：q 语义的像素半径轴标 R (pixel)；A^-1 输入内部已
    ×10 归一到 nm^-1，故 q 分支统一标 nm^-1）"""
    if x_semantic == 'q':
        return 'R (pixel)' if x_unit == 'pixel' else 'q (nm$^{-1}$)'
    return 'phi (deg)'


def _render_png(path, x_tokens, Y0, Yf, mask_codes, fill_report, x_semantic,
                x_unit='nm^-1'):
    """标记图（图内英文标签，避免渲染端中文字体依赖）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = np.array([float(t) for t in x_tokens])
    n_seg = min(len(fill_report), 4)
    fig = plt.figure(figsize=(11.0, 7.2))
    gs = fig.add_gridspec(2, max(n_seg, 1), height_ratios=[1.0, 0.9],
                          hspace=0.42, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(x, Y0, color='0.65', lw=1.0, label='raw')
    ax0.plot(x, Yf, color='tab:blue', lw=1.2, label='filled')
    lbl = {1: 'deep_gap_core', 2: 'shoulder', 3: 'isolated_outlier',
           4: 'broad_shadow', 5: 'filled'}
    col = {1: 'red', 2: 'orange', 3: 'magenta', 4: 'gold', 5: 'lime'}
    for code in (1, 2, 3, 4):
        m = mask_codes == code
        if m.any():
            ax0.plot(x[m], Y0[m], '.', color=col[code], ms=5, label=lbl[code])
    m = mask_codes == 5
    if m.any():
        ax0.plot(x[m], Yf[m], '.', color=col[5], ms=5, label=lbl[5])
    ax0.set_yscale('log')
    ax0.set_xlabel(_x_label(x_semantic, x_unit))
    ax0.set_ylabel('I (a.u., log)')
    ax0.set_title('gap fill overview (raw grey vs filled blue)')
    ax0.legend(fontsize=7, loc='best', ncol=3)

    for k, r in enumerate(fill_report[:4]):
        ax = fig.add_subplot(gs[1, k])
        lo, hi = r['start_idx'], r['end_idx']
        pad = max(3 * (hi - lo + 1), 12)
        a, b = max(0, lo - pad), min(len(x), hi + pad + 1)
        ax.plot(x[a:b], Y0[a:b], '.', color='0.65', ms=4, label='raw')
        ax.plot(x[a:b], Yf[a:b], '-', color='tab:blue', lw=1.2, label='filled')
        m = mask_codes[a:b] == 5
        if m.any():
            ax.plot(x[a:b][m], Yf[a:b][m], '.', color='lime', ms=6, label='filled pts')
        ax.set_title('%s [%.4g, %.4g] | order=%d rms=%.3g win=%dpts'
                     % (r['label'], r['x_start'], r['x_end'], r['order'],
                        r['rms'], r['window']['n_eff']), fontsize=8)
        ax.set_xlabel('x', fontsize=8)
        ax.set_ylabel('I', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- detect 产物


def write_detect_outputs(out_dir, stem, col, x_internal, raw, segments, base,
                         notes, card, source_path):
    """detect 产物：候选标记图 PNG + gap_table.json（供用户手改后回传 fill）。
    返回 [png_path, table_path]。"""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # gap 表（x 用原始 token 单位值——用户可读可手改；fill 端最近邻映射回索引）
    table = {
        'tool': TOOL_TAG, 'source': os.path.basename(source_path),
        'timestamp': ts, 'x_semantic': card['x_semantic'], 'x_unit': card['x_unit'],
        'column_1based': col + 1,
        'usage': '手改本文件后传给 saxs_gap_fill 的 gap_table_file；'
                 '每段字段：x_start/x_end（x 单位）、code(1/2/3/4)、note 可加、'
                 'decision_source 可改为 AGENT/USER',
        'segments': [
            {'x_start': s['x_start'], 'x_end': s['x_end'], 'code': s['code'],
             'confidence': s['confidence'], 'note': s['note'],
             'decision_source': 'AUTO'} for s in segments],
        'notes': notes,
    }
    tpath = os.path.join(out_dir, stem + '_gaptable_col%d.json' % (col + 1))
    with open(tpath, 'w', encoding='utf-8') as f:
        json.dump(table, f, ensure_ascii=False, indent=2)

    # 候选标记图
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x_tokens = [t[0] for t in raw['raw_rows']]
    x = np.array([float(t) for t in x_tokens])
    I = raw['Y'][:, col]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(x, I, color='0.55', lw=1.0, label='raw')
    seg_col = {1: 'red', 2: 'orange', 3: 'magenta', 4: 'gold'}
    seen = set()
    for s in segments:
        sl = slice(s['start_idx'], s['end_idx'] + 1)
        ax.plot(x[sl], I[sl], '.', color=seg_col[s['code']], ms=6,
                label=('%s(c%s)' % (s['label'], s['confidence'][0].upper()))
                if s['code'] not in seen else None)
        seen.add(s['code'])
    with np.errstate(all='ignore'):
        base_lin = np.power(10.0, base)
    ax.plot(x, base_lin, '--', color='tab:blue', lw=1.0, label='robust baseline')
    ax.set_yscale('log')
    ax.set_xlabel(_x_label(card['x_semantic'], card['x_unit']))
    ax.set_ylabel('I (a.u., log)')
    ax.set_title('gap candidates, col=%d (%s) — red=core orange=shoulder '
                 'magenta=outlier gold=shadow?' % (col + 1, card['x_semantic']),
                 fontsize=10)
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    png = os.path.join(out_dir, stem + '_gapdetect_col%d.png' % (col + 1))
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return [png, tpath]


# ---------------------------------------------------------------- 工具主入口


def detect_card(path, x_semantic, x_unit, col=1, params=None, out_dir=None):
    """saxs_gap_detect 主入口：读盘 + 检测 + 标记图/gap 表落盘。col: 1-based。
    out_dir 缺省 runs/<stem>_gapdetect/。返回 JSON 可序列化 dict。"""
    if not os.path.isfile(path):
        raise ValueError('文件不存在: %s' % path)
    raw = _read_columns(path)
    if col < 1 or col > raw['n_col'] - 1:   # n_col 含 x 列，因变量列数 = n_col-1
        raise ValueError('column=%d 超出范围（文件含 %d 列因变量）' % (col, raw['n_col']))
    x = _internal_x(raw['raw_rows'], x_semantic, x_unit)
    _check_x_monotonic(x)
    segments, base, notes = detect_gaps(x, raw['Y'], col - 1, params)

    stem = os.path.splitext(os.path.basename(path))[0]
    if out_dir is None:
        out_dir = workdir.default_out(stem + '_gapdetect')
    files = write_detect_outputs(out_dir, stem, col - 1, x, raw, segments, base,
                                 notes, {'x_semantic': x_semantic,
                                         'x_unit': x_unit}, path)

    Yc = raw['Y'][:, col - 1]
    pos = Yc[np.isfinite(Yc) & (Yc > 0)]
    card = {
        'tool': TOOL_TAG, 'mode': 'detect', 'path': path,
        'x_semantic': x_semantic, 'x_unit': x_unit, 'column_1based': col,
        'n_rows': raw['n_rows'], 'n_y_columns': raw['n_col'],
        'n_bad_lines': raw['n_bad'],
        'I_min': float(Yc.min()), 'I_max': float(Yc.max()),
        'dynamic_range': float(pos.max() / pos.min()) if len(pos) >= 2 else None,
        'x_range': [float(x[0]), float(x[-1])],
        'n_segments': len(segments),
        'segment_counts': {str(k): sum(1 for s in segments if s['code'] == k)
                           for k in (1, 2, 3, 4)},
        'notes': notes,
        'files': {'png': files[0], 'gap_table': files[1]},
        'segments': segments,
    }
    card['编排提示'] = ('下一步：Agent 目检标记图（Tier B 主判，可调 params 重跑 ≤2 次）'
                    '→ 需用户裁决的段（Tier C）征询用户 → 手改 gap_table 后传 '
                    'saxs_gap_fill 的 gap_table_file')
    return card


def fill_card(path, x_semantic, x_unit, col=1, gap_table=None, params=None,
              fit_log=None, out_dir=None):
    """saxs_gap_fill 主入口：读盘 + 填充 + 五件套写盘。col: 1-based。
    gap_table: list[dict] 或 JSON 文件路径（detect 产物，可经用户手改）。
    out_dir 缺省 runs/<stem>_gapfix/。返回 dict（files + summary）。"""
    if not os.path.isfile(path):
        raise ValueError('文件不存在: %s' % path)
    if isinstance(gap_table, str):
        if not os.path.isfile(gap_table):
            raise ValueError('gap_table_file 不存在: %s' % gap_table)
        with open(gap_table, 'r', encoding='utf-8') as f:
            gap_table = json.load(f)
        if isinstance(gap_table, dict) and 'segments' in gap_table:
            gap_table = gap_table['segments']
    if not isinstance(gap_table, list) or not gap_table:
        raise ValueError('gap_table 须为非空段列表（list[dict]）或含 segments 的 JSON 文件')

    raw = _read_columns(path)
    if col < 1 or col > raw['n_col'] - 1:   # n_col 含 x 列，因变量列数 = n_col-1
        raise ValueError('column=%d 超出范围（文件含 %d 列因变量）' % (col, raw['n_col']))
    x = _internal_x(raw['raw_rows'], x_semantic, x_unit)
    _check_x_monotonic(x)

    Y_filled, mask_codes, fill_report, issues, use_log, frac = fill_gaps(
        x, raw['Y'], col - 1, gap_table, x_semantic, params, fit_log)

    stem0 = os.path.splitext(os.path.basename(path))[0]
    stem = stem0 + '_col%d' % col   # 批处理防覆盖：五件套文件名必须带列号
    if out_dir is None:
        out_dir = workdir.default_out(stem0 + '_gapfix')
    files = write_outputs(out_dir, stem, raw, col - 1, fill_report, mask_codes,
                          Y_filled, issues, path, x_semantic, x_unit)
    return {
        'tool': TOOL_TAG, 'mode': 'fill', 'path': path,
        'x_semantic': x_semantic, 'x_unit': x_unit, 'column_1based': col,
        'fit_log': use_log, 'n_rows': raw['n_rows'],
        'n_segments': len(fill_report), 'filled_fraction': frac,
        'issues': issues,
        'segments_summary': [
            {'x': [r['x_start'], r['x_end']], 'code': r['code'],
             'label': r['label'], 'order': r['order'], 'rms': r['rms'],
             'window': r['window'], 'disclosure': r['disclosure']}
            for r in fill_report],
        'files': files, 'out_dir': out_dir,
    }


def fill_card_batch(path, x_semantic, x_unit, columns, gap_tables,
                    params=None, fit_log=None, out_dir=None):
    """批量填充（提速三件套 T-B）：内部循环逐列调 fill_card——单列路径逐位
    不变，本函数纯聚合不做新算法。columns 与 gap_tables 须等长对齐（显式
    gaptable JSON 路径列表）；任一列失败 → 硬报错列明列号（P7 报错不静默跳过）。
    返回逐列摘要（filled_fraction/n_segments/issues/主产物路径）。"""
    columns = [int(c) for c in columns]
    if len(gap_tables) != len(columns):
        raise ValueError('gap_tables(%d) 与 columns(%d) 数量不一致（须等长对齐）'
                         % (len(gap_tables), len(columns)))
    results, out_dir_eff = [], None
    for col, gt in zip(columns, gap_tables):
        try:
            card = fill_card(path, x_semantic, x_unit, col=col, gap_table=gt,
                             params=params, fit_log=fit_log, out_dir=out_dir)
        except Exception as e:
            raise ValueError('批量填充 col%d 失败（P7 纪律：报错不静默跳过）：%s'
                             % (col, e)) from e
        out_dir_eff = card['out_dir']
        f = card['files']
        results.append({'column_1based': card['column_1based'],
                        'fit_log': card['fit_log'],
                        'filled_fraction': card['filled_fraction'],
                        'n_segments': card['n_segments'],
                        'issues': card['issues'],
                        'files': {'filled': f[0], 'mask': f[1], 'png': f[2],
                                  'report': f[3], 'json': f[4]}})
    return {'tool': TOOL_TAG, 'mode': 'fill_batch', 'path': path,
            'x_semantic': x_semantic, 'x_unit': x_unit,
            'n_columns': len(results), 'columns': results,
            'total_issues': sum(len(r['issues']) for r in results),
            'out_dir': out_dir_eff}


def merge_filled_book(src_path, out_dir, columns=None, filled_paths=None):
    """批量整备合并总表（Origin 拖入友好）：把 out_dir 内各列填充产物拼成一个
    与源文件同形的数据总表 <srcstem>_filled_book.dat——x 与未合并列取源文件
    token，第 k 因变量列取 <srcstem>_col<k>_filled.dat 的第 k 列 token（gap 行
    为填充值、其余行与源文件逐 token 一致），供 Origin 与原始数据 book 并排
    拖入对照。纯打包步骤不做拟合；列文件缺失/行数或 x 不一致按 P7 报错。
    columns: 1-based 因变量列号列表（缺省 = out_dir 内实际存在填充产物的全部列，
    按列号升序合并）；filled_paths: 显式指定各列 filled.dat（须与 columns 等长
    对齐，缺省按命名惯例在 out_dir 自动发现）。"""
    raw = _read_columns(src_path)
    n, n_col = raw['n_rows'], raw['n_col']
    n_ycol = n_col - 1
    srcstem = os.path.splitext(os.path.basename(src_path))[0]
    if not os.path.isabs(out_dir):        # MCP 宿主 cwd 不可控（A 类 09-28/09-29）
        out_dir = os.path.join(_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 定位各列 filled 文件
    if filled_paths:
        if columns is None:
            raise ValueError('filled_paths 与 columns 必须成对给出（等长对齐）')
        if len(filled_paths) != len(columns):
            raise ValueError('filled_paths(%d) 与 columns(%d) 数量不一致'
                             % (len(filled_paths), len(columns)))
        pairs = list(zip(columns, filled_paths))
    else:
        cand = sorted(columns) if columns else list(range(1, n_ycol + 1))
        pairs, none_found = [], True
        for k in cand:
            fp = os.path.join(out_dir, '%s_col%d_filled.dat' % (srcstem, k))
            if os.path.isfile(fp):
                pairs.append((k, fp))
                none_found = False
            elif columns:
                raise ValueError('col%d 填充产物缺失: %s；P7 纪律：报错不合并'
                                 % (k, fp))
        if none_found:
            raise ValueError('out_dir 内未发现任何 <srcstem>_col<k>_filled.dat '
                             '（%s_col1_filled.dat 等）；P7 纪律：报错不合并' % srcstem)
    pairs.sort(key=lambda t: t[0])
    ks = [k for k, _ in pairs]

    # ---- 分隔符嗅探（源文件末数据行；输出与源同 delimiter 便于 Origin 对照拖入）
    with open(src_path, 'rb') as f:
        last_ln = f.read().rstrip().split(b'\n')[-1]
    delim = '\t' if b'\t' in last_ln else ' '

    # ---- 逐列装载与校验（行数 + x 逐行一致性 + 列宽）
    merged_cols, fracs, sem_lines = {}, {}, {}
    for k, fp in pairs:
        fr = _read_columns(fp)
        if fr['n_rows'] != n or fr['n_col'] != n_col:
            raise ValueError('col%d 填充文件形状不一致（%d 行 %d 列，应为 %d 行 %d 列）：'
                             '%s；P7 纪律：报错不合并'
                             % (k, fr['n_rows'], fr['n_col'], n, n_col, fp))
        for i in range(n):
            if not math.isclose(float(fr['raw_rows'][i][0]),
                                float(raw['raw_rows'][i][0]),
                                rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError('col%d 填充文件 x 第 %d 行与源文件不一致'
                                 '（%s vs %s）；P7 纪律：报错不合并'
                                 % (k, i + 1, fr['raw_rows'][i][0],
                                    raw['raw_rows'][i][0]))
        merged_cols[k] = [t[k] for t in fr['raw_rows']]   # 0-based: x 占 0，col k 占 k
        rj = os.path.join(out_dir, '%s_col%d_result.json' % (srcstem, k))
        if os.path.isfile(rj):
            with open(rj, 'r', encoding='utf-8') as f:
                res = json.load(f)
            fracs[k] = float(res.get('filled_fraction', float('nan')))
            sem_lines.setdefault('sem', (res.get('x_semantic'),
                                         res.get('x_unit_in')))
        else:
            fracs[k] = None

    # ---- 写总表
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    out_lines = [
        '# SAXS gap-filled book | saxs_gap_merge v0.1.0',
        '# source_file: %s' % os.path.basename(src_path),
        '# timestamp: %s' % ts,
        '# x 与未合并列: 源文件 token 照抄 | 合并列 gap 行 = 填充值',
        '# 合并列: %s' % ', '.join('col%d' % k for k in ks),
    ]
    for k in ks:
        out_lines.append('# col%d: filled_fraction=%s'
                         % (k, ('%.2f%%' % (100.0 * fracs[k]))
                            if fracs[k] is not None else '见各列 result.json'))
    if 'sem' in sem_lines:
        out_lines.append('# x_semantic: %s | x_unit_in: %s' % sem_lines['sem'])
    out_lines.append('# 填充值=条件值（插值整备，非信号恢复）：下游使用含填充段'
                     '窗口时必须披露填充占比；逐列披露见各列 _整备报告.md')
    for i in range(n):
        toks = list(raw['raw_rows'][i])
        for k in ks:
            toks[k] = merged_cols[k][i]
        out_lines.append(delim.join(toks))
    out_path = os.path.join(out_dir, srcstem + '_filled_book.dat')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines) + '\n')

    n_filled = sum(int(round(fracs[k] * n)) for k in ks
                   if fracs[k] is not None)
    return {
        'tool': 'saxs_gap_merge', 'mode': 'merge', 'source': src_path,
        'out_dir': out_dir, 'n_rows': n, 'n_columns_merged': len(ks),
        'columns_merged': ks, 'delimiter': repr(delim),
        'filled_fraction_per_column': {str(k): fracs[k] for k in ks},
        'total_filled_cells': n_filled,
        'files': [out_path],
    }

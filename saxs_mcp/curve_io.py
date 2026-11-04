# -*- coding: utf-8 -*-
"""1D SAXS 曲线读取（M2.5-1 通用化版）

两层结构：
- probe_format(path)：只读探测，返回格式卡（编码/分隔符/表头/列数/单位提示/疑似异常）。
  探测只做"无歧义的唯一判定"，有歧义如实列入 ambiguities，绝不猜——裁决权在用户。
- load_curve(path, q_unit, qmin, qmax, fmt=None)：
  * fmt=None：M1 原始严格路径（三列空白分隔，行为逐位保持，回归基准依赖此路径）
  * fmt=确认后的格式卡：按显式格式读入（编码/分隔符/列号/跳行/单位/双块/2θ）
    q_unit 仍必须显式（探测器的单位提示只是提示，须经用户确认后随 fmt 传入）

硬纪律（源自废弃计划"决定 1"，M2.5-1 正式落地）：
- 编码只自动判定 BOM/UTF-8/GBK，判不出如实报 unknown
- 分隔符必须唯一，歧义不裁决
- 无 σ 显式标记 unweighted（perr 被低估），不静默按 1
- 首列疑似 2θ/方位角/帧号 → 探测卡标记，读取需显式给波长或拒绝
"""
import math
import os

import numpy as np

# ---------------------------------------------------------------- 数据文件大小门
# （2026-11-04 用户裁决，部署优化）：确认的数据类文件读入不得超过 1MB，
# 防误传 2D 原始帧 / 打包大文件拖死共用主机。仅作用于正式数据读入通道
# （load_curve / gap_fill._read_columns）；saxs_probe_curve 为类型诊断通道
# 不设限——先用探测卡确认文件类型，再决定是否作为数据读入。
MAX_DATA_FILE_MB = 1.0


def check_data_file_size(path):
    """数据类文件大小硬门：超 1MB 报错拒读（不截断不降采样，裁决权在用户）。"""
    size = os.path.getsize(path)
    if size > MAX_DATA_FILE_MB * 1024 * 1024:
        raise ValueError(
            '数据文件 %.2f MB 超过上限 %.0f MB，拒绝读入: %s\n'
            '若确为 1D 数据文件请先截取感兴趣区间或降采样后再读；'
            '若不确定文件类型，先用 saxs_probe_curve 探测格式卡'
            % (size / 1048576.0, MAX_DATA_FILE_MB, path))


# ---------------------------------------------------------------- 内部工具

_BOMS = (
    ('utf-8-sig', b'\xef\xbb\xbf'),
    ('utf-16', b'\xff\xfe'),      # LE（含 BE 由 utf-16 codec 按 BOM 自动分辨，BE 单列如下）
    ('utf-16', b'\xfe\xff'),
)


def _decode_raw(raw):
    """按 BOM → UTF-8 严格 → GBK 严格 链解码，返回 (text, encoding, evidence)"""
    for enc, bom in _BOMS:
        if raw.startswith(bom):
            try:
                return raw.decode(enc), enc, 'BOM'
            except (UnicodeDecodeError, LookupError):
                pass
    for enc in ('utf-8', 'gbk'):
        try:
            return raw.decode(enc), enc, 'strict-%s' % enc
        except UnicodeDecodeError:
            continue
    return None, 'unknown', 'decode-failed'


_UNIT_HINTS = (
    ('A-1', 'A^-1'), ('Å-1', 'A^-1'), ('1/A', 'A^-1'), ('1/Å', 'A^-1'),
    ('NM-1', 'nm^-1'), ('NM^-1', 'nm^-1'),
    ('2THETA', '2theta'), ('2Θ', '2theta'), ('TWO-THETA', '2theta'),
    ('AZIMUTH', 'azimuth'), ('方位角', 'azimuth'), ('帧号', 'frame_index'),
)


def _unit_hint_from_text(text):
    """在表头文本里找机器可读单位线索（大小写不敏感），返回 (提示值, 命中原文)"""
    up = text.upper()
    for pat, val in _UNIT_HINTS:
        i = up.find(pat.upper())
        if i >= 0:
            return val, text[max(0, i - 12):i + len(pat) + 12].strip()
    return None, None


def _is_num(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _split_data_line(line, delim):
    if delim == 'comma':
        parts = [p.strip() for p in line.split(',')]
    elif delim == 'tab':
        parts = [p.strip() for p in line.split('\t')]
    else:
        parts = line.split()
    return [p for p in parts if p != '']


def _detect_delimiter(data_lines):
    """分隔符唯一性判定：逗号/Tab 各自"在所有数据行上一致出现"才算命中；多命中=歧义"""
    hits = []
    for d, ch in (('comma', ','), ('tab', '\t')):
        counts = {ln.count(ch) for ln in data_lines}
        if len(counts) == 1 and counts != {0}:
            hits.append(d)
    if len(hits) == 1:
        return hits[0], []
    if len(hits) > 1:
        return None, ['分隔符歧义：逗号与 Tab 同时一致出现，需用户指定 fmt.delimiter']
    return 'space', []


# ---------------------------------------------------------------- 探测器

def probe_format(path):
    """只读探测 1D 曲线文件，返回格式卡 dict。不做任何修改、不做猜测性裁决。"""
    with open(path, 'rb') as f:
        raw = f.read()
    text, encoding, enc_evidence = _decode_raw(raw)
    if text is None:
        return {'path': path, 'ok': False, 'issues': ['编码无法判定（非 BOM/UTF-8/GBK）'],
                'encoding': 'unknown'}
    lines_all = text.split('\n')
    n_crlf = text.count('\r\n')
    bare_lf = text.count('\n') - n_crlf
    endings = 'CRLF/LF 混杂' if (n_crlf and bare_lf) else ('CRLF' if n_crlf else 'LF')

    # 表头结构：连续的"非数据行"归为表头区
    header_lines, data_start = [], None
    for i, ln in enumerate(lines_all):
        s = ln.strip()
        if not s:
            if data_start is None:
                continue  # 表头区前的空行
            else:
                break    # 数据区后的空行不管
        toks = _split_data_line(s, 'space') if (',' not in s and '\t' not in s) \
            else _split_data_line(s, 'comma') if s.count(',') else _split_data_line(s, 'tab')
        if toks and all(_is_num(t) for t in toks) and len(toks) >= 2:
            data_start = i
            break
        header_lines.append(s)

    header_text = '\n'.join(header_lines)
    edf_like = ('{' in header_text) or ('= ' in header_text and header_text.count('=') >= 3)
    n_hash = sum(1 for h in header_lines if h.startswith('#'))
    plain_header = [h for h in header_lines if not h.startswith('#')]

    # 数据行集合与列数（raw_data_lines 保留原始字符串供分隔符判定）
    data_lines, raw_data_lines, bad = [], [], 0
    if data_start is not None:
        for ln in lines_all[data_start:]:
            s = ln.strip()
            if not s:
                continue
            delim0 = 'comma' if (s.count(',') and (s.count(',') >= s.count('\t'))) else \
                     'tab' if s.count('\t') else 'space'
            toks = _split_data_line(s, delim0)
            if toks and all(_is_num(t) for t in toks):
                data_lines.append(toks)
                raw_data_lines.append(s)
            elif toks and all(_is_num(t) for t in toks if t.lower() not in ('nan', 'na', '--')):
                data_lines.append(toks)  # NaN 行也计列数（dim⑦ 由 load 阶段丢弃）
                raw_data_lines.append(s)
            else:
                bad += 1
    n_cols = max((len(t) for t in data_lines), default=0)

    delim, ambiguities = _detect_delimiter(raw_data_lines) if data_lines else (None, ['未找到数据行'])

    # 单位提示（表头内嵌元数据）
    hint_val, hint_text = _unit_hint_from_text(header_text) if header_text else (None, None)

    # 首列性质：整数等差 → 疑似帧号/方位角；等差但非整数 → 可能 2θ 步进
    suspect_nonq = ''
    if data_lines and n_cols >= 2:
        try:
            c0 = [float(t[0]) for t in data_lines]
            diffs = {round(c0[i + 1] - c0[i], 9) for i in range(len(c0) - 1)}
            all_int = all(abs(v - round(v)) < 1e-9 for v in c0)
            if len(diffs) == 1 and all_int:
                suspect_nonq = 'first_col_integer_equispaced（疑似帧号/方位角，非 q）'
            elif hint_val in ('2theta', 'azimuth'):
                suspect_nonq = 'header_indicates_%s' % hint_val
        except ValueError:
            pass

    # 双块疑似：4 列且第 0 列 ≈ 第 2 列
    dual = False
    if data_lines and n_cols == 4:
        try:
            same = sum(1 for t in data_lines
                       if abs(float(t[0]) - float(t[2])) <= 1e-6 * max(1.0, abs(float(t[0]))))
            dual = same >= len(data_lines) * 0.9
        except ValueError:
            pass

    # 建议的 fmt（除 q_unit 需用户确认外均可直接采用）
    suggested = {'encoding': encoding, 'delimiter': delim, 'skip_rows': data_start or 0,
                 'col_q': 0, 'col_i': 1, 'col_s': 2 if n_cols >= 3 else None}
    if dual:
        suggested['block'] = 0
        suggested['col_q'], suggested['col_i'] = 0, 1
        suggested['col_s'] = None               # 双块 q I q I 无 σ 列
    if suspect_nonq.startswith('header_indicates_2theta'):
        suggested['q_unit'] = '2theta'          # 还需用户给 two_theta_wavelength_A
    else:
        suggested['q_unit'] = {'A^-1': 'A^-1', 'nm^-1': 'nm^-1'}.get(hint_val or '', None)

    card = {
        'path': path, 'ok': True,
        'encoding': encoding, 'encoding_evidence': enc_evidence,
        'line_endings': endings,
        'delimiter': delim, 'n_data_columns': n_cols,
        'header': {'edf_like': edf_like, 'hash_meta_lines': n_hash,
                   'plain_text_header': plain_header or None,
                   'data_start_line': (data_start if data_start is not None else 0) + 1},
        'unit_hint': {'value': hint_val, 'evidence': hint_text} if hint_val else None,
        'suspect_nonq': suspect_nonq or None,
        'suspect_dual_block': dual,
        'n_data_lines': len(data_lines), 'n_bad_lines_estimate': bad,
        'ambiguities': ambiguities,
        'suggested_fmt': suggested,
    }
    if not data_lines:
        card['ambiguities'].append('未解析到任何数据行，请人工检查文件')
    return card


# ---------------------------------------------------------------- 读取器

def load_curve(path, q_unit='A^-1', qmin=0.3, qmax=4.4, fmt=None):
    """读取 1D 曲线，返回 (q, I, sigma, summary)。

    qmin/qmax：感兴趣窗口 (nm^-1)；传 None = 全范围通道（261104 部署优化：
    窗口建议引擎与"初始默认=完整数据范围"用；仅剔除 σ≤0 点，不裁 q）。
    qmin/qmax 同时显式给定时行为与历史逐位一致。

    fmt=None：M1 原始路径（三列空白分隔，q_unit 必须 'A^-1'/'nm^-1'）。
    fmt=dict：显式格式卡（probe_format 的 suggested_fmt 经用户确认后传入）：
        encoding, delimiter('space'|'tab'|'comma'), skip_rows,
        col_q, col_i, col_s(None=无σ→unweighted), q_unit('A^-1'|'nm^-1'|'2theta'),
        block(双块选块, 默认0), two_theta_wavelength_A(2θ 必给, 单位 Å)
    """
    check_data_file_size(path)   # 数据类文件 >1MB 拒读（261104 用户裁决）
    # ---------- M1 原始路径（逐位保持，勿动） ----------
    if fmt is None:
        if q_unit not in ('A^-1', 'nm^-1'):
            raise ValueError("q_unit 只支持 'A^-1' 或 'nm^-1'，不做单位猜测")
        factor = 10.0 if q_unit == 'A^-1' else 1.0

        q_all, I_all, s_all = [], [], []
        n_bad = 0
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                p = line.split()
                if len(p) != 3:
                    continue
                try:
                    q_all.append(float(p[0]) * factor)   # A^-1 -> nm^-1
                    I_all.append(float(p[1]))
                    s_all.append(float(p[2]))
                except ValueError:
                    n_bad += 1
        if not q_all:
            raise ValueError('未解析到任何数据行: %s' % path)
    # ---------- fmt 显式路径（M2.5-1） ----------
    else:
        f_unit = fmt.get('q_unit') or q_unit
        if f_unit not in ('A^-1', 'nm^-1', '2theta'):
            raise ValueError("fmt.q_unit 只支持 'A^-1'/'nm^-1'/'2theta'，不做单位猜测")
        if f_unit == '2theta' and not fmt.get('two_theta_wavelength_A'):
            raise ValueError('2θ 读入必须显式给 two_theta_wavelength_A（Å），不做波长猜测')

        delim = fmt.get('delimiter', 'space')
        c_q, c_i, c_s = fmt['col_q'], fmt['col_i'], fmt.get('col_s')
        skip = int(fmt.get('skip_rows', 0))
        block = int(fmt.get('block', 0))
        lam = fmt.get('two_theta_wavelength_A')

        with open(path, 'rb') as f:
            text, _, _ = _decode_raw(f.read())
        if text is None:
            raise ValueError('fmt.encoding=%s 解码失败: %s' % (fmt.get('encoding'), path))
        lines_all = text.split('\n')

        q_all, I_all, s_all = [], [], []
        n_bad = 0
        for ln in lines_all[skip:]:
            s = ln.strip()
            if not s:
                continue
            toks = _split_data_line(s, delim)
            if block and len(toks) >= 4:            # 双块：block=1 取第 2 组 (q,I)
                off = 2
            else:
                off = 0
            try:
                if c_i + off >= len(toks) or (c_s is not None and c_s + off >= len(toks)):
                    raise IndexError
                qv = float(toks[c_q + off])
                iv = float(toks[c_i + off])
                if toks[c_q + off].lower() in ('nan', 'na', '--') or \
                   toks[c_i + off].lower() in ('nan', 'na', '--'):
                    raise ValueError
                if f_unit == '2theta':
                    th = math.radians(qv / 2.0)     # qv 此处是 2θ（度）
                    factor = 40.0 * math.pi / lam   # q[nm^-1] = 10 × 4π sinθ / λ[Å]
                    q_all.append(math.sin(th) * factor)
                else:
                    q_all.append(qv * (10.0 if f_unit == 'A^-1' else 1.0))
                I_all.append(iv)
                if c_s is None:
                    s_all.append(1.0)
                else:
                    sv = toks[c_s + off]
                    if sv.lower() in ('nan', 'na', '--'):
                        raise ValueError
                    s_all.append(float(sv))
            except (ValueError, IndexError):
                n_bad += 1
        if not q_all:
            raise ValueError('未解析到任何数据行: %s' % path)
        q_unit = f_unit

    q_all, I_all, s_all = map(np.asarray, (q_all, I_all, s_all))
    m = s_all > 0
    if qmin is not None:
        m &= q_all >= qmin
    if qmax is not None:
        m &= q_all <= qmax
    q, I, sig = q_all[m], I_all[m], s_all[m]
    if len(q) == 0:
        if qmin is None or qmax is None:
            raise ValueError('全范围通道下无有效数据点（σ>0）: %s' % path)
        raise ValueError('窗口 [%.2f, %.2f] 内无有效数据点' % (qmin, qmax))

    summary = {
        'n_total': int(len(q_all)), 'n_used': int(len(q)),
        'n_bad_lines': int(n_bad), 'n_dropped_window': int(len(q_all) - len(q)),
        'q_min': float(q[0]), 'q_max': float(q[-1]),
        'q_unit_final': 'nm^-1',
    }
    if fmt is not None:
        summary['fmt_used'] = {k: v for k, v in fmt.items()}
        if fmt.get('col_s') is None:
            summary['unweighted'] = True   # 无 σ：perr 被低估，不得当加权结果解读
    return q, I, sig, summary

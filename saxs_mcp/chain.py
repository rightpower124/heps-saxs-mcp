# -*- coding: utf-8 -*-
"""Ruland 全流程链式编排器（orientation chain v0.1.0，提速三件套 T-C）

规格卡：.trae/documents/提速三件套-beq扫描内置-批量填充-链式编排-规划.md（§3）
动机：000064 全流程实测 32 次 MCP 调用，瓶颈在编排而非计算；本模块把机械接线
（逐列 detect 循环 / gaptable-mask 路径推导 / 批量填充 + 合并）收拢为三阶段
thin dispatcher，全流程调用数 32 → 4。

三阶段（每阶段纯转发既有 card 函数 + 路径接线，不做任何新算法）：
  stage='detect'   逐列 detect_card（内部循环）→ runs/<stem>_chain/det/col%02d/
                   → 停点：Agent Tier B 逐列目检标记图（lessons.md 批量执行教训）
  stage='prepare'  fill_card_batch（gaptable 按确定性约定发现）+
                   merge_filled_book → filled_book 总表
  stage='bpi2'     bpi2_card（filled_book + mask 路径自动接线 + q_values 走
                   Q9 文件头/显式 + q_unit 必填）→ 停点：B 表/QC 图目检
beq 终点走 saxs_ruland_beq（scan_qmins 内置扫描，T-A），不设重复阶段。

硬纪律：
  - 不提供 detect→fill 一键贯通 stage——必须逐列 detect→目检→fill
    （lessons.md 批量执行教训：col10 肩部段出现在峰麓上升区，靠目检拦截）
  - 路径发现全部确定性约定（col%02d 目录 / <stem>_gaptable_col<k>.json /
    <stem>_col<k>_mask.csv），缺失即硬报错（P7 报错不猜测）
  - 填充值条件值 / bpi2 填充依赖披露 / beq 采纳窗语义均由底层既有函数承载，
    本模块不复制任何口径逻辑（零双实现漂移）
"""
import os

from gap_fill import (_read_columns, detect_card, fill_card_batch,
                      merge_filled_book)
from ruland import bpi2_card
import workdir                     # 会话级工作目录（部署优化 261104）：缺省产物重定向

__version__ = '0.1.0'

TOOL_TAG = 'saxs_ruland_chain v%s' % __version__

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stem(path):
    """链内锚定 stem：filled_book 输入剥去 _filled_book 后缀，还原原始文件
    stem——保证 detect/prepare/bpi2 三阶段落到同一条 runs/<stem>_chain/ 链上"""
    s = os.path.splitext(os.path.basename(path))[0]
    if s.endswith('_filled_book'):
        s = s[:-len('_filled_book')]
    return s


def _resolve_out_dir(out_dir, stem):
    if out_dir is None:
        out_dir = workdir.default_out(stem + '_chain')
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(_ROOT, out_dir)   # MCP 宿主 cwd 不可控（A 类）
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _columns_or_all(path, columns):
    raw = _read_columns(path)
    n_ycol = raw['n_col'] - 1   # n_col 含 x 列（_read_columns 契约 Y=[n, n_col-1]）
    cols = sorted(int(c) for c in columns) if columns else \
        list(range(1, n_ycol + 1))
    for c in cols:
        if c < 1 or c > n_ycol:
            raise ValueError('column=%d 超出范围（文件含 %d 列因变量）'
                             % (c, n_ycol))
    return cols


def chain_detect(path, x_semantic, x_unit, columns=None, params=None,
                 out_dir=None):
    """stage='detect'：逐列 detect_card（内部循环，产物 runs/<stem>_chain/
    det/col%02d/）。返回逐列段数/标记图/gaptable 路径——停点：Tier B 目检。"""
    if not os.path.isfile(path):
        raise ValueError('文件不存在: %s' % path)
    cols = _columns_or_all(path, columns)
    stem = _stem(path)
    out = _resolve_out_dir(out_dir, stem)
    per_col = []
    for c in cols:
        card = detect_card(path, x_semantic, x_unit, col=c, params=params,
                           out_dir=os.path.join(out, 'det', 'col%02d' % c))
        per_col.append({'column_1based': c,
                        'n_segments': card['n_segments'],
                        'segment_counts': card['segment_counts'],
                        'notes': card['notes'],
                        # detect_card 已把 write_detect_outputs 的 [png, table]
                        # 包装成 dict（L1386 契约，070930 乌龙教训：包装层才是
                        # 返回契约，勿只核 write 函数）
                        'png': card['files']['png'],
                        'gap_table': card['files']['gap_table']})
    return {'tool': TOOL_TAG, 'stage': 'detect', 'path': path, 'out_dir': out,
            'n_columns': len(per_col), 'columns': per_col,
            '编排提示': ('停点：逐列目检标记图（Tier B 主判，可调 params 重跑 '
                     "本阶段 ≤2 次）→ 确认后进入 stage='prepare'（批量填充+"
                     '合并总表）→ stage=\'bpi2\'（峰积分，mask 自动接线）→ '
                     'saxs_ruland_beq（scan_qmins 扫描）')}


def _discover_gap_tables(det_dir, stem, cols):
    """确定性约定发现各列 gaptable：<det>/col%02d/<stem>_gaptable_col<k>.json。
    缺失即硬报错（P7：报错不猜测，不静默跳列）。"""
    gts = []
    for c in cols:
        gt = os.path.join(det_dir, 'col%02d' % c,
                          '%s_gaptable_col%d.json' % (stem, c))
        if not os.path.isfile(gt):
            raise ValueError('col%d gap 表缺失: %s（P7 纪律：报错不猜测；'
                             '请先跑 stage=detect 或核对 det 目录约定）' % (c, gt))
        gts.append(gt)
    return gts


def chain_prepare(path, x_semantic, x_unit, columns=None, params=None,
                  fit_log=None, out_dir=None):
    """stage='prepare'：fill_card_batch（逐列批量填充）+ merge_filled_book
    （总表）。gaptable 由确定性约定自 stage='detect' 产物发现。"""
    if not os.path.isfile(path):
        raise ValueError('文件不存在: %s' % path)
    cols = _columns_or_all(path, columns)
    stem = _stem(path)
    out = _resolve_out_dir(out_dir, stem)
    gts = _discover_gap_tables(os.path.join(out, 'det'), stem, cols)
    batch = fill_card_batch(path, x_semantic, x_unit, cols, gts, params=params,
                            fit_log=fit_log,
                            out_dir=os.path.join(out, 'gapfix'))
    merged = merge_filled_book(path, os.path.join(out, 'gapfix'))
    return {'tool': TOOL_TAG, 'stage': 'prepare', 'path': path, 'out_dir': out,
            'n_columns': batch['n_columns'], 'columns': batch['columns'],
            'total_issues': batch['total_issues'],
            'filled_book': merged['files'][0],
            'filled_fraction_per_column': merged['filled_fraction_per_column'],
            'total_filled_cells': merged['total_filled_cells'],
            '编排提示': ('停点：目检各列标记图/整备报告（填充值=条件值）→ '
                     '确认后进入 stage=\'bpi2\'（输入 filled_book）')}


def chain_bpi2(path, q_unit, q_values=None, columns=None, phi_ref=None,
               params=None, out_dir=None):
    """stage='bpi2'：对 filled_book 峰积分（mask 路径按约定自动接线，
    q_values 走 Q9 文件头/显式，q_unit 必填不猜）。
    停点：目检 B 表 CSV + QC 网格图 → saxs_ruland_beq（scan_qmins）收官。"""
    if not os.path.isfile(path):
        raise ValueError('filled_book 不存在: %s' % path)
    if q_unit not in ('nm^-1', 'A^-1'):
        raise ValueError("q_unit 只支持 'nm^-1'/'A^-1'，必填不猜")
    stem = _stem(path)
    out = _resolve_out_dir(out_dir, stem)
    cols = _columns_or_all(path, columns)
    fix_dir = os.path.join(out, 'gapfix')
    masks = []
    for c in cols:
        m = os.path.join(fix_dir, '%s_col%d_mask.csv' % (stem, c))
        if not os.path.isfile(m):
            raise ValueError('col%d mask 缺失: %s（P7 纪律：报错不猜测；'
                             '请先跑 stage=prepare）' % (c, m))
        masks.append(m)
    card = bpi2_card(path, q_unit, q_values=q_values, columns=cols,
                     phi_ref=phi_ref, params=params, mask_files=masks,
                     out_dir=os.path.join(out, 'bpi2'))
    card['tool'] = TOOL_TAG
    card['stage'] = 'bpi2'
    card['mask_source'] = 'auto（chain 约定 <out>/gapfix/<stem>_col<k>_mask.csv）'
    card['编排提示'] = ('停点：目检 B 表 CSV + QC 网格图 → 确认后调用 '
                     'saxs_ruland_beq（传 scan_qmins 扫描自动采纳，或传 '
                     'q_window 人工裁决单窗）产出 B_eq / f_v')
    return card

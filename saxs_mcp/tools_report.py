# -*- coding: utf-8 -*-
"""报告工具（M2 通用化）：make_plots 三联图 + saxs_report 分析闭环交付物
通用引擎按 model_loader 加载的模型插件工作，不含模型知识：
  参数标签/文献元数据/样品名 ← manifest.json（labels / refs / modes 字段）
  方法说明/结果解读/结论文本 ← 插件 report 钩子（liposome 文本逐字移植自 M1，
  即 2026-09-23 用户已审核版本）
saxs_report 生成四件套：分析报告.md / 参数表.csv / 拟合曲线.csv / result.json
"""
import csv
import datetime
import json
import os
import re

import matplotlib
matplotlib.use('Agg')          # MCP stdio 进程内禁用交互后端
import matplotlib.pyplot as plt
import numpy as np

import model_loader
import quality_gate
import workdir
from curve_io import load_curve

# M1 版硬编码的参数标签 _LABELS 与文献元数据 _REF 已移入
# model_library/validated/liposome_membrane/manifest.json（labels / refs 字段）

# Δρ(z) 剖面 z 网格常用步长（nm）——交付物约定（项目记忆）
Z_STEP = 0.02

# M2.5-2 渲染器面板原语（布局由插件 plot_layout 声明，本文件不含模型知识）
_PANEL_KINDS = ('fit', 'residual', 'profile', 'curve')


def _render_panel(ax, d, ctx):
    """按布局声明绘制单个面板。d: panel DSL；ctx: 渲染上下文（数据与钩子）"""
    kind = d['kind']
    if kind == 'fit':
        ax.errorbar(ctx['q'], ctx['I'], ctx['sig'], fmt='.', ms=2, lw=0.4,
                    alpha=0.4, color='k', label='data')
        ml = d.get('model_label', 'fit')     # 标签支持 manifest 字段模板（如 {fit_label}）
        if isinstance(ml, str):
            ml = ml.format(**ctx['mmeta'])
        ax.plot(ctx['qq'], ctx['model_v'](ctx['qq']), 'r-', lw=1.5,
                label=ml)
        for comp in ctx['cfg'].get('components', []):
            ax.plot(ctx['qq'], comp['calc'](ctx['qq'], ctx['params'], ctx['fixed']),
                    comp.get('style', 'b--'), lw=1.0, label=comp['label'])
        if ctx['compare_params'] is not None and ctx['cfg'].get('compare'):
            ccfg = ctx['cfg']['compare']
            ax.plot(ctx['qq'], ccfg['calc'](_vec(ctx['compare_params'], ccfg['names']),
                                            ctx['qq'], ctx['fixed']),
                    color='0.5', ls=':', lw=1.5,
                    label=ctx['mmeta'].get('compare_label', 'compare model'))
        # 轴范围数据锚定（2026-10-09 缺陷修复）：fit 面板未显式声明 ylim 时，
        # 以窗内数据 min/max 锚定 Y 轴（含边距），分量延拓线由 matplotlib 裁剪。
        # 同时杜绝两类历史缺陷：分量线撑爆 autoscale（10-06 unified）、
        # 手工常量 ylim 错锚量级截断数据（10-09 fuzzy，低 q 段整体不可见）。
        # 插件可声明 'ylim_data': False 退出本行为
        if d.get('ylim') is None and d.get('ylim_data', True):
            I_f = ctx['I'][np.isfinite(ctx['I'])]
            if d.get('yscale') == 'log':
                I_f = I_f[I_f > 0]
            if I_f.size:
                lo, hi = float(np.min(I_f)), float(np.max(I_f))
                if d.get('yscale') == 'log':
                    top = hi * 2.2
                    # 底界在数据下界外再放一个量级：给分量曲线（form factor 侧瓣
                    # 等）显示空间，使各子曲线显示效果均衡（2026-10-09 用户定调）；
                    # 零点下探类曲线在 log 轴深度无限，"显示全"数学上不可能，
                    # 侧瓣形状可见即达标，深部仍由裁剪兜底。总跨度封顶 5 个量级，
                    # 防噪声低尾把数据本体压扁
                    bot = lo / 10.0
                    if top / bot > 1e5:
                        bot = top / 1e5
                    ax.set_ylim(bot, top)
                else:
                    pad = 0.08 * (hi - lo if hi > lo else max(abs(hi), 1.0))
                    ax.set_ylim(lo - pad, hi + pad)
    elif kind == 'residual':
        ax.plot(ctx['q'], (ctx['model_v'](ctx['q']) - ctx['I']) / ctx['I'], '.', ms=2)
        ax.axhline(0, color='r', lw=0.8)
        # 残差面板数据锚定（2026-10-09）：robust 定界——个别极值离群点（如相对
        # 残差 −152）会把 autoscale 撑爆，主体压成"一条直线"；取 |r| 的 95 分位
        # ×2 对称定界，界外极值点不显示（数值已在 result/报告披露）。
        # 插件 'ylim' 声明或 'ylim_data': False 可退出
        if d.get('ylim') is None and d.get('ylim_data', True):
            rr = (ctx['model_v'](ctx['q']) - ctx['I']) / ctx['I']
            rr = rr[np.isfinite(rr)]
            if rr.size:
                h = 2.0 * float(np.quantile(np.abs(rr), 0.95))
                if h <= 0:
                    h = float(np.max(np.abs(rr)))
                if h > 0:
                    ax.set_ylim(-h, h)
    elif kind == 'profile':
        cfg = ctx['cfg']
        zz = np.linspace(cfg['z_range'][0], cfg['z_range'][1], cfg['profile_n'])
        ax.plot(zz, cfg['profile'](zz, ctx['params'], ctx['fixed']), 'g-')
        ax.axhline(0, color='gray', lw=0.6)
    elif kind == 'curve':
        for c in d.get('curves', []):
            ax.plot(ctx['qq'], c['calc'](ctx['qq'], ctx['params'], ctx['fixed']),
                    c.get('style', 'b-'), lw=1.0, label=c.get('label', ''))
        if 'hline' in d:
            ax.axhline(d['hline'], color='gray', lw=0.6)
    # 通用轴装饰与标题
    if d.get('xlabel'):
        ax.set_xlabel(d['xlabel'])
    if d.get('ylabel'):
        ax.set_ylabel(d['ylabel'])
    if d.get('yscale'):
        ax.set_yscale(d['yscale'])
    if d.get('xscale'):
        ax.set_xscale(d['xscale'])
    if d.get('ylim') is not None:
        ax.set_ylim(d['ylim'])
    t = d.get('title')
    if t is None and kind == 'profile':      # profile 标题缺省回落到插件钩子
        t = ctx['cfg']['profile_title'](ctx['params'], ctx['derived'])
    if callable(t):
        t = t(ctx['params'], ctx['derived'])
    elif isinstance(t, str):
        t = t.format(title=ctx['extra_title'], **ctx['mmeta'])
    if t:
        ax.set_title(t)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8)


def _figure_from_layout(layout, ctx):
    """按布局 DSL 组装图窗（grid + panel 列表），返回 fig"""
    fig = plt.figure(figsize=tuple(layout.get('figsize', (10, 7))))
    grid = tuple(layout['grid'])
    for d in layout['panels']:
        ax = plt.subplot2grid(grid, tuple(d['at']),
                              colspan=d.get('colspan', 1), rowspan=d.get('rowspan', 1))
        _render_panel(ax, d, ctx)
    return fig


def register(mcp):
    @mcp.tool()
    def make_plots(model: str, mode: str, path: str, params: dict, out_png: str,
                   q_unit: str = 'A^-1', qmin: float = 0.3, qmax: float = 4.4,
                   fixed: dict = None, compare_params: dict = None,
                   title: str = '', derived: dict = None,
                   fmt: dict = None) -> str:
        """生成标准图（布局由模型插件 plot_layout 声明），返回 PNG 路径。
        本工具是纯渲染器（M2.5-2）：panel 原语 fit / residual / profile / curve，
        布局（grid / figsize / panel 列表）全部来自插件声明，不预设任何具体模型。
        model/mode: 模型名与拟合模式；params: saxs_fit 返回的 params；
        声明了 fixed_keys 的模式需传 fixed（如跨样品固定参数）；
        compare_params: 对照模型参数（模式声明 compare 时画对照虚线）；
        derived: saxs_fit 返回的 derived（剖面标题用）；title: 附加标题（如 'chi2/dof = 2.09'）；
        fmt: 数据格式显式声明（saxs_probe_curve 确认卡，None = 常规三列路径）。"""
        info = model_loader.load_model(model)
        man, cfg = info['manifest'], info['modes'][mode]
        mmeta = man['modes'][mode]
        layout = cfg.get('plot_layout')
        if layout is None:
            raise ValueError('模式 %s/%s 未声明 plot_layout（M2.5-2 渲染器契约），'
                             '无法绘图；请在插件中声明布局 DSL' % (model, mode))
        path, out_png = _resolve(path), _resolve(out_png)
        q, I, sig, _ = load_curve(path, q_unit, qmin, qmax, fmt)
        qq = np.linspace(q[0], q[-1], 800)
        pv = _vec(params, cfg['names'])
        ctx = {'q': q, 'I': I, 'sig': sig, 'qq': qq,
               'model_v': lambda x: cfg['calc'](pv, x, fixed),
               'cfg': cfg, 'mmeta': mmeta, 'params': params, 'fixed': fixed,
               'compare_params': compare_params, 'derived': derived, 'extra_title': title}

        fig = _figure_from_layout(layout, ctx)
        os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return out_png

    @mcp.tool()
    def saxs_report(model: str, mode: str, path: str, params: dict, perr: dict,
                    chi2_dof: float, dof: int, out_dir: str = None, fixed: dict = None,
                    at_bound: list = None, nopeg_compare: dict = None,
                    compare_params: dict = None,
                    sample_note: str = '', png_name: str = '',
                    q_unit: str = 'A^-1', qmin: float = 0.3, qmax: float = 4.4,
                    fmt: dict = None) -> dict:
        """生成完整分析闭环交付物（4 个文件），返回文件清单。
        model/mode: 模型与模式；params/perr/chi2_dof/dof/at_bound/nopeg_compare 来自 saxs_fit 返回值；
        peg 模式需传 fixed（固定膜参数）；compare_params: 对照模型参数（nopeg_compare.params，
        传入且模式支持 compare 时，拟合曲线.csv 增加对照曲线列 I_compare）；
        sample_note: 样品备注；png_name: 同目录三联图文件名；
        out_dir: 交付物目录（缺省 = 会话工作目录的 HEPS_SAXS_analysis/<数据stem>/
        子目录，未设工作目录时兜底工作区 runs/<数据stem>/）；
        fmt: 数据格式显式声明（saxs_probe_curve 确认卡，None = 常规三列路径）。"""
        info = model_loader.load_model(model)
        man, cfg = info['manifest'], info['modes'][mode]
        mmeta = man['modes'][mode]
        labels = man['labels']
        path = _resolve(path)
        q, I, sig, summ = load_curve(path, q_unit, qmin, qmax, fmt)
        if out_dir is None:      # 261104：缺省 → 会话工作目录（未设则兜底 runs/）
            out_dir = workdir.default_out(os.path.splitext(os.path.basename(path))[0])
        os.makedirs(out_dir, exist_ok=True)
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        at = list(at_bound or [])
        names = cfg['names']
        # 分级与 quality_gate 同源（grade_params 单一实现，杜绝双实现漂移）——
        # 2026-09-24 实测曾漏归一量（桥接模型 A/B 误标"稳健"）；2026-10-09 起
        # 同源启用 perr 爆表降级（perr/|value| > 1 → 条件值）
        grade = quality_gate.grade_params(names, at,
                                          cfg.get('normalized_params'),
                                          params, perr)
        der = cfg['derived'](params, perr, fixed)
        model_v = lambda x: cfg['calc'](_vec(params, names), x, fixed)

        # ---- 1. result.json（机器可读全量结果）----
        result = {'schema': 'heps-saxs/result-v0', 'kind': mode, 'sample': mmeta['sample'],
                  'created': now, 'model': man['description'],
                  'model_version': '%s %s' % (man['name'], man.get('version', '')),
                  'params': params, 'perr': perr, 'at_bound': at, 'grade': grade,
                  'chi2_dof': chi2_dof, 'dof': int(dof), 'derived': der,
                  'nopeg_compare': nopeg_compare,
                  'provenance': {'data_file': path,
                                 'q_unit_input': q_unit + (' (x10 -> nm^-1)' if q_unit == 'A^-1' else ''),
                                 'qmin': qmin, 'qmax': qmax, 'n_points': int(len(q)),
                                 'fitting': 'scipy least_squares trf, x_scale=jac',
                                 'fixed': fixed,
                                 'fmt': fmt}}
        rj = os.path.join(out_dir, '%s_result.json' % mode)
        with open(rj, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # ---- 2. 参数表.csv ----
        pc = os.path.join(out_dir, '%s_参数表.csv' % mode)
        with open(pc, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['参数', '含义', '值', '误差', '单位', '撞界', '可靠性分级'])
            for n in names:
                lab, unit = labels.get(n, (n, ''))
                w.writerow([n, lab, '%.6g' % params[n], '%.3g' % perr[n], unit,
                            '是' if n in at else '', grade[n]])
            for k, v in der.items():
                if k.endswith('_err') or not isinstance(v, (int, float)):
                    continue        # conformation 等字符串派生量不进 CSV 数值列
                lab, unit = labels.get(k, (k, ''))
                w.writerow([k, '派生量: ' + lab, '%.6g' % v, '%.3g' % der.get(k + '_err', float('nan')),
                            unit, '', ''])

        # ---- 3. 拟合曲线.csv（逐点；图中绘制的全部数据都在本表新增列）----
        cc = os.path.join(out_dir, '%s_拟合曲线.csv' % mode)
        Ifit = model_v(q)
        comps = cfg.get('components', [])
        comp_arrs = [c['calc'](q, params, fixed) for c in comps]
        extras = list(zip([c['column'] for c in comps], comp_arrs))
        # T5：对照模型曲线列（传入 compare_params 且模式支持 compare 时新增，位于 I_fit 之后）
        ccfg = cfg.get('compare')
        use_cmp = compare_params is not None and ccfg is not None
        if use_cmp:
            cmp_arr = ccfg['calc'](_vec(compare_params, ccfg['names']), q, fixed)
            extras.append((mmeta.get('compare_column', 'I_compare'), cmp_arr))
        # Δρ(z) 剖面：按常用步长 0.02 nm 生成 z 网格，作为新增列；与 q 不等长时短列留空。
        # 模式未声明 z_range（无剖面钩子，如 M2.5-4 桥接模型）时跳过 z 列
        has_prof = cfg.get('z_range') is not None
        if has_prof:
            zmin, zmax = cfg['z_range']
            zz = np.arange(zmin, zmax + Z_STEP / 2, Z_STEP)
            dr = cfg['profile'](zz, params, fixed)
        n_rows = len(q)
        # 背底独立列（2026-10-27 用户裁决，生成规则）：模型声明常数背景参数 'B'
        # （引擎背景参数命名契约）时，拟合曲线.csv 增列 I_background（逐点常数 = B），
        # 位于各分量/对照列之后、rel_residual 之前。配套契约：插件分量列为**纯项**
        # （不含 B，命名 I_power/I_closedpore/...），背底由本独立列承载，
        # 并以 components_sum_to_fit=True 接受下方加和硬门
        if 'B' in params:
            extras.append(('I_background',
                           np.full(n_rows, float(params['B']), dtype=float)))
        # 加和硬门（2026-10-27 用户裁决，强限制纪律）：插件声明
        # components_sum_to_fit=True 时，I_fit 必须逐点等于
        # Σ分量列 + I_background（相对偏差 >1e-6 即报错拒绝生成），
        # 保证总拟合曲线恒等于子拟合曲线（含背底）的简单加和
        if comps and cfg.get('components_sum_to_fit'):
            ssum = comp_arrs[0].copy()
            for _a in comp_arrs[1:]:
                ssum = ssum + _a
            if 'B' in params:
                ssum = ssum + float(params['B'])
            dev = float(np.max(np.abs(ssum - Ifit)
                               / np.maximum(np.abs(Ifit), 1e-300)))
            if dev > 1e-6:
                raise ValueError(
                    '拟合曲线加和硬门未通过（max_rel_dev=%.3e > 1e-6）：'
                    'I_fit ≠ Σ分量列 + I_background；'
                    '检查插件 components 声明与模型式的一致性' % dev)
        header = (['q_nm-1', 'I_obs', 'I_sigma', mmeta['fit_column']]
                  + [name for name, _ in extras]
                  + ['rel_residual'] + (['z_nm', 'drho_z_relative'] if has_prof else []))
        nq = 4 + len(extras) + 1
        with open(cc, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(n_rows):
                qf = (['%.6f' % q[i], '%.6e' % I[i], '%.6e' % sig[i], '%.6e' % Ifit[i]]
                      + ['%.6e' % arr[i] for _, arr in extras]
                      + ['%.4f' % ((Ifit[i] - I[i]) / I[i])]) if i < len(q) else [''] * nq
                zf = (['%.4f' % zz[i], '%.6e' % dr[i]] if has_prof and i < len(zz)
                      else (['', ''] if has_prof else []))
                w.writerow(qf + zf)

        # ---- 4. 分析报告.md ----
        md = _build_md(man, mmeta, cfg, mode, path, params, perr, chi2_dof, dof, at,
                       grade, der, nopeg_compare, sample_note, summ, now, png_name,
                       labels, fixed)
        mp = os.path.join(out_dir, '%s_分析报告.md' % mode)
        with open(mp, 'w', encoding='utf-8') as f:
            f.write(md)
        return {'files': [rj, pc, cc, mp], 'out_dir': out_dir}


def _vec(d, names):
    """参数 dict -> numpy 向量（顺序按 names）"""
    return np.array([float(d[n]) for n in names], dtype=float)


def _md_param_table(names, params, perr, grade, at, labels):
    rows = []
    for n in names:
        lab, unit = labels.get(n, (n, ''))
        rows.append('| %s | %s | %.4g ± %.4g | %s | %s |'
                    % (n, lab, params[n], perr[n], unit or '—',
                       '条件值（撞界）' if n in at else grade[n]))
    return '\n'.join(rows)


def _build_md(man, mmeta, cfg, mode, path, params, perr, chi2_dof, dof, at, grade,
              der, nopeg_compare, sample_note, summ, now, png_name, labels, fixed):
    """分析报告正文：框架通用（五节结构），模型语义由 manifest 元数据 + 插件钩子提供"""
    refs = man['refs']
    base = os.path.basename(path)
    png = png_name or '（未生成，可调用 make_plots 补充）'

    head = ['# %s SAXS 分析报告' % mmeta['sample'],
            '', '- 生成时间：%s' % now,
            '- 分析系统：HEPS SAXS Agent（M2 通用引擎，模型插件 %s %s）' % (man['name'], man.get('version', '')),
            '- 数据文件：`%s`（窗口内 %d 点，q = %.2f–%.2f nm⁻¹）'
            % (base, summ['n_used'], summ['q_min'], summ['q_max'])]
    # 样品备注分条（workflow §5：舍细节留结论、每条一行；按 ；/;/换行 拆分）
    _note_items = [s.strip() for s in re.split(r'[；;\n]+', sample_note or '') if s.strip()]
    if _note_items:
        head.append('- 样品备注：')
        head += ['- %s' % s for s in _note_items]
    else:
        head.append('- 样品备注：（未提供）')
    head.append('')

    z_desc = ('z_nm/drho_z_relative（Δρ(z) 电子密度剖面）' if cfg.get('z_range') is not None
              else '该模式无剖面列')
    files_md = ['## 1. 文件说明', '',
                '| 文件 | 说明 |', '|---|---|',
                '| `%s` | 原始 1D 数据（窗口内列语义见数据自带说明） |' % base,
                '| `%s_拟合曲线.csv` | 绘图全数据（单表逐行对齐）：q、I_obs、I_sigma（误差棒）、I_fit、各分量列、I_background（常数背底独立列，模型声明 B 参数时）、相对残差 + 新增列 %s |' % (mode, z_desc),
                '| `%s_参数表.csv` | 全部拟合参数与派生量（值/误差/单位/撞界/分级） |' % mode,
                '| `%s_result.json` | 机器可读全量结果与溯源信息 |' % mode,
                '| `%s_分析报告.md` | 本报告 |' % mode,
                '| `%s`（同目录） | 拟合图（布局由模型声明：%s），dpi=200 |'
                % (png, '拟合曲线(log) + 相对残差 + Δρ(z) 剖面'
                   if cfg.get('z_range') is not None else '拟合曲线(log) + 相对残差'), '']

    ref_md = ['## 2. 参考文献说明', '',
              '- 主文献：%s' % refs['paper'],
              '- 补充材料：%s' % refs['si'],
              '- 采用模型：%s' % man['description']]
    # 模型假设分条（workflow §5：每条一行；refs.assumptions 为列表，逐条出弹点）
    _assume_items = [a.strip() for a in refs['assumptions'] if a.strip()]
    if _assume_items:
        ref_md.append('- 模型假设：')
        ref_md += ['  - %s' % a for a in _assume_items]
    else:
        ref_md.append('- 模型假设：（未声明）')
    ref_md.append('')

    method = cfg['report']['method'](fixed)
    res_rows = _md_param_table(cfg['names'], params, perr, grade, at, labels)
    results = cfg['report']['results_md'](der, chi2_dof, dof, at, nopeg_compare)
    grade_md = ['## 4. 分析结果说明', '',
                '| 参数 | 含义 | 值 ± 误差 | 单位 | 可靠性分级 |', '|---|---|---|---|---|',
                res_rows, ''] + results
    concl = cfg['report']['conclusions'](params, perr, chi2_dof, dof, at, der,
                                         nopeg_compare, sample_note)
    concl_head = ['## 5. 结论', '']

    # 「算法与公式」节（2026-10-22 用户新需求，方式一）：插件可选声明 report.algo 钩子，
    # 声明即追加为第 6 节；未声明模型零变化（向后兼容）
    algo_md = []
    algo_hook = cfg['report'].get('algo')
    if algo_hook is not None:
        algo_md = ['## 6. 算法与公式', ''] + list(algo_hook())

    return '\n'.join(head + files_md + ref_md + method + grade_md
                     + concl_head + concl + algo_md)


def _resolve(path):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(os.getcwd(), path))

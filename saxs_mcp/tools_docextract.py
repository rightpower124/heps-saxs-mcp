# -*- coding: utf-8 -*-
"""saxs_doc_extract：文献/说明文档全文提取工具（PDF / docx / pptx -> 纯文本）

背景（2026-10-23 用户指令）：文献学习（方法学判定、SI 核对）是分析会话的常规
前置动作，此前靠临时脚本 + 用户手动代跑（Shell 沙盒崩溃期完全阻塞）。本工具把
该能力内化为 MCP 工具，Agent 自助提取，不再依赖用户操作。

支持格式（按扩展名分派，不猜内容）：
- .pdf  : pypdf 逐页提取（页分隔符 ===== PAGE i/N =====），扫描版 PDF 提取出
          空文本时如实报 empty_text 不伪造成功（OCR 不在本工具范围）
- .docx : zip 直读 word/document.xml，<w:t> run 拼接 + <w:p> 段落断行
          （含表格内嵌文本；XML 实体反转义）
- .pptx : zip 直读 ppt/slides/slideN.xml，<a:t> run 拼接（逐 slide 分节）

落盘纪律（lessons 09-28 / 09-29）：
- 一切路径参数用绝对路径；相对路径按宿主 cwd 解析的坑以 _resolve 显式归一
- 缺省输出目录以 _ROOT 锚定工作区根（runs/doc_extract/<stem>/），不相对 cwd
- runs 产物生命周期照旧：用户 IDE 目检后手动清空，本工具不做删除

注意：本模块为核心模块，改动后需重启 MCP 服务器生效（lessons 2026-10-06）。
"""
import os
import re
import zipfile

import workdir          # 会话级工作目录（部署优化 261104）：缺省产物重定向

_TOOL_VER = '0.1.0'
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SUPPORTED = ('.pdf', '.docx', '.pptx')


def _resolve(path):
    """绝对路径原样返回；相对路径按工作区根锚定（不按宿主 cwd，lessons 09-28）。"""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_ROOT, path))


def _xml_entities(text):
    return (text.replace('&amp;', '&').replace('&lt;', '<')
                .replace('&gt;', '>').replace('&quot;', '"')
                .replace('&apos;', "'"))


def _extract_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    n = len(reader.pages)
    pages = []
    empty_pages = 0
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ''
        except Exception as e:
            pages.append('\n===== PAGE %d/%d EXTRACT_ERROR: %s =====\n' % (i + 1, n, e))
            continue
        if not t.strip():
            empty_pages += 1
        pages.append('\n===== PAGE %d/%d =====\n%s' % (i + 1, n, t))
    return ''.join(pages), {'pages': n, 'empty_text_pages': empty_pages}


def _extract_docx(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read('word/document.xml').decode('utf-8', errors='replace')
    xml = re.sub(r'<w:p[ >/]', '\n<w:p ', xml)
    runs = re.findall(r'<w:t[^>]*>(.*?)</w:t>|(\n)', xml, flags=re.S)
    text = ''.join(t if t else '\n' for t, br in runs)
    return _xml_entities(text), {}


def _extract_pptx(path):
    with zipfile.ZipFile(path) as z:
        slides = sorted((m for m in z.namelist()
                         if re.match(r'ppt/slides/slide\d+\.xml$', m)),
                        key=lambda s: int(re.search(r'(\d+)', s).group(1)))
        parts = []
        for s in slides:
            xml = z.read(s).decode('utf-8', errors='replace')
            texts = re.findall(r'<a:t>(.*?)</a:t>', xml, flags=re.S)
            parts.append('--- %s ---\n%s' % (s, '\n'.join(_xml_entities(t) for t in texts)))
        return '\n'.join(parts), {'slides': len(slides)}


def register(mcp):
    @mcp.tool()
    def saxs_doc_extract(path: str, out_dir: str = None) -> dict:
        """文献/说明文档全文提取（PDF/docx/pptx -> 纯文本 .txt），供文献方法学学习。

        path: 文档绝对路径（支持 .pdf / .docx / .pptx，按扩展名分派不猜内容）；
        out_dir: 输出目录绝对路径（缺省 runs/doc_extract/<文件名stem>/，
        以工作区根锚定）。提取文本落盘为 <stem>_extract.txt（UTF-8），
        返回输出路径 / 页数或 slide 数 / 字符数 / warnings。

        纪律：扫描版 PDF（图片型）提取为空文本时如实报 empty_text=true，
        不伪造成功；OCR 不在本工具范围。核心模块，改动需重启服务器生效。
        """
        warnings = []
        src = _resolve(path)
        if not os.path.isfile(src):
            return {'ok': False, 'error': 'file not found: %s' % src}
        ext = os.path.splitext(src)[1].lower()
        if ext not in _SUPPORTED:
            return {'ok': False, 'error': 'unsupported extension %r (supported: %s)'
                    % (ext, ', '.join(_SUPPORTED))}
        stem = os.path.splitext(os.path.basename(src))[0]
        if out_dir:
            od = _resolve(out_dir)
        else:
            od = workdir.default_out('doc_extract', stem)
        os.makedirs(od, exist_ok=True)
        out_path = os.path.join(od, stem + '_extract.txt')

        if ext == '.pdf':
            text, meta = _extract_pdf(src)
        elif ext == '.docx':
            text, meta = _extract_docx(src)
        else:
            text, meta = _extract_pptx(src)

        if not text.strip():
            warnings.append('empty text extracted (scanned/image PDF? OCR not supported)')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)

        return {
            'ok': True,
            'tool_version': _TOOL_VER,
            'source': src,
            'output': out_path,
            'chars': len(text),
            **meta,
            'warnings': warnings,
        }

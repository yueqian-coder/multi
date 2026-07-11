from __future__ import annotations

import html
import re
import shutil
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "course-report.md"
OUTPUT = ROOT / "deliverables" / "ClaimScope-course-report.pdf"
SUBMISSION = ROOT / "deliverables" / "自然语言处理大作业-刘子谦-23354118.pdf"
INK = colors.HexColor("#17212B")
BLUE = colors.HexColor("#1F5A7A")
MUTED = colors.HexColor("#60717A")
LINE = colors.HexColor("#CBD5DA")


def register_fonts() -> None:
    font_dir = Path(r"C:\Windows\Fonts")
    pdfmetrics.registerFont(TTFont("SimSun", str(font_dir / "simsun.ttc")))
    pdfmetrics.registerFont(TTFont("MicrosoftYaHei", str(font_dir / "msyh.ttc")))
    pdfmetrics.registerFont(TTFont("MicrosoftYaHei-Bold", str(font_dir / "msyhbd.ttc")))
    pdfmetrics.registerFont(TTFont("Arial", str(font_dir / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(font_dir / "arialbd.ttf")))
    pdfmetrics.registerFontFamily(
        "SimSun",
        normal="SimSun",
        bold="MicrosoftYaHei-Bold",
        italic="SimSun",
        boldItalic="MicrosoftYaHei-Bold",
    )


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover": ParagraphStyle(
            "Cover",
            parent=base["Title"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=22,
            leading=32,
            alignment=TA_CENTER,
            textColor=BLUE,
            spaceAfter=22,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            fontName="MicrosoftYaHei-Bold",
            fontSize=15,
            leading=22,
            alignment=TA_CENTER,
            textColor=INK,
        ),
        "meta": ParagraphStyle(
            "Meta",
            fontName="MicrosoftYaHei",
            fontSize=11,
            leading=19,
            alignment=TA_CENTER,
            textColor=MUTED,
        ),
        "h1": ParagraphStyle(
            "H1",
            fontName="MicrosoftYaHei-Bold",
            fontSize=16,
            leading=22,
            textColor=BLUE,
            spaceBefore=12,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            fontName="MicrosoftYaHei-Bold",
            fontSize=13,
            leading=19,
            textColor=INK,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            fontName="SimSun",
            fontSize=10.2,
            leading=16.2,
            textColor=INK,
            alignment=TA_JUSTIFY,
            firstLineIndent=2 * 10.2,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            fontName="SimSun",
            fontSize=10,
            leading=15,
            textColor=INK,
            leftIndent=14,
            firstLineIndent=-8,
            bulletIndent=4,
            spaceAfter=3,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "Caption",
            fontName="MicrosoftYaHei",
            fontSize=8.5,
            leading=12,
            alignment=TA_CENTER,
            textColor=MUTED,
            spaceAfter=8,
        ),
        "code": ParagraphStyle(
            "Code",
            fontName="Arial",
            fontSize=8.5,
            leading=13,
            leftIndent=12,
            textColor=colors.HexColor("#34444D"),
            backColor=colors.HexColor("#F2F5F6"),
            borderPadding=7,
            spaceAfter=6,
        ),
        "toc": ParagraphStyle(
            "TOC",
            fontName="MicrosoftYaHei",
            fontSize=11,
            leading=18,
            leftIndent=18,
            textColor=INK,
        ),
    }


def inline_markup(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`([^`]+)`", r'<font color="#1F5A7A">\1</font>', escaped)
    return escaped


def markdown_table(lines: list[str], style: ParagraphStyle, width: float) -> Table:
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    rows = [rows[0]] + rows[2:]
    rendered = [
        [Paragraph(inline_markup(cell), style) for cell in row]
        for row in rows
    ]
    column_widths = [width / len(rows[0])] * len(rows[0])
    table = Table(rendered, colWidths=column_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEFF1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), INK),
                ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def draw_page(canvas, document) -> None:
    canvas.saveState()
    if document.page > 1:
        canvas.setStrokeColor(colors.HexColor("#D9E0E3"))
        canvas.line(2.4 * cm, A4[1] - 1.55 * cm, A4[0] - 2.4 * cm, A4[1] - 1.55 * cm)
        canvas.setFont("MicrosoftYaHei", 8)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(A4[0] - 2.4 * cm, A4[1] - 1.3 * cm, "ClaimScope  |  自然语言处理课程大作业")
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, str(document.page))
    canvas.restoreState()


def build_pdf() -> Path:
    register_fonts()
    text_styles = styles()
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    page_width = A4[0] - 4.8 * cm
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=2.4 * cm,
        rightMargin=2.4 * cm,
        topMargin=2.0 * cm,
        bottomMargin=1.8 * cm,
        title="ClaimScope：面向科研想法生成前的主张、假设与负面证据发现智能体",
        author="刘子谦",
        subject="自然语言处理课程大作业",
    )
    story = [Spacer(1, 3.5 * cm)]
    cover_lines = [line for line in lines[:8] if line.strip()][:4]
    story.append(Paragraph(inline_markup(cover_lines[0].removeprefix("# ")), text_styles["cover"]))
    story.append(Paragraph(inline_markup(cover_lines[1]), text_styles["subtitle"]))
    story.append(Spacer(1, 4.0 * cm))
    story.append(Paragraph(inline_markup(cover_lines[2]), text_styles["meta"]))
    story.append(Paragraph(inline_markup(cover_lines[3]), text_styles["meta"]))
    story.append(PageBreak())
    story.append(Paragraph("目录", text_styles["cover"]))
    for line in lines:
        if line.startswith("## ") and line != "## 参考文献":
            story.append(Paragraph(inline_markup(line[3:]), text_styles["toc"]))
    story.append(PageBreak())

    index = lines.index("## 摘要")
    in_code = False
    code_lines: list[str] = []
    figure_number = 0
    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        if line.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), text_styles["code"]))
                code_lines = []
            in_code = not in_code
            index += 1
            continue
        if in_code:
            code_lines.append(raw)
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and re.fullmatch(r"[|:\- ]+", lines[index + 1].strip()):
            table_lines = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            story.append(markdown_table(table_lines, text_styles["body"], page_width))
            story.append(Spacer(1, 6))
            continue
        image_match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
        if image_match:
            figure_number += 1
            image_path = SOURCE.parent / image_match.group(2)
            image = Image(str(image_path))
            scale = min(page_width / image.imageWidth, 10.2 * cm / image.imageHeight)
            image.drawWidth = image.imageWidth * scale
            image.drawHeight = image.imageHeight * scale
            image.hAlign = "CENTER"
            story.append(image)
            story.append(Paragraph(f"图 {figure_number}　{html.escape(image_match.group(1))}", text_styles["caption"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), text_styles["h2"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), text_styles["h1"]))
        elif line.startswith("- "):
            story.append(Paragraph(inline_markup(line[2:]), text_styles["bullet"], bulletText="•"))
        elif line:
            story.append(Paragraph(inline_markup(line), text_styles["body"]))
        index += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    shutil.copy2(OUTPUT, SUBMISSION)
    return OUTPUT


if __name__ == "__main__":
    print(build_pdf())

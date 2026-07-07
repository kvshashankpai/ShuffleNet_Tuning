#!/usr/bin/env python3
"""
Convert Report.md to Report.pdf using fpdf2.
Parses Markdown manually into styled PDF with headings, tables, code blocks, etc.
"""

import re
from pathlib import Path
from fpdf import FPDF


def sanitize(text: str) -> str:
    """Replace unicode characters with ASCII equivalents for latin-1 fonts."""
    replacements = {
        '\u2013': '-',   # en-dash
        '\u2014': '--',  # em-dash
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2026': '...', # ellipsis
        '\u2192': '->',  # right arrow
        '\u2190': '<-',  # left arrow
        '\u2191': '^',   # up arrow
        '\u2193': 'v',   # down arrow
        '\u00d7': 'x',   # multiplication sign
        '\u2265': '>=',  # greater than or equal
        '\u2264': '<=',  # less than or equal
        '\u2248': '~=',  # approximately equal
        '\u2260': '!=',  # not equal
        '\u221e': 'inf', # infinity
        '\u03b5': 'e',   # epsilon
        '\u2022': '-',   # bullet
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Final fallback: replace any remaining non-latin-1 chars
    return text.encode('latin-1', errors='replace').decode('latin-1')


class ReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_page()
        self.set_auto_page_break(auto=True, margin=15)
        # Use built-in fonts
        self.set_font("Helvetica", size=10)

    def chapter_title(self, title, level=1):
        sizes = {1: 18, 2: 15, 3: 13}
        self.ln(4 if level > 1 else 8)
        self.set_font("Helvetica", "B", sizes.get(level, 12))
        self.multi_cell(0, 8, sanitize(title))
        if level == 1:
            self.set_draw_color(41, 128, 185)
            self.set_line_width(0.8)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(3)
        else:
            self.ln(2)
        self.set_font("Helvetica", size=10)

    def body_text(self, text):
        self.set_font("Helvetica", size=10)
        self.multi_cell(0, 5.5, sanitize(text))
        self.ln(2)

    def bold_text(self, text):
        self.set_font("Helvetica", "B", 10)
        self.multi_cell(0, 5.5, sanitize(text))
        self.set_font("Helvetica", size=10)
        self.ln(1)

    def code_block(self, text):
        self.set_fill_color(240, 240, 240)
        self.set_font("Courier", size=9)
        for line in text.split("\n"):
            self.cell(0, 5, "  " + sanitize(line), ln=True, fill=True)
        self.ln(3)
        self.set_font("Helvetica", size=10)

    def table(self, headers, rows):
        self.set_font("Helvetica", "B", 9)
        col_count = len(headers)
        available_width = 190
        col_w = available_width / col_count

        # Header row
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        for h in headers:
            safe = sanitize(h.strip())
            self.cell(col_w, 7, safe, border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", size=9)
        self.set_text_color(0, 0, 0)
        alt = False
        for row in rows:
            if alt:
                self.set_fill_color(245, 245, 245)
            else:
                self.set_fill_color(255, 255, 255)
            for cell in row:
                safe = sanitize(cell.strip())
                self.cell(col_w, 6, safe, border=1, fill=True)
            self.ln()
            alt = not alt
        self.ln(3)

    def bullet(self, text):
        self.set_font("Helvetica", size=10)
        safe = sanitize(text)
        x = self.get_x()
        self.cell(5, 5.5, "-")
        self.multi_cell(0, 5.5, " " + safe)
        self.ln(1)

    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"ShuffleNetV2 BO Report - Page {self.page_no()}/{{nb}}", align="C")


def parse_and_render(md_path: Path, pdf_path: Path):
    """Parse markdown and render to PDF."""
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    pdf = ReportPDF()
    pdf.alias_nb_pages()

    # Title page
    pdf.set_font("Helvetica", "B", 22)
    pdf.ln(30)
    pdf.multi_cell(0, 12, "ShuffleNetV2 Bayesian\nHyperparameter Optimization", align="C")
    pdf.ln(5)
    pdf.set_font("Helvetica", size=14)
    pdf.multi_cell(0, 8, "Technical Report", align="C")
    pdf.ln(10)
    pdf.set_draw_color(41, 128, 185)
    pdf.set_line_width(1.5)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7,
        "Multi-Objective MOTPE & Hypervolume Maximization\n"
        "CPU-Only Training & INT8 Post-Training Quantization\n"
        "MedMNIST PathMNIST (9-class Histopathology)",
        align="C"
    )
    pdf.add_page()

    i = 0
    in_code = False
    code_buf = []
    in_table = False
    table_headers = []
    table_rows = []

    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.strip().startswith("```"):
            if in_code:
                pdf.code_block("\n".join(code_buf))
                code_buf = []
                in_code = False
            else:
                # Flush any pending table
                if in_table:
                    pdf.table(table_headers, table_rows)
                    in_table = False
                    table_headers = []
                    table_rows = []
                in_code = True
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # Table rows
        if "|" in line and line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # Check if separator row
            if all(re.match(r'^[-:]+$', c) for c in cells):
                i += 1
                continue
            if not in_table:
                in_table = True
                table_headers = cells
            else:
                table_rows.append(cells)
            i += 1
            continue
        else:
            if in_table:
                pdf.table(table_headers, table_rows)
                in_table = False
                table_headers = []
                table_rows = []

        # Headings
        if line.startswith("### "):
            pdf.chapter_title(line[4:].strip(), level=3)
            i += 1
            continue
        if line.startswith("## "):
            pdf.chapter_title(line[3:].strip(), level=2)
            i += 1
            continue
        if line.startswith("# "):
            # Skip the top-level title (already on title page)
            i += 1
            continue

        # Horizontal rules
        if line.strip() in ("---", "***", "___"):
            pdf.ln(3)
            pdf.set_draw_color(200, 200, 200)
            pdf.set_line_width(0.3)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)
            i += 1
            continue

        # Bullet points
        if line.strip().startswith("- ") or line.strip().startswith("* "):
            text = line.strip()[2:]
            # Remove markdown bold/italic
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
            text = re.sub(r'\*(.*?)\*', r'\1', text)
            text = re.sub(r'`(.*?)`', r'\1', text)
            pdf.bullet(text)
            i += 1
            continue

        # Numbered items
        m = re.match(r'^\d+\.\s+', line.strip())
        if m:
            text = line.strip()[m.end():]
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
            text = re.sub(r'`(.*?)`', r'\1', text)
            pdf.bullet(text)
            i += 1
            continue

        # Bold lines
        if line.strip().startswith("**") and line.strip().endswith("**"):
            clean = line.strip().strip("*")
            pdf.bold_text(clean)
            i += 1
            continue

        # Regular paragraph text
        stripped = line.strip()
        if stripped:
            # Clean markdown formatting
            stripped = re.sub(r'\*\*(.*?)\*\*', r'\1', stripped)
            stripped = re.sub(r'\*(.*?)\*', r'\1', stripped)
            stripped = re.sub(r'`(.*?)`', r'\1', stripped)
            pdf.body_text(stripped)

        i += 1

    # Flush any remaining table
    if in_table:
        pdf.table(table_headers, table_rows)

    pdf.output(str(pdf_path))
    print(f"PDF saved to {pdf_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parse_and_render(root / "Report.md", root / "Report.pdf")

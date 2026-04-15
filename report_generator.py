"""
Copyright 2026 Péter Lénárt

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""
================================================================================
HDX-MS EQUIVALENCE TESTING - PDF REPORT GENERATOR
================================================================================

PURPOSE:
    Captures terminal output and generates a PDF report with the same formatting.

USAGE:
    with OutputCapture() as capture:
        # ... run pipeline (all print statements are captured)
        pass

    generate_pdf_report(capture.get_output(), output_path)

================================================================================
"""

import sys
import io
import os
from datetime import datetime
from typing import Optional


class OutputCapture:
    """
    Context manager that captures stdout while still printing to terminal.

    Usage:
        with OutputCapture() as capture:
            print("This is captured and printed")

        text = capture.get_output()
    """

    def __init__(self):
        self._buffer = io.StringIO()
        self._original_stdout = None
        self._tee = None

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._tee = TeeStream(self._original_stdout, self._buffer)
        sys.stdout = self._tee
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        return False

    def get_output(self) -> str:
        """Get all captured output as a string."""
        return self._buffer.getvalue()


class TeeStream:
    """Stream that writes to two outputs simultaneously."""

    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, data):
        self.stream1.write(data)
        self.stream2.write(data)

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()


def _replace_unicode_chars(text: str) -> str:
    """
    Replace Unicode characters with ASCII equivalents for PDF compatibility.

    Parameters
    ----------
    text : str
        Text with potential Unicode characters

    Returns
    -------
    str
        Text with ASCII-safe replacements
    """
    replacements = {
        # Greek letters
        '\u0394': 'Delta',      # Δ
        '\u03b1': 'alpha',      # α
        '\u03b2': 'beta',       # β
        '\u03c3': 'sigma',      # σ
        # Subscripts 
        '\u2080': '0',          # ₀
        '\u2081': '1',          # ₁
        '\u2082': '2',          # ₂
        '\u2083': '3',          # ₃
        '\u2084': '4',          # ₄
        '\u2085': '5',          # ₅
        '\u2086': '6',          # ₆
        '\u2087': '7',          # ₇
        '\u2088': '8',          # ₈
        '\u2089': '9',          # ₉
        # Superscripts
        '\u00b2': '^2',         # ²
        '\u00b3': '^3',         # ³
        # Math symbols
        '\u2264': '<=',         # ≤
        '\u2265': '>=',         # ≥
        '\u00b1': '+/-',        # ±
        '\u2248': '~=',         # ≈
        '\u00d7': 'x',          # ×
        # Latin letters with diacritics that are not Latin-1 encodable
        '\u0160': 'S',          # Š
        # Arrows
        '\u2192': '->',         # →
        '\u2190': '<-',         # ←
    }

    for unicode_char, ascii_equiv in replacements.items():
        text = text.replace(unicode_char, ascii_equiv)

    return text


def generate_pdf_report(text: str, output_path: str, title: str = "HDX-MS TOST Analysis Report") -> str:
    """
    Generate a PDF report from captured terminal output.

    Parameters
    ----------
    text : str
        The captured terminal output text
    output_path : str
        Path to save the PDF file
    title : str, optional
        Title for the report header

    Returns
    -------
    str
        Path to the generated PDF file
    """
    try:
        from fpdf import FPDF
    except ImportError:
        # Fallback: save as text file if fpdf not available
        txt_path = output_path.replace('.pdf', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Note: fpdf not installed. Report saved as text: {txt_path}")
        print("Install fpdf2 for PDF output: pip install fpdf2")
        return txt_path

    # Create PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Use Courier (monospace) font for terminal-like appearance
    pdf.set_font("Courier", size=8)
    effective_page_width = pdf.w - pdf.l_margin - pdf.r_margin

    # Process text line by line
    lines = text.split('\n')

    for line in lines:
        # Replace known Unicode characters with ASCII equivalents
        safe_line = _replace_unicode_chars(line)

        # Handle any remaining special characters
        safe_line = safe_line.encode('latin-1', errors='replace').decode('latin-1')

        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(effective_page_width, 4, safe_line)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Save PDF
    pdf.output(output_path)

    return output_path


def generate_report_from_file(text_file: str, pdf_path: Optional[str] = None) -> str:
    """
    Generate PDF from a text file.

    Parameters
    ----------
    text_file : str
        Path to text file with terminal output
    pdf_path : str, optional
        Output PDF path. If None, uses text_file name with .pdf extension.

    Returns
    -------
    str
        Path to generated PDF
    """
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    if pdf_path is None:
        pdf_path = text_file.rsplit('.', 1)[0] + '.pdf'

    return generate_pdf_report(text, pdf_path)


if __name__ == "__main__":
    # Test the module
    print("Report Generator Module - TEST")
    print("Run main.py to execute the full pipeline.")

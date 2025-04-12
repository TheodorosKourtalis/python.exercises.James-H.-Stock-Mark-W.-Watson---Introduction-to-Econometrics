#!/usr/bin/env python3
import sys
import io
from fpdf import FPDF

def main():
    sample_text = sys.stdin.read()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in sample_text.splitlines():
        pdf.multi_cell(0, 10, line)
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    sys.stdout.buffer.write(pdf_buffer.getvalue())

if __name__ == "__main__":
    main()

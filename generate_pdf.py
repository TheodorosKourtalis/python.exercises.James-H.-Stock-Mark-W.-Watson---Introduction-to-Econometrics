#!/usr/bin/env python3
import sys
import io
from fpdf import FPDF

def generate_pdf_from_markdown(sample_text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in sample_text.splitlines():
        pdf.multi_cell(0, 10, line)
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_data = pdf_buffer.getvalue()
    pdf_buffer.close()
    return pdf_data

if __name__ == "__main__":
    sample_text = sys.stdin.read()
    pdf_data = generate_pdf_from_markdown(sample_text)
    sys.stdout.buffer.write(pdf_data)

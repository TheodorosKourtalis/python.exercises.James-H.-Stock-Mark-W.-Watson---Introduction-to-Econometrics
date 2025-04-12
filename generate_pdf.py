#!/usr/bin/env python3
import sys
from fpdf import FPDF

def generate_pdf_from_markdown(sample_text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in sample_text.splitlines():
        pdf.multi_cell(0, 10, line)
    # Χρησιμοποίησε dest="S" για να πάρεις το PDF ως string και μετά το κωδικοποίησε σε bytes.
    pdf_str = pdf.output(dest="S")
    return pdf_str.encode("latin1")

if __name__ == "__main__":
    sample_text = sys.stdin.read()
    pdf_data = generate_pdf_from_markdown(sample_text)
    sys.stdout.buffer.write(pdf_data)

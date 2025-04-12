#!/usr/bin/env python3
import sys
import io
from fpdf import FPDF

def main():
    # Read the sample answer text from stdin.
    sample_text = sys.stdin.read()
    
    # Create a new PDF.
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Split the text into lines and add them (multi_cell allows automatic wrapping).
    for line in sample_text.splitlines():
        pdf.multi_cell(0, 10, line)
    
    # Write the PDF to a BytesIO buffer.
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    
    # Output PDF data to stdout as bytes.
    sys.stdout.buffer.write(pdf_buffer.getvalue())

if __name__ == "__main__":
    main()

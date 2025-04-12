#!/usr/bin/env python3
import sys
import io
from fpdf import FPDF

def generate_pdf_from_markdown(sample_text: str) -> bytes:
    pdf = StyledPDF()
    pdf.add_markdown_text(sample_text)
    # Επιστρέφουμε το PDF ως bytes με τη χρήση dest="S"
    pdf_str = pdf.output(dest="S")
    return pdf_str.encode("latin1")

class StyledPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        # Χρησιμοποιούμε τη Helvetica ως βασική γραμματοσειρά
        self.set_font("Helvetica", size=12)

    def add_markdown_text(self, text: str):
        # Αντικατάσταση ορισμένων χαρακτήρων (π.χ. Unicode minus)
        replacements = {
            "−": "-",  # Unicode minus σε ASCII
            "≤": "<=",
            "≈": "~="
        }
        for key, val in replacements.items():
            text = text.replace(key, val)
        
        # Διαίρεση του κειμένου σε γραμμές
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            # Εάν ολόκληρη η γραμμή είναι bold (ξεκινά και τελειώνει με **)
            if line.startswith("**") and line.endswith("**"):
                self.set_font("Helvetica", size=12, style="B")
                self.multi_cell(0, 10, line.strip("*").strip())
                self.set_font("Helvetica", size=12, style="")
            # Inline bold: χωρίζουμε τη γραμμή στα ** και επεξεργαζόμαστε
            elif "**" in line:
                parts = line.split("**")
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # bold τμήμα
                        self.set_font("Helvetica", size=12, style="B")
                        self.write(10, part)
                        self.set_font("Helvetica", size=12, style="")
                    else:
                        self.write(10, part)
                self.ln(10)
            else:
                self.multi_cell(0, 10, line)

if __name__ == "__main__":
    sample_text = sys.stdin.read()
    pdf_data = generate_pdf_from_markdown(sample_text)
    sys.stdout.buffer.write(pdf_data)

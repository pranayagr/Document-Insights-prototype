import fitz
import docx

file = "/Users/pagrawal140/document-insights-prototype/data/a_2.1_financial_policy_manual_lubbock_chamber_of_commerce_11.19.pdf"

def load_pdf(file):
    doc = fitz.open(file)
    text = "\n".join(page.get_text() for page in doc)
    return text

def load_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

if __name__ == "__main__":
    data = load_pdf(file)
    print(data)

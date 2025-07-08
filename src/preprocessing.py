import base64
import io
import os
import sys
import pandas as pd
import numpy as np
import json
import fitz
import re
import difflib
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.llm_call import get_chat_completion

def convert_pdf_to_images_with_pymupdf(pdf_path, dpi=200):
    """Convert each page of a PDF into a PIL Image using PyMuPDF."""
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        images.append(image)
    return images

def image_to_base64(pil_image):
    """Convert a PIL Image to a base64-encoded PNG."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def extract_text_from_pdf_pages(pdf_path, output_json_path):
    """
    1. Converts PDF pages to images for GPT OCR.
    2. Sends each image to GPT for text and table extraction.
    3. Outputs the final JSON.
    """
    doc = fitz.open(pdf_path)
    images = convert_pdf_to_images_with_pymupdf(pdf_path)
    all_pages_data = []

    for idx, img in enumerate(images):
        page = doc.load_page(idx)
        base64_img = image_to_base64(img)

        system_prompt = (
            "You are a helpful assistant that extracts text and tables from images. "
            "If any tables are present, please convert them into Markdown format. "
            "Return your answer as valid JSON. "
            "Return ONLY valid JSON. Do not wrap your JSON in triple backticks or code fences. "
            "Structure it as a JSON array of objects, where each object has a single key (the heading) "
            "and the value is the text (including any tables in Markdown) that follows until the next heading. "
            "If there is text with no heading, place it under the key 'No Heading'. "
            "Do not include any extra commentary—only return valid JSON."
        )

        user_prompt = [
            {
                "type": "text",
                "text": f"This is page {idx + 1} of the PDF. Please extract all text and tables in JSON format as described."
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}"}
            }
        ]

        raw_response = get_chat_completion([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        gpt_response = raw_response.choices[0].message.content
        cleaned_response = re.sub(r"```(?:json)?", "", gpt_response).replace("```", "").strip()

        try:
            page_data = json.loads(cleaned_response)
        except json.JSONDecodeError:
            page_data = [{"Error": cleaned_response}]

        all_pages_data.append({
            "page_number": idx + 1,
            "extracted_data": page_data
        })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_pages_data, f, indent=2)
    print(f"Extraction complete! Output saved to {output_json_path}")

def chunk_text(text, max_words=250, overlap=50):
    # print("Text : ", text)
    text = str(text)
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks

def clean_text_for_embedding(text: str) -> str:
    # Remove pipe characters, newlines, and tabs
    text = re.sub(r'[\|\n\t]', ' ', text)
    # Reduce multiple hyphens to one
    text = re.sub(r'-{2,}', '-', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunking(df, filename):
    chunked_rows = []
    id = 0
    for i, row in df.iterrows():
        topic = row["Keyword"]
        full_text = row["Context"]
        page = row["Page_Number"]

        if full_text == np.nan:
            continue
        
        chunks = chunk_text(full_text)
        for chunk in chunks:
            metadata = {"chunk_id" : id, "topic" : topic, "source" : filename, "page" : page}
            id += 1
            chunked_rows.append({
                "chunk_text": clean_text_for_embedding(chunk),
                "metadata": json.dumps(metadata)
            })
    chunks_df = pd.DataFrame(chunked_rows)
    return chunks_df

def post_processing(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    final_rows = []

    for page in data_list:
        page_number = page.get("page_number", -1)
        extracted_data = page.get("extracted_data", [])
        for item in extracted_data:
            for key, value in item.items():
                keyword = key.strip()
                if keyword.lower() == "no heading":
                    keyword = "General"
                context = value.strip()
                
                final_rows.append({
                    "Context": context,
                    "Keyword": keyword,
                    "Page_Number" : page_number
                })

    df = pd.DataFrame(final_rows)[['Context', 'Keyword', 'Page_Number']]
    print(df)

    output_csv_path = f"/Users/pagrawal140/document-insights-prototype/output/extracted_data/{filepath.split('/')[-1][:-5]}.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"CSV file saved at: {output_csv_path}")

    chunks_df = chunking(df,filepath.split('/')[-1][:-5])
    return chunks_df

if __name__ == "__main__":
    data_files = {
        "Lubbock_Chamber" : "/Users/pagrawal140/document-insights-prototype/data/a_2.1_financial_policy_manual_lubbock_chamber_of_commerce_11.19.pdf",
        "Financial_Policies" : "/Users/pagrawal140/document-insights-prototype/data/Financial Policies (PDF).pdf",
        "Fin_Management_Policy" : "/Users/pagrawal140/document-insights-prototype/data/sample_fin_mgmt_policy.pdf",
        "NonProfit_Financial_Policies_and_Procedures" : "/Users/pagrawal140/document-insights-prototype/data/Sample-Nonprofit-Financial-Policies-and-Procedures-Manual-Resource.pdf"
    }
    final_kb = pd.DataFrame(columns = ["chunk_text", "metadata"])
    for file_name, file in data_files.items():
        if file.endswith('.pdf'):
            print(f"{file} - begins!")
            pdf_file_path = file
            output_json_file_path = f"/Users/pagrawal140/document-insights-prototype/output/extracted_data/{file_name}.json"
            extract_text_from_pdf_pages(pdf_file_path, output_json_file_path)
            chunks_df = post_processing(output_json_file_path)
            final_kb = pd.concat([final_kb, chunks_df])
        else:
            print("Input file should be a PDF file")
    final_kb.to_csv("/Users/pagrawal140/document-insights-prototype/output/chunked_kb.csv", index = False)


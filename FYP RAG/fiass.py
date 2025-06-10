import os, re, pdfplumber, faiss, pickle
import numpy as np
from sentence_transformers import SentenceTransformer

BOOKS = {
    "Pakistan Penal Code": r"D:\FYP WEBSITE\summerization-app\RAG books\pakistan penal code_removed.pdf",
    "Code of Criminal Procedure": r"D:\FYP WEBSITE\summerization-app\RAG books\code of criminal procedure_removed.pdf",
    "Code of Civil Procedure": r"D:\FYP WEBSITE\summerization-app\RAG books\code of civil procedure_removed_removed.pdf",
    "Constitution of Pakistan": r"D:\FYP WEBSITE\summerization-app\RAG books\constitution of pakistan_removed.pdf",
    "Rules of Business": r"D:\FYP WEBSITE\summerization-app\RAG books\rules of business_removed.pdf"
}

def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        return ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    except:
        return ""

def clean_text(text):
    text = re.sub(r'Page\s*\d+\s*of\s*\d+', '', text)
    text = re.sub(r'\bPage\s*\d+\b', '', text)
    text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text)
    return text.strip()

def split_sections_by_number(text):
    pattern = r"(\d+[A-Z]?(?:\(\d+\))?)\.\s*(.*?)(?=\n\d+[A-Z]?(?:\(\d+\))?\.)"
    return [{"section_id": s[0].strip(), "content": s[1].strip()} for s in re.findall(pattern, text, re.DOTALL)]

def create_faiss_index(structured_data, model):
    corpus = [section['content'] for section in structured_data]
    corpus_embeddings = model.encode(corpus, show_progress_bar=True)
    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(corpus_embeddings))
    return index

if __name__ == "__main__":
    model = SentenceTransformer('all-MiniLM-L6-v2')

    all_data = {}
    all_indices = {}

    for name, path in BOOKS.items():
        print(f"Processing {name}...")
        text = extract_text_from_pdf(path)
        clean = clean_text(text)
        sections = split_sections_by_number(clean)
        index = create_faiss_index(sections, model)

        all_data[name] = sections
        all_indices[name] = index

    with open("data/legal_data.pkl", "wb") as f:
        pickle.dump(all_data, f)

    for name, index in all_indices.items():
        faiss.write_index(index, f"data/{name.replace(' ', '_')}_faiss.index")

    print("All data processed and saved.")

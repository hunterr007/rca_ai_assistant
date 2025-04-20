import os
import pandas as pd
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv
from uuid import uuid4
import pickle

# --- Configs ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = os.getenv("GEMINI_API_URL")

# --- Embedder ---
model = SentenceTransformer(EMBEDDING_MODEL)

# --- Load and Chunk PDFs with page numbers ---
def load_documents(directory):
    records = []
    for file in os.listdir(directory):
        if file.endswith(".pdf"):
            path = os.path.join(directory, file)
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    records.append({
                        "id": str(uuid4()),
                        "content": text,
                        "page": i + 1,
                        "file": file
                    })
    return pd.DataFrame(records)

# --- Save/load embeddings and index ---
def save_index(index, embeddings, df):
    faiss.write_index(index, "data/index.faiss")
    with open("data/embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    df.to_csv("data/documents.csv", index=False)

def load_existing_index():
    if os.path.exists("data/index.faiss") and os.path.exists("data/embeddings.pkl") and os.path.exists("data/documents.csv"):
        index = faiss.read_index("data/index.faiss")
        with open("data/embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        df = pd.read_csv("data/documents.csv")
        return index, embeddings, df
    return None, None, None

# --- Embed documents and store in FAISS ---
def build_index(df):
    embeddings = model.encode(df['content'].tolist(), show_progress_bar=True)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# --- Gemini API Call ---
def query_gemini(prompt):
    params = {"key": API_KEY}
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    response = requests.post(API_URL, params=params, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        raise Exception(f"Gemini API error {response.status_code}: {response.text}")

# --- Main Flow ---
def main():
    print("🔧 AI-Powered Root Cause Analysis Console App")

    index, embeddings, doc_df = load_existing_index()
    if index is None:
        print("🔍 Embeddings not found. Building new index...")
        doc_df = load_documents("data/manuals")
        index, embeddings = build_index(doc_df)
        save_index(index, embeddings, doc_df)
    else:
        print("✅ Loaded existing FAISS index and document embeddings.")

    while True:
        print("\nOptions:")
        print("1) Diagnose based on defect description")
        print("2) Diagnose by Asset Number")
        print("3) Diagnose by Work Order Number")
        print("Type 'exit' to quit.")

        choice = input("Choose an option (1/2/3): ").strip()

        if choice.lower() == 'exit':
            break

        asset = ""
        wonum = ""
        issue = ""

        if choice == "1":
            issue = input("Describe the defect: ").strip()
        elif choice == "2":
            asset = input("Enter Asset Number: ").strip()
            asset_df = pd.read_csv("data/asset_master.csv")
            asset_info = asset_df[asset_df['assetnum'] == asset].iloc[0]
            issue = f"Diagnose issue for Asset: {asset_info['description']} | Type: {asset_info['type']} | Model: {asset_info['model']} | Manufacturer: {asset_info['manufacturer']}"
        elif choice == "3":
            wonum = input("Enter Work Order Number: ").strip()
            wo_df = pd.read_csv("data/wo_master.csv")
            wo_info = wo_df[wo_df['wonum'] == wonum].iloc[0]
            issue = f"Diagnose issue from WO Description: {wo_info['description']} | Details: {wo_info['longdescription']} | Asset Info: {wo_info['asset_desc']} | Model: {wo_info['model']}"
        else:
            print("❌ Invalid choice. Try again.")
            continue

        query_embed = model.encode([issue])
        _, indices = index.search(query_embed, TOP_K)

        relevant_docs = doc_df.iloc[indices[0]]
        context = "\n\n".join([
            f"| File | Page | Content |\n|------|------|---------|\n| {row['file']} | {row['page']} | {row['content'][:300].replace('\n', ' ')}... |"
            for _, row in relevant_docs.iterrows()
        ])

        prompt = f"""
        You are a root cause analysis expert.
        Based on the documents provided below, identify possible root causes for the issue.

        Issue:
        {issue}

        Documents Table:
        {context}

        Summarize the most likely root causes based only on the table above. Mention supporting evidence and reference file names and page numbers.
        """

        try:
            answer = query_gemini(prompt)
            print("\n📘 Suggested Root Cause(s):\n")
            print(answer)
            print("\n---\n")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

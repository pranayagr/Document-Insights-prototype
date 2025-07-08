import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.llm_call import get_embeddings

def load_vectorized_kb(csv_path: str):
    df = pd.read_csv(csv_path)
    df['embedding'] = df['embedding'].apply(lambda x: np.array(json.loads(x)))
    return df

def embed_questions(questions: list):
    embeddings = []
    for q in questions:
        embeddings.append(get_embeddings(q)['data'][0]['embedding'])
    return embeddings

def retrieve_answers(user_questions: list, kb_df: pd.DataFrame, top_k=None, top_p=0.9):
    q_embeddings = embed_questions(user_questions)
    results = []

    for q_idx, q_embed in enumerate(q_embeddings):
        similarities = cosine_similarity([q_embed], list(kb_df['embedding'].values))[0]
        sorted_indices = similarities.argsort()[::-1]
        sorted_sims = similarities[sorted_indices]

        if top_k is not None:
            selected_indices = sorted_indices[:top_k]
        else:
            cumulative = 0.0
            selected_indices = []
            total = sum(sorted_sims)
            for i, score in enumerate(sorted_sims):
                cumulative += score
                selected_indices.append(sorted_indices[i])
                if cumulative >= top_p * total:
                    break

        answers = []
        for i in selected_indices:
            row = kb_df.iloc[i]
            metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
            answers.append({
                "score": float(similarities[i]),
                "context": row['chunk_text'],
                "source": metadata['source'],
                "section": metadata['topic']
            })

        results.append({
            "question": user_questions[q_idx],
            "retrieved_context": answers
        })

    return results

def save_results(results, output_path="/Users/pagrawal140/document-insights-prototype/output/query_results.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    questions = [
        "What is the procedure for financial approval outlined in the policies?",
        "How does the organization handle conflicts of interest in financial decisions?",
        "What are the guidelines for budgeting and budget revisions?",
        "Describe the process and requirements for employee expense reimbursement",
        "What internal controls are established for auditing and financial oversight?"
    ]

    kb_csv_path = "/Users/pagrawal140/document-insights-prototype/output/vectorized_kb.csv"
    kb_df = load_vectorized_kb(kb_csv_path)

    result = retrieve_answers(questions, kb_df, top_k=None, top_p=0.8)
    save_results(result)
    print(f"Retrieval complete. Results saved to output/query_results.json")

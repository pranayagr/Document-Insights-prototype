import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.llm_call import get_embeddings

questions = [
        "What is the procedure for financial approval outlined in the policies?",
        "How does the organization handle conflicts of interest in financial decisions?",
        "What are the guidelines for budgeting and budget revisions?",
        "Describe the process and requirements for employee expense reimbursement",
        "What internal controls are established for auditing and financial oversight?"
    ]

for q in questions:
    q_embedding = get_embeddings(q)['data'][0]['embedding']
    kb_df = pd.read_csv("/Users/pagrawal140/document-insights-prototype/output/vectorized_kb.csv")
    kb_df['embedding'] = kb_df['embedding'].apply(lambda x: np.array(json.loads(x)))
    sims = cosine_similarity([q_embedding], list(kb_df['embedding'].values))[0]
    plt.plot(sorted(sims, reverse=True))
    plt.title("Similarity score distribution")
    plt.xlabel("Chunk rank")
    plt.ylabel("Cosine similarity")
    plt.grid(True)
    plt.show()

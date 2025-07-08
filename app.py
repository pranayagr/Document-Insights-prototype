import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import dotenv
import logging

from litellm import embedding
from src.answer_generation import generate_answer, build_prompt
from src.retrieval import load_vectorized_kb
from sklearn.metrics.pairwise import cosine_similarity

dotenv.load_dotenv(override = True)
logger = logging.getLogger(__name__)

api_key = os.getenv("OPENAI_API_KEY")
base = os.getenv("OPENAI_API_BASE")

def get_embeddings(text):
    model_name = "azure.text-embedding-3-large" 
    
    model_kwargs = {"dimensions": 1024}
    
    response = embedding( 
        input=text, 
        model=f"openai/{model_name}",
        api_key=api_key,
        api_base=base,
        **model_kwargs
    )
    
    return response

@st.cache_data
def load_kb():
    df = pd.read_csv("/Users/pagrawal140/document-insights-prototype/output/vectorized_kb.csv")
    return df

def retrieve_answers(user_question: list, kb_df: pd.DataFrame, top_k=None, top_p=0.9):
    q_embeddings = get_embeddings(user_question)['data'][0]['embedding']

    similarities = cosine_similarity([q_embeddings], list(kb_df['embedding'].values))[0]
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

    return kb_df.iloc[selected_indices]

if __name__ == "__main__":
    # Streamlit UI
    st.set_page_config(page_title="Document QA Assistant", layout="wide")
    st.title("Document Insights Prototype")

    kb_df = load_kb()

    question = st.text_input("Go ahead with you query:", placeholder="e.g. What is the procedure for financial approval?")

    if st.button("Get Answer") and question:
        with st.spinner("Retrieving context and generating answer..."):
            top_contexts = retrieve_answers(question, kb_df)
            prompt = build_prompt(question, top_contexts.to_dict(orient="records"))
            answer = generate_answer(prompt)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Show retrieved context"):
            for _, row in top_contexts.iterrows():
                st.markdown(f"**{row.get('source', '')} > {row.get('section', '')}** (Page {int(row.get('page', -1))})")
                st.markdown(row['chunk'])
                st.markdown("---")

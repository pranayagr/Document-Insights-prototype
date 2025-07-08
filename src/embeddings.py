import os
import pandas as pd
import numpy as np
import dotenv
import logging
from litellm import embedding

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

if __name__ == "__main__":
    chunks_df = pd.read_csv("/Users/pagrawal140/document-insights-prototype/output/chunked_kb.csv")

    chunks_df['embedding'] = chunks_df['chunk_text'].apply(lambda x : get_embeddings(x)['data'][0]['embedding'])

    embeddings = np.array(chunks_df['embedding'].tolist())
    print(embeddings.shape)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    norm_embeddings = embeddings/np.linalg.norm(embeddings, axis = 1, keepdims=True)
    chunks_df['norm_embedding'] = norm_embeddings.tolist()

    chunks_df.to_csv("/Users/pagrawal140/document-insights-prototype/output/vectorized_kb.csv", index = False)
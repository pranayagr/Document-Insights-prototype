import os
import dotenv
import logging

from litellm import completion, embedding

dotenv.load_dotenv(override = True)
logger = logging.getLogger(__name__)

api_key = os.getenv("OPENAI_API_KEY")
base = os.getenv("OPENAI_API_BASE")
model = os.getenv("MODEL")
max_tokens = os.getenv("MAX_TOKENS")
temperature = os.getenv("TEMPERATURE")

def get_chat_completion(chat_history, context=""):
    logger.info(f"chat_history: {chat_history}")
    model_kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    response = completion(
        model=f"openai/{model}",
        messages=chat_history,
        base_url=base,
        api_key=api_key,
        **model_kwargs
    )
    return response

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

# if __name__ == "__main__":
#     print(get_chat_completion([{"role" : "user", "content" : "Hi"}]))

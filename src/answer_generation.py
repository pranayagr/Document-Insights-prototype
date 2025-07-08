import os
import sys
import json
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.llm_call import get_chat_completion

def build_prompt(question: str, contexts: List[Dict]) -> str:
    context_blocks = "\n\n".join([
        f"[From {ctx.get('source', '')} > {ctx.get('section', '')} (Page {ctx.get('page', '?')}):]\n{ctx['context']}"
        for ctx in contexts
    ])
    return (
        f"You are a helpful assistant designed to answer policy-related questions using only the provided context.\n"
        f"Answer clearly, concisely, and accurately based on the sources.\n"
        f"If multiple answers seem to contradict, mention that.\n"
        f"\n\nContext:\n{context_blocks}\n\nQuestion: {question}\nAnswer:"
    )

def generate_answer(prompt: str) -> str:
    try:
        response = get_chat_completion(
            [
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0]['message']['content'].strip()
    except Exception as e:
        return f"Error generating answer: {e}"

def generate_answers_from_retrieval(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        retrieval_data = json.load(f)

    results = []
    for item in retrieval_data:
        question = item['question']
        contexts = item['retrieved_context']
        prompt = build_prompt(question, contexts)
        answer = generate_answer(prompt)

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Generated answers saved to {output_path}")

if __name__ == "__main__":
    input_file = "/Users/pagrawal140/document-insights-prototype/output/query_results.json"
    output_file = "/Users/pagrawal140/document-insights-prototype/output/generated_answers.json"
    generate_answers_from_retrieval(input_file, output_file)

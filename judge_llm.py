import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.llm_call import get_chat_completion

def call_llm_judge(question, answer, context):
    prompt = f"""
                You are an expert evaluator for an AI question-answering system. Your task is to rate the answer on 4 criteria:
                1. Accuracy: Is the answer factually correct according to the provided context?
                2. Completeness: Does it fully answer the question, covering all major aspects?
                3. Bias: Is the answer overly dependent on only one source or viewpoint?
                4. Hallucination: Does it include any information not present in the context?

                Provide scores from 1 (poor) to 5 (excellent) for each criterion, and also a brief comment.

                ---

                Question: {question}

                Answer:
                {answer}

                Context:
                {context}

                Please output your evaluation as strict JSON like:
                {{
                "accuracy": 1–5,
                "completeness": 1–5,
                "bias": 1–5,
                "hallucination": 1–5,
                "comment": "your summary"
                }}
            """.strip()

    try:
        response = get_chat_completion(
            [
                {"role": "system", "content": "You are an expert QA system evaluator."},
                {"role": "user", "content": prompt}
            ]
        )
        output = response['choices'][0]['message']['content'].strip()
        return json.loads(output)
    except Exception as e:
        print("Error:", e)
        return {"error": str(e)}

def format_context_snippets(context_list):
    return "\n---\n".join(
        f"[{c.get('source', '')} > {c.get('section', '')} (Page {c.get('page', -1)})]\n{c['context']}"
        for c in context_list
    )

def main():
    input_path = "/Users/pagrawal140/document-insights-prototype/output/generated_answers.json"
    output_path = "/Users/pagrawal140/document-insights-prototype/validation/judge_llm_scores.json"

    with open(input_path, "r") as f:
        data = json.load(f)

    results = []
    for item in data:
        q = item["question"]
        a = item["answer"]
        c = format_context_snippets(item.get("contexts", []))

        print(f"Evaluating: {q[:60]}...")
        eval_result = call_llm_judge(q, a, c)
        results.append({
            "question": q,
            "answer": a,
            "evaluation": eval_result
        })

        time.sleep(2)  # to avoid rate limits

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done! Evaluation saved to: {output_path}")

if __name__ == "__main__":
    main()

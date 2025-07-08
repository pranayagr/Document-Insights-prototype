# Document Insights Prototype

This project is a RAG-based intelligent assistant designed to help internal employees of a financial organization ask questions about internal policy documents and receive accurate, contextual answers.

Built as part of a take-home assignment, the prototype demonstrates the complete GenAI pipeline: document ingestion, preprocessing, vector embedding, semantic retrieval, LLM-based answer generation, and validation of retrieval & generation process.

## Objective

To design and build an AI assistant that:
- Ingests unstructured policy documents (PDFs, DOCX)
- Converts them into a semantically searchable knowledge base
- Answers user questions using document-grounded information
- Validates system performance without gold-labeled ground truth

## Folder Structure

document-insights-prototype/
├── data/                      # Input financial policy PDFs
├── output/                   # System outputs
│   ├── extracted_data/       # Chunked and cleaned text (JSON/CSV)
│   ├── chunked_kb.csv        # Cleaned text + metadata
│   ├── vectorized_kb.csv     # Embedded chunks
│   ├── query_results.json    # Retrieved context per question
│   ├── generated_answers.json# Final LLM-generated answers
├── src/                      # Core pipeline scripts
│   ├── ingestion.py          # PDF parsing and raw text extraction
│   ├── preprocessing.py      # Cleaning, chunking, metadata attachment
│   ├── embeddings.py         # Indexing & vectorizing the chunks
│   ├── retrieval.py          # Semantic retrieval based on user_query
│   ├── answer_generation.py  # Prompt building and final answer generation
│   └── utils/llm_call.py     # Contains Wrapper for LLM/embedding APIs
├── validation/               # Evaluation utilities
│   ├── retrieval_eval.py     # Plot cosine similarity distributions
│   ├── judge_llm.py          # LLM-based rubric scoring
│   └── judge_llm_scores.json # Judgments for completeness, accuracy, etc.
├── app.py                    # Streamlit app (UI entrypoint)
└── README.md                 # This file

## Strategy Summary

1. Ingestion  
   Used PyMuPDF to extract high-quality text per page from each document.

2. Chunking  
   Hybrid strategy combining section header detection with sentence-level splitting (~100–150 words), 20-word overlaps, and full metadata tagging.

3. Embedding & Indexing  
   Used `all-MiniLM-L6-v2` from `sentence-transformers` and stored results in `vectorized_kb.csv`.

4. Retrieval  
   Implemented Top-p (nucleus) retrieval, allowing dynamic context size based on cumulative similarity mass.

5. Answer Generation  
   Constructed structured prompts with labeled context sections, then passed to OpenAI's `gpt-3.5-turbo` for answer generation.

6. Validation  
   - Retrieval quality visualized with cosine similarity plots.
   - Judged answer quality using GPT-4 on four rubrics: completeness, accuracy, hallucination, and bias.

## Sample Workflow

1. Place PDFs in `data/`

2. Run ingestion + preprocessing:
   ```
   python src/preprocessing.py
   ```

3. Generate embeddings:
   ```
   python src/embeddings.py
   ```
4. Retrieve answers:
   ```
   python src/retrieval.py
   ```
5. Generate final LLM responses:
   ```
   python src/answer_generation.py
   ```
6. Evaluate using LLM:
   ```
   python validation/judge_llm.py
   ```

## Implementation Challenges

1. Forming coherent structured data for effective answer generation
   Solved via chunking with overlap and section & source document based metadata design

2. Context overload  
   Switched from Top-k to Top-p nucleus sampling to prevent inclusion of low-signal chunks.

3. No gold truth for evaluation  
   Created rubric-based LLM judge and visual retrieval diagnostics.

## License

This project was created as part of a take-home GenAI assessment for American Express. Do not reuse without permission.

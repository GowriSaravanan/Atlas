# Production Models Benchmark

Generated: 2026-07-20T19:27:45.930886+00:00

## Components (all real)

- **embedder**: `SentenceTransformerEmbedder` (BAAI/bge-base-en-v1.5)
- **reranker**: `CrossEncoderReranker` (BAAI/bge-reranker-base)
- **llm**: `OpenRouterProviderLLM` (meta-llama/llama-3.1-8b-instruct)
- **citation_formatter**: `EvidenceCitationFormatter` (—)

## Timings (ms)

- ingestion_ms: 9019.5
- retrieval_pipeline_ms: 15466.67
- original_analyze_ms: 0.21
- rerank_ms: 5659.54
- dense_ms: 353.76
- sparse_ms: 0.77
- fusion_ms: 0.82
- answer_generation_ms: 4320.38
- citation_formatting_ms: 0.47

## Example Output

**Query:** How many sick leave days are allowed per year?
**Answer:** Based on the provided evidence, I can answer the question as follows:

The evidence states that employees may take up to 10 sick leave days per year with manager approval. There is no information in the evidence that contradicts or modifies this statement, so the answer is:

10 sick leave days per year.
**Citations:** 2
**Used chunks:** ['53a59edb-234f-42e4-9b91-16c108cd435d', '92f07c05-cfc2-467f-85b1-7ac705dc7032']
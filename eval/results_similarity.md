# Evaluation results (similarity)

| Metric | Value |
|---|---|
| overall_accuracy | 1.0 |
| answerable_accuracy | 8/8 |
| correct_refusals | 2/2 |
| false_refusals | 0 |
| citation_accuracy | 8/8 |
| retrieval_hit_at_k | 8/8 |
| avg_latency_ms | 24563 |
| max_latency_ms | 106520 |
| errors | 0 |

Config: `{"llm_model": "nvidia/nemotron-3.5-lightning:free", "fallbacks": ["google/gemma-4-31b-it:free", "poolside/laguna-s-2.1:free"], "search_type": "similarity", "rerank": false, "top_k": 4, "chunk_size": 800, "chunk_overlap": 120}`

| id | correct | refused | citation | retrieval hit | latency (ms) | answer |
|---|---|---|---|---|---|---|
| q01 | True | False | True | True | 106520 | The guardrail is mandatory on all open sides and ends of every scaffold platform where the platform is 2 metres or more above the lower level. [scaffolding_work |
| q02 | True | False | True | True | 52988 | Within 24 hours of an incident, a worker must take the following actions:  - Report the incident verbally to their supervisor straight away, then complete the I |
| q03 | True | False | True | True | 22296 | For concrete cutting, the required PPE (in addition to the site minimum of helmet, high-visibility vest, and S3 safety footwear) is:  - Safety glasses plus full |
| q04 | True | False | True | True | 15631 | Hearing protection is mandatory when noise exceeds 85 dB(A). [ppe_rules.pdf, p.2]  - If you have to raise your voice to be understood by someone 1 metre away, t |
| q05 | True | False | True | True | 17162 | A scaffold must be inspected at least every 7 days while in use. [scaffolding_work_at_height.pdf, p.2] |
| q06 | True | False | True | True | 7061 | A worker suspended in a harness after a fall must be rescued within **15 minutes** because of the risk of suspension trauma. [scaffolding_work_at_height.pdf, p. |
| q07 | True | False | True | True | 7405 | According to the safety documents, excavations deeper than 1.2 metres must be shored, sloped or protected by a trench box before anyone enters. [site_safety_reg |
| q08 | True | False | True | True | 5751 | 0.6 litres per minute [equipment_manual_cutoff_saw.txt, p.1] |
| q09 | True | True | None | None | 5759 | I could not find this in the safety documents. |
| q10 | True | True | None | None | 5062 | I could not find this in the safety documents. |

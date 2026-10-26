# Evaluation results (bm25_offline)

| Metric | Value |
|---|---|
| overall_accuracy | 1.0 |
| answerable_accuracy | 8/8 |
| correct_refusals | 2/2 |
| false_refusals | 0 |
| citation_accuracy | 8/8 |
| retrieval_hit_at_k | 8/8 |
| avg_latency_ms | 9272 |
| max_latency_ms | 19291 |
| errors | 0 |

Config: `{"llm_model": "nvidia/nemotron-3.5-lightning:free", "fallbacks": ["google/gemma-4-31b-it:free", "poolside/laguna-s-2.1:free"], "search_type": "bm25 (offline run)", "rerank": false, "top_k": 4, "chunk_size": 800, "chunk_overlap": 120}`

| id | correct | refused | citation | retrieval hit | latency (ms) | answer |
|---|---|---|---|---|---|---|
| q01 | True | False | True | True | 13630 | According to the scaffolding document, guardrails are mandatory on scaffold platforms that are **2 metres or more** above the lower level.  - Guardrails (top ra |
| q02 | True | False | True | True | 19291 | Within 24 hours of an incident, a worker must complete the following actions ([incident_reporting_procedure.md, p.1]):  - Report the incident verbally to the su |
| q03 | True | False | True | True | 16128 | For concrete cutting, the required PPE (in addition to the minimum site PPE of helmet, high-visibility vest, and S3 safety footwear) is:  - Safety glasses and f |
| q04 | True | False | True | True | 5947 | Hearing protection is mandatory when the noise level exceeds 85 dB(A). [ppe_rules.pdf, p.2]  - If you have to raise your voice to be understood by someone 1 met |
| q05 | True | False | True | True | 6568 | A scaffold must be inspected at least every 7 days while it is in use. [scaffolding_work_at_height.pdf, p.2]  Additional inspection requirements from the same s |
| q06 | True | False | True | True | 9854 | Within 15 minutes because of the risk of suspension trauma. [scaffolding_work_at_height.pdf, p.3] |
| q07 | True | False | True | True | 5296 | Excavations deeper than 1.2 metres must be shored, sloped or protected by a trench box before anyone enters. [site_safety_regulations.pdf, p.3] |
| q08 | True | False | True | True | 5598 | The minimum water flow needed for wet cutting with the CS-350 cut-off saw is **0.6 litres per minute**.   [equipment_manual_cutoff_saw.txt, p.1] |
| q09 | True | True | None | None | 5674 | I could not find this in the safety documents. |
| q10 | True | True | None | None | 4736 | I could not find this in the safety documents. |

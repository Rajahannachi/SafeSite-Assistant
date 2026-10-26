# Evaluation results (hybrid)

| Metric | Value |
|---|---|
| overall_accuracy | 1.0 |
| answerable_accuracy | 8/8 |
| correct_refusals | 2/2 |
| false_refusals | 0 |
| citation_accuracy | 8/8 |
| retrieval_hit_at_k | 8/8 |
| avg_latency_ms | 13352 |
| max_latency_ms | 31489 |
| errors | 0 |

Config: `{"llm_model": "nvidia/nemotron-3.5-lightning:free", "fallbacks": ["google/gemma-4-31b-it:free", "poolside/laguna-s-2.1:free"], "search_type": "hybrid", "rerank": false, "top_k": 4, "chunk_size": 800, "chunk_overlap": 120}`

| id | correct | refused | citation | retrieval hit | latency (ms) | answer |
|---|---|---|---|---|---|---|
| q01 | True | False | True | True | 13041 | Guardrails become mandatory on any scaffold platform that is **2 metres or more** above the lower level. Below 2 metres, guardrails are recommended but not mand |
| q02 | True | False | True | True | 17783 | Within 24 hours of any incident, a worker must complete the following actions [incident_reporting_procedure.md, p.1]:  - Report the incident verbally to the sup |
| q03 | True | False | True | True | 16678 | For concrete cutting, the required PPE consists of the site minimum PPE plus the following task-specific items:  - Safety glasses and a full face shield [ppe_ru |
| q04 | True | False | True | True | 11243 | Hearing protection is mandatory when noise exceeds 85 dB(A). [ppe_rules.pdf, p.2]   - If you have to raise your voice to be understood by someone 1 metre away,  |
| q05 | True | False | True | True | 10295 | The scaffold must be inspected at least every 7 days while in use. [scaffolding_work_at_height.pdf, p.2]  Additional inspection triggers (from the same source): |
| q06 | True | False | True | True | 31489 | Within 15 minutes [scaffolding_work_at_height.pdf, p.3]. |
| q07 | True | False | True | True | 7962 | Excavations deeper than 1.2 metres must be shored, sloped or protected by a trench box before anyone enters. [site_safety_regulations.pdf, p.3] |
| q08 | True | False | True | True | 14527 | The minimum water flow required for wet cutting with the CS-350 cut-off saw is 0.6 litres per minute.    [equipment_manual_cutoff_saw.txt, p.1] |
| q09 | True | True | None | None | 4058 | I could not find this in the safety documents. |
| q10 | True | True | None | None | 6444 | I could not find this in the safety documents. |

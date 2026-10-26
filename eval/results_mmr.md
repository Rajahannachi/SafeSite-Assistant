# Evaluation results (mmr)

| Metric | Value |
|---|---|
| overall_accuracy | 1.0 |
| answerable_accuracy | 8/8 |
| correct_refusals | 2/2 |
| false_refusals | 0 |
| citation_accuracy | 8/8 |
| retrieval_hit_at_k | 8/8 |
| avg_latency_ms | 17093 |
| max_latency_ms | 62142 |
| errors | 0 |

Config: `{"llm_model": "nvidia/nemotron-3.5-lightning:free", "fallbacks": ["google/gemma-4-31b-it:free", "poolside/laguna-s-2.1:free"], "search_type": "mmr", "rerank": false, "top_k": 4, "chunk_size": 800, "chunk_overlap": 120}`

| id | correct | refused | citation | retrieval hit | latency (ms) | answer |
|---|---|---|---|---|---|---|
| q01 | True | False | True | True | 13694 | Guardrails are mandatory on scaffold platforms that are **2 metres or more** above the lower level. Below 2 metres, guardrails are recommended but not mandatory |
| q02 | True | False | True | True | 62142 | Within the first 24 hours after an incident, a worker must complete the reporting and immediate safety actions outlined in the procedure.   - Report the inciden |
| q03 | True | False | True | True | 8799 | Based on [ppe_rules.pdf, p.3], the PPE required for concrete cutting (in addition to the minimum helmet, high-visibility vest, and S3 safety footwear) is:  - Sa |
| q04 | True | False | True | True | 9076 | Hearing protection is mandatory when the noise level exceeds 85 dB(A). [ppe_rules.pdf, p.2] |
| q05 | True | False | True | True | 6663 | According to the scaffold inspection requirements, a scaffold must be inspected at least every 7 days while in use. [scaffolding_work_at_height.pdf, p.2]  - Ins |
| q06 | True | False | True | True | 8481 | A worker suspended in a harness after a fall must be rescued within **15 minutes** because of the risk of suspension trauma. [scaffolding_work_at_height.pdf, p. |
| q07 | True | False | True | True | 36536 | Excavations deeper than 1.2 metres must be shored, sloped or protected by a trench box. [site_safety_regulations.pdf, p.3] |
| q08 | True | False | True | True | 5343 | The minimum water flow needed for wet cutting with the CS-350 cut-off saw is **0.6 litres per minute**.   [equipment_manual_cutoff_saw.txt, p.1] |
| q09 | True | True | None | None | 13787 | I could not find this in the safety documents. |
| q10 | True | True | None | None | 6416 | I could not find this in the safety documents. |

# Dataset storage

The generated data is intentionally separated by lifecycle:

- `raw/feeds/YYYY-MM-DD/<source_id>/`: immutable feed responses.
- `interim/feed_items.jsonl`: append-only parsed records with publication labels and provenance.
- `processed/publication_proxy_headlines.csv`: validated model-ready snapshot.
- `reports/dataset_report.json`: counts, missingness and rejection reasons for the snapshot.

Raw and generated records are ignored by Git because repeated feed polling will grow them quickly. The registry and collection code are versioned, which makes each snapshot reproducible when the raw responses are retained separately.


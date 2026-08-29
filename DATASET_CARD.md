# Dataset card: publication-proxy headlines

## Status

Politics-only milestone snapshot built on **2026-08-27**. It contains **1,033 accepted records from 17 publications**, with at least three configured publications per bias class. Politics-specific feeds and publisher RSS archive pages are used where the publisher provides working endpoints.

| Weak outlet label | Records in pilot |
|---|---:|
| Left | 226 |
| Lean Left | 164 |
| Center | 153 |
| Lean Right | 347 |
| Right | 143 |

This initial snapshot is intentionally not balanced. Feed depths vary substantially; balancing a one-time poll would discard data and would not solve temporal or topical confounding. Repeated collection should accumulate an archive first, after which training snapshots can be capped by `class × outlet × month`.

## Label definition

`weak_label` is inherited from the publication's AllSides rating snapshot. Every output record has:

```text
label_scope = outlet
label_provider = AllSides
```

There is no human headline annotation and no claim that the label describes the individual headline. The valid modelling claim is that a system predicts the AllSides category associated with a publication from its headline text.

## Sources and holdouts

The source registry is [config/outlets.csv](config/outlets.csv). Five publications—one per class—are marked `holdout=true` for later unseen-publication evaluation. Do not train on those records.

Rating pages, rating scope, rating check date, feed endpoint and source domain are versioned in the registry and copied into each collected record. AllSides ratings are available for attributed noncommercial research under its current terms; commercial use requires permission or a licence.

## Fields

The processed CSV includes publication identity, headline, feed summary, blank/reserved subheadline fields, canonical URL, publication and observation timestamps, rating provenance, exact-normalization hashes, inherited label, holdout status, political-relevance reason and cross-publication exact-headline count.

`feed_summary` is not automatically treated as a subheadline. It can be an abstract, teaser, video description or generic feed text. Processed values are capped at 1,000 characters so feeds containing a full article body cannot dominate the model. A later experiment may compare headline-only features with headline-plus-summary features.

## Pilot quality report

- 1,033 politics-relevant records accepted.
- 990 non-political records excluded from the processed snapshot but preserved upstream.
- 32 rejected because their headlines contained fewer than three tokens.
- 48 duplicate story versions and 26 off-domain URLs excluded.
- 5 accepted records lacked a parseable publisher timestamp but retain acquisition time.
- 1,018 records contained a non-empty feed summary.
- 22 accepted records share an exact normalized headline with another publication and are marked for grouped deduplication before splitting.
- 496 records belong to the five publication-level holdouts.
- A repeat collection of four feeds added 0 records, confirming record-level idempotency.

Machine-readable details are written to `data/reports/dataset_report.json` whenever the processed snapshot is rebuilt.

## Known limitations

- This is publication classification through headline language/content, not independently established headline bias.
- A model may learn publication fingerprints, recurring authors/topics, formatting conventions and story-selection patterns.
- The politics-only archive has passed its 1,000-record milestone but remains class-imbalanced. It must not be used for a final performance claim.
- Feed content is not equivalent across publishers: feed depth, section coverage and summary policy differ.
- Political relevance is a deterministic heuristic. It intentionally prioritizes precision, but borderline political items can still be included or missed.
- The current pilot does not yet identify near-duplicate or syndicated stories; it only provides exact hashes.
- Opinion/news separation still requires per-source URL/section auditing even though the registry requests online-written-news scope.

## Next release gate

Accumulate at least three months before freezing the first serious training snapshot. Before training:

1. audit a random sample from each publication for feed scope and summary meaning;
2. add near-duplicate and syndicated-story clustering;
3. cap publication contribution within each class rather than globally downsampling a class;
4. freeze a future time window in addition to the five unseen-publication holdouts;
5. report random-row results only as a diagnostic, never as the primary score.

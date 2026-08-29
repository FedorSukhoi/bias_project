# Model card: first interpretable publication-proxy baseline

## Status

Pilot model trained and evaluated on **2026-08-27**. This is an interpretable research baseline,
not a production model and not a validated detector of bias in an individual headline.

## Intended prediction

Given a political headline, predict the five-way AllSides label inherited from its publication:

```text
Left | Lean Left | Center | Lean Right | Right
```

The model receives the headline only. It does not receive publication identity, URL, author,
section, RSS summary or other metadata. The target remains a weak outlet label.

## Model

- Word-unigram TF-IDF selected over word unigram/bigram TF-IDF on temporal validation macro-F1.
- Lowercasing and Unicode accent normalization.
- No stopword removal, stemming or lemmatization.
- `min_df=2`, `max_df=0.98`, sublinear term frequency and L2 normalization.
- Balanced multinomial Logistic Regression with `C=1.0` and seed 42.

Logistic Regression was chosen for the first iteration because every vocabulary feature has a
class-specific coefficient and each prediction can be decomposed into TF-IDF × coefficient terms.
Its probability outputs have **not** been calibrated.

## Split

- 419 older non-holdout records for training.
- 118 newer non-holdout records for temporal validation.
- 496 records from five frozen unseen publications for OOD testing.
- No outlet or exact normalized headline crosses the development/OOD boundary.
- The final model is refitted on all 537 development rows before the OOD test.

## Results

| Evaluation | Macro-F1 | Accuracy | Ordinal MAE | Within one class |
|---|---:|---:|---:|---:|
| Temporal validation, selected unigram model | 0.407 | 0.407 | 1.136 | 0.678 |
| Frozen unseen-publication test | 0.197 | 0.220 | 1.510 | 0.548 |

The large generalization gap is the primary result. It indicates that the development score relies
substantially on publication-specific vocabulary, recurring series/author terms, topic selection or
other source-associated patterns that do not transfer reliably to unseen outlets.

## Interpretation findings

Coefficient inspection surfaced ordinary political/topic terms alongside likely shortcut signals,
including recurring branded or author-like tokens. Those coefficients are associations in this
specific weakly labelled sample; they are not a general-purpose ideological lexicon and should not
be interpreted causally.

## Main limitations

- Only two or three development publications represent each class.
- Archive depth and dates differ substantially by publication.
- The weak label describes the outlet, not the individual headline.
- Topic/story choice and headline framing are not separated.
- Exact duplicates are isolated, but near-duplicate and event clusters are not yet available.
- The OOD set has very uneven support because entire natural publication archives are retained.
- Probabilities are uncalibrated and must not be presented as trustworthy confidence estimates.

## Next model-development step

Using the same frozen development split, compare character TF-IDF, LinearSVC and ComplementNB;
add named-entity/source-token masking as a shortcut diagnostic. Do not tune against the already
observed OOD test. A genuinely new future/outlet test will be required after further selection.

# Inclusion Criteria

Rules for deciding whether a candidate paper should be added to the reviews. When in doubt, err on the side of **defer** rather than **accept** — deferrals are cheap, rushed accepts are expensive to correct later.

## Global Criteria (apply to both reviews)

A paper is **in scope** if it meets all of these:

1. **Methodological contribution.** It proposes, analyses, or validates a method — not just applies an existing one to a new dataset.
2. **Empirical or theoretical substance.** It includes either real-data validation on a nontrivial dataset, or a formal result (theorem, identification condition, convergence rate). Thought pieces without either are out of scope.
3. **Relevant domain.** Online experimentation, advertising measurement, causal inference for decision-making, or a closely adjacent field (e.g., epidemiology methods that transfer to ads).
4. **Peer-reviewed, preprint on a reputable server, or published by a credible industry lab.** arXiv with a serious author list counts; blog posts from major labs count (Meta, Google, Netflix, Amazon, Microsoft) if they present a method with validation.

A paper is **out of scope** if any of these apply:

1. **Descriptive only.** Surveys that don't take a position are fine for the reading list but usually not worth a full entry.
2. **Domain mismatch.** Pure healthcare RCT analysis with no generalisable methodology. Pure financial econometrics without treatment-effect framing.
3. **Duplicate method.** Same idea under a different name — include the canonical citation, not every rebranding.
4. **Preliminary or unvalidated.** Workshop papers with only simulated validation where the claim is practical feasibility.

## Review-Specific Criteria

### MDE Reduction Review

A paper belongs here if it addresses **one of these problems**:

- **Reducing the minimum detectable effect** of an experiment at a fixed sample size
- **Reducing the sample size** needed to detect a given effect at fixed power
- **Reducing time-to-decision** via sequential testing or early-read methods
- **Handling interference** in a way that affects variance or bias of the estimator
- **Variance estimation validity** (diagnostics, corrections, heavy-tail handling)
- **Bias correction** for selection effects within the experimentation system (e.g., winner's curse)

It does **not** belong here if:

- The focus is on measurement *without* an experiment (that's the Prediction review)
- It's about CATE estimation for targeting, not ATE estimation for measurement
- It's about experimental design for system testing rather than causal effect estimation

### Prediction of Intervention Performance Review

A paper belongs here if it addresses **one of these problems**:

- **Estimating causal effects from observational data** (with or without experimental calibration)
- **Extending a single RCT's results** to new populations, time periods, or campaigns (transportability, generalisation)
- **Combining experimental and observational data** (data fusion, shrinkage, experimental grounding)
- **Predicting effects of new interventions** from a corpus of past experiments (PIE, meta-learning)
- **Measuring ad incrementality at scale** (attribution, MMM calibration)
- **Using short-term proxies for long-term effects** (surrogates)

It does **not** belong here if:

- The focus is on making a single experiment more sensitive (that's the MDE Reduction review)
- It's about design choices within one experiment (e.g., stratification, adaptive allocation)
- It's pure uplift modeling for within-experiment targeting (border case — include only if there's a cross-experiment extension)

### Papers That Belong in Both

Some papers touch both reviews. Examples:

- **Cross-experiment learning for HTE** — relevant to both, because it reduces effective MDE for subgroups (MDE review) and enables prediction across experiments (Prediction review)
- **Calibrated surrogates** — relevant to both, because short-term proxies reduce experiment duration (MDE) and enable long-term prediction (Prediction)
- **Heavy-tail robust estimators** — MDE review for the variance reduction angle, Prediction review if applied to observational measurement

When a paper fits both, add it to both with section-appropriate framing. Do not duplicate content verbatim — each review should emphasise the angle relevant to its scope.

## Decision Criteria for Triage

When triaging a candidate, score it on these dimensions. Use the scoring to justify the disposition.

| Criterion | Accept if | Defer if | Reject if |
|-----------|-----------|----------|-----------|
| **Method novelty** | Introduces a new technique or a meaningful refinement | Incremental improvement on existing method | Essentially reprises existing method |
| **Empirical validation** | Real data, meaningful comparison, honest limitations | Validation is thin or only simulated | No validation beyond toy examples |
| **Production evidence** | Adopted by a credible org (Meta, Google, Amazon, Netflix, major advertiser) | Unclear adoption | No adoption evidence |
| **Author credibility** | Established causal inference or experimentation researcher | Known author in adjacent field | First-time authors on a highly technical claim |
| **Relevance to review scope** | Directly addresses one of the review's problems | Tangential — method could be stretched to fit | Not within the listed problems |
| **Superseded?** | Addresses a gap our current coverage misses | Overlaps with existing entry but adds refinement | Essentially duplicates existing entry |
| **Citation signal** | Cited by recent high-quality work | Too new to judge | Low citations despite age |

### Scoring Rules

- **Accept** if the paper is clearly in scope, has strong empirical or theoretical grounding, and adds something not already in the review
- **Defer** if it's plausibly relevant but one of: too new to judge, marginally in scope, overlaps significantly with existing coverage, or would require major review restructuring to include well
- **Reject** if it fails on scope or substance. **Record a specific reason** — this prevents re-evaluation next quarter

## Escalation

If a paper is difficult to classify and the automated triage is uncertain, mark it **defer** and raise it explicitly in the triage report with:

- The specific question the reviewer needs to decide
- The criterion where the paper is on the borderline
- What the implication would be of each decision

This is better than a confident wrong classification.

## Notes on Industry Papers

Blog posts, whitepapers, and tech reports from major labs are often the most practically relevant sources. Standards:

- **Include** if: describes a real method, provides empirical results, is authored by a team with domain credibility
- **Cite carefully** — blog posts disappear. If possible, cite the archived version (Wayback Machine) alongside the live URL
- **Do not over-rely** on single-source industry claims. One paper from Meta saying "this works in production" is a data point, not validation. Balance against independent work.

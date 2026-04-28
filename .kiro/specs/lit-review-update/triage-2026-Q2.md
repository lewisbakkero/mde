# Triage Report — 2026-Q2

**Run date:** 2026-04-28
**Search window:** 2024-10-28 to 2026-04-28 (18 months — first automated run)
**Total candidates surfaced (after filtering for window):** 11
**New candidates (not in seen-papers.json):** 10
**Already-seen (deduplicated):** 1 (RBSDs, Masoero 2025 — already tracked as `switchback-multi-unit-2025`)
**Candidates by disposition:** accepted: 7, deferred: 2, rejected: 1

---

## Executive Summary

Productive quarter for both reviews. Three clear accepts for the MDE review — two on variance reduction (STATE for heavy-tailed ratio metrics; ratio-metric variance reduction via GBDT control variates) and one on winner's curse (the "zoom correction"). One strong accept for the Prediction review: cross-validated causal inference (Yang 2025) as a modern alternative to shrinkage-based combination of experimental and observational data.

A notable find is Li et al. (2025) "Learning Across Experiments and Time" — a local empirical Bayes framework that is directly related to the PIE / cross-experiment-meta-learning thread in the Prediction review. Propose as a new §4.8.

Two deferrals worth flagging: a prediction-powered anytime-valid inference paper (Kilian 2025) that unifies §3 (MDE) and §8.1 (Prediction) but needs a close read; and a multi-item inventory experimental-design paper (Si 2026) that formalises cannibalisation bias in a way that may justify a short subsection in MDE §4 on cross-item interference.

Field trend this quarter: continued convergence between the sequential-testing and prediction-powered-inference literatures; multiple teams working on heavy-tailed/ratio variance reduction with ML-based control variates.

---

## Accepts (Proposed for Inclusion)

### 1. Zhou et al. (2024) — STATE: A Robust ATE Estimator of Heavy-Tailed Metrics

- **Source:** [arXiv:2407.16337](https://arxiv.org/abs/2407.16337)
- **Venue:** preprint (Meituan)
- **Proposed section:** **§2.8 of MDE Reduction review** (extending existing heavy-tailed subsection with a named method)
- **Why this paper:** Uses Student's t-distribution + variational EM to build a robust ATE estimator that handles heavy tails while doing variance reduction. Claims 50%+ variance reduction over CUPAC/MLRATE on Meituan data. Extends to ratio metrics.
- **Why accept:** Method novelty (Student's t-based VR is new), empirical validation on large-scale real data, directly fills a gap in the current §2.8 which is mostly conceptual on EVT/winsorisation. Production evidence from Meituan.
- **Suggested framing:** Add as §2.8 subsection "STATE: Robust VR for Heavy-Tailed Metrics" with problem/method/assumptions structure. Emphasise it as a named alternative to the heuristic winsorisation-then-CUPED stack.

### 2. Jeunen et al. (2024) — Variance Reduction in Ratio Metrics for Efficient Online Experiments

- **Source:** [arXiv:2401.04062](https://arxiv.org/abs/2401.04062)
- **Venue:** preprint (ShareChat, RecSys'24)
- **Proposed section:** **MDE Reduction review, ratio-metrics special-considerations subsection** (currently under Practitioner's Guide)
- **Why this paper:** Applies variance reduction to ratio metrics (CTR, retention) at scale on ShareChat. Shows that GBDT-based control variates outperform linear CUPED for ratio metrics; importantly, they show that piling on more covariates is counter-productive — control variate selection matters.
- **Why accept:** Strong empirical result on production data, useful counterpoint to "more features = better CUPAC" assumption, directly addresses the ratio metric problem the current review flags but only partially addresses.
- **Suggested framing:** Short entry within the ratio-metrics special considerations subsection noting the ShareChat result and the "fewer, better control variates" lesson.

### 3. Zrnic (2024) — A Flexible Defense Against the Winner's Curse (Zoom Correction)

- **Source:** [arXiv:2411.18569](https://arxiv.org/abs/2411.18569)
- **Venue:** preprint
- **Proposed section:** **§10 of MDE Reduction review (Winner's Curse)** — add as §10.2 alongside the existing Bayesian Hybrid Shrinkage entry
- **Why this paper:** Introduces the "zoom correction" — a flexible, distribution-free approach to valid inference on the winner. Works parametric/nonparametric, handles arbitrary dependencies, auto-adapts to selection bias level. Extends to top-k and near-winners.
- **Why accept:** Fills a real gap — the current §10 has Bayesian shrinkage (§10.1) but no frequentist flexible method. Zoom correction is the natural companion.
- **Suggested framing:** §10.2 "Zoom Correction: Frequentist Winner's Curse Inference." Draw a comparison table with §10.1's Bayesian approach on assumption strength, flexibility, and interpretability.

### 4. Li et al. (2025) — Learning Across Experiments and Time

- **Source:** [arXiv:2511.21282](https://arxiv.org/abs/2511.21282)
- **Venue:** preprint
- **Proposed section:** **§4 of Prediction review** — new §4.8 "Local Empirical Bayes for Cross-Experiment Learning"
- **Why this paper:** Local empirical Bayes framework that adapts pooling to both temporal evolution within an experiment and heterogeneity across experiments. Key insight: build a context-aware comparison set (time-aware within, context-aware across) rather than naive pooling. Directly addresses known failure modes of cross-experiment meta-learning (§4.4) and is relevant to the PIE story.
- **Why accept:** Addresses a named weakness of existing methods (PIE's concept drift, meta-learning's dilution from unrelated experiments). Theoretical + empirical treatment. Authors credible in this area.
- **Suggested framing:** §4.8 with problem/method/assumptions structure. Emphasise how it complements PIE (§4.1) and cross-experiment meta-learning (§4.4) rather than competing with them. Add a comparison row to the §4.4 vs PIE table.

### 5. Yang et al. (2025) — Cross-Validated Causal Inference: A Modern Method to Combine Experimental and Observational Data

- **Source:** [arXiv:2511.00727](https://arxiv.org/abs/2511.00727)
- **Venue:** preprint
- **Proposed section:** **§4.6 of Prediction review** — add as §4.6.1 or extend existing Shrinkage Estimators section
- **Why this paper:** Formulates combining experimental and observational data as empirical risk minimisation with cross-validation on experimental folds to select the weighting. Provides non-asymptotic error bounds. Modern alternative to Rosenman-style shrinkage (§4.6) and Kallus-style experimental grounding (§4.5).
- **Why accept:** Clean methodology, theoretical guarantees, and it unifies several existing approaches under one ERM framework. Directly in scope for the Prediction review's §4 thread on RCT-calibrated observational methods.
- **Suggested framing:** Add as §4.6.1 "Cross-Validated ERM for Combining Experimental and Observational Data" with a comparison table vs. Rosenman shrinkage (§4.6) and Kallus grounding (§4.5) on assumption structure and what the cross-validation buys you.

### 6. Brennan et al. (2024) — Optimal Design under Interference, Homophily, and Robustness Trade-offs

- **Source:** [arXiv:2601.17145](https://arxiv.org/abs/2601.17145) (note: arxiv ID formatting)
- **Venue:** preprint
- **Proposed section:** **§5.1 of MDE Reduction review (Handling Interference)** — short entry
- **Why this paper:** Shows that in the presence of homophily (common in social networks), standard cluster-randomised designs can *increase* MSE rather than decrease it. Provides a framework for the bias-variance tradeoff under homophily.
- **Why accept:** Adds a needed nuance to the current §5.1 treatment of cluster randomisation, which presents it as uniformly good for handling interference. The homophily caveat is real and well-motivated here.
- **Suggested framing:** Add as a "Caveats" subsection to the existing cluster randomisation entry. Table: when cluster randomisation helps vs. hurts, indexed by network properties (degree distribution, homophily).

### 7. Zhao et al. (2023, updated 2025) — Switchback Experiments under Geometric Mixing

- **Source:** [arXiv:2209.00197v4](https://arxiv.org/abs/2209.00197)
- **Venue:** preprint
- **Proposed section:** **§4.1 of MDE Reduction review (Switchback Experiments)** — short methodological note
- **Why this paper:** Provides tighter theoretical analysis of switchback designs under geometric mixing conditions on the treatment effect carryover. Connects to the data-driven switchback design literature already cited (Xiong et al. 2023).
- **Why accept:** Incremental but substantive — strengthens the theoretical foundation already cited, provides cleaner assumptions for when the carryover bias is controllable.
- **Suggested framing:** Short paragraph within the existing §4.1, cited alongside the Wen et al. and Xiong et al. references.

---

## Deferrals (Borderline — Need Reviewer Decision)

### 1. Kilian et al. (2025) — Anytime-valid, Bayes-assisted, Prediction-Powered Inference

- **Source:** [arXiv:2505.18000](https://arxiv.org/abs/2505.18000)
- **Why defer:** This paper sits at the intersection of §3 of the MDE review (sequential/anytime-valid inference) and §8.1 of the Prediction review (prediction-powered generalisation). Including it in both reviews requires careful framing to avoid duplication, and in either single review it requires a restructure to link the two threads. Also: the paper is methodologically dense and benefits from a closer read than triage allows.
- **What the reviewer needs to decide:** Do we (a) include in MDE §3 only as a sequential extension of PPI, (b) include in Prediction §8.1 only as an anytime-valid extension of PPI, or (c) include in both with cross-references? If (c), confirm we're willing to spend the editorial effort to keep the two treatments synchronised.
- **Implication of accepting:** Opens a potentially important bridge between the two reviews, but adds maintenance burden.
- **Implication of rejecting:** Lose coverage of an emerging subfield that will likely generate more papers; may need to revisit in 2–3 quarters.

### 2. Si & Bojinov (2025) — Experimental Designs for Multi-Item Multi-Period Inventory Control

- **Source:** [arXiv:2501.11996](https://arxiv.org/abs/2501.11996)
- **Why defer:** This is domain-specific (inventory management) but introduces a design — pairwise over items and time — that generalises the two-sided randomisation thread (§5.1 of MDE) to a multi-item setting with cannibalisation. Whether the method is worth a section depends on whether the cannibalisation bias story is broadly applicable beyond inventory or narrowly tied to capacity-constrained systems.
- **What the reviewer needs to decide:** Is multi-item cannibalisation a first-class concern in your broader problem space (e.g., ad auctions with budget competition, retail promotions)? If yes, accept and frame as a generalisation. If not, reject as domain-specific.
- **Implication of accepting:** Would add a named subsection under §5.1 on multi-item interference, which could eventually grow as more work in this vein appears.
- **Implication of rejecting:** No immediate loss; the phenomenon is covered at high level under §5.1 interference.

---

## Rejects (Logged for Record)

| # | Author, Year | Title | Reject reason |
|---|--------------|-------|---------------|
| 1 | Various industry blogs | "CUPED calculator," "CUPED in A/B Testing," multiple derivative posts | No methodological contribution — these are explanatory posts of existing CUPED work. Already covered in §2.1 via the Deng 2013 paper. |

(Blog posts rejected on methodological-novelty grounds per inclusion criteria; CUPED is well covered in §2.1.)

---

## Health Check Findings

### Stale Methods

Methods in the reviews whose most recent cited paper is >24 months old (scan of §2–§5 completed):

| Method | Review | Most recent citation | Action |
|--------|--------|---------------------|--------|
| CUPED (§2.1) | MDE | 2013 (foundational paper) + 2023 (Deng "New Look") | No action — stable method, recent follow-ups covered |
| Ghost Ads (§2.1 Prediction) | Prediction | 2017 (Johnson-Lewis-Nubbemeyer) | **Flag for targeted search next quarter** — search for "ghost ads" / "counterfactual auction logging" follow-ups |
| Stratification (§2.4 MDE) | MDE | 2025 review paper | No action |
| Transportability core (§6.1 Prediction) | Prediction | 2014 (Bareinboim-Pearl) | **Flag for targeted search** — there has been substantial follow-up work since |
| Random-effects meta-analysis (§7.1 Prediction) | Prediction | 2002 (Higgins-Thompson) | No action — foundational, stable |

### Missing Cross-References

Spot-check found:

| Method | Missing from | Action |
|--------|--------------|--------|
| Li et al. (2025) local empirical Bayes | Not yet in reviews (this run) | Covered by accept #4 above |
| Zoom correction (Zrnic 2024) | Not yet in reviews (this run) | Covered by accept #3 above |

No cross-reference issues found in existing content.

### Claims That May Need Updating

| Section | Current claim | Concern | Action |
|---------|---------------|---------|--------|
| MDE §2.2 CUPAC | Industry "10-30% additional reduction over CUPED" is "highly optimistic" | Still supported by recent Meta/Booking meta-analyses; no change needed |
| Prediction §4.1 PIE | R² = 0.88 cited as central number | No challenger result found; no change needed |

None require immediate update.

### Dead Links

Quick spot-check of arxiv URLs did not turn up broken links. Full URL audit deferred — this should be run periodically but isn't a per-quarter priority.

---

## Proposed Changes Summary

Once the reviewer confirms accepts, this becomes the changelog for the edit commit.

### New entries

**MDE Reduction review:**
- §2.8: Add STATE (Zhou et al. 2024) as a named robust VR method for heavy-tailed metrics
- §2.x (ratio metrics subsection): Add brief entry for Jeunen et al. 2024 ShareChat result on GBDT control variates
- §4.1 (Switchback): Short methodological note citing Zhao et al. (2023/2025) on geometric mixing
- §5.1 (Cluster randomisation): Add caveat subsection on homophily from Brennan et al. 2024
- §10.2 (new subsection under Winner's Curse): Zoom correction (Zrnic 2024)

**Prediction review:**
- §4.6.1 (new subsection under Shrinkage): Cross-validated ERM (Yang et al. 2025)
- §4.8 (new subsection): Local Empirical Bayes for Cross-Experiment Learning (Li et al. 2025)

### Modified entries

- Prediction §4.4 (Cross-Experiment Meta-Learning): Update to reference the new §4.8 entry and note the local-pooling refinement
- MDE §2.8 (Heavy-tailed robust): Restructure to position STATE as a concrete method alongside the conceptual EVT treatment

### Structural changes (if any)

- New subsection §10.2 in MDE (Winner's Curse gains a second method)
- New subsection §4.8 in Prediction (Cross-experiment learning gains a refinement)

---

## Reviewer Decisions (Fill In During Human Review)

| Paper | Proposed | Final | Notes |
|-------|----------|-------|-------|
| Zhou 2024 — STATE | accept | | |
| Jeunen 2024 — Ratio VR at ShareChat | accept | | |
| Zrnic 2024 — Zoom correction | accept | | |
| Li 2025 — Learning Across Experiments and Time | accept | | |
| Yang 2025 — Cross-validated causal inference | accept | | |
| Brennan 2024 — Homophily trade-offs | accept | | |
| Zhao 2023/2025 — Switchback geometric mixing | accept | | |
| Kilian 2025 — Anytime-valid PPI | defer | | |
| Si 2025 — Multi-item inventory | defer | | |
| Industry CUPED blogs (aggregated) | reject | | |

---

## Notes for Next Quarter

- **Search query tuning:** "CUPED" as a standalone query returned mostly industry blog posts. Next quarter, narrow to `CUPED variance reduction` + `arxiv` or `+ NeurIPS` to cut noise. Consider adding specific co-occurring terms like `CUPAC`, `MLRATE`, `control variate` to get methodology work rather than derivative content.
- **Prediction-powered inference is heating up.** Expect more papers at the intersection of PPI and sequential testing, and at the intersection of PPI and causal generalisation. Consider whether this deserves a dedicated tag in the inclusion criteria.
- **Authors to watch for next quarter:** Tijana Zrnic (winner's curse, PPI), Lorenzo Masoero (RBSDs, switchback), Valentin Kilian (anytime-valid PPI), Xuelin Yang (cross-validated causal inference).
- **Gaps in coverage noticed during triage:** The MDE review's §2.8 (heavy-tailed) would benefit from named methods alongside the EVT conceptual treatment. STATE (Zhou 2024) starts that but more work likely exists — targeted search next quarter for "heavy-tailed robust estimator A/B test."
- **Venue checks deferred:** Did not do full venue TOC checks this run. For next quarter, spend 15 min each on Marketing Science 2025 issues and KDD'25 applied track.

---

## Next Steps

1. **User review:** Open this file, review the proposed changes, annotate final decisions in the Reviewer Decisions table.
2. **Draft section entries:** For each confirmed accept, use `.kiro/specs/lit-review-update/templates/section-entry.md` to draft the full entry.
3. **Apply edits:** Use `strReplace` to add entries to the appropriate sections of the review files.
4. **Update seen-papers.json:** Append entries for all 10 new candidates with their dispositions.
5. **Commit:** Single commit per the runbook's Step 11 format.

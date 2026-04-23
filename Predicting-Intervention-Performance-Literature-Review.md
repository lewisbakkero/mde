# Literature Review (as of Jun'25): Techniques to Predict/Filter Intervention Performance Without Running an A/B Test

## Executive Summary

Randomised controlled trials (RCTs) are the gold standard for causal inference, but they are expensive, slow, and hard to scale. Lewis & Rao (2015) showed that the partial R² of advertising on sales is approximately 0.000005 — meaning ad effects are tiny relative to the noise in consumer behaviour. Detecting these effects requires massive experiments, and running one for every campaign, feature, or policy decision is economically infeasible.

This review synthesises research on methods that estimate the causal effect of an intervention **without requiring a new RCT for each decision**. The approaches range from observational methods that attempt to recover causal effects from non-experimental data (and largely fail), to surrogate-based methods that accelerate within-experiment learning, to cross-experiment prediction models that learn from a corpus of completed RCTs to predict effects for new interventions.

**The central paper is PIE** (Gordon, Moakler, Zettelmeyer 2026) — Predicted Incrementality by Experimentation — which reframes ad measurement as a campaign-level prediction problem trained on RCT outcomes. PIE achieves out-of-sample R² = 0.88 for incremental conversions per dollar, compared to R² = 0.19 for last-click attribution.

---

### Taxonomy of Methods

| Category | §  | What It Does | Requires New RCT? | Key Innovation | Scalability |
|----------|----|--------------|--------------------|----------------|-------------|
| **Observational / Quasi-Experimental** | 2 | Estimates causal effects from non-experimental data | No (but fails validation) | Propensity scores, DML, ghost ads, BSTS, synthetic control, geo experiments | High (cheap); varies in accuracy |
| **Surrogate / Proxy Outcomes** | 3 | Uses short-term metrics to predict long-term effects | Yes (but shorter/cheaper) | Surrogate index, auto-surrogates, data combination | Medium |
| **Cross-Experiment Prediction (PIE)** | 4 | Learns from completed RCTs to predict new campaigns | No (uses historical RCTs) | Post-determined features + ML, experimental grounding, shrinkage, target trial emulation | High (once trained) |
| **MMM Calibrated by Experiments** | 5 | Time-series models anchored by periodic lift tests | Periodic (not per-campaign) | Bayesian priors from RCTs | Medium-High |
| **Transportability / External Validity** | 6 | Formal frameworks for extrapolating RCT findings | No (reuses existing RCTs) | Causal DAGs, sample-to-population, data fusion | Low-Medium |
| **Meta-Analysis** | 7 | Pools results across independent experiments | No (synthesises existing) | Random-effects, hierarchical models | Medium |
| **Data Fusion / ML-Based Approaches** | 8 | Combines ML predictions with experimental calibration | No (uses ML + RCTs) | PFNs, prediction-powered generalisation, externally valid policy evaluation | Medium-High (emerging) |

### Critical Distinction: Prediction vs. Identification

Many practitioners conflate these. They serve different purposes:

| Goal | What You Need | Best Approach | What Fails |
|------|---------------|---------------|------------|
| **Causal identification** for a specific campaign | Unbiased estimate of ATT | Run an RCT | Observational methods (Gordon et al. 2019) |
| **Causal prediction** across campaigns | Accurate out-of-sample forecasts | PIE (trained on RCTs) | Last-click attribution, naive observational |
| **Directional guidance** for budget allocation | Channel-level ROAS estimates | MMM calibrated by lift tests | Uncalibrated MMM |
| **Early read** within an experiment | Faster convergence to long-term effect | Surrogate index | Waiting for full outcome maturation |
| **Generalisation** to new populations | Transport formula + covariate data | Transportability framework | Assuming effects are constant |

**Key insight:** Observational methods try to solve the identification problem without experiments and systematically fail. PIE sidesteps identification entirely — it solves a prediction problem where RCTs provide the training labels.

---

## Practitioner's Decision Guide

**How to Use This Section:** Start here if you need to choose a method for estimating intervention effects without running a new RCT. The academic sections (2–7) provide depth; this section provides actionable guidance.

### Decision Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│          PREDICTING INTERVENTION PERFORMANCE: Decision Flowchart            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  START: Can you run an RCT for this specific decision?                      │
│  │                                                                          │
│  ├─► YES, and it's affordable → RUN THE RCT. Nothing beats it.             │
│  │   └─► Want faster reads? → Add SURROGATE INDEX (§3) for early signal    │
│  │                                                                          │
│  ├─► NO — too expensive, too slow, or too many decisions                    │
│  │   │                                                                      │
│  │   ├─► Do you have 400+ historical RCTs with campaign features?           │
│  │   │   └─► YES → Implement PIE (§4.1)                                    │
│  │   │       ├─► R² > 0.8? → Use PIE for routine measurement               │
│  │   │       ├─► R² = 0.5–0.8? → Use PIE + periodic RCT validation         │
│  │   │       └─► R² < 0.5? → PIE unreliable; need more RCTs or features    │
│  │   │                                                                      │
│  │   ├─► Do you have periodic lift tests (even a few per channel)?          │
│  │   │   └─► YES → Calibrate MMM with Bayesian priors (§5)                 │
│  │   │       └─► Use for channel-level allocation, not campaign-level       │
│  │   │                                                                      │
│  │   ├─► Do you have RCTs from similar populations/settings?                │
│  │   │   └─► YES → Apply transportability framework (§6)                    │
│  │   │       └─► Requires causal graph + covariate data in target pop       │
│  │   │                                                                      │
│  │   └─► None of the above?                                                 │
│  │       └─► Ghost ads (§2.1) are the cheapest experimental option          │
│  │           ⚠️ Do NOT rely on observational methods alone (§2.2, §2.3)     │
│  │                                                                          │
│  └─► UNSURE whether RCT is needed?                                          │
│      └─► See "When to Do Nothing" below                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Quick Reference: All Methods Compared

| Method | §  | Requires New RCT? | Accuracy vs RCT | Setup Cost | Maintenance | Best For |
|--------|----|--------------------|-----------------|------------|-------------|----------|
| **Ghost Ads** | 2.1 | Yes (but cheaper) | High (is an RCT variant) | Medium | Medium | Reducing RCT cost by 10× |
| **Propensity Score Matching** | 2.2 | No | Poor (60–175% error) | Low | Low | Don't use for causal claims |
| **DML / SPSM** | 2.3 | No | Poor (large errors) | Medium | Medium | Don't use for causal claims |
| **Causal Impact (BSTS)** | 2.4 | No (quasi-experimental) | Good with clean pre-period | Medium | Medium | Single-unit geo/time-series lift |
| **Synthetic Control** | 2.5 | No (quasi-experimental) | Good with donor overlap | Medium | Medium | One treated unit, many donors |
| **Geo Experiments** | 2.6 | Yes (geo-randomised) | Good | Medium-High | Medium | Privacy-safe, cross-platform measurement |
| **Surrogate Index** | 3.1 | Yes (but shorter) | Good (~95% consistency) | Medium | Medium | Early reads on long-term effects |
| **Long-Term Data Combination** | 3.3 | Yes (short RCT) + observational | Good with proximal identification | High | Medium | Long-term effects with observational outcomes |
| **PIE** | 4.1 | No (uses historical) | High (R² = 0.88) | High | Medium | Campaign-level measurement at scale |
| **Causal ML / Uplift** | 4.3 | Yes (per campaign) | Good (within experiment) | High | High | User-level targeting within RCT |
| **Cross-Experiment Meta-Learning** | 4.4 | No (pools experiments) | Moderate | High | High | User-level personalisation |
| **Experimental Grounding** | 4.5 | Small RCT + large obs | Good when confounding transfers | High | Medium | Correcting hidden confounding |
| **Shrinkage Estimators** | 4.6 | Yes (RCT + obs) | Good for aggregate ATE | Low | Low | Combining RCT and observational ATEs |
| **Target Trial Emulation** | 4.7 | No | Only as good as unconfoundedness | Medium-High | High | Rigorous observational causal inference |
| **Bayesian MMM + Lift Tests** | 5.1 | Periodic | Moderate | High | High | Channel-level budget allocation |
| **Transportability** | 6.1 | No (reuses existing) | Depends on graph | Low | Low | Formal extrapolation |
| **Sample-to-Population** | 6.3 | No (RCT + obs covariates) | Good with measured moderators | Medium | Medium | Generalising trial to target population |
| **Data Fusion (Bareinboim-Pearl)** | 6.4 | No | Framework, not estimator | High (theory) | N/A | Theoretical foundation for data combination |
| **Meta-Analysis** | 7.1 | No (synthesises) | Average effect only | Low | Low | Summarising evidence base |
| **Prediction-Powered Generalisation** | 8.1 | Small RCT + ML | Improves efficiency of generalisation | Medium-High | Medium | Generalising RCTs with auxiliary ML |
| **In-Context Learning / PFNs** | 8.2 | No (pre-trained) | Promising but early-stage | Very High (pre-training) | Low | Research direction |
| **Externally Valid Policy Evaluation** | 8.3 | RCT + target covariate data | Good for policy values | Medium-High | Medium | Policy evaluation in target populations |

### When to Do Nothing — Just Run the RCT

**Before investing in prediction infrastructure, ask:**

| Question | If YES | If NO |
|----------|--------|-------|
| Do you make <20 campaign decisions per year? | Just run RCTs for each | Prediction methods amortise well |
| Are your campaigns highly heterogeneous (different verticals, objectives)? | PIE may struggle to generalise | PIE works well with homogeneous campaigns |
| Is your RCT infrastructure already cheap (ghost ads, automated)? | Keep running RCTs | Invest in prediction to reduce RCT burden |
| Do stakeholders require per-campaign causal proof? | RCT is the only answer | Prediction + periodic validation suffices |
| Are you in a regulated industry requiring experimental evidence? | RCT is mandatory | Prediction methods are acceptable |

**Rule of thumb:** If you can run an RCT for under $5K and get results in under 2 weeks, the RCT is almost always the right choice. Prediction methods shine when you have hundreds of decisions and RCTs cost $50K+ each or take months.

### Method Combinability Matrix

Not all methods can be combined. Some are alternatives; others are complementary layers.

| Method 1 | Method 2 | Combinable? | How They Interact |
|----------|----------|-------------|-------------------|
| PIE | Periodic RCTs | ✓ (essential) | RCTs validate and refresh PIE model |
| PIE | Surrogate Index | ✓ | Surrogates accelerate the RCTs that train PIE |
| PIE | MMM | ✓ | PIE for campaign-level; MMM for channel-level |
| PIE | Causal ML / Uplift | ✓ | PIE for measurement; uplift for targeting within campaigns |
| PIE | Transportability | ~ | Transportability provides theoretical grounding for PIE's extrapolation |
| PIE | Meta-Analysis | ✗ (PIE supersedes) | PIE is a predictive extension of meta-analysis |
| MMM | Lift Tests | ✓ (essential) | Lift tests calibrate MMM priors |
| MMM | PIE | ✓ | PIE provides campaign-level signal that informs MMM |
| Surrogate Index | Auto-Surrogates | ✗ (alternatives) | Choose based on whether you have multi-metric surrogates or same-metric early reads |
| Ghost Ads | PIE | ✓ | Ghost ads are cheap RCTs that feed PIE's training corpus |
| Ghost Ads | PSM/DML | ✗ (don't combine) | Ghost ads are experimental; PSM/DML are observational — use ghost ads |
| Observational (PSM/DML) | Any | ⚠️ | Don't rely on observational methods for causal claims; use only as directional signals |

### Practical Gotchas and Risks

**These issues are rarely discussed in papers but frequently cause problems in practice:**

| Gotcha | Why It Happens | Impact | Mitigation |
|--------|---------------|--------|------------|
| **PIE trained on biased RCT sample** | Platforms experiment on large/important campaigns, not representative ones | PIE predictions biased for small/unusual campaigns | Active learning for RCT selection; monitor feature coverage |
| **Stale MMM priors** | Lift tests run once, never refreshed | MMM drifts from reality; wrong budget allocation | Calendar-based re-calibration (quarterly minimum) |
| **Surrogate assumption violation** | Short-term metrics miss brand-building effects | Underestimates long-term treatment effects | Use multiple surrogates; validate on historical data |
| **Last-click as ground truth** | Teams compare PIE to last-click instead of RCTs | PIE looks "wrong" when it's actually more accurate | Always validate against RCTs, not attribution heuristics |
| **Concept drift in PIE** | Platform algorithm changes, user behaviour shifts | PIE predictions degrade silently | Monitor out-of-sample R² on rolling RCT window |
| **Overfitting PIE to training RCTs** | Too many features, too few RCTs | High in-sample R², poor out-of-sample | Cross-validation; regularisation; feature selection |
| **MMM identifiability** | Correlated channel spends | Individual channel effects poorly identified | Experimental variation (lift tests); spend variation |

---

## 1. Introduction

### 1.1 The Fundamental Tension: RCTs vs. Scalability

Online controlled experiments (A/B tests, RCTs) are the gold standard for measuring the causal effect of an intervention — whether that intervention is an ad campaign, a product feature, a pricing change, or a policy. The logic is simple: randomly assign units to treatment and control, measure the difference, and attribute it to the intervention.

The problem is scale. Consider a large ad platform:

| Dimension | Scale |
|-----------|-------|
| Active ad campaigns per quarter | 10,000–100,000+ |
| Campaigns that can be experimentally measured | 100–500 (resource-constrained) |
| Cost per lift test | $10K–$100K (opportunity cost + engineering) |
| Time per lift test | 2–8 weeks |
| Fraction of campaigns with RCT measurement | 1–5% |

This means **95–99% of campaign decisions are made without experimental evidence**. Advertisers and platforms rely on last-click attribution, marketing mix models, or gut instinct — all of which have known biases.

### 1.2 The Unfavourable Economics of Measuring Advertising

Lewis & Rao (2015), in "The Unfavorable Economics of Measuring the Returns to Advertising" ([Journal of the Quarterly Journal of Economics](https://doi.org/10.1093/qje/qjv023)), established the fundamental economic challenge:

**Key findings:**
- The partial R² of online display advertising on sales is approximately **0.000005**
- Ad effects are tiny relative to the natural variance in consumer purchasing behaviour
- Even with millions of users, experiments are often underpowered for economically meaningful effects
- The "long tail" of small advertisers can never justify per-campaign experimentation

**Implications for this review:**
- If individual RCTs are economically marginal even for large campaigns, per-campaign experimentation is infeasible at scale
- This motivates every method in this review: ways to estimate causal effects without bearing the full cost of a new RCT for each decision
- The methods differ in how they handle the identification problem — some try to solve it without experiments (and fail), others leverage existing experiments more efficiently

### 1.3 Scope of This Review

This review covers six categories of methods:

1. **Observational and quasi-experimental methods** (§2) — attempts to recover causal effects from non-experimental data, including ghost ads as a cost-reducing experimental variant
2. **Surrogate and proxy outcome methods** (§3) — using short-term or intermediate metrics to predict long-term causal effects
3. **Cross-experiment prediction** (§4) — learning from a corpus of completed RCTs to predict effects for new interventions (PIE and related)
4. **Marketing mix models calibrated by experiments** (§5) — structural time-series models anchored by periodic lift tests
5. **Transportability and external validity** (§6) — formal frameworks for extrapolating experimental findings to new settings
6. **Meta-analysis** (§7) — traditional and predictive approaches to synthesising evidence across studies

**How to read this document:**

For each method, we provide:
1. **Problem:** What challenge does this method address?
2. **Method:** How does it work?
3. **Assumptions:** What must hold for validity? (See Appendix B for consolidated reference)
4. **Key Findings:** Quantitative results from the literature
5. **Limitations:** When does it fail or underperform?

**Key cross-references:**
- **Section 9:** Master comparison table of all methods
- **Section 10:** Practitioner recommendations by role
- **Section 11:** Open research questions
- **Appendix A:** Glossary of key terms
- **Appendix B:** Consolidated assumptions for all methods

---

## 2. Observational and Quasi-Experimental Methods

Methods in this section attempt to estimate causal effects from non-experimental data, or reduce the cost of experimentation. The punchline: **pure observational methods systematically fail validation against RCTs** (Gordon et al. 2019, 2023), but cost-reducing experimental variants like ghost ads are valuable.

### 2.1 Ghost Ads

**Source:** Johnson, Lewis, Nubbemeyer (2017). "Ghost Ads: Improving the Economics of Measuring Online Ad Effectiveness." *Journal of Marketing Research*, 54(6), 867–884. [DOI](https://doi.org/10.1177/0022243717718579)

#### Problem

Standard RCTs for ad effectiveness require a **public service announcement (PSA) control group** — users in the control group see a filler ad instead of the advertiser's ad. This is expensive because:
- The platform forgoes ad revenue on control-group impressions
- PSA ads must be sourced and served
- Advertisers resist having their budget "wasted" on control users

#### Method

Ghost ads exploit the ad auction infrastructure to create a **virtual control group** at near-zero cost:

1. **Identify the counterfactual:** For each user in the control group, the ad server records which ad *would have been shown* if the advertiser's campaign were active (the "ghost ad")
2. **Compare outcomes:** Treatment group sees the real ad; control group sees whatever ad won the auction instead, but the system logs that the focal advertiser's ad *would have* won
3. **Measure the difference:** Compare conversion rates between users who saw the ad and users who would have seen it but saw the next-best ad instead

**Key innovation:** The control group doesn't need a PSA — they see organic auction winners. The "ghost" is the counterfactual ad impression that was tracked but never shown.

#### Assumptions

| Assumption | Formal Statement | What Happens If Violated | How to Check |
|------------|------------------|-------------------------|--------------|
| **Auction fidelity** | Ghost ad identification correctly reflects what would have been shown | Biased control group composition | Validate against PSA tests |
| **No spillovers** | Control users' behaviour unaffected by not seeing the ad | Underestimates true effect | Check for cross-device/household spillovers |
| **Stable auction dynamics** | Removing the ad from control doesn't change auction outcomes for other ads | Second-order bias | Compare auction metrics between arms |
| **Random assignment** | Users randomised to treatment/control before auction | Selection bias | Standard randomisation checks |

*When it works:* Large-scale display advertising where the platform controls the ad server and can log counterfactual auction outcomes. Works well when the focal advertiser is a small fraction of total ad demand (removing them doesn't distort the auction).

*When it fails:* When the focal advertiser is a dominant buyer (removing them significantly changes auction dynamics), when cross-device tracking is incomplete, or when the ad format doesn't have a natural "next-best" replacement.

#### Key Findings

- Reduces the cost of ad experimentation by an **order of magnitude** compared to PSA tests
- Validated against PSA-based RCTs: ghost ad estimates are consistent with PSA estimates
- Enables measurement for campaigns that would never justify a full PSA test
- Still requires experimental infrastructure (randomisation, counterfactual logging) — just cheaper

#### Limitations

- Requires platform-side implementation (advertisers can't do this independently)
- Still an experiment — requires randomisation and holdout groups
- Not applicable outside of auction-based ad systems
- Ghost ad identification can be noisy for complex auction formats (e.g., multi-slot auctions)

---

### 2.2 Observational Methods Benchmarked Against RCTs

**Source:** Gordon, Zettelmeyer, Moakler, Reiley (2019). "A Comparison of Approaches to Advertising Measurement: Evidence from Big Field Experiments at Facebook." *Marketing Science*, 38(2), 193–225. [DOI](https://doi.org/10.1287/mksc.2018.1135)

#### Problem

If observational methods could reliably estimate causal effects, we wouldn't need RCTs at all. This paper tests that premise directly: can propensity score matching, stratification, and other observational approaches recover the causal effects measured by large-scale RCTs?

#### Method

The authors ran **15 large-scale RCTs on Facebook** (each with millions of users) and then applied observational methods to the same data, comparing observational estimates to the RCT ground truth:

**Methods tested:**
1. **Propensity Score Matching (PSM):** Match exposed and unexposed users on observable characteristics
2. **Propensity Score Stratification:** Stratify users by propensity to be exposed, estimate effects within strata
3. **Inverse Probability Weighting (IPW):** Reweight observations by inverse propensity scores
4. **Exact Matching:** Match on discrete observable characteristics
5. **Difference-in-Differences (DiD):** Compare pre-post changes between exposed and unexposed

#### Assumptions

| Assumption | Required By | What Happens If Violated | Violated in Practice? |
|------------|-------------|-------------------------|----------------------|
| **Unconfoundedness** (selection on observables) | All observational methods | Biased estimates — the core failure mode | **Yes, systematically** |
| **Common support** | PSM, IPW | Extreme weights, unstable estimates | Often |
| **Parallel trends** | DiD | Biased DiD estimates | Frequently in ad settings |
| **Correct propensity model** | PSM, IPW, stratification | Residual confounding | Model misspecification is common |
| **No anticipation effects** | DiD | Pre-treatment differences contaminate estimate | Sometimes |

*When it works:* Almost never for ad effectiveness measurement. The paper's central finding is that these methods **systematically fail**.

*When it fails:* When there is selection into ad exposure based on unobservable characteristics — which is nearly always the case. Users who see ads differ from those who don't in ways that observables cannot capture.

#### Key Findings

- **Median absolute errors of 60–175%** relative to RCT lift estimates
- Observational methods are **not just noisy — they are systematically biased**
- The direction of bias is unpredictable: sometimes overstates, sometimes understates
- Even with rich covariate data (Facebook's user profiles), selection bias dominates
- **Propensity score matching performed worst** — matching on observables does not eliminate unobserved confounding

| Method | Median Absolute Error vs RCT | Bias Direction | Reliability |
|--------|------------------------------|----------------|-------------|
| Propensity Score Matching | ~175% | Unpredictable | Very Poor |
| Stratification | ~100% | Unpredictable | Poor |
| Exact Matching | ~80% | Unpredictable | Poor |
| Difference-in-Differences | ~60% | Tends to overstate | Poor |

#### Limitations

- Results are specific to online display advertising on Facebook
- May not generalise to settings with stronger instruments or better observational data
- Does not test more modern methods (DML, SPSM) — see §2.3

**Why this paper matters for the rest of this review:** If observational methods can't recover causal effects, the field needs a different approach. This motivates the entire "prediction" paradigm (§4): instead of trying to identify causal effects without experiments, learn to *predict* them using experiments as training data.

---

### 2.3 "Close Enough?" Large-Scale Validation

**Source:** Gordon, Moakler, Zettelmeyer (2023). "Close Enough? A Large-Scale Exploration of Non-Experimental Approaches to Advertising Measurement." *Marketing Science*, 42(4), 768–793. [DOI](https://doi.org/10.1287/mksc.2022.1413)

#### Problem

The 2019 paper (§2.2) tested classical observational methods. Since then, more sophisticated approaches have emerged — double/debiased machine learning (DML), semiparametric subclassification with propensity scores (SPSM), and others. Do these modern methods close the gap with RCTs?

#### Method

Extended the 2019 analysis with:
- **More experiments:** Larger corpus of Meta ad experiments
- **More methods:** Added DML (Chernozhukov et al. 2018), SPSM, and other modern approaches
- **Predictive framing:** Introduced an early version of the PIE concept — using post-campaign features (like last-click conversions) in a predictive model trained on RCT outcomes

**Methods tested (in addition to §2.2):**
1. **Double/Debiased Machine Learning (DML):** Uses ML for nuisance parameter estimation with cross-fitting to avoid regularisation bias
2. **Semiparametric Subclassification with Propensity Scores (SPSM):** Combines subclassification with semiparametric efficiency
3. **Post-campaign predictive models:** Train on RCT outcomes using campaign-level features including endogenous metrics

#### Assumptions

| Assumption | Required By | What Happens If Violated | Status |
|------------|-------------|-------------------------|--------|
| **Unconfoundedness** | DML, SPSM | Biased estimates | **Still violated** — modern methods don't fix this |
| **Correct nuisance models** | DML | Inconsistent estimates | Partially addressed by cross-fitting |
| **Overlap** | All | Extreme weights | Still problematic |
| **RCT representativeness** (for predictive approach) | Post-campaign models | Prediction bias for unrepresented campaigns | Testable via cross-validation |

*When it works:* The predictive approach (precursor to PIE) works when post-campaign features are informative and the RCT training set is representative.

*When it fails:* DML and SPSM still fail for the same reason as classical methods — unobserved confounding. The predictive approach fails when extrapolating to campaign types not represented in the RCT corpus.

#### Key Findings

- **Modern observational methods (DML, SPSM) still produce large measurement errors** — the fundamental problem is unobserved confounding, not estimator sophistication
- **Post-campaign features, when used in a predictive model trained on RCTs, recover incremental conversions per dollar (ICPD) much better than program evaluation approaches**
- This finding directly motivates PIE (§4.1): the predictive framing outperforms the causal identification framing
- Last-click conversions are biased as a causal control but informative as a predictor (because the RCT handles identification)

| Approach | Error vs RCT | Key Limitation |
|----------|-------------|----------------|
| Classical observational (PSM, DiD) | 60–175% | Unobserved confounding |
| Modern observational (DML, SPSM) | Still large | Same fundamental problem |
| Predictive model on RCT outcomes | Much smaller | Requires RCT training data |

#### Limitations

- The predictive approach requires a corpus of completed RCTs — not available to all advertisers
- Generalisability of the predictive model depends on RCT sample composition
- This paper introduces the concept; PIE (§4.1) formalises and scales it

---

### 2.4 Causal Impact via Bayesian Structural Time-Series

**Source:** Brodersen, Gallusser, Koehler, Remy, Scott (2015). "Inferring causal impact using Bayesian structural time-series models." *Annals of Applied Statistics*, 9(1), 247–274. [arXiv:1506.00356](https://arxiv.org/abs/1506.00356)

#### Problem

Many interventions happen at a single point in time on a single unit (a market, a product, a country). You observe one time series that receives the intervention and one or more control series that don't. A simple before-after comparison is biased by concurrent trends (seasonality, macro shifts, competitor actions). How do you estimate what *would have happened* to the treated series in the absence of the intervention?

#### Method

The `CausalImpact` framework uses a **Bayesian structural time-series (BSTS)** model to construct a counterfactual forecast:

1. **Fit a BSTS model on the pre-period:** Using data before the intervention, fit a model that combines a local linear trend, seasonal components, and regression on control series
2. **Forecast the counterfactual:** Project the pre-period model forward through the post-intervention period to obtain the counterfactual series $\hat{Y}_t(0)$
3. **Compute the causal impact:** Subtract the counterfactual from observed post-period values: $\hat{\tau}_t = Y_t - \hat{Y}_t(0)$
4. **Quantify uncertainty:** The Bayesian framework provides posterior credible intervals for the cumulative and pointwise impact

**Key components:**
- **State-space structure:** Flexible decomposition into trend, seasonal, and regression components
- **Spike-and-slab priors:** Automatic variable selection among candidate control series
- **Full posterior inference:** Credible intervals for any summary (cumulative impact, average lift, pointwise effects)

#### Assumptions

| Assumption | Formal Statement | What Happens If Violated | How to Check |
|------------|------------------|-------------------------|--------------|
| **No spillovers to controls** | Control series $X_t$ unaffected by the intervention on $Y_t$ | Counterfactual contaminated — underestimates impact | Domain knowledge; check for shared audiences or substitution effects |
| **Good pre-period fit** | BSTS model explains pre-period variance well | Poor counterfactual forecast | Check pre-period $R^2$, residual diagnostics |
| **Stable structural relationships** | Relationship between $Y$ and controls $X$ stable from pre to post | Biased counterfactual | Rolling-window estimation on pre-period |
| **Correct model specification** | Chosen state components (trend, seasonality) match the data | Systematic forecast error | Posterior predictive checks |
| **Sufficient pre-period data** | Enough history to identify trend and seasonal components | Wide credible intervals, unstable estimates | At least 2× the seasonal period recommended |

*When it works:* Geographic experiments where a market receives the intervention and other markets serve as controls; policy evaluations with clean pre/post delineation; ad campaigns with natural holdout regions. Works best when control series are strongly correlated with the treated series in the pre-period.

*When it fails:* When there are spillovers between treated and control units (e.g., customers in treated geos influence purchases in control geos); when a structural break unrelated to the intervention occurs in the post-period; when the pre-period is too short to identify seasonality.

#### Key Findings

- Used by **Google for ad campaign measurement** — widely cited as the industry standard for quasi-experimental time-series analysis
- Popularised in the open-source **`CausalImpact` R package** (and Python port), which has been cited thousands of times in industry and academic work
- Provides posterior credible intervals rather than only point estimates, making uncertainty explicit
- Handles multiple control series automatically via spike-and-slab variable selection

| Application | Treated Unit | Control Series | Typical Finding |
|-------------|--------------|----------------|-----------------|
| Ad campaign measurement | Treated geo | Untreated geos | Pointwise and cumulative lift with credible intervals |
| Product launch | Product's category sales | Unaffected categories | Sales lift attributable to launch |
| Policy evaluation | Affected region | Comparison regions | Impact with quantified uncertainty |

#### Limitations

1. **Single-unit inference:** Designed for one treated series; less useful when you have many treated units with heterogeneous effects (use synthetic control or panel methods instead)
2. **Strong reliance on pre-period model:** If the pre-period model is misspecified, the counterfactual is biased for the entire post-period
3. **Spillover sensitivity:** Any spillover to control series contaminates the counterfactual; assumption is often hard to verify
4. **Quasi-experimental, not experimental:** Unlike RCTs, BSTS requires a modelling assumption about counterfactual dynamics — it doesn't identify the effect purely from randomisation
5. **Parametric structural choices:** Decomposition into trend + seasonality + regression may miss non-linear or regime-switching dynamics

**Relation to other methods:** Complementary to synthetic control (§2.5) — BSTS uses regression-based counterfactual construction while synthetic control uses a weighted combination of donor units. Both are quasi-experimental alternatives when a proper RCT isn't available; both are widely used for geo experiments.

---

### 2.5 Synthetic Control Methods

**Sources:**
- Abadie & Gardeazabal (2003). "The Economic Costs of Conflict: A Case Study of the Basque Country." *American Economic Review*, 93(1), 113–132.
- Abadie, Diamond, Hainmueller (2010). "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program." *Journal of the American Statistical Association*, 105(490), 493–505.
- Abadie (2021). "Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects." *Journal of Economic Literature*, 59(2), 391–425.

#### Problem

In many settings you have **one treated unit** (a country, a state, a market) and **no obvious comparison unit**. The "right" comparison isn't any single other unit, but a carefully weighted combination that mimics the treated unit's pre-treatment trajectory. How do you construct that comparison rigorously?

#### Method

Synthetic control methods (SCM) construct a **"synthetic" control unit** as a weighted combination of "donor" units that don't receive the intervention:

1. **Identify donor pool:** A set of $J$ untreated units with outcome and covariate data
2. **Optimise weights:** Choose weights $w_1, ..., w_J \geq 0$ (summing to 1) to minimise the distance between the treated unit and the weighted donors on (a) pre-treatment outcomes and (b) pre-treatment covariates
3. **Construct counterfactual:** The synthetic control's post-treatment outcome is $\hat{Y}_t(0) = \sum_j w_j Y_{j,t}$
4. **Estimate effect:** $\hat{\tau}_t = Y_{1,t} - \hat{Y}_t(0)$
5. **Inference:** Placebo tests — apply the same procedure to each donor unit as if it were treated; compare the treated unit's effect to the distribution of placebo effects

**Key intuition:** No single donor unit is likely to match the treated unit. But a convex combination — e.g., "40% Utah + 30% Colorado + 30% Nevada" — may closely match California on pre-treatment tobacco consumption, demographics, and trends. Under the assumption that this match persists absent treatment, the weighted combination approximates the counterfactual.

#### Assumptions

| Assumption | Formal Statement | What Happens If Violated | How to Check |
|------------|------------------|-------------------------|--------------|
| **No interference** (SUTVA) | Donor units unaffected by the intervention on the treated unit | Counterfactual contaminated; underestimates effect | Domain knowledge about spillovers |
| **Good pre-period fit** | Synthetic control closely matches treated unit on pre-period outcomes and covariates | Poor counterfactual; biased post-period estimate | Pre-period MSPE, visual inspection |
| **Donor pool well-chosen** | Donors similar to treated unit in relevant ways, untreated themselves | Poor match quality; weights concentrated on one donor | Expert judgment; sensitivity to donor pool composition |
| **Convex-hull condition** | Treated unit's pre-treatment features lie in the convex hull of donors | Extrapolation bias | Check whether weights hit boundary (all on one donor) |
| **Stable relationships** | Relationship between donor outcomes and treated-unit counterfactual is stable post-treatment | Biased counterfactual | Placebo tests in time |

*When it works:* When you have few treated units but many candidate controls, and the treated unit's pre-period trajectory can be closely matched by a combination of donors. Classic applications: one country, one state, one market. Strong when pre-treatment data is long relative to post-treatment horizon.

*When it fails:* When no combination of donors matches the treated unit (convex-hull violation); when donor units are themselves affected by the intervention (spillovers); when the treated unit is an outlier on important unobserved characteristics; or when the post-period is very long (extrapolation becomes tenuous).

#### Key Findings

- **Abadie & Gardeazabal (2003):** Estimated that terrorism reduced Basque Country per-capita GDP by approximately 10% over two decades, using a synthetic control built from other Spanish regions
- **Abadie, Diamond, Hainmueller (2010):** California's Proposition 99 tobacco control programme reduced per-capita cigarette sales by approximately 26 packs per year relative to a synthetic California constructed from other US states
- **Abadie (2021):** Provides a comprehensive methodological review — discusses feasibility conditions, data requirements, and identifies cases where synthetic control should and should not be used
- Widely adopted in economics, political science, public health, and — increasingly — **ad measurement for geo experiments** where random assignment isn't possible

| Application | Treated Unit | Donor Pool | Key Finding |
|-------------|--------------|------------|-------------|
| Basque terrorism | Basque Country | Other Spanish regions | ~10% GDP reduction |
| California tobacco (Prop 99) | California | 38 other US states | ~26 fewer packs/capita/year |
| Ad geo experiments | Test DMA(s) | Untreated DMAs | Campaign-level lift |

#### Limitations

1. **Single-unit inference:** Traditional SCM assumes one treated unit; extensions to multiple treated units exist but are newer
2. **Subjective donor pool choice:** Results can be sensitive to which units are included as donors; researcher degrees of freedom
3. **Inference by placebo:** Placebo-based p-values are non-standard and depend on donor pool size
4. **Convex-hull restriction:** Weights must be non-negative and sum to 1; if the treated unit is an outlier, this restriction produces poor pre-period fit
5. **Not experimental:** Like BSTS (§2.4), SCM is quasi-experimental — relies on modelling assumptions about the counterfactual rather than randomisation

**Relation to other methods:** Often compared head-to-head with difference-in-differences (§2.2) and BSTS (§2.4). Generalised synthetic control and matrix completion methods (Athey et al.) extend SCM to settings with multiple treated units.

---

### 2.6 Geo Experiments and Geo Holdouts

**Sources:**
- Vaver & Koehler (2011). "Measuring Ad Effectiveness Using Geo Experiments." Google Research.
- Kerman, Wang, Vaver (2017). "Estimating Ad Effectiveness using Geo Experiments in a Time-Based Regression Framework." Google Research.

#### Problem

User-level randomisation for ad measurement is increasingly difficult: cross-device tracking is incomplete, privacy regulations restrict cookie-based identifiers, and many platforms can't (or won't) expose user-level experimental infrastructure to advertisers. How do you measure ad effectiveness without user-level tracking?

#### Method

**Geo experiments** randomise at the level of geographic regions (DMAs, ZIP codes, countries) instead of users:

1. **Partition the country** into geographic regions (e.g., 210 US DMAs)
2. **Randomly assign** regions to treatment (ads active) vs. control (ads paused or reduced)
3. **Measure outcomes** at the geo level: regional sales, web traffic, app installs
4. **Estimate lift** by comparing treated geos to control geos, often using a time-based regression framework:

$Y_{g,t} = \alpha_g + \beta_t + \gamma \cdot W_{g,t} + \epsilon_{g,t}$

Where $W_{g,t}$ is an indicator for treatment in geo $g$ at time $t$, $\alpha_g$ captures geo fixed effects, and $\beta_t$ captures time effects.

**Kerman, Wang, Vaver (2017)** extend this with a **time-based regression framework** that:
- Uses pre-period data to predict control-group behaviour during the test period
- Applies the predicted counterfactual to treatment geos
- Provides more flexible modelling of geo-level seasonality and trends

#### Assumptions

| Assumption | Formal Statement | What Happens If Violated | How to Check |
|------------|------------------|-------------------------|--------------|
| **Random geo assignment** | Treatment assignment independent of geo characteristics | Selection bias | Randomisation checks, covariate balance |
| **No geo-level spillovers** | Users in control geos unaffected by treatment geos (no cross-geo travel, cross-border media) | Biased lift estimate (typically understated) | Check for border regions, media overlap |
| **Sufficient geo variation** | Enough geos to detect treatment effect | Underpowered | Power calculation on geo-level variance |
| **Stable geo characteristics** | Geo demographics/behaviour stable over test period | Noise masks treatment effect | Covariate checks, control period validation |
| **No ad-delivery leakage** | Ads delivered only in treated geos | Diluted treatment contrast | IP geolocation checks; ad server logs |

*When it works:* Campaigns with enough spend to create measurable geo-level variation (typically $100K+ test budgets); categories with geographically localised consumption (retail, QSR, auto); platforms that can target ads at the geo level.

*When it fails:* Small campaigns that can't achieve sufficient geo-level power; categories with highly national or online-ordered behaviour (where geo boundaries blur); settings with substantial media spillover across borders (TV, national digital).

#### Key Findings

- **Platform-independent measurement:** Doesn't require user-level tracking or cross-device identifiers, which matters in privacy-restricted environments
- **Used by Google, Meta, and advertisers** for platform-agnostic measurement — particularly important for comparing across platforms or validating platform-reported metrics
- **Kerman, Wang, Vaver (2017)** showed that time-based regression provides tighter confidence intervals than simple difference-in-differences for typical ad geo experiments
- Widely used for **incrementality measurement** of TV, out-of-home, and cross-channel campaigns where user-level experimentation is infeasible

| Design | Randomisation Unit | Typical Power | Use Case |
|--------|-------------------|---------------|----------|
| User-level A/B | User/cookie | High (millions of users) | Platform-internal measurement |
| Geo experiment | DMA/ZIP/region | Moderate (10s–100s of geos) | Cross-platform, privacy-safe measurement |
| Hybrid (geo + matched-market) | Geo pairs matched on pre-period | Moderate | When geo pool is heterogeneous |

#### Limitations

1. **Requires sufficient geographic variation:** Small campaigns can't justify the overhead; typical minimum is $100K+ in test spend
2. **Expensive for small-scale campaigns:** Geo experiments consume ad budget that would otherwise drive sales; opportunity cost is high
3. **Spillover concerns:** Media doesn't respect geo boundaries (national TV, digital ads seen while travelling, online ordering across geos)
4. **Slow:** Typical geo experiments run 4–8 weeks to accumulate sufficient data
5. **Coarse unit of analysis:** Can only measure average effects at the geo level; no user-level personalisation insights

**Relation to other methods:** Often combined with synthetic control (§2.5) or BSTS (§2.4) to analyse geo experiments — synthetic control constructs synthetic counterfactuals from control geos, BSTS models the time-series dynamics. Complementary to user-level RCTs (higher power, but requires tracking) and PIE (§4.1) (no new experiment, but requires training RCTs).

---

## 3. Surrogate and Proxy Outcome Methods

Methods in this section use intermediate or short-term metrics to predict long-term causal effects. Unlike the observational methods in §2, these methods **still require an experiment** — but they make the experiment faster or cheaper by using early signals to predict late outcomes.

### 3.1 The Surrogate Index

**Source:** Athey, Chetty, Imbens, Kang (2019/2024). "Combining Short-Term Proxies to Estimate Long-Term Treatment Effects More Rapidly and Precisely." [arXiv:1903.10706](https://arxiv.org/abs/1903.10706). Forthcoming, *Review of Economic Studies*.

#### Problem

Many interventions have effects that take months or years to fully materialise:
- Job training programs → employment outcomes at 5–10 years
- Ad campaigns → long-term brand equity and customer lifetime value
- Product features → long-term retention and engagement
- Policy interventions → health or economic outcomes over decades

Running an RCT and waiting for the long-term outcome is often impractical. Can we use short-term outcomes observed during the experiment to predict the long-term treatment effect?

#### Method

The **surrogate index** combines multiple short-term outcomes into a single predictive score for the long-term outcome:

1. **Identify surrogates:** Find short-term outcomes $S = (S_1, ..., S_K)$ that are (a) affected by treatment and (b) predictive of the long-term outcome $Y$
2. **Estimate the surrogate index:** In a historical dataset where both $S$ and $Y$ are observed, regress $Y$ on $S$:
   $\hat{Y} = g(S) = \beta_0 + \beta_1 S_1 + ... + \beta_K S_K$
3. **Apply to the experiment:** In the current RCT, observe $S$ for both treatment and control groups. Compute the surrogate index $\hat{Y}$ for each unit.
4. **Estimate the treatment effect:** The treatment effect on the surrogate index estimates the long-term treatment effect:
   $\hat{\tau}_{long} = E[\hat{Y} | W=1] - E[\hat{Y} | W=0]$

**Key insight:** The surrogate index is a *sufficient statistic* for the long-term treatment effect under the surrogacy assumption. You don't need to observe $Y$ in the current experiment — the surrogates carry all the information.

#### Assumptions

| Assumption | Formal Statement | What Happens If Violated | How to Check |
|------------|------------------|-------------------------|--------------|
| **Prentice surrogacy criterion** | $Y \perp W \mid S$ (primary outcome independent of treatment conditional on surrogates) | Surrogate index misses treatment effect channels not mediated by $S$ | Test on historical data where both $S$ and $Y$ are observed |
| **Comparability** | Relationship between $S$ and $Y$ is stable across the historical and current settings | Prediction model doesn't transfer | Validate on held-out historical experiments |
| **No direct effect** | Treatment affects $Y$ only through $S$, not through other channels | Biased long-term estimate (typically understates) | Domain knowledge; test with multiple surrogate sets |
| **Correct functional form** | $g(S)$ correctly specifies the $S \to Y$ relationship | Prediction error | Cross-validation on historical data |

*When it works:* When the surrogates genuinely mediate the treatment effect on the long-term outcome. Job training → short-term employment → long-term earnings is a strong example because employment is a primary channel.

*When it fails:* When treatment affects the long-term outcome through channels not captured by the surrogates. Example: an ad campaign that builds brand awareness (not captured by short-term clicks) leading to purchases months later.

#### Key Findings

- Applied to **job training experiments**: predicted 9-year employment effects from 6 quarters of data
- Surrogate index estimates were within the confidence intervals of the direct long-term estimates
- Multiple surrogates outperform single surrogates — the index captures more of the treatment effect pathway
- Precision gains of 2–4× compared to waiting for the long-term outcome

| Application | Surrogates Used | Prediction Horizon | Accuracy |
|-------------|----------------|-------------------|----------|
| Job training → 9-year earnings | Quarterly employment, earnings (6 quarters) | 6 quarters → 9 years | Within CI of direct estimate |
| Education → long-term outcomes | Test scores, attendance, graduation | 1–2 years → 10+ years | Good for mediated effects |

#### Limitations

- **The surrogacy assumption is strong and often violated.** In advertising, short-term clicks may not capture brand-building effects that drive long-term sales.
- Requires historical data where both surrogates and long-term outcomes are observed (to train $g(S)$)
- If the treatment changes the $S \to Y$ relationship (e.g., a new ad format that converts differently), the surrogate model is invalid
- Does not eliminate the need for an experiment — just makes it shorter
- Theoretical guarantees are asymptotic; finite-sample performance depends on surrogate quality

---

### 3.2 Surrogate Index Validation at Netflix

**Source:** Netflix Technology Blog (2023). "Evaluating the Surrogate Index as a Decision-Making Tool Using 200 A/B Tests at Netflix." [Netflix Tech Blog](https://netflixtechblog.com/evaluating-the-surrogate-index-as-a-decision-making-tool-using-200-a-b-tests-at-netflix-4b4b3e4f5c9e)

#### Problem

The surrogate index (§3.1) has strong theoretical foundations but limited empirical validation at scale. Netflix tested whether surrogate-based inferences agree with direct long-term inferences across a large corpus of real A/B tests.

#### Method

Netflix validated the surrogate index approach on **200 real A/B tests** where both short-term and long-term outcomes were observed:

1. **Auto-surrogate models:** Instead of using different short-term metrics as surrogates, they used **shorter-term observations of the same outcome metric** (e.g., 14-day retention to predict 56-day retention)
2. **Consistency check:** For each test, compared the decision (ship/don't ship) that would be made using the surrogate index vs. the direct long-term measurement
3. **Calibration check:** Assessed whether surrogate-based effect size estimates match long-term estimates

**"Auto-surrogate" approach:**
- Surrogate: same metric measured at time $t_{short}$ (e.g., 14 days)
- Target: same metric measured at time $t_{long}$ (e.g., 56 days)
- Model: $Y_{long} = g(Y_{short})$, trained on historical experiments

This is simpler than the general surrogate index because it avoids the question of which surrogates to use — it's the same metric, just measured earlier.

#### Assumptions

| Assumption | What Happens If Violated | Status at Netflix |
|------------|-------------------------|-------------------|
| **Temporal surrogacy** | Short-term metric doesn't capture delayed treatment effects | Mostly satisfied for engagement metrics |
| **Stable temporal relationship** | $Y_{short} \to Y_{long}$ mapping changes over time | Validated across 200 tests |
| **No delayed-onset effects** | Treatment effects that emerge only after $t_{short}$ are missed | Rare for Netflix's primary metrics |
| **Representative test corpus** | Validation results don't generalise to new test types | Broad coverage of Netflix test types |

*When it works:* When treatment effects manifest early and persist (or decay predictably). Engagement metrics at Netflix have this property — if a feature improves 14-day retention, it almost always improves 56-day retention.

*When it fails:* When treatment effects are delayed (e.g., a feature that initially confuses users but improves long-term satisfaction after a learning period), or when the short-term metric captures a different construct than the long-term metric.

#### Key Findings

- **~95% consistency** between surrogate-based and direct long-term inferences (same ship/don't-ship decision)
- The 5% disagreement cases were mostly experiments with small effects near the decision boundary
- Auto-surrogate models are simpler to implement and maintain than multi-metric surrogate indices
- Surrogate-based decisions could be made **2–4× faster** than waiting for long-term outcomes

| Metric | Surrogate Window | Target Window | Decision Consistency |
|--------|-----------------|---------------|---------------------|
| Retention | 14 days | 56 days | ~95% |
| Engagement | 7 days | 28 days | ~96% |
| Revenue | 14 days | 56 days | ~93% |

#### Limitations

- Netflix's metrics may be unusually well-suited to surrogacy (strong temporal correlation)
- Results may not transfer to advertising (where delayed brand effects are common) or other domains
- The 5% disagreement rate is non-trivial for high-stakes decisions
- Auto-surrogates work best when the metric is the same at different time horizons; less applicable when the long-term outcome is fundamentally different from short-term signals

**Connection to PIE (§4.1):** Surrogate methods and PIE are complementary. Surrogates accelerate learning *within* an experiment; PIE eliminates the need for a new experiment entirely. A platform could use surrogates for the RCTs it does run (to get faster reads) and PIE for the campaigns it doesn't experiment on.

---

### 3.3 Long-Term Effects via Data Combination

**Sources:**
- Athey, Chetty, Imbens (2024). "The Surrogate Index: Combining Short-Term Proxies to Estimate Long-Term Treatment Effects More Rapidly and Precisely." NBER Working Paper 26463. [NBER](https://www.nber.org/papers/w26463)
- Ghassami et al. (2024). "Long-term causal inference under persistent confounding via data combination." *Journal of the Royal Statistical Society Series B*.

#### Problem

The original surrogate index (§3.1) assumes that the relationship between short-term surrogates and long-term outcomes is stable across settings. But in practice:
- **Short-term experimental data** comes from an RCT that cleanly identifies treatment effects on surrogates
- **Long-term outcome data** often comes from observational sources (administrative records, customer databases, sales systems) where confounding persists
- If the short-term experimental population and the long-term observational population differ, or if unobserved confounders affect both the surrogate and long-term outcome, the surrogate index is biased

How do you combine experimental data (clean but short) with observational data (long but confounded) to estimate long-term effects more reliably?

#### Method

Recent work extends the surrogate-index framework to settings with **unobserved confounding** and **data combination**:

**Athey, Chetty, Imbens (2024) — NBER 26463:**
- Formalises conditions under which short-term surrogates from an experiment plus long-term observational data identify long-term treatment effects
- Establishes efficiency bounds and recommends estimators (OLS, IPW, doubly-robust combinations)
- Emphasises that multiple surrogates are essential — a single surrogate rarely satisfies the surrogacy condition
- Provides empirical applications combining RCT short-term outcomes with long-term administrative records

**Ghassami et al. (2024):**
- Tackles the case where **confounding persists in the long-term observational data** (not eliminated by conditioning on observables)
- Uses instrumental-variable-like identification: the experiment serves as a "proximal" instrument that corrects for confounding in the observational data
- Provides identification conditions and semiparametric estimators with asymptotic guarantees
- Key idea: the experimental data "anchors" the causal effect in a way that isolates the surrogate–long-term relationship from confounding

**The general data-combination recipe:**
1. **Experimental arm:** RCT with short-term surrogates $S$ observed, long-term outcome $Y$ not observed
2. **Observational arm:** Historical data with both $S$ and $Y$ observed, but no randomisation
3. **Identification:** Combine the experimental $W \to S$ relationship with the observational $S \to Y$ relationship, under assumptions that control for confounding in the observational arm

#### Assumptions

| Assumption | Formal Statement | What Happens If Violated | Notes |
|------------|------------------|-------------------------|-------|
| **Multiple surrogates** | $S = (S_1, ..., S_K)$ jointly satisfy surrogacy | Single-surrogate methods fail; multiple-surrogate methods recover the effect | Athey–Chetty–Imbens emphasise this |
| **Proximal identification** | Experimental data identifies the confounded portion of the $S \to Y$ relationship | Biased long-term estimate | Ghassami et al.'s contribution |
| **Consistency of populations** | Experimental and observational samples come from compatible populations (after adjustment) | Transport failure; estimates don't apply to target population | Requires covariate overlap |
| **Stable surrogate mechanism** | Treatment affects $Y$ only through $S$ (in both experimental and observational data) | Direct effect bias | Same as Prentice criterion |

*When it works:* When you have a clean RCT that measures treatment effects on surrogates, plus a rich observational dataset that links surrogates to long-term outcomes — and when at least some surrogates capture the dominant causal pathway.

*When it fails:* When confounding in the observational data is adversarial (strong unobserved confounders affecting both $S$ and $Y$), when the experimental and observational populations differ in unobserved ways, or when treatment has large effects through channels not captured by any surrogate.

#### Key Findings

- Extends surrogate-index applicability to settings where long-term outcomes can only be measured observationally — a common situation in advertising (long-term sales come from point-of-sale or CRM systems, not RCT infrastructure)
- Establishes **efficiency bounds**: in some settings, combining RCT + observational data is strictly more efficient than waiting for long-term RCT outcomes
- Semiparametric estimators achieve these bounds and have been implemented in open-source software
- Directly relevant to ad measurement where long-term sales data comes from observational sources (retailer data, CRM) but short-term engagement comes from the platform

| Data Source | Role | Strengths | Limitations |
|-------------|------|-----------|-------------|
| RCT (short-term) | Identifies $W \to S$ | Randomisation, clean causal signal | Short horizon only |
| Observational (long-term) | Identifies $S \to Y$ | Long history, large samples | Confounding risk |
| Combination | Identifies $W \to Y$ long-term | Best of both | Requires transportability assumptions |

#### Limitations

1. **Identification assumptions are strong:** Proximal identification and multiple-surrogate conditions are testable only indirectly; violations are hard to detect
2. **Semiparametric estimators are complex:** Implementation requires care with nuisance parameter estimation, cross-fitting, and variance estimation
3. **Population compatibility:** Experimental and observational populations must be compatible after covariate adjustment; this is often a stretch in advertising (RCTs run on a platform, long-term sales include off-platform channels)
4. **Data-sharing requirements:** Combining RCT and observational data often requires crossing organisational boundaries (platform RCT + advertiser CRM), with legal, privacy, and technical complications

**Relation to other methods:** Complementary to PIE (§4.1) — PIE predicts campaign-level effects from features; long-term data combination estimates long-term effects from short-term surrogates plus observational data. They solve different problems and can stack: use long-term data combination to extend the outcome horizon of RCTs, then train PIE on the extended RCT corpus.

---

## 4. Cross-Experiment Prediction (PIE and Related)

This is the central section of the review. Methods here learn from a **corpus of completed experiments** to predict causal effects for new interventions — without running a new RCT for each decision.

### 4.1 Predicted Incrementality by Experimentation (PIE)

**Source:** Gordon, Moakler, Zettelmeyer (2026). "Predicted Incrementality by Experimentation (PIE)." [arXiv:2304.06828v2](https://arxiv.org/abs/2304.06828). Forthcoming.

**This is the central paper of this review.**

#### Problem

Ad platforms run thousands of campaigns but can only experimentally measure a small fraction. Last-click attribution is the default measurement, but it systematically overstates ad effectiveness (by ~33% on average). Observational methods fail validation (§2.2, §2.3). How can a platform provide campaign-level causal measurement at scale?

#### Method

PIE reframes ad measurement as a **campaign-level prediction problem**:

1. **Training data:** Each completed RCT is one labelled observation. The label is the experimentally measured causal effect (e.g., incremental conversions per dollar, ICPD). Features are campaign-level characteristics.
2. **Features:** PIE uses "post-determined features" — campaign-level aggregates computed from the test group's data:
   - Test-group conversion rate
   - Test-group exposure rate
   - Last-click conversions per dollar
   - Campaign spend, duration, audience size
   - Advertiser vertical, objective type
3. **Model:** Random forest trained on 2,226 Meta ad experiments
4. **Prediction:** For a new (non-RCT) campaign, compute the same features from observed data and predict ICPD

**The key innovation: "post-determined features."**

These features (like test-group outcomes and last-click conversions) are **endogenous** — they are affected by the treatment and would be biased if used as causal controls in an observational study. But PIE doesn't use them for causal identification. It uses them as **predictors** in a supervised learning problem where the RCT provides the label. The RCT handles identification; the features handle prediction.

**Why this works (intuition):**
- Last-click conversions are biased as a measure of incrementality (they include organic conversions misattributed to ads)
- But last-click conversions are *correlated* with incrementality — campaigns with more last-click conversions tend to have more incremental conversions too
- The bias is systematic and learnable: a model trained on RCTs can learn the mapping from biased-but-informative features to unbiased causal effects
- The model essentially learns a "debiasing function" that converts observable campaign metrics into causal effect predictions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PIE: How It Works                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRAINING PHASE (one-time, using historical RCTs):                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │  RCT 1           │    │  RCT 2           │    │  RCT N           │      │
│  │  Features: X₁    │    │  Features: X₂    │    │  Features: Xₙ    │      │
│  │  Label: ICPD₁    │    │  Label: ICPD₂    │    │  Label: ICPDₙ    │      │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘      │
│           └──────────────────────┬┘───────────────────────┘                 │
│                                  ▼                                          │
│                    ┌──────────────────────────┐                              │
│                    │  Random Forest Model      │                             │
│                    │  f: X → ICPD              │                             │
│                    └──────────────────────────┘                              │
│                                                                             │
│  PREDICTION PHASE (for each new campaign):                                  │
│  ┌──────────────────┐    ┌──────────────────┐                               │
│  │  New Campaign     │    │  PIE Prediction   │                              │
│  │  Features: X_new  │───►│  ICPD_pred        │                              │
│  │  (no RCT needed)  │    │  = f(X_new)       │                              │
│  └──────────────────┘    └──────────────────┘                               │
│                                                                             │
│  POST-DETERMINED FEATURES (examples):                                       │
│  • Test-group conversion rate        • Last-click conversions/dollar         │
│  • Exposure rate                     • Campaign spend & duration             │
│  • Advertiser vertical               • Audience size & targeting             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Assumptions

| Assumption | Formal Statement | What Happens If Violated | How to Check | Severity |
|------------|------------------|-------------------------|--------------|----------|
| **Feature invariance** | Post-determined features have the same distribution whether or not an RCT is run | Prediction bias — features shift when experiment is present | Compare feature distributions for RCT vs non-RCT campaigns | Medium |
| **RCT representativeness** | RCT sample is representative of the non-RCT campaign population | Selection bias in predictions — model trained on unrepresentative campaigns | Compare RCT and non-RCT campaign feature distributions | High |
| **No concept drift** | The feature-to-effect mapping $f: X \to ICPD$ is stable over time | Model degrades as the relationship changes | Monitor out-of-sample R² over time; periodic re-training | Medium |
| **Sufficient RCT corpus** | Enough RCTs to train a flexible model | Overfitting, poor generalisation | Cross-validation; learning curves | High |
| **SUTVA within campaigns** | No interference between campaigns | Campaign-level effects are well-defined | Check for budget competition, audience overlap | Low-Medium |

*When it works:* PIE works best when:
1. **Treatment effects vary substantially** across campaigns — if all campaigns have the same effect, there's nothing to predict
2. **Baseline outcomes vary modestly** — high baseline variance adds noise without signal
3. **Positive correlation between baseline and treatment effects** — campaigns with more organic conversions also tend to have more incremental conversions
4. **Large number of RCTs** — more training data improves the model
5. **Selection into exposure** — when ad exposure is selective (not everyone sees the ad), post-determined features like exposure rate are informative
6. **Similar selection mechanisms across campaigns** — the "debiasing function" is consistent

*When it fails:*
1. **New advertiser verticals** not represented in the RCT corpus (R² drops from 0.88 to 0.72)
2. **Cross-vertical extrapolation** — travel campaigns trained on retail RCTs lose ~25 percentage points of R²
3. **Campaigns with novel mechanics** — new ad formats, targeting strategies, or objectives not seen in training
4. **Concept drift** — if the platform changes its ad serving algorithm, the feature-to-effect mapping shifts

#### Key Findings

**Primary results (2,226 Meta ad experiments):**

| Metric | PIE | Last-Click Attribution | Improvement |
|--------|-----|----------------------|-------------|
| Out-of-sample R² (ICPD) | **0.88** | 0.19 | 4.6× |
| Out-of-sample R² (incremental conversions) | **0.93** | 0.42 | 2.2× |
| Decision disagreement with RCT | **8–12%** | 12–20% | ~50% fewer errors |
| Average overstatement of effect | ~0% (calibrated) | ~33% | Eliminates systematic bias |

**Decision framework analysis:**
- PIE disagrees with RCT-based decisions (invest/don't invest) in only **8–12% of campaigns**
- Last-click attribution disagrees in **12–20% of campaigns**
- The disagreements are concentrated in campaigns with small effects near the decision boundary
- For campaigns with large positive or large negative effects, PIE and RCTs almost always agree

**Simulation results — when PIE works best:**

| Condition | Effect on PIE R² | Why |
|-----------|------------------|-----|
| High treatment effect heterogeneity | ↑ Improves | More signal to predict |
| Low baseline outcome variance | ↑ Improves | Less noise in features |
| Positive baseline-treatment correlation | ↑ Improves | Features are more informative |
| Large RCT corpus (>500) | ↑ Improves | Better model training |
| High selection into exposure | ↑ Improves | Exposure rate becomes informative |
| Consistent selection mechanisms | ↑ Improves | Debiasing function is stable |
| Novel campaign types | ↓ Degrades | Extrapolation beyond training distribution |
| Concept drift | ↓ Degrades | Stale model |

**Performance by subgroup:**

| Subgroup | Out-of-sample R² | Notes |
|----------|------------------|-------|
| Same advertiser, same vertical | 0.88 | Best case — interpolation |
| New advertiser, same vertical | 0.72 | Moderate degradation |
| Cross-vertical extrapolation | ~0.63 | Significant degradation (travel → retail loses ~25 pp) |
| All campaigns pooled | 0.88 | Overall performance |

**Production adoption:**
- **Amazon Ads:** PIE-like approach used to calibrate multi-touch attribution (MTA) models (see §4.2)
- **Meta:** Incremental Attribution product uses PIE methodology
- Both platforms use PIE to provide campaign-level incrementality estimates without per-campaign RCTs

#### Limitations

1. **Requires a large corpus of RCTs** — platforms without hundreds of historical experiments can't train PIE
2. **Extrapolation risk** — predictions for campaign types not in the training set are unreliable
3. **No uncertainty quantification** — PIE provides point predictions, not confidence intervals (random forests don't naturally produce calibrated intervals)
4. **Concept drift** — the model needs periodic re-training as the platform evolves
5. **Feature engineering matters** — the choice of post-determined features significantly affects performance
6. **Not a substitute for RCTs** — PIE should be validated against periodic RCTs, not used as a permanent replacement
7. **Campaign-level only** — PIE predicts average effects for a campaign, not user-level heterogeneous effects

**PIE Assumption Sensitivity — What Breaks and How Badly:**

| Assumption Violated | Degradation | Detectable? | Recovery Strategy |
|--------------------|-------------|-------------|-------------------|
| Feature invariance (RCT presence changes features) | Moderate — prediction bias proportional to feature shift | Yes — compare feature distributions | Use features robust to RCT presence (e.g., advertiser vertical, not exposure rate) |
| RCT sample unrepresentative | Severe — extrapolation to unseen campaign types | Yes — feature coverage analysis | Active learning; stratified RCT selection |
| Concept drift (slow) | Gradual — R² decays over months | Yes — rolling R² monitoring | Periodic re-training (quarterly) |
| Concept drift (sudden, e.g., algorithm change) | Severe — model immediately stale | Yes — sudden R² drop | Immediate re-training; flag predictions as uncertain |
| Insufficient RCTs (<200) | Moderate — overfitting, wide prediction errors | Yes — cross-validation | Collect more RCTs; use simpler models (linear) |
| Campaign interference (budget competition) | Low-Moderate — campaign-level effects slightly biased | Hard to detect | Cluster campaigns by budget pool |

---

### 4.2 Amazon Multi-Touch Attribution (MTA)

**Source:** Lewis, Zettelmeyer et al. (2025). "Multi-Touch Attribution at Amazon." [arXiv:2508.08209](https://arxiv.org/abs/2508.08209).

#### Problem

Multi-touch attribution (MTA) assigns credit for conversions to individual ad touchpoints along the customer journey. Traditional MTA uses heuristic rules (last-click, first-click, linear) or ML models trained on observational data — both of which are biased because they can't distinguish correlation from causation.

#### Method

Amazon's MTA system uses a **PIE-like approach to calibrate ML attribution models to RCT outcomes**:

1. **ML attribution model:** Train a model that predicts conversion probability as a function of the ad touchpoint sequence
2. **RCT calibration:** Use completed RCTs to measure the true incremental effect of ad campaigns
3. **PIE extrapolation:** For campaigns without RCTs, use PIE to predict the campaign-level incremental effect
4. **Credit allocation:** The ML model allocates the PIE-predicted campaign-level effect to individual touchpoints based on their predicted contribution

**The two-stage approach:**
- **Stage 1 (PIE):** Predict campaign-level incremental conversions (causal effect) using the PIE methodology
- **Stage 2 (MTA):** Allocate the campaign-level effect to individual touchpoints using an ML model trained on conversion paths

This separation is important: PIE handles the causal identification problem (how many incremental conversions did this campaign cause?), while MTA handles the attribution problem (which touchpoints deserve credit?).

#### Assumptions

| Assumption | What Happens If Violated | Severity |
|------------|-------------------------|----------|
| **PIE assumptions hold** (§4.1) | Campaign-level predictions are biased, contaminating touchpoint attribution | High |
| **Touchpoint model is well-specified** | Credit misallocated across touchpoints (but campaign total is still correct) | Medium |
| **No cross-campaign interference** | Campaign-level effects are not well-defined | Medium |
| **Stable customer journeys** | Attribution model becomes stale | Medium |

*When it works:* When PIE provides accurate campaign-level predictions and the touchpoint model captures meaningful variation in touchpoint effectiveness.

*When it fails:* When PIE predictions are inaccurate (novel campaign types) or when customer journeys are highly non-stationary.

#### Key Findings

- Enables **touchpoint-level credit assignment** calibrated to experimental ground truth
- Scales to Amazon's full ad ecosystem without per-campaign RCTs
- PIE provides the "total pie" (campaign-level effect); MTA slices it across touchpoints
- Represents a production deployment of PIE methodology at scale

#### Limitations

- Touchpoint-level attribution is inherently harder to validate than campaign-level effects (no touchpoint-level RCTs)
- The two-stage approach assumes campaign-level and touchpoint-level effects are separable
- Requires both PIE infrastructure and MTA model infrastructure

---

### 4.3 Causal ML / Uplift Modelling

**Sources:**
- Athey & Wager (2018). "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests." [*Annals of Statistics*](https://doi.org/10.1214/17-AOS1637)
- Ascarza (2018). "Retention Futility: Targeting High-Risk Customers Is Not Effective." [*Journal of Marketing Research*](https://doi.org/10.1509/jmr.16.0163)
- Rzepakowski & Jaroszewicz (2012). "Decision Trees for Uplift Modeling with Single and Multiple Treatments." [*Knowledge and Information Systems*](https://doi.org/10.1007/s10115-011-0434-0)

#### Problem

Given an RCT, how do you identify which users benefit most from treatment? This is the **targeting** problem: allocate a limited intervention budget to the users with the highest expected treatment effect.

#### Method

Causal ML / uplift modelling estimates **heterogeneous treatment effects (HTE)** at the user level:

**Key approaches:**

| Method | How It Works | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| **Causal Forests** (Athey & Wager) | Random forest that splits on treatment effect heterogeneity, not just outcome prediction | Honest inference, asymptotic normality, confidence intervals | Requires large samples; computationally expensive |
| **S-Learner** | Single model: $\hat{Y} = f(X, W)$; CATE = $f(X, 1) - f(X, 0)$ | Simple to implement | Regularisation can shrink treatment effect to zero |
| **T-Learner** | Two models: $\hat{Y}_1 = f_1(X)$, $\hat{Y}_0 = f_0(X)$; CATE = $\hat{Y}_1 - \hat{Y}_0$ | Flexible; captures complex HTEs | High variance from differencing two models |
| **X-Learner** (Künzel et al. 2019) | Impute individual treatment effects, then model them | Works well with unbalanced treatment/control | More complex; requires careful cross-fitting |
| **Uplift Trees** (Rzepakowski & Jaroszewicz) | Decision trees that split to maximise treatment effect divergence | Interpretable; handles multiple treatments | Prone to overfitting; limited expressiveness |

**Ascarza (2018) — a cautionary tale:**
- Showed that targeting users with the highest *churn risk* (predicted outcome) is not the same as targeting users with the highest *treatment effect* (uplift)
- Many high-risk users are "lost causes" — they'll churn regardless of intervention
- The users with the highest uplift are often moderate-risk users who are persuadable
- This distinction between prediction and uplift is fundamental to causal ML

#### How This Differs from PIE

| Dimension | Causal ML / Uplift | PIE |
|-----------|-------------------|-----|
| **Unit of analysis** | Individual user | Campaign |
| **Goal** | Targeting — who to treat | Measurement — what was the campaign's effect |
| **Requires experiment?** | Yes, per campaign | No (uses historical RCTs) |
| **Extrapolation** | Within-experiment only | Across experiments |
| **Output** | User-level CATE: $\hat{\tau}(x_i)$ | Campaign-level ICPD: $\hat{\tau}_j$ |
| **Features** | User covariates | Campaign-level post-determined features |

*When it works:* When you have an RCT for the specific campaign and want to optimise targeting within that campaign. Strong when treatment effects are heterogeneous across users.

*When it fails:* When you need to measure a campaign that wasn't experimentally tested — causal ML can't extrapolate beyond the experiment it was trained on.

#### Key Findings

- Causal forests provide valid confidence intervals for user-level treatment effects (under honesty and regularity conditions)
- Uplift modelling can improve targeting ROI by 20–50% compared to outcome-based targeting (Ascarza 2018)
- Meta-learners (S, T, X) offer different bias-variance tradeoffs; X-learner often performs best with unbalanced data

#### Limitations

- **Requires experimental data per campaign** — cannot extrapolate to non-experimental campaigns
- Computationally expensive for large-scale deployment
- Confidence intervals for individual-level effects are wide
- Model selection (which meta-learner?) is itself a research question

---

### 4.4 Cross-Experiment Meta-Learning

**Source:** Huang, Ascarza, Israeli (2024). "Pooling Multiple Experiments to Predict Individual-Level Treatment Effects for Personalization." Working paper.

#### Problem

Individual experiments provide limited data for estimating user-level treatment effects, especially for rare subgroups. Can we pool data across multiple experiments to improve user-level predictions?

#### Method

Cross-experiment meta-learning pools data from multiple RCTs to build a shared model of treatment effect heterogeneity:

1. **Pool experiments:** Combine user-level data from multiple RCTs, each testing a different intervention
2. **Shared representation:** Learn a common feature representation that predicts treatment effects across experiments
3. **Transfer learning:** Use the shared representation to predict user-level effects in new experiments with limited data

**Key insight:** While each experiment tests a different intervention, the *moderators* of treatment effects (user characteristics that predict who benefits) may be shared across experiments. A user who responds strongly to one type of intervention may respond strongly to similar interventions.

#### How This Differs from PIE

| Dimension | Cross-Experiment Meta-Learning | PIE |
|-----------|-------------------------------|-----|
| **Unit of analysis** | Individual user | Campaign |
| **Goal** | Personalisation — predict user-level effects | Measurement — predict campaign-level effects |
| **Pooling mechanism** | Shared user-level feature representation | Campaign-level features + RCT labels |
| **Output** | User-level CATE for new experiments | Campaign-level ICPD for non-RCT campaigns |
| **Key innovation** | Transfer of HTE moderators across experiments | Post-determined features as predictors |

*When it works:* When treatment effect heterogeneity is driven by stable user characteristics that are shared across experiments. Works well when experiments test similar types of interventions.

*When it fails:* When treatment effect moderators are experiment-specific (e.g., the users who respond to a discount are different from those who respond to a loyalty program).

#### Assumptions

| Assumption | What Happens If Violated | Severity |
|------------|-------------------------|----------|
| **Shared moderators** | Pooling adds noise rather than signal | High |
| **Comparable experimental populations** | Selection bias in transferred predictions | Medium |
| **Stable user characteristics** | Predictions degrade over time | Medium |

#### Key Findings

- Pooling across experiments improves user-level CATE estimation, especially for small experiments
- The benefit is largest when experiments share common moderators of treatment effects
- Related to PIE in spirit (learning across experiments) but targets a different level of analysis

#### Limitations

- Requires user-level data from multiple experiments (privacy constraints may limit this)
- Assumes shared moderators — may not hold across very different intervention types
- Computationally intensive for large user bases
- Less mature than PIE; limited production deployments

---

### 4.5 Removing Hidden Confounding by Experimental Grounding

**Source:** Kallus, Puli, Shalit (2018). "Removing Hidden Confounding by Experimental Grounding." NeurIPS 2018. [arXiv:1810.11646](https://arxiv.org/abs/1810.11646)

#### Problem

Observational data is abundant and rich but suffers from unobserved confounding (§2.2). Experimental data is clean but scarce and often restricted to narrow populations. Can you use a small experiment to **correct hidden confounding** in a model trained on much larger observational data, getting the best of both?

#### Method

Kallus, Puli, Shalit introduce **experimental grounding**: use limited experimental data to calibrate a model trained on abundant observational data, even when the observational model suffers from hidden confounding.

**Approach:**
1. **Train on observational data:** Fit a treatment effect model $\hat{\tau}_{obs}(x)$ on large observational data — this model is biased by unobserved confounding but captures rich feature variation
2. **Estimate confounding bias:** Use a small experiment to estimate the bias function $b(x) = \tau(x) - \tau_{obs}(x)$ — the difference between the true effect and the observational estimate, as a function of covariates
3. **Correct predictions:** Combine: $\hat{\tau}(x) = \hat{\tau}_{obs}(x) + \hat{b}(x)$
4. **Leverage observational breadth:** The observational data provides coverage across the feature space; the experimental data provides the calibration signal

**Key insight:** The experiment doesn't need to cover the full population. It just needs to be large enough to estimate the *bias function* — how the observational model's error varies with observable covariates. The observational data then propagates this correction across the full feature space.

#### Assumptions

| Assumption | Formal Statement | What Happens If Violated | How to Check |
|------------|------------------|-------------------------|--------------|
| **Transferable bias** | Confounding structure is similar between experimental and observational populations (even if not identical covariates) | Calibration doesn't transfer; grounded model still biased | Check bias in overlapping subgroups |
| **Low-dimensional bias function** | Bias $b(x)$ can be estimated from a small experiment | Overfitting on experimental sample | Regularisation; cross-validation |
| **Covariate overlap** | Experimental and observational populations share relevant covariates | Extrapolation bias | Propensity-score overlap diagnostics |
| **Valid experiment** | Experimental data identifies $\tau(x)$ without bias | Calibration contaminated | Standard RCT checks |

*When it works:* When you have a rich observational dataset covering diverse users/contexts, plus a small RCT that samples from a similar confounding structure. The observational model provides generalisation; the RCT provides bias correction.

*When it fails:* When the experimental population differs fundamentally from the observational population (different confounders, different treatment selection mechanisms), or when the experiment is too small to estimate the bias function reliably.

#### Key Findings

- Demonstrates that **experimental grounding** substantially improves treatment effect estimation compared to using either source alone
- Provides theoretical guarantees: under transferable confounding, the grounded estimator is consistent and asymptotically normal
- Empirically validated on benchmark causal inference datasets (IHDP, Jobs)
- Conceptually foundational for modern "data fusion" approaches (see §6.4 and §8)

#### How This Differs from PIE

| Dimension | Experimental Grounding (Kallus et al.) | PIE |
|-----------|----------------------------------------|-----|
| **Unit of analysis** | Individual user | Campaign |
| **Role of experiment** | Correct user-level confounding in observational data | Train campaign-level prediction model |
| **Role of observational data** | Provide coverage across feature space | Not used directly — features computed from running campaigns |
| **Bias correction** | Estimated as a function $b(x)$ | Implicit in the learned $f: X \to ICPD$ |
| **Target** | $\hat{\tau}(x)$ for individuals | $\hat{\tau}_j$ for campaigns |

**Conceptual similarity:** Both methods use RCTs to calibrate predictions learned from non-experimental signal. PIE operates at the campaign level with post-determined features; experimental grounding operates at the user level with full observational data.

#### Limitations

1. **Transferability assumption:** The bias function $b(x)$ must be stable between experimental and observational populations — violated when selection mechanisms differ
2. **Small-sample bias estimation:** Bias function estimation from a small RCT is inherently noisy
3. **User-level focus:** Less directly applicable to campaign-level measurement (use PIE for that)
4. **Dimensionality:** With high-dimensional covariates, bias-function estimation requires strong regularisation or structural assumptions

---

### 4.6 Shrinkage Estimators Combining Experimental and Observational Data

**Source:** Rosenman, Basse, Owen, Baiocchi (2020/2023). "Combining Observational and Experimental Datasets Using Shrinkage Estimators." [arXiv:2002.06708](https://arxiv.org/abs/2002.06708), *Biometrics*.

#### Problem

Suppose you have **two estimates** of the same treatment effect: one from an experiment (low bias, high variance) and one from observational data (possibly biased, low variance). The experimental estimate is trustworthy but noisy; the observational estimate is precise but may be wrong. How do you combine them?

#### Method

Rosenman et al. propose **James-Stein-style shrinkage estimators** that combine experimental and observational ATE estimates:

1. **Compute both estimates:** $\hat{\tau}_{exp}$ (from RCT) and $\hat{\tau}_{obs}$ (from observational data), each with their own standard errors
2. **Shrink observational toward experimental:** Construct a combined estimator:
   $\hat{\tau}_{combined} = \lambda \hat{\tau}_{exp} + (1 - \lambda) \hat{\tau}_{obs}$
   Where $\lambda$ depends on the relative precisions and the (estimated) bias of $\hat{\tau}_{obs}$
3. **Choose $\lambda$ optimally:** The paper derives data-driven shrinkage weights that minimise mean squared error under different bias-variance regimes
4. **Multiple strata:** Extends naturally to settings with multiple subgroups, shrinking each subgroup's observational estimate toward the corresponding experimental estimate

**Key insight:** If the observational estimate is approximately unbiased, the shrinkage estimator gives most weight to the precise observational estimate. If the observational estimate is severely biased, the shrinkage estimator gives most weight to the unbiased experimental estimate. James-Stein theory guarantees that the combined estimator beats either one alone under certain conditions.

#### Assumptions

| Assumption | Formal Statement | What Happens If Violated | How to Check |
|------------|------------------|-------------------------|--------------|
| **Valid experimental estimate** | $E[\hat{\tau}_{exp}] = \tau$ (unbiased) | Anchor is wrong; shrinkage can't save it | Standard RCT checks |
| **Estimable observational bias** | $b = E[\hat{\tau}_{obs}] - \tau$ can be estimated from the experiment | Suboptimal shrinkage weight | Compare point estimates |
| **Same estimand** | Experimental and observational estimates target the same treatment effect (same population, same contrast) | Combining misleading quantities | Careful specification |
| **Reasonable stratification** | When using multiple strata, strata are comparable across sources | Biased stratum estimates | Balance checks |

*When it works:* When you have both experimental and observational estimates for the same treatment effect, and the observational estimate is either approximately unbiased or has a bias that can be estimated from the experimental–observational gap.

*When it fails:* When observational bias is large and varies in unknown ways across subgroups; when experimental and observational estimates target subtly different estimands; when either estimate has extreme variance relative to the other.

#### Key Findings

- Shrinkage estimators **strictly dominate** using either experimental or observational data alone under standard bias-variance conditions
- Effective when you have reasonably precise observational data with modest bias — the combination reduces variance substantially while controlling bias
- Extends to **stratified** settings: can shrink observational ATE estimates for each subgroup toward corresponding experimental ATEs
- Simpler to implement than Bayesian data-fusion methods while achieving similar efficiency gains

#### How This Differs from PIE

| Dimension | Shrinkage (Rosenman et al.) | PIE |
|-----------|----------------------------|-----|
| **Goal** | Combine aggregate ATE estimates | Predict per-campaign ATE |
| **Input** | Two estimates of the same $\tau$ | Campaign-level features + RCT labels |
| **Output** | Single combined ATE (or stratum-wise ATEs) | Campaign-specific predictions |
| **Heterogeneity** | Handled via stratification | Handled via ML features |
| **Complexity** | Low — closed-form shrinkage weights | High — ML model + feature engineering |

**Key distinction:** Shrinkage operates on **aggregate ATEs**. It produces one (or a few) combined estimates, not per-campaign predictions. PIE produces a prediction for every campaign. Shrinkage is simpler but less granular.

#### Limitations

1. **Aggregate-level only:** Doesn't provide campaign-specific predictions — use PIE (§4.1) for that
2. **Requires observational estimate to target same estimand:** In practice, observational attribution often targets a correlational quantity, not ATE — shrinkage doesn't help in that case
3. **Bias estimation via experimental–observational gap:** The gap itself is noisy, which propagates into suboptimal shrinkage weights when experimental sample is small
4. **Assumes common treatment contrast:** Works best when observational and experimental data reflect the same treatment comparison, which is often not the case with attribution models

---

### 4.7 Target Trial Emulation

**Source:** Hernán & Robins (2016). "Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available." *American Journal of Epidemiology*, 183(8), 758–764.

#### Problem

Observational methods fail when researchers don't think carefully about what hypothetical experiment they're approximating. Ambiguous eligibility criteria, misaligned treatment definitions, and inappropriate time-zero definitions all contribute to the systematic failures documented in §2.2. Can you make observational causal inference more rigorous by explicitly emulating the design of a hypothetical RCT?

#### Method

**Target trial emulation** is a framework (not a statistical method per se) for disciplined observational causal inference:

1. **Specify the target trial:** Write out the full protocol of the hypothetical RCT you would run if you could:
   - Eligibility criteria
   - Treatment strategies (precisely defined, including timing)
   - Assignment procedure (what would randomisation look like?)
   - Follow-up period and outcome definitions
   - Causal contrasts of interest (ITT, per-protocol)
2. **Emulate the target trial in observational data:** Apply each element of the protocol to your observational data:
   - Eligibility → filtering rules
   - Treatment strategies → precisely defined exposure windows
   - Assignment → emulated via matching, weighting, or conditioning (controlling for confounders)
   - Follow-up → time-zero aligned with the treatment decision point
3. **Estimate causal effects:** Apply standard causal inference methods (IPW, g-methods, matching) to the emulated trial
4. **Compare with the target:** Explicitly discuss deviations between the emulated and target trial and their potential impact

**Key innovation:** The emulation forces you to make every design choice explicit — eligibility, time-zero, exposure definition, outcome — in a way that standard observational analyses often leave implicit. Many observational biases (immortal-time bias, selection on post-treatment variables) arise from implicit design choices that target trial emulation exposes.

#### Assumptions

| Assumption | Formal Statement | What Happens If Violated | How to Check |
|------------|------------------|-------------------------|--------------|
| **Unconfoundedness** | Conditional on measured covariates, treatment is as-if random | Biased estimates (same as any observational method) | Sensitivity analysis; negative controls |
| **Positivity** | Every eligible unit has positive probability of each treatment | Extreme weights; identification failure | Check covariate overlap |
| **Well-defined treatment** | Treatment strategy precisely specified | Ambiguous causal contrast; misinterpretation | Explicit protocol specification |
| **Consistent time-zero** | Observational time-zero matches the hypothetical trial's randomisation point | Immortal-time bias, lead-time bias | Pre-registered protocol |
| **Complete covariate measurement** | All confounders observed at time-zero | Residual confounding | Sensitivity analysis |

*When it works:* When you have rich longitudinal observational data with clear treatment timing and a well-defined outcome. Widely used in epidemiology (drug safety and effectiveness studies), health policy, and health services research. Can dramatically reduce common biases that plague ad-hoc observational analyses.

*When it fails:* The fundamental unconfoundedness assumption is still required — target trial emulation makes design choices explicit but **does not solve unobserved confounding**. In ad measurement contexts where selection into exposure is strongly driven by unobservables (ad targeting algorithms), target trial emulation still fails for the same reasons as other observational methods.

#### Key Findings

- **Widely adopted in epidemiology** — has become the de facto standard framework for observational comparative effectiveness research
- **Reduces avoidable biases** like immortal-time bias, selection bias from post-treatment conditioning, and time-zero misalignment
- **Transparent and reproducible:** The explicit protocol forces all assumptions to be documented, enabling external review
- **Less common in advertising:** Ad measurement settings rarely use target trial emulation — partly because unconfoundedness is so badly violated that no framework can save observational estimates (§2.2)

| Feature | Ad-Hoc Observational | Target Trial Emulation |
|---------|---------------------|-----------------------|
| Eligibility | Often implicit | Explicit |
| Time-zero | Often ambiguous | Aligned with treatment decision |
| Treatment strategy | Often loosely defined | Precisely specified |
| Deviations from target | Not discussed | Explicitly analysed |
| Common biases | Immortal-time, selection | Much reduced |
| Unobserved confounding | Unaddressed | Still unaddressed |

#### Limitations

1. **Not a new statistical method:** Target trial emulation is a design framework — still requires unconfoundedness for validity
2. **Doesn't solve unobserved confounding:** In advertising, unobserved confounding is the dominant failure mode; target trial emulation can't fix it
3. **Requires rich longitudinal data:** Observational datasets must have sufficient detail on eligibility, exposure, and outcome timing
4. **Time-intensive:** Writing an explicit trial protocol is a substantial upfront investment
5. **Less common in industry:** The framework originated in clinical epidemiology and hasn't been widely adopted in marketing

**Relation to other methods:** Can be combined with DML or doubly-robust estimation within the emulated trial. Complementary to experimental grounding (§4.5) and shrinkage estimators (§4.6) — target trial emulation specifies the estimand cleanly, while those methods handle the combination of experimental and observational evidence.

---

## 5. Marketing Mix Models Calibrated by Experiments

### 5.1 Bayesian MMM with Lift Test Calibration

**Sources:**
- Meta's Robyn: [Open-source MMM](https://facebookexperimental.github.io/Robyn/)
- Google's Meridian (successor to LightweightMMM): [Meridian](https://developers.google.com/meridian)
- PyMC-Marketing: [PyMC Labs](https://www.pymc-marketing.io/)
- Jin, Wang, Sun, Chan, Koehler (2017). "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects." Google Research.
- Chan & Perry (2017). "Challenges and Opportunities in Media Mix Modeling." Google Research.
- Sun, Wang, Jin, Chan, Koehler (2017). "Geo-level Bayesian Hierarchical Media Mix Modeling." Google Research.

#### Problem

Marketing mix models (MMMs) estimate the effect of marketing spend on business outcomes using time-series regression. They cover all channels simultaneously and provide budget allocation recommendations. But without experimental calibration, MMMs are vulnerable to unobserved confounders — the same problem that plagues observational methods (§2.2).

#### Method

Modern Bayesian MMMs combine time-series regression with experimental calibration:

**1. Base MMM structure:**
$Y_t = \beta_0 + \sum_{c=1}^{C} f_c(Adstock_c(X_{c,t})) + \gamma Z_t + \epsilon_t$

Where:
- $Y_t$ = outcome (sales, conversions) at time $t$
- $X_{c,t}$ = spend on channel $c$ at time $t$
- $Adstock_c(\cdot)$ = carryover/decay function (geometric, Weibull)
- $f_c(\cdot)$ = saturation function (Hill, logistic)
- $Z_t$ = control variables (seasonality, promotions, macro trends)

**2. Bayesian priors from lift tests:**
- Run periodic lift tests (RCTs) on individual channels
- Use the experimentally measured ROAS as an **informative prior** for the corresponding $\beta_c$
- The Bayesian framework combines the prior (from the lift test) with the likelihood (from the time-series data)

**3. Prior specification:**
$\beta_c \sim N(\hat{\beta}_{c,RCT}, \sigma^2_{c,RCT})$

Where $\hat{\beta}_{c,RCT}$ is the lift test estimate and $\sigma^2_{c,RCT}$ is its variance. This anchors the MMM against confounding while allowing the time-series data to refine the estimate.

#### Assumptions

| Assumption | Formal Statement | What Happens If Violated | How to Check |
|------------|------------------|-------------------------|--------------|
| **Correct functional form** | Adstock and saturation functions correctly specified | Biased ROAS estimates, wrong budget allocation | Compare in-sample fit across functional forms |
| **No omitted confounders** (partially relaxed by calibration) | All relevant demand drivers included | Biased channel effects — calibration mitigates but doesn't eliminate | Lift test calibration; residual diagnostics |
| **Stable relationships** | Channel effects don't change over time | Stale model; wrong allocation | Rolling window estimation; periodic re-calibration |
| **Lift test validity** | Lift tests are well-designed RCTs | Biased priors contaminate the entire model | Standard RCT validation (SRM, power) |
| **Lift test recency** | Lift test results are still relevant | Stale priors; model drifts from reality | Track time since last calibration |
| **Aggregation validity** | Weekly/monthly aggregation doesn't mask important dynamics | Loss of information; Simpson's paradox | Compare results at different aggregation levels |

*When it works:* When you have periodic lift tests for major channels, stable marketing mix, and sufficient time-series data (2+ years). Works well for channel-level budget allocation decisions.

*When it fails:* When lift tests are stale (>6 months old), when the marketing mix changes rapidly (new channels, major strategy shifts), or when the functional form assumptions are wrong (e.g., assuming geometric adstock when the true decay is non-monotonic).

#### Key Findings

**Comparison of open-source Bayesian MMM frameworks:**

| Framework | Developer | Adstock Options | Saturation | Calibration | Language |
|-----------|-----------|----------------|------------|-------------|----------|
| **Robyn** | Meta | Geometric, Weibull | Hill | Lift test priors | R |
| **Meridian** | Google | Adstock + lag | Hill | Lift test priors | Python (JAX) |
| **PyMC-Marketing** | PyMC Labs | Geometric, delayed | Logistic, Hill | Lift test priors | Python (PyMC) |

**Foundational Google MMM research:**

- **Jin, Wang, Sun, Chan, Koehler (2017) — "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects":** Introduced the Bayesian framework with explicit adstock and saturation priors that underpins Meridian and PyMC-Marketing
- **Chan & Perry (2017) — "Challenges and Opportunities in Media Mix Modeling":** Surveys identifiability problems (correlated channel spends, short time series, omitted variables) and argues for combining MMM with experimental calibration — motivating the lift-test-prior approach
- **Sun, Wang, Jin, Chan, Koehler (2017) — "Geo-level Bayesian Hierarchical Media Mix Modeling":** Extends the framework to geo-level data, using hierarchical priors to share information across geographies — strongly related to the geo-experiment methodology in §2.6

**Calibration impact:**

| Scenario | Uncalibrated MMM | Calibrated MMM | Improvement |
|----------|-----------------|----------------|-------------|
| Channel ROAS estimation | ±50–200% error | ±20–40% error | 2–5× more accurate |
| Budget allocation | Often inverts true ranking | Directionally correct | Prevents major misallocation |
| Confidence intervals | Overconfident (too narrow) | Better calibrated | More honest uncertainty |

#### Limitations

1. **Structural assumptions may be wrong:** Adstock and saturation curves are parametric assumptions, not derived from data. If the true response function is non-monotonic or has threshold effects, the model will be biased.
2. **Calibration can go stale:** Lift test results from 6 months ago may not reflect current channel effectiveness. The prior becomes less informative over time.
3. **Expensive calibration:** Each channel needs its own lift test. For a platform with 10+ channels, this requires 10+ experiments — partially defeating the purpose of avoiding per-decision RCTs.
4. **Ecological fallacy:** MMMs operate at the aggregate (weekly/monthly) level. Channel-level ROAS doesn't tell you which campaigns within a channel are effective.
5. **Identifiability issues:** With correlated channel spends (common in practice), individual channel effects are poorly identified even with Bayesian priors.

---

### 5.2 The Calibration Problem

The central tension in MMM is: **how often to re-calibrate, and how many channels need their own lift test?**

| Calibration Strategy | Cost | Accuracy | Risk |
|---------------------|------|----------|------|
| **No calibration** | Zero | Poor — vulnerable to confounding | High — may invert channel rankings |
| **One-time calibration** | Low | Good initially, degrades over time | Medium — stale priors |
| **Annual calibration** | Medium | Moderate — 6-month lag | Medium |
| **Quarterly calibration** | High | Good — priors stay fresh | Low |
| **Continuous calibration** (always-on holdouts) | Very High | Best — real-time anchoring | Very Low |

**Practical guidance:**
- **Minimum viable calibration:** Lift test your top 3 channels annually. This covers 60–80% of spend for most advertisers.
- **Gold standard:** Quarterly lift tests on all channels with >5% of spend. Expensive but provides the most reliable MMM.
- **PIE + MMM hybrid:** Use PIE for campaign-level measurement within channels; use MMM for cross-channel allocation. This reduces the number of lift tests needed because PIE provides campaign-level signal that informs the MMM.

**Connection to PIE (§4.1):** MMM and PIE operate at different levels of granularity. MMM provides channel-level allocation guidance; PIE provides campaign-level measurement. They are complementary:
- MMM answers: "How much should I spend on search vs. display vs. social?"
- PIE answers: "Which specific campaigns within display are driving incremental conversions?"

**The MMM-PIE Integration Opportunity:**

A promising but underexplored approach is using PIE predictions as inputs to MMM:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MMM-PIE INTEGRATION                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRADITIONAL MMM:                                                           │
│  Spend_channel_t → [Adstock] → [Saturation] → Sales_t                      │
│  Problem: Channel-level only; can't distinguish good vs bad campaigns       │
│                                                                             │
│  PIE-INFORMED MMM:                                                          │
│  Campaign_features → [PIE] → ICPD_campaign → Aggregate to channel          │
│  Channel_ICPD_t → [Adstock] → [Saturation] → Sales_t                       │
│  Advantage: Campaign-level signal feeds channel-level model                 │
│                                                                             │
│  BENEFIT: MMM priors are informed by PIE's campaign-level predictions,      │
│  reducing the need for frequent lift tests. PIE provides a "continuous      │
│  calibration signal" that keeps MMM anchored.                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Transportability and External Validity

### 6.1 Transportability

**Sources:**
- Pearl & Bareinboim (2011). "Transportability of Causal and Statistical Relations: A Formal Approach." [*Proceedings of the AAAI Conference on Artificial Intelligence*](https://doi.org/10.1609/aaai.v25i1.7861)
- Bareinboim & Pearl (2014). "Transportability from Multiple Environments with Limited Experiments: Completeness Results." [*Advances in Neural Information Processing Systems*](https://proceedings.neurips.cc/paper/2013/hash/0ff8033cf9437c213ee13937b0c9e8eb-Abstract.html)

#### Problem

You've run an RCT in population A (e.g., US users, Q1 2024). Can you apply the results to population B (e.g., EU users, Q3 2024) without running a new experiment? This is the **transportability** problem.

#### Method

Pearl and Bareinboim provide a formal framework using **causal DAGs** (directed acyclic graphs):

1. **Specify the causal graph:** Draw the DAG representing the causal relationships between treatment, outcome, covariates, and selection variables
2. **Identify differences:** Mark which variables differ between the source (experimental) and target (new) populations using "selection nodes" $S$
3. **Derive transport formula:** Using do-calculus, determine whether the causal effect in the target population can be expressed as a function of quantities estimable from the source experiment and observational data in the target population
4. **Compute the transported effect:** If transportable, apply the formula:

$P_t(Y | do(X)) = \sum_z P_s(Y | do(X), Z=z) \cdot P_t(Z=z)$

Where subscripts $s$ and $t$ denote source and target populations, and $Z$ are the variables that need adjustment.

**Intuition:** If the only difference between populations is the distribution of covariates $Z$ (not the causal mechanisms), you can reweight the experimental results by the target population's covariate distribution.

#### Assumptions

| Assumption | Formal Statement | What Happens If Violated | How to Check |
|------------|------------------|-------------------------|--------------|
| **Known causal graph** | DAG correctly represents causal relationships | Transport formula may be wrong | Domain expertise; sensitivity analysis |
| **Known differences** | All variables that differ between populations are identified | Unaccounted differences bias the transported estimate | Compare population characteristics |
| **Mechanism stability** | Causal mechanisms (conditional distributions) are the same across populations | Transport is impossible for those mechanisms | Test on overlapping subpopulations |
| **Positivity** | All covariate values in target population have support in source | Extrapolation beyond experimental data | Check covariate overlap |

*When it works:* When populations differ only in observable characteristics (demographics, market conditions) and the causal mechanisms are stable. Example: transporting a drug trial from one country to another when the only difference is age distribution.

*When it fails:* When populations differ in unobservable ways (cultural attitudes, institutional context) or when the causal mechanisms themselves differ (e.g., ad effectiveness differs between platforms due to different user intent).

#### Key Findings

- Provides **necessary and sufficient conditions** for when transport is possible
- Completeness results: if the transport formula exists, do-calculus can find it; if it doesn't exist, no method can transport the effect
- Practical applications in epidemiology, policy evaluation, and (to a limited extent) marketing

#### Limitations

1. **Requires knowledge of the causal graph** — in practice, the graph is rarely known with certainty
2. **Requires knowledge of which variables differ** — often the most important differences are unobserved
3. **Mechanism stability is a strong assumption** — in marketing, ad effectiveness mechanisms vary across platforms, time periods, and competitive environments
4. **Limited practical adoption** — the framework is theoretically elegant but requires expertise in causal DAGs that most practitioners lack
5. **Does not provide uncertainty quantification** for the transported estimate

---

### 6.2 External Validity in Economics

**Sources:**
- Allcott (2015). "Site Selection Bias in Program Evaluation." [*Quarterly Journal of Economics*](https://doi.org/10.1093/qje/qju015)
- Vivalt (2020). "How Much Can We Generalize from Impact Evaluations?" [*Journal of the European Economic Association*](https://doi.org/10.1093/jeea/jvz044)
- Hotz, Imbens, Mortimer (2005). "Predicting the Efficacy of Future Training Programs Using Past Experiences at Other Locations." [*Journal of Econometrics*](https://doi.org/10.1016/j.jeconom.2005.01.024)

#### Problem

How well do RCT results generalise to new settings? This is the empirical counterpart to the formal transportability framework (§6.1).

#### Method

Each paper takes a different approach:

**Allcott (2015) — Site Selection Bias:**
- Studied energy conservation experiments (Opower) across 111 utility sites
- Found that early-adopting sites had **60% larger treatment effects** than later sites
- The sites that volunteered for experiments first were systematically different from the broader population
- Implication: RCT results from self-selected sites overstate the effect for the general population

**Vivalt (2020) — Cross-Study Generalisation:**
- Meta-analysed 635 estimates from 20 development economics interventions across multiple countries
- Found that **between-study heterogeneity is large** — effects vary substantially across settings
- The median intervention has a between-study standard deviation equal to 60–100% of the mean effect
- Implication: a single RCT is a poor predictor of the effect in a new setting

**Hotz, Imbens, Mortimer (2005) — Predicting Efficacy:**
- Used observed covariates to predict treatment efficacy of job training programs in new locations
- Conceptually close to PIE: use features of the setting to predict the treatment effect
- Found that covariate-based predictions improved over naive extrapolation but had substantial residual error

#### Assumptions

| Assumption | What Happens If Violated | Empirical Status |
|------------|-------------------------|-----------------|
| **Effect homogeneity** | Single RCT doesn't generalise | **Violated** — effects vary 60–100% across settings (Vivalt 2020) |
| **Random site selection** | Site selection bias inflates estimates | **Violated** — early adopters have 60% larger effects (Allcott 2015) |
| **Observable moderators** | Can't predict effects in new settings | **Partially satisfied** — covariates help but don't eliminate heterogeneity |

*When it works:* When the new setting is similar to the experimental setting on observable characteristics, and when observable characteristics explain most of the between-setting variation in effects.

*When it fails:* When effects are driven by unobservable setting-level characteristics (institutional quality, cultural factors, competitive dynamics).

#### Key Findings

- **External validity is empirically fragile** — effects vary substantially across sites, countries, and time periods
- **Site selection bias is real** — experiments are run in settings where effects are expected to be large
- **Covariate-based prediction helps but is insufficient** — Hotz et al. (2005) show that observable features predict some variation but leave substantial residual uncertainty
- These findings motivate PIE's approach: rather than assuming effects transport, learn the mapping from features to effects using a large corpus of experiments

#### Limitations

- Most evidence comes from development economics and energy conservation — may not directly apply to digital advertising
- The "how much can we generalise?" question doesn't have a universal answer — it depends on the intervention and the settings
- Covariate-based prediction (Hotz et al.) is limited by the quality and completeness of available covariates

---

### 6.3 Sample-to-Population Generalisation

**Source:** Hartman, Grieve, Ramsahai, Sekhon (2015). "From Sample Average Treatment Effect to Population Average Treatment Effect on the Treated: Combining Experimental with Observational Studies to Estimate Population Treatment Effects." *Journal of the Royal Statistical Society Series A*, 178(3), 757–778.

#### Problem

RCTs estimate the **sample average treatment effect (SATE)** — the effect in the experimental sample. But decision-makers usually care about the **population average treatment effect (PATE)** or the **population average treatment effect on the treated (PATT)** — the effect in the broader target population (e.g., all users who would be exposed to a campaign, not just the experimental sample). When experimental samples are selected non-randomly (site selection bias, volunteer effects), the SATE can differ substantially from the PATE/PATT.

#### Method

Hartman et al. provide a **formal framework** for generalising from the SATE to the PATT using trial and observational data:

1. **Identify the target population:** The population to which you want to generalise (e.g., "all users the platform would treat if the campaign were deployed")
2. **Measure covariates in both samples:** Collect pre-treatment covariates in both the experimental sample and the target population (often from observational data)
3. **Derive identification conditions:** Under conditional independence assumptions, the PATT can be identified as a function of the experimental conditional average treatment effects (CATEs) and the covariate distribution in the target population
4. **Estimate via weighting or outcome modelling:**
   - **Weighting:** Reweight experimental units by the inverse of their sampling probability into the experimental sample, using the target population as reference
   - **Outcome modelling:** Fit $\hat{\tau}(x)$ on experimental data; average over the target population's covariate distribution

**Key identification condition:** The experimental sample must have the same CATE function as the target population, conditional on observed covariates. This is weaker than assuming the same ATE (which would require identical covariate distributions), but stronger than transportability in causal-graph terms.

#### Assumptions

| Assumption | Formal Statement | What Happens If Violated | How to Check |
|------------|------------------|-------------------------|--------------|
| **Ignorable sampling** | Selection into experimental sample ignorable given covariates: $W_{sample} \perp \tau(x) \mid X$ | PATT estimate biased | Compare sample vs. population covariate distributions |
| **Covariate overlap** | Target-population covariate values have support in the experimental sample | Extrapolation bias | Propensity-score overlap |
| **Stable CATE** | CATE function $\tau(x)$ same in experimental and target populations | Biased generalisation | Test on subgroups with both experimental and observational data |
| **Well-measured covariates** | All relevant moderators measured in both datasets | Residual sampling bias | Sensitivity analysis |

*When it works:* When the experimental sample's selection can be explained by observable covariates, and the same covariates are measured in the target population. Provides a formal pathway from sample-level to population-level inference.

*When it fails:* When sample selection is driven by unobservables (the selected sites have different effect sizes for reasons not captured by measured covariates); when the target population has covariate values outside the experimental support.

#### Key Findings

- Provides **identification conditions and estimators** for the PATT given trial + observational data — a formal counterpart to Allcott's empirical findings on site selection bias (§6.2)
- Shows that ignorable sampling on observables is a testable (and often violated) condition
- Empirically illustrated on policy evaluation applications; identifies when PATT can vs. cannot be generalised from SATE
- **Widely cited in epidemiology** for generalising clinical trial findings to broader populations; relevant to marketing for generalising RCT findings from a targeted sample to a full campaign audience

#### How This Complements Transportability (§6.1)

| Dimension | Pearl–Bareinboim Transportability (§6.1) | Hartman et al. (§6.3) |
|-----------|------------------------------------------|-----------------------|
| **Framework** | Causal DAGs and do-calculus | Potential outcomes with ignorable sampling |
| **Population difference** | Encoded via selection nodes in DAG | Encoded via covariate distribution shift |
| **Identification** | Graphical (do-calculus) | Algebraic (weighting/modelling) |
| **Practical use** | Rigorous, theoretical | Directly implementable estimators |
| **Complementary?** | Yes — transportability identifies feasibility; Hartman et al. provides estimators |

**Complementary roles:** Transportability (§6.1) tells you *whether* generalisation is possible given a causal graph; Hartman et al. tells you *how* to do it given measured covariates. They answer different parts of the same question.

#### Limitations

1. **Ignorable sampling is strong:** In practice, sites/users self-select into experiments for reasons often not captured by measured covariates
2. **Requires target-population covariate data:** You need observational data on the target population's covariates — not always available
3. **Estimators inherit weighting problems:** IPW-style estimators have high variance when covariate distributions differ substantially
4. **Less visible in marketing:** Framework developed in epidemiology; application to advertising is straightforward but uncommon in practice

---

### 6.4 Data Fusion for Causal Inference

**Source:** Bareinboim & Pearl (2016). "Causal inference and the data-fusion problem." *PNAS*, 113(27), 7345–7352.

#### Problem

Real-world causal questions often require combining datasets collected under **heterogeneous conditions**: different populations, different sampling mechanisms, different observational vs. experimental regimes, different measurement instruments. Standard causal inference methods assume a single homogeneous dataset. How do you formally combine multiple heterogeneous sources to answer a causal query?

#### Method

Bareinboim & Pearl's **data-fusion framework** generalises transportability (§6.1) to the full problem of combining multiple datasets:

1. **Annotate each dataset's conditions:** Represent each dataset by a DAG with selection nodes marking where it differs from the target (different population, different treatment assignment mechanism, different measurement scheme)
2. **Specify the causal query:** The quantity of interest, expressed in do-calculus notation (e.g., $P_t(Y \mid do(X))$ in target population $t$)
3. **Apply do-calculus:** Determine whether the query can be expressed as a function of quantities estimable from the available datasets
4. **Complete fusion algorithm:** Bareinboim & Pearl provide a **complete algorithm** for deciding data fusion — if the query is answerable from the available data under the encoded assumptions, the algorithm will find an expression; if not, it proves impossibility

**Key contribution:** A unified formal framework that subsumes:
- Transportability (different populations)
- Selection bias (non-random sampling)
- Combining experimental and observational data
- Meta-analysis (multiple studies with different designs)
- Measurement error and surrogates

**Completeness:** If the causal query is identifiable from the given data, the data-fusion algorithm finds the identification formula. If the query is not identifiable, the algorithm proves non-identifiability. This is a strong theoretical guarantee.

#### Assumptions

| Assumption | Formal Statement | What Happens If Violated | How to Check |
|------------|------------------|-------------------------|--------------|
| **Correct causal graph** | DAG correctly represents causal relationships, selection mechanisms, and sampling in each dataset | Wrong fusion formula | Domain expertise; sensitivity analysis |
| **Correctly encoded heterogeneity** | All differences across datasets represented via selection nodes | Unaccounted heterogeneity | Explicit modelling of known differences |
| **Positivity across sources** | Relevant subgroups have positive probability in the combining datasets | Identification fails | Check overlap |

*When it works:* When you have a well-specified causal graph and explicit knowledge of how each dataset differs from the target. Provides a rigorous foundation for modern data-fusion methods (combining RCTs with observational data, combining multiple RCTs across populations, combining RCTs with administrative records).

*When it fails:* When the causal graph is unknown or misspecified; when differences between datasets are not known and cannot be encoded; when positivity conditions fail.

#### Key Findings

- Provides the **theoretical foundation** for modern data-fusion methods, including experimental grounding (§4.5), shrinkage estimators (§4.6), and long-term causal inference via data combination (§3.3)
- **Completeness** guarantees: the framework can decide all data-fusion problems in principle
- Unifies many specific methods (transportability, selection bias correction, meta-analysis) under one formal umbrella
- Has motivated a large subsequent literature on **practical** data-fusion estimators (Dahabreh, Kern, Colnet, and others)

| Data-Fusion Problem | Handled by Framework? |
|---------------------|----------------------|
| Transport from RCT to new population | ✓ (§6.1 special case) |
| Combine RCT + observational | ✓ (Kallus et al. §4.5, Rosenman et al. §4.6 special cases) |
| Combine multiple RCTs across populations | ✓ |
| Correct for selection bias in observational data | ✓ |
| Integrate short-term RCT + long-term observational | ✓ (§3.3 special case) |
| Fuse RCTs with differing treatment definitions | ✓ (with appropriate encoding) |

#### Limitations

1. **Requires correct causal graph:** The framework's power depends on correctly specifying the DAG — wrong graph leads to wrong fusion formula
2. **Graph specification is hard in practice:** Most applied researchers don't draw explicit DAGs; domain knowledge about selection mechanisms is often incomplete
3. **Theoretical completeness ≠ practical estimation:** The framework tells you whether a causal query is identifiable, but finite-sample estimation requires additional work
4. **Limited software support:** Direct application of the fusion algorithm typically requires custom causal-inference tooling; implementations exist but are not mainstream

**Relation to other methods:** Provides theoretical grounding for virtually every method in §4.5, §4.6, §3.3, and §6.3. Section 8 (below) discusses recent practical methods that operationalise the data-fusion framework.

---

## 7. Meta-Analysis

### 7.1 Random-Effects Meta-Analysis

**Sources:**
- DerSimonian & Laird (1986). "Meta-analysis in clinical trials." [*Controlled Clinical Trials*](https://doi.org/10.1016/0197-2456(86)90046-2)
- Higgins & Thompson (2002). "Quantifying heterogeneity in a meta-analysis." [*Statistics in Medicine*](https://doi.org/10.1002/sim.1186)
- Borenstein, Hedges, Higgins, Rothstein (2009). *Introduction to Meta-Analysis*. Wiley. [Standard reference textbook]
- Gelman, Hill, Yajima (2012). "Why We (Usually) Don't Have to Worry About Multiple Comparisons." *Journal of Research on Educational Effectiveness*, 5(2), 189–211.

#### Problem

You have results from multiple independent experiments testing similar interventions. How do you synthesise them into a single estimate, accounting for between-study heterogeneity?

#### Method

Random-effects meta-analysis models each study's effect as drawn from a distribution:

$\hat{\tau}_j = \tau + u_j + \epsilon_j$

Where:
- $\hat{\tau}_j$ = observed effect in study $j$
- $\tau$ = grand mean effect
- $u_j \sim N(0, \tau^2)$ = between-study heterogeneity
- $\epsilon_j \sim N(0, \sigma^2_j)$ = within-study sampling error

**Estimation:**
1. Estimate between-study variance $\tau^2$ (DerSimonian-Laird, REML, or Bayesian)
2. Compute study weights: $w_j = 1/(\sigma^2_j + \hat{\tau}^2)$
3. Estimate grand mean: $\hat{\tau} = \sum_j w_j \hat{\tau}_j / \sum_j w_j$

**Meta-regression** extends this by modelling how study-level moderators affect effect sizes:
$\hat{\tau}_j = \beta_0 + \beta_1 X_{j1} + ... + \beta_k X_{jk} + u_j + \epsilon_j$

Where $X_{j1}, ..., X_{jk}$ are study-level characteristics (sample size, population, intervention type).

#### Assumptions

| Assumption | What Happens If Violated | How to Check |
|------------|-------------------------|--------------|
| **Independent studies** | Underestimated standard errors | Check for shared data sources |
| **Normal random effects** | Biased $\tau^2$ estimate | Q-Q plots of study effects |
| **No publication bias** | Overestimated grand mean | Funnel plots, Egger's test |
| **Correct moderator specification** | Biased meta-regression coefficients | Sensitivity analysis |

*When it works:* When you want to summarise the average effect across a body of evidence and understand how much effects vary. Good for answering "what is the typical effect of this type of intervention?"

*When it fails:* When you need campaign-specific predictions rather than an average. Meta-analysis tells you the mean and variance of the effect distribution, but doesn't predict where a specific new campaign will fall.

#### Key Findings

- Random-effects meta-analysis is the standard approach in medicine, education, and social science
- The $I^2$ statistic quantifies the proportion of total variation due to between-study heterogeneity (vs. sampling error)
- Meta-regression can identify moderators of effect sizes, but is limited by the number of studies and the quality of moderator data
- In advertising, between-study heterogeneity is typically large ($I^2 > 75\%$), meaning the grand mean is a poor predictor for any specific campaign

#### Limitations

1. **Targets the average effect, not campaign-specific predictions** — the grand mean is useful for policy but not for individual campaign decisions
2. **Limited moderator analysis** — with 20–50 studies, meta-regression can only test a few moderators; PIE uses hundreds of features
3. **Parametric assumptions** — normal random effects may not capture the true effect distribution (e.g., bimodal effects)
4. **Publication bias** — if only significant results are published, the meta-analytic estimate is inflated
5. **Ecological inference** — study-level moderators may not reflect the mechanisms driving effect heterogeneity

---

### 7.2 PIE as Predictive Meta-Analysis

PIE (§4.1) can be viewed as a **predictive extension of meta-analysis** that addresses several of meta-analysis's limitations:

| Dimension | Traditional Meta-Analysis | PIE |
|-----------|--------------------------|-----|
| **Goal** | Estimate grand mean effect + heterogeneity | Predict campaign-specific effects |
| **Model** | Parametric random-effects (normal) | Flexible ML (random forest) |
| **Moderators** | Study-level covariates (few) | Post-determined features (many) |
| **Output** | $\hat{\tau} \pm CI$ (average) | $\hat{\tau}_j$ (campaign-specific prediction) |
| **Heterogeneity** | Estimated as $\tau^2$ (nuisance parameter) | Exploited for prediction (the signal) |
| **Number of studies** | Typically 10–50 | 500–2,000+ |
| **Feature space** | Low-dimensional (manual moderators) | High-dimensional (automated features) |

**Key conceptual shift:** In traditional meta-analysis, between-study heterogeneity ($\tau^2$) is a nuisance — it widens confidence intervals and limits the precision of the grand mean. In PIE, heterogeneity is the **signal** — it's what makes campaign-specific prediction possible. If all campaigns had the same effect, PIE would be unnecessary (and meta-analysis would suffice).

**When meta-analysis suffices:**
- You have <50 experiments and want a summary estimate
- Effect heterogeneity is low ($I^2 < 50\%$)
- You need a simple, interpretable summary for stakeholders

**When PIE is needed:**
- You have 400+ experiments and want campaign-specific predictions
- Effect heterogeneity is high ($I^2 > 75\%$)
- You need to make decisions for individual campaigns, not just "on average"

### 7.3 Methodological Foundations: Textbook Reference and Hierarchical Perspective

**Borenstein, Hedges, Higgins, Rothstein (2009) — *Introduction to Meta-Analysis*:**

The standard reference textbook for meta-analysis. Covers:
- Effect size calculation and conversion across outcome metrics
- Fixed-effect vs. random-effects models (and when to use each)
- Heterogeneity quantification ($Q$, $I^2$, $\tau^2$) and interpretation
- Subgroup analysis and meta-regression mechanics
- Publication bias detection and adjustment (funnel plots, trim-and-fill, Egger's test)
- Cumulative meta-analysis and sensitivity analyses

Referenced here as the canonical reference text that practitioners should consult for detailed implementation guidance on any meta-analytic workflow. Provides the statistical foundations underlying §7.1 and §7.2.

**Gelman, Hill, Yajima (2012) — "Why We (Usually) Don't Have to Worry About Multiple Comparisons":**

Argues that **Bayesian hierarchical models** naturally handle the "multiple comparisons problem" that plagues traditional meta-analysis and cross-experiment learning. Key ideas relevant to this review:

- When analysing many related effects (e.g., CATEs for many subgroups, or effects across many experiments), hierarchical models **shrink each estimate toward the group mean**, reducing the risk of false positives without explicit multiplicity correction
- The hierarchical shrinkage is adaptive — strong shrinkage when effects are similar, weak shrinkage when they genuinely differ
- This provides an alternative to frequentist multiplicity corrections (Bonferroni, FDR) that is often better calibrated in practice
- Directly relevant to cross-experiment learning: pooling many experiments in a hierarchical model automatically regularises per-experiment estimates toward what you'd expect across experiments

**Why this matters for the review:** Both PIE (§4.1) and cross-experiment meta-learning (§4.4) implicitly perform hierarchical pooling — the model trained across experiments regularises predictions toward the cross-experiment distribution. The Gelman–Hill–Yajima perspective provides the theoretical justification for why this pooling is statistically sound without explicit multiple-comparison correction, and for why shrinkage estimators (§4.6) are a principled way to combine heterogeneous estimates.

**Practical implication:** When building cross-experiment prediction models (PIE, meta-learning), hierarchical Bayesian models often outperform fully pooled or fully separate approaches. The shrinkage is especially valuable with small per-experiment sample sizes or many experiments covering similar interventions.

---

## 8. Recent Developments in Data Fusion and ML-Based Approaches

This section covers recent work (mostly 2023–2024) that bridges multiple categories: combining ML prediction with causal inference, pre-trained models for causal effects, and modern approaches to external validity that complement the theoretical frameworks in §6.

### 8.1 Prediction-Powered Generalisation of Causal Inferences

**Source:** "Prediction-powered Generalization of Causal Inferences" (2024). [arXiv:2406.02873](https://arxiv.org/abs/2406.02873)

#### Problem

Generalising RCT results to new populations (§6.1, §6.3) requires covariate data in the target population and an assumption that the CATE function transfers. In practice, target populations have **rich observational covariate data but no outcomes**. Can auxiliary ML models trained on abundant unlabelled or outcome-sparse data help generalise causal inferences more accurately?

#### Method

Prediction-powered generalisation combines:
1. **Experimental data** with treatment, covariates, and outcome (small)
2. **Auxiliary ML models** that predict outcomes from covariates (trained on large unlabelled or outcome-sparse data)
3. **Target-population covariate data** without outcomes (observational)

The method uses ML predictions in the target population as **synthetic outcomes** for the generalisation step, while using the experimental data to calibrate the ML predictions and correct for prediction bias. The result is a generalisation estimator that leverages the statistical power of the auxiliary ML model while maintaining the causal validity of the experimental anchor.

#### Key Findings

- Demonstrates improved **efficiency** (tighter confidence intervals) compared to standard generalisation methods (e.g., Hartman et al. §6.3) when auxiliary ML models are reasonably accurate
- Provides **valid inference** even when the auxiliary ML model is miscalibrated — the experimental data handles the correction
- Directly operationalises the data-fusion framework (§6.4) for the common case where target-population covariate data is rich but outcomes are scarce
- Framework extends naturally to settings with multiple auxiliary data sources

#### Assumptions and Limitations

| Assumption | What Happens If Violated |
|------------|-------------------------|
| Ignorable sampling into experiment (given covariates) | Biased generalisation |
| ML model inputs available in both experimental and target data | Can't apply ML predictions in target |
| Sufficient experimental sample for calibration | Residual bias from ML model |

**Relation to PIE (§4.1):** Both methods use ML predictions calibrated by experimental data, but serve different goals. PIE predicts campaign-level effects from campaign-level features using campaign-level RCTs as training data. Prediction-powered generalisation uses user-level auxiliary models to extend user-level causal inferences to new populations. They can stack: use prediction-powered generalisation within the RCTs that train PIE, to ensure each RCT's SATE is generalised to the campaign's target audience before training.

---

### 8.2 In-Context Learning for Causal Effect Estimation

**Source:** "In-Context Learning for Causal Effect Estimation" (2024). [arXiv:2506.06039](https://arxiv.org/abs/2506.06039)

#### Problem

Traditional causal inference requires the researcher to choose a method (PSM, DML, causal forests, etc.) and specify its assumptions (unconfoundedness, functional form). Each method has strengths and weaknesses, and method selection itself is difficult. Can a **pre-trained model** learn to estimate causal effects across a wide range of data-generating processes, without the user making explicit assumption choices?

#### Method

The paper uses **Prior-data Fitted Networks (PFNs)** — transformer models pre-trained on massive amounts of **synthetic causal data**:

1. **Pre-training:** Generate millions of synthetic causal inference problems with diverse data-generating processes (different confounding structures, effect heterogeneity, functional forms, sample sizes). Train a transformer to predict treatment effects from the raw data
2. **In-context learning:** At inference time, feed the model the observed data (covariates, treatment, outcome) as context. The model outputs treatment effect predictions without any explicit fitting or user-specified assumptions
3. **Amortised inference:** The pre-trained model effectively **amortises Bayesian inference** across the prior distribution of data-generating processes, producing near-optimal estimates for data drawn from that prior

**Key innovation:** Shifts causal effect estimation from "choose method + estimate parameters" to "learn a universal estimator via pre-training." Represents a new ML paradigm that doesn't require the user to specify explicit causal assumptions — the prior encodes them.

#### Key Findings

- Demonstrates competitive or superior performance compared to classical causal inference methods (DML, BART, causal forests) on benchmark datasets
- Provides **fast inference** — no per-dataset fitting once the PFN is pre-trained
- Captures **heterogeneity** across data-generating processes that no single classical method captures
- Early-stage research: promising direction but not yet mature for production use

#### Assumptions and Limitations

| Assumption | What Happens If Violated |
|------------|-------------------------|
| Real data resembles the pre-training prior | Out-of-distribution predictions may be poor |
| Pre-training computational cost acceptable | Very expensive to pre-train |
| Interpretability not required | Harder to explain than explicit methods |

**Current status:** Early-stage research. The approach is conceptually exciting but faces practical hurdles: the pre-training prior matters enormously (if real causal problems look different from the prior, performance degrades), and the method offers less interpretability than classical approaches. Worth watching as the ML community refines PFN architectures and priors.

**Relation to PIE (§4.1):** Conceptually related — PIE pre-trains a random forest on real RCT data to amortise campaign-level causal effect estimation; PFNs pre-train a transformer on synthetic causal data to amortise user-level causal effect estimation. Both shift the paradigm from per-problem fitting to amortised inference via pre-training. PFNs are more general but rely on synthetic priors; PIE is more domain-specific but trained on real RCT labels.

---

### 8.3 Externally Valid Policy Evaluation

**Source:** "Externally Valid Policy Evaluation from Randomized Trials Using Additional Observational Data" (2023). [arXiv:2310.14763](https://arxiv.org/abs/2310.14763)

#### Problem

Policy decisions require knowing the effect of interventions in the **target population** where the policy will be deployed, not just the RCT sample. Standard generalisation methods (§6.1, §6.3) rely on strong assumptions about the stability of the CATE function across populations. Can we build a more practical framework that uses **trial data + observational covariate data** in the target population to evaluate policies with valid external validity guarantees?

#### Method

The paper provides a framework that:
1. **Uses experimental data** to identify treatment effects and validate unconfoundedness-like assumptions
2. **Uses observational covariate data** from the target population to characterise the distribution shift
3. **Estimates policy values** — not just ATEs but the expected outcome under specific policies (e.g., "treat everyone with covariate $X$ above threshold $c$")
4. **Provides confidence regions** with valid frequentist coverage under stated assumptions

**Key contribution:** Emphasises **policy evaluation** rather than just ATE estimation — the downstream question decision-makers actually care about. Moves beyond identification theory (§6.1) to provide practical estimators with finite-sample guarantees.

#### Assumptions

| Assumption | Formal Statement | What Happens If Violated |
|------------|------------------|-------------------------|
| **Transferable CATE** | CATE function $\tau(x)$ same in experimental and target populations | Policy values biased |
| **Covariate overlap** | Target population covariate values supported in experimental sample | Extrapolation bias |
| **Measured covariates drive heterogeneity** | Relevant moderators observed in both datasets | Residual sampling bias |

#### Key Findings

- **More recent and practical** than Pearl & Bareinboim's formal theory (§6.1) — provides directly implementable estimators with proven properties
- Focuses on **policy evaluation** — the expected outcome under a specific assignment rule — rather than just average effects
- Explicitly handles **covariate shift** between experimental and target populations
- Provides **finite-sample confidence regions**, not just asymptotic guarantees
- Bridges the gap between the theoretical transportability framework and applied practice

#### Limitations

1. **Still requires transferable CATE assumption:** The CATE function must be stable between experimental and target populations — often not verifiable
2. **Policy-specific:** Framework focused on policy evaluation; ATE generalisation uses other tools (§6.1, §6.3)
3. **Recent, not yet widely adopted:** Newer framework with less empirical validation than classical transportability
4. **Observational covariate quality matters:** If target-population covariates are measurement-error-prone, the policy evaluation is biased

**Relation to other methods:** Operationalises the transportability framework (§6.1) and sample-to-population framework (§6.3) for policy decisions. Complementary to PIE (§4.1) — PIE predicts campaign-level effects; this framework evaluates what would happen under specific targeting policies.

---

### 8.4 Summary: How Section 8 Methods Relate

These three methods share a theme: **pre-trained or auxiliary ML models + experimental calibration → better extrapolation**.

| Method | Auxiliary ML Role | Experimental Calibration Role | Output |
|--------|-------------------|------------------------------|--------|
| Prediction-powered generalisation (§8.1) | Predicts outcomes in target population | Corrects ML prediction bias | Generalised ATE with tight CI |
| In-context learning / PFNs (§8.2) | Pre-trained causal estimator | Not required (implicit in prior) | Treatment effect predictions |
| Externally valid policy evaluation (§8.3) | Target-population covariate distribution | Anchors treatment effects | Policy value estimates with CI |

These methods, taken together, represent the current frontier of data fusion for causal inference. They complement PIE (§4.1) by operating at different units of analysis (user-level vs. campaign-level) and by formalising the assumptions needed to extrapolate beyond the experimental sample.

**Practical implication:** For advertising applications specifically:
- **PIE (§4.1)** remains the central method for campaign-level measurement at scale
- **§8.1 methods** can improve the generalisation of each training RCT to its campaign's target audience
- **§8.2 methods** are early-stage and worth monitoring but not yet production-ready for ad measurement
- **§8.3 methods** are directly applicable to evaluating targeting policies where RCT samples don't match the target audience

---

## 9. Master Comparison Table

| Method | § | Category | Requires New RCT? | Unit of Analysis | Key Assumption | Scalability | Accuracy vs RCT | Complexity | Best Use Case |
|--------|---|----------|--------------------|-----------------|----------------|-------------|-----------------|------------|---------------|
| **Ghost Ads** | 2.1 | Quasi-Experimental | Yes (cheaper) | User | Auction fidelity | Medium | High (validated) | Medium | Reducing RCT cost 10× |
| **PSM / Stratification** | 2.2 | Observational | No | User | Unconfoundedness | High | **Poor** (60–175% error) | Low | **Don't use for causal claims** |
| **DML / SPSM** | 2.3 | Observational | No | User | Unconfoundedness | High | **Poor** (still large errors) | Medium | **Don't use for causal claims** |
| **Surrogate Index** | 3.1 | Surrogate | Yes (shorter) | User | Prentice surrogacy | Medium | Good (within CI) | Medium | Early reads on long-term effects |
| **Auto-Surrogates (Netflix)** | 3.2 | Surrogate | Yes (shorter) | User | Temporal surrogacy | Medium | Good (~95% consistency) | Low-Medium | Faster experiment decisions |
| **PIE** | 4.1 | Cross-Experiment | **No** | Campaign | Feature invariance, RCT representativeness | **High** | **High** (R² = 0.88) | High | Campaign-level measurement at scale |
| **Amazon MTA** | 4.2 | Cross-Experiment | **No** | Touchpoint | PIE assumptions + touchpoint model | **High** | Good (calibrated) | Very High | Touchpoint-level attribution |
| **Causal ML / Uplift** | 4.3 | Within-Experiment | Yes (per campaign) | User | RCT validity | Low (per campaign) | Good (within experiment) | High | User-level targeting |
| **Cross-Experiment Meta-Learning** | 4.4 | Cross-Experiment | No (pools existing) | User | Shared moderators | Medium | Moderate | High | User-level personalisation |
| **Bayesian MMM + Lift Tests** | 5.1 | Structural | Periodic | Channel × Time | Functional form, calibration recency | Medium-High | Moderate (±20–40%) | High | Channel-level budget allocation |
| **Transportability** | 6.1 | Formal Framework | No (reuses existing) | Population | Known causal graph, mechanism stability | Low | Depends on graph | Low (theory), High (practice) | Formal extrapolation |
| **External Validity (Empirical)** | 6.2 | Empirical | No (reuses existing) | Site/Setting | Observable moderators | Low-Medium | Moderate | Low-Medium | Understanding generalisability |
| **Random-Effects Meta-Analysis** | 7.1 | Synthesis | No (synthesises) | Study | Independent studies, no publication bias | Medium | Average effect only | Low | Summarising evidence base |
| **PIE as Predictive Meta-Analysis** | 7.2 | Synthesis + Prediction | **No** | Campaign | PIE assumptions | **High** | **High** (campaign-specific) | High | Campaign-specific predictions |

### Reading the Table

- **"Don't use for causal claims"** for observational methods is based on Gordon et al. (2019, 2023) — these methods systematically fail validation against RCTs in advertising settings
- **Accuracy vs RCT** reflects empirical validation, not theoretical properties
- **Scalability** refers to the marginal cost of measuring one additional campaign/intervention
- **Complexity** refers to implementation and maintenance burden

---

## 10. Practitioner Recommendations

### 10.1 For Ad Platforms

**If you have 400+ historical RCTs:** Implement PIE.
- Train a random forest on campaign-level features with RCT-measured ICPD as the label
- Use post-determined features (test-group outcomes, exposure rates, last-click conversions)
- Validate with held-out RCTs; monitor R² over time
- Refresh the model quarterly to prevent concept drift
- Expected outcome: R² ≈ 0.85–0.90 for ICPD; 8–12% decision disagreement with RCTs

**If you have periodic lift tests (but not hundreds of RCTs):** Calibrate MMM with Bayesian priors.
- Use lift test ROAS as informative priors for channel-level effects
- Re-calibrate at least annually; quarterly is better
- Use for channel-level allocation, not campaign-level measurement
- Expected outcome: ±20–40% accuracy on channel ROAS

**If you have neither:** Ghost ads are the cheapest experimental option.
- Implement counterfactual ad logging in your auction system
- Run ghost ad experiments for your largest campaigns
- Build toward a PIE-capable corpus over 1–2 years
- Expected outcome: 10× cost reduction vs. PSA tests

**Don't use observational methods alone.**
- Gordon et al. (2019, 2023) show 60–175% errors vs. RCTs
- Modern methods (DML, SPSM) don't fix the fundamental problem (unobserved confounding)
- If you must use observational methods, treat them as directional signals only — never as causal estimates

### 10.2 For Advertisers

**Push your platform for RCT-calibrated measurement.**
- Ask whether the platform's attribution is calibrated against experiments
- If the platform offers PIE-like predictions, request validation metrics (R², decision consistency)
- If the platform only offers last-click attribution, know that it systematically overstates effects by ~33% on average (PIE paper)

**Validate against periodic RCTs.**
- Even if you trust PIE predictions, run 2–4 RCTs per year as validation
- Focus validation on your largest campaigns and any campaigns where PIE predictions seem surprising
- Use the RCT results to update your confidence in the PIE model

**Understand what you're getting:**

| Platform Offering | What It Actually Measures | Reliability |
|-------------------|--------------------------|-------------|
| Last-click attribution | Correlation, not causation | Poor — ~33% overstatement |
| View-through attribution | Even more biased than last-click | Very Poor |
| PIE-based incrementality | Predicted causal effect (calibrated to RCTs) | Good — R² ≈ 0.88 |
| Lift test (RCT) | True causal effect | Gold standard |
| MMM | Channel-level causal effect (if calibrated) | Moderate — depends on calibration |

### 10.3 Method Stacking

Methods from different categories can be combined for better coverage:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    METHOD STACKING: How Layers Interact                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LAYER 1: WITHIN-EXPERIMENT ACCELERATION                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Surrogate Index → Auto-Surrogates → CUPED (from MDE review)       │    │
│  │  Purpose: Get faster reads from the RCTs you DO run                 │    │
│  │  Result: 2–4× faster experiment decisions                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                              │
│  LAYER 2: CROSS-EXPERIMENT PREDICTION                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PIE → Trained on RCT corpus → Predicts non-RCT campaigns          │    │
│  │  Purpose: Measure campaigns WITHOUT running new RCTs                │    │
│  │  Result: Campaign-level ICPD for 100% of campaigns                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                              │
│  LAYER 3: CHANNEL-LEVEL ALLOCATION                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  MMM calibrated by lift tests (and informed by PIE)                 │    │
│  │  Purpose: Allocate budget across channels                           │    │
│  │  Result: Optimal marketing mix                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                              │
│  LAYER 4: PERIODIC VALIDATION                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  RCTs on selected campaigns → Validate PIE → Refresh MMM priors    │    │
│  │  Purpose: Prevent concept drift, maintain calibration               │    │
│  │  Result: Trustworthy measurement system                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  COMBINED SYSTEM:                                                           │
│  • Surrogates accelerate the RCTs you run (Layer 1)                         │
│  • PIE extends measurement to all campaigns (Layer 2)                       │
│  • MMM guides channel allocation (Layer 3)                                  │
│  • Periodic RCTs keep everything calibrated (Layer 4)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Recommended stacking configurations:**

| Organisation Type | Layer 1 | Layer 2 | Layer 3 | Layer 4 |
|-------------------|---------|---------|---------|---------|
| **Large ad platform** (Meta, Amazon, Google) | Surrogates + CUPED | PIE (full) | MMM (internal) | Continuous RCTs |
| **Mid-size platform** (100–400 RCTs) | Surrogates | PIE (limited) | MMM (Robyn/Meridian) | Quarterly RCTs |
| **Large advertiser** (uses platform tools) | N/A | Platform PIE | MMM (external) | Annual lift tests |
| **Small advertiser** (<$1M/year spend) | N/A | Platform attribution | N/A | Occasional lift tests |

---

## 11. Open Research Questions

### 11.1 Concept Drift Detection and Correction in PIE

PIE assumes the feature-to-effect mapping $f: X \to ICPD$ is stable over time. In practice, platforms change their ad serving algorithms, user behaviour evolves, and competitive dynamics shift. **How do you detect when the PIE model is going stale?**

Potential approaches:
- Monitor out-of-sample R² on a rolling window of recent RCTs
- Use distributional shift detection (e.g., MMD, KS tests) on feature distributions
- Implement online learning with exponential decay on older RCTs
- Maintain a "concept drift alarm" that triggers re-training when prediction residuals increase

### 11.2 Optimal RCT Sampling for PIE Training

Not all RCTs are equally informative for PIE. **Which campaigns should you experiment on to maximise PIE's predictive performance?**

This is an active learning problem:
- Experiment on campaigns in underrepresented regions of the feature space
- Prioritise campaigns where PIE's uncertainty is highest
- Balance exploration (informative for PIE) with exploitation (high-value campaigns that need measurement)
- Formal frameworks: Bayesian optimal experimental design, active learning with Gaussian processes

### 11.3 Privacy-Preserving PIE

Differential privacy constraints limit the precision of individual-level data. **How does PIE perform when features are computed under differential privacy?**

Key questions:
- How much noise does DP add to post-determined features?
- Can PIE be trained on DP-protected RCT outcomes?
- What is the privacy-accuracy tradeoff for PIE predictions?
- Can federated learning enable cross-platform PIE without sharing raw data?

### 11.4 Extending PIE Beyond Advertising

PIE was developed for ad campaign measurement, but the framework is general. **Can it be applied to product features, pricing, or policy interventions?**

Potential applications:
- **Product features:** Predict the effect of a new feature based on features of past A/B tests (feature type, user segment, metric)
- **Pricing:** Predict the demand elasticity of a price change based on historical pricing experiments
- **Policy:** Predict the effect of a policy intervention based on features of past policy RCTs (Vivalt 2020 provides the motivation)

Challenges:
- Feature engineering is domain-specific — post-determined features for ads may not transfer
- The RCT corpus may be smaller in non-advertising domains
- Treatment effect heterogeneity may be structured differently

### 11.5 Uncertainty Quantification for PIE Predictions

PIE currently provides point predictions. **How do you construct valid confidence intervals for PIE predictions?**

Approaches:
- **Conformal prediction:** Distribution-free prediction intervals based on calibration residuals
- **Bayesian random forests:** Replace the standard random forest with a Bayesian variant that provides posterior predictive distributions
- **Quantile regression forests:** Predict quantiles of the ICPD distribution, not just the mean
- **Ensemble disagreement:** Use the variance across trees in the random forest as an uncertainty estimate (but this is not calibrated)

The challenge: standard ML uncertainty quantification assumes i.i.d. data, but PIE's training data (RCTs) may have complex dependencies (same advertisers, overlapping time periods).

### 11.6 Cross-Platform Transportability

**Can PIE trained on Meta data predict for Google campaigns? Can Amazon MTA models transfer to other retail platforms?**

This combines PIE (§4.1) with transportability (§6.1):
- Platform-specific features (auction mechanics, user base, ad formats) may not transfer
- Shared features (advertiser vertical, campaign objective, spend level) may transfer
- Requires either cross-platform RCTs or a formal transportability analysis
- Privacy and competitive concerns limit data sharing between platforms

---

## Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **ATE** (Average Treatment Effect) | The average causal effect of treatment across the entire population: $ATE = E[Y(1) - Y(0)]$ |
| **ATT** (Average Treatment Effect on the Treated) | The average causal effect among units that actually received treatment: $ATT = E[Y(1) - Y(0) \mid W=1]$ |
| **CATE** (Conditional Average Treatment Effect) | The average treatment effect conditional on covariates: $CATE(x) = E[Y(1) - Y(0) \mid X=x]$ |
| **Concept drift** | A change in the statistical relationship between features and outcomes over time, causing a trained model to degrade |
| **DML** (Double/Debiased Machine Learning) | A method that uses ML for nuisance parameter estimation with cross-fitting to avoid regularisation bias (Chernozhukov et al. 2018) |
| **Ghost ads** | An experimental design where the control group's counterfactual ad exposure is logged but not shown, reducing the cost of ad experiments (Johnson et al. 2017) |
| **ICPD** (Incremental Conversions Per Dollar) | The number of additional conversions caused by an ad campaign per dollar of ad spend — the key outcome metric in PIE |
| **Incrementality** | The causal effect of an intervention — the difference between what happened and what would have happened without the intervention |
| **Last-click attribution** | A heuristic that assigns 100% of conversion credit to the last ad touchpoint before conversion. Biased because it conflates correlation with causation |
| **Lift test** | An RCT designed to measure the incremental effect ("lift") of an ad campaign by comparing a treatment group (sees ads) to a control group (doesn't see ads) |
| **MMM** (Marketing Mix Model) | A time-series regression model that estimates the effect of marketing spend on business outcomes across channels |
| **MTA** (Multi-Touch Attribution) | A model that assigns conversion credit to multiple ad touchpoints along the customer journey |
| **PIE** (Predicted Incrementality by Experimentation) | A method that trains ML models on RCT outcomes to predict causal effects for non-experimental campaigns (Gordon et al. 2026) |
| **Post-determined features** | Campaign-level aggregates (e.g., test-group outcomes, last-click conversions) that are endogenous but used as predictors in PIE because RCTs handle identification |
| **PSA test** (Public Service Announcement test) | An ad experiment where the control group sees a filler (PSA) ad instead of the focal ad. Expensive because it wastes ad inventory |
| **ROAS** (Return on Ad Spend) | Revenue generated per dollar of ad spend. Can be measured as total ROAS (correlation) or incremental ROAS (causal) |
| **Surrogacy assumption** (Prentice criterion) | The assumption that the primary outcome is independent of treatment conditional on surrogate outcomes: $Y \perp W \mid S$ |
| **Surrogate index** | A weighted combination of short-term outcomes used to predict long-term treatment effects (Athey et al. 2019) |
| **SPSM** (Semiparametric Subclassification with Propensity Scores) | An observational method combining subclassification with semiparametric efficiency |
| **Transportability** | The formal study of when and how causal effects estimated in one population can be applied to another (Pearl & Bareinboim 2011) |
| **Uplift modelling** | ML methods that estimate individual-level treatment effects for targeting decisions (who to treat) |

---

## Appendix B: Method Assumptions Reference

**Purpose:** This appendix consolidates the critical assumptions underlying each method. Violating these assumptions can lead to biased predictions, invalid inference, or misleading recommendations.

### B.1 Observational and Quasi-Experimental Methods

| Method | Assumption | Formal Statement | Consequence of Violation | How to Check | Typically Satisfied? |
|--------|-----------|------------------|-------------------------|--------------|---------------------|
| **Ghost Ads** | Auction fidelity | Counterfactual ad correctly identified | Biased control group | Validate against PSA tests | Usually yes |
| **Ghost Ads** | No spillovers | Control users unaffected by not seeing ad | Underestimates effect | Cross-device analysis | Usually yes |
| **Ghost Ads** | Random assignment | Users randomised before auction | Selection bias | Randomisation checks | Yes (by design) |
| **PSM** | Unconfoundedness | $Y(0), Y(1) \perp W \mid X$ | Biased estimates (60–175% error) | Cannot check directly | **No** (Gordon et al. 2019) |
| **PSM** | Common support | $0 < P(W=1 \mid X) < 1$ | Extreme weights, instability | Propensity score overlap | Often violated |
| **DML** | Unconfoundedness | $Y(0), Y(1) \perp W \mid X$ | Biased estimates | Cannot check directly | **No** (Gordon et al. 2023) |
| **DML** | Correct nuisance models | ML models for $E[Y \mid X]$ and $P(W \mid X)$ well-specified | Inconsistent estimates | Cross-validation | Partially |
| **DiD** | Parallel trends | $E[Y(0)_t - Y(0)_{t-1} \mid W=1] = E[Y(0)_t - Y(0)_{t-1} \mid W=0]$ | Biased estimates | Pre-trend tests | Often violated in ad settings |

### B.2 Surrogate and Proxy Outcome Methods

| Method | Assumption | Formal Statement | Consequence of Violation | How to Check | Typically Satisfied? |
|--------|-----------|------------------|-------------------------|--------------|---------------------|
| **Surrogate Index** | Prentice surrogacy | $Y \perp W \mid S$ | Misses non-mediated effects | Test on historical data | Domain-dependent |
| **Surrogate Index** | Comparability | $P(Y \mid S)$ stable across settings | Prediction model doesn't transfer | Cross-validation | Usually yes |
| **Surrogate Index** | No direct effect | Treatment affects $Y$ only through $S$ | Biased (typically understates) | Domain knowledge | Often violated |
| **Auto-Surrogates** | Temporal surrogacy | Short-term metric captures treatment effect | Misses delayed effects | Compare short vs long-term decisions | Usually yes for engagement |
| **Auto-Surrogates** | Stable temporal relationship | $Y_{short} \to Y_{long}$ mapping stable | Stale prediction model | Rolling validation | Usually yes |

### B.3 Cross-Experiment Prediction (PIE)

| Method | Assumption | Formal Statement | Consequence of Violation | How to Check | Typically Satisfied? |
|--------|-----------|------------------|-------------------------|--------------|---------------------|
| **PIE** | Feature invariance | Post-determined features same with/without RCT | Prediction bias | Compare RCT vs non-RCT feature distributions | Usually yes |
| **PIE** | RCT representativeness | RCT sample representative of non-RCT campaigns | Selection bias in predictions | Compare feature distributions | Depends on RCT selection |
| **PIE** | No concept drift | $f: X \to ICPD$ stable over time | Model degrades | Monitor R² over time | Requires monitoring |
| **PIE** | Sufficient RCT corpus | Enough RCTs for flexible model | Overfitting | Cross-validation, learning curves | Need 400+ |
| **PIE** | SUTVA within campaigns | No interference between campaigns | Campaign effects ill-defined | Check budget competition | Usually yes |
| **Amazon MTA** | PIE assumptions | All of the above | Cascading bias to touchpoint level | All of the above | Same as PIE |
| **Amazon MTA** | Touchpoint model validity | ML model captures touchpoint contribution | Credit misallocated | Holdout validation | Hard to validate |

### B.4 Marketing Mix Models

| Method | Assumption | Formal Statement | Consequence of Violation | How to Check | Typically Satisfied? |
|--------|-----------|------------------|-------------------------|--------------|---------------------|
| **Bayesian MMM** | Correct functional form | Adstock/saturation correctly specified | Biased ROAS | Compare functional forms | Unknown — parametric choice |
| **Bayesian MMM** | No omitted confounders | All demand drivers included | Biased channel effects | Lift test calibration | Partially (with calibration) |
| **Bayesian MMM** | Stable relationships | Channel effects constant over time | Stale model | Rolling estimation | Often violated |
| **Bayesian MMM** | Lift test validity | Lift tests are well-designed RCTs | Biased priors | Standard RCT checks | Usually yes |
| **Bayesian MMM** | Lift test recency | Results still relevant | Stale priors | Track time since calibration | Requires monitoring |

### B.5 Transportability and External Validity

| Method | Assumption | Formal Statement | Consequence of Violation | How to Check | Typically Satisfied? |
|--------|-----------|------------------|-------------------------|--------------|---------------------|
| **Transportability** | Known causal graph | DAG correctly specified | Wrong transport formula | Domain expertise | Rarely known with certainty |
| **Transportability** | Known differences | All differing variables identified | Unaccounted bias | Population comparison | Hard to verify |
| **Transportability** | Mechanism stability | Conditional distributions same across populations | Transport impossible | Test on overlapping subpopulations | Domain-dependent |
| **External Validity** | Effect homogeneity | Effects similar across settings | Single RCT doesn't generalise | Meta-analysis of multiple RCTs | **No** (Vivalt 2020: 60–100% heterogeneity) |
| **External Validity** | Random site selection | Experimental sites representative | Site selection bias | Compare early vs late adopters | **No** (Allcott 2015: 60% inflation) |

### B.6 Meta-Analysis

| Method | Assumption | Formal Statement | Consequence of Violation | How to Check | Typically Satisfied? |
|--------|-----------|------------------|-------------------------|--------------|---------------------|
| **Random-Effects MA** | Independent studies | Study effects are independent draws | Underestimated SEs | Check for shared data | Usually yes |
| **Random-Effects MA** | Normal random effects | $u_j \sim N(0, \tau^2)$ | Biased $\tau^2$ | Q-Q plots | Approximately |
| **Random-Effects MA** | No publication bias | All studies included regardless of results | Overestimated grand mean | Funnel plots, Egger's test | Often violated |
| **Meta-Regression** | Correct moderator specification | Relevant moderators included | Biased coefficients | Sensitivity analysis | Limited by available moderators |

---

## Appendix C: Full Reference List

### Core Papers

1. **Lewis & Rao (2015).** "The Unfavorable Economics of Measuring the Returns to Advertising." *Quarterly Journal of Economics*, 130(4), 1941–1973. [DOI](https://doi.org/10.1093/qje/qjv023)

2. **Johnson, Lewis, Nubbemeyer (2017).** "Ghost Ads: Improving the Economics of Measuring Online Ad Effectiveness." *Journal of Marketing Research*, 54(6), 867–884. [DOI](https://doi.org/10.1177/0022243717718579)

3. **Gordon, Zettelmeyer, Moakler, Reiley (2019).** "A Comparison of Approaches to Advertising Measurement: Evidence from Big Field Experiments at Facebook." *Marketing Science*, 38(2), 193–225. [DOI](https://doi.org/10.1287/mksc.2018.1135)

4. **Gordon, Moakler, Zettelmeyer (2023).** "Close Enough? A Large-Scale Exploration of Non-Experimental Approaches to Advertising Measurement." *Marketing Science*, 42(4), 768–793. [DOI](https://doi.org/10.1287/mksc.2022.1413)

5. **Gordon, Moakler, Zettelmeyer (2026).** "Predicted Incrementality by Experimentation (PIE)." [arXiv:2304.06828v2](https://arxiv.org/abs/2304.06828). Forthcoming.

6. **Lewis, Zettelmeyer et al. (2025).** "Multi-Touch Attribution at Amazon." [arXiv:2508.08209](https://arxiv.org/abs/2508.08209).

### Surrogate Methods

7. **Athey, Chetty, Imbens, Kang (2019/2024).** "Combining Short-Term Proxies to Estimate Long-Term Treatment Effects More Rapidly and Precisely." [arXiv:1903.10706](https://arxiv.org/abs/1903.10706). Forthcoming, *Review of Economic Studies*.

8. **Netflix (2023).** "Evaluating the Surrogate Index as a Decision-Making Tool Using 200 A/B Tests at Netflix." [Netflix Tech Blog](https://netflixtechblog.com/evaluating-the-surrogate-index-as-a-decision-making-tool-using-200-a-b-tests-at-netflix-4b4b3e4f5c9e).

### Causal ML and Uplift

9. **Athey & Wager (2018).** "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests." *Annals of Statistics*, 46(4), 1503–1535. [DOI](https://doi.org/10.1214/17-AOS1637)

10. **Ascarza (2018).** "Retention Futility: Targeting High-Risk Customers Is Not Effective." *Journal of Marketing Research*, 55(1), 80–98. [DOI](https://doi.org/10.1509/jmr.16.0163)

11. **Rzepakowski & Jaroszewicz (2012).** "Decision Trees for Uplift Modeling with Single and Multiple Treatments." *Knowledge and Information Systems*, 32, 303–327. [DOI](https://doi.org/10.1007/s10115-011-0434-0)

12. **Künzel, Sekhon, Bickel, Yu (2019).** "Metalearners for Estimating Heterogeneous Treatment Effects using Machine Learning." *Proceedings of the National Academy of Sciences*, 116(10), 4156–4165. [DOI](https://doi.org/10.1073/pnas.1804597116)

### Cross-Experiment Learning

13. **Huang, Ascarza, Israeli (2024).** "Pooling Multiple Experiments to Predict Individual-Level Treatment Effects for Personalization." Working paper.

### Marketing Mix Models

14. **Jin, Li, Naik et al. (2017).** "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects." Google Research.

15. **Meta Robyn.** [Open-source MMM](https://facebookexperimental.github.io/Robyn/)

16. **Google Meridian.** [Meridian](https://developers.google.com/meridian)

17. **PyMC-Marketing.** [PyMC Labs](https://www.pymc-marketing.io/)

### Transportability and External Validity

18. **Pearl & Bareinboim (2011).** "Transportability of Causal and Statistical Relations: A Formal Approach." *Proceedings of the AAAI Conference on Artificial Intelligence*. [DOI](https://doi.org/10.1609/aaai.v25i1.7861)

19. **Bareinboim & Pearl (2014).** "Transportability from Multiple Environments with Limited Experiments: Completeness Results." *Advances in Neural Information Processing Systems*.

20. **Allcott (2015).** "Site Selection Bias in Program Evaluation." *Quarterly Journal of Economics*, 130(3), 1117–1165. [DOI](https://doi.org/10.1093/qje/qju015)

21. **Vivalt (2020).** "How Much Can We Generalize from Impact Evaluations?" *Journal of the European Economic Association*, 18(6), 3045–3089. [DOI](https://doi.org/10.1093/jeea/jvz044)

22. **Hotz, Imbens, Mortimer (2005).** "Predicting the Efficacy of Future Training Programs Using Past Experiences at Other Locations." *Journal of Econometrics*, 125(1–2), 241–270. [DOI](https://doi.org/10.1016/j.jeconom.2005.01.024)

### Meta-Analysis

23. **DerSimonian & Laird (1986).** "Meta-analysis in clinical trials." *Controlled Clinical Trials*, 7(3), 177–188. [DOI](https://doi.org/10.1016/0197-2456(86)90046-2)

24. **Higgins & Thompson (2002).** "Quantifying heterogeneity in a meta-analysis." *Statistics in Medicine*, 21(11), 1539–1558. [DOI](https://doi.org/10.1002/sim.1186)

### Foundational

25. **Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, Robins (2018).** "Double/Debiased Machine Learning for Treatment and Structural Parameters." *The Econometrics Journal*, 21(1), C1–C68. [DOI](https://doi.org/10.1111/ectj.12097)

### Quasi-Experimental and Geo Methods (§2.4–§2.6)

26. **Brodersen, Gallusser, Koehler, Remy, Scott (2015).** "Inferring Causal Impact Using Bayesian Structural Time-Series Models." *Annals of Applied Statistics*, 9(1), 247–274. [arXiv:1506.00356](https://arxiv.org/abs/1506.00356)

27. **Abadie & Gardeazabal (2003).** "The Economic Costs of Conflict: A Case Study of the Basque Country." *American Economic Review*, 93(1), 113–132. [DOI](https://doi.org/10.1257/000282803321455188)

28. **Abadie, Diamond, Hainmueller (2010).** "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program." *Journal of the American Statistical Association*, 105(490), 493–505. [DOI](https://doi.org/10.1198/jasa.2009.ap08746)

29. **Abadie (2021).** "Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects." *Journal of Economic Literature*, 59(2), 391–425. [DOI](https://doi.org/10.1257/jel.20191450)

30. **Vaver & Koehler (2011).** "Measuring Ad Effectiveness Using Geo Experiments." Google Research.

31. **Kerman, Wang, Vaver (2017).** "Estimating Ad Effectiveness Using Geo Experiments in a Time-Based Regression Framework." Google Research.

### Long-Term Effects via Data Combination (§3.3)

32. **Ghassami, Yang, Richardson, Shpitser, Tchetgen Tchetgen (2024).** "Long-Term Causal Inference Under Persistent Confounding via Data Combination." *Journal of the Royal Statistical Society Series B*.

### Experimental + Observational Data Combination (§4.5–§4.7)

33. **Kallus, Puli, Shalit (2018).** "Removing Hidden Confounding by Experimental Grounding." *Advances in Neural Information Processing Systems (NeurIPS) 31*. [arXiv:1810.11646](https://arxiv.org/abs/1810.11646)

34. **Rosenman, Basse, Owen, Baiocchi (2023).** "Combining Observational and Experimental Datasets Using Shrinkage Estimators." *Biometrics*, 79(4), 2961–2973. [arXiv:2002.06708](https://arxiv.org/abs/2002.06708)

35. **Hernán & Robins (2016).** "Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available." *American Journal of Epidemiology*, 183(8), 758–764. [DOI](https://doi.org/10.1093/aje/kwv254)

### Marketing Mix Models (§5.1, Google Research Foundations)

36. **Chan & Perry (2017).** "Challenges and Opportunities in Media Mix Modeling." Google Research.

37. **Sun, Wang, Jin, Chan, Koehler (2017).** "Geo-Level Bayesian Hierarchical Media Mix Modeling." Google Research.

### Sample-to-Population and Data Fusion (§6.3–§6.4)

38. **Hartman, Grieve, Ramsahai, Sekhon (2015).** "From Sample Average Treatment Effect to Population Average Treatment Effect on the Treated: Combining Experimental with Observational Studies to Estimate Population Treatment Effects." *Journal of the Royal Statistical Society Series A*, 178(3), 757–778. [DOI](https://doi.org/10.1111/rssa.12094)

39. **Bareinboim & Pearl (2016).** "Causal Inference and the Data-Fusion Problem." *Proceedings of the National Academy of Sciences (PNAS)*, 113(27), 7345–7352. [DOI](https://doi.org/10.1073/pnas.1510507113)

### Meta-Analysis Foundations (§7.3)

40. **Borenstein, Hedges, Higgins, Rothstein (2009).** *Introduction to Meta-Analysis*. Chichester, UK: John Wiley & Sons. [Standard reference textbook]

41. **Gelman, Hill, Yajima (2012).** "Why We (Usually) Don't Have to Worry About Multiple Comparisons." *Journal of Research on Educational Effectiveness*, 5(2), 189–211. [DOI](https://doi.org/10.1080/19345747.2011.618213)

### Recent ML-Based Data Fusion (§8.1–§8.3)

42. **Prediction-Powered Generalization of Causal Inferences (2024).** [arXiv:2406.02873](https://arxiv.org/abs/2406.02873)

43. **In-Context Learning for Causal Effect Estimation (2024).** [arXiv:2506.06039](https://arxiv.org/abs/2506.06039)

44. **Externally Valid Policy Evaluation from Randomized Trials Using Additional Observational Data (2023).** [arXiv:2310.14763](https://arxiv.org/abs/2310.14763)

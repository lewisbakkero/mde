# Quick Literature Review (as of Dec'25): Techniques to Reduce Minimum Detectable Effect (MDE) in A/B Test Platforms

## Executive Summary

This literature review synthesizes research on methods to reduce the Minimum Detectable Effect (MDE) in online experimentation platforms, with particular focus on advertising contexts. MDE represents the smallest treatment effect that an experiment can reliably detect given its statistical power, sample size, and significance level. Reducing MDE enables platforms to detect smaller but meaningful effects, accelerating decision-making and improving experimentation efficiency.

---

## Practitioner's Decision Guide

**How to Use This Section:** Start here if you need to choose an MDE reduction method. The academic sections (2-6) provide depth; this section provides actionable guidance.

### When to Use What: Quick Reference Table

| Method | Variance Reduction | Implementation Complexity | Data Requirements | Computational Cost | Best For |
|--------|-------------------|--------------------------|-------------------|-------------------|----------|
| **CUPED** | 20-50% | Low (days) | Pre-experiment metric | Negligible | Any experiment with historical data |
| **CUPAC** | 30-60% | Medium (weeks) | ML training data | Medium (model training) | High-traffic experiments |
| **Stratification** | 10-30% | Low (days) | Categorical covariates | Negligible | Known heterogeneous segments |
| **Interleaving** | 50-90% | Medium (weeks) | Paired user sessions | Low | Ranking/recommendation changes |
| **Switchback** | 20-40% | High (months) | Time-series capability | Low | Marketplace/supply-side experiments |
| **Sequential (mSPRT)** | N/A (time savings) | Medium (weeks) | Continuous data stream | Low | Experiments needing early decisions |
| **Budget-Split** | 30-50% vs cluster | High (months) | Budget isolation infra | Medium | Ad marketplace experiments |

### Method Selection by Experiment Type

**Not all methods are interchangeable.** The table below maps experiment types to appropriate methods:

| Experiment Type | Primary Method | Secondary Method | Avoid |
|-----------------|---------------|------------------|-------|
| **UI/UX changes** | CUPED | Stratification | Interleaving (not applicable) |
| **Ranking algorithm** | Interleaving | CUPED (additive) | Switchback (overkill) |
| **Pricing changes** | Switchback | Cluster randomisation | CUPED alone (interference) |
| **Ad creative testing** | CUPED | Stratification by advertiser | Budget-split (unnecessary) |
| **Bidding algorithm** | Budget-split | Switchback | Standard A/B (interference) |
| **Recommendation model** | Interleaving | CUPED | Cluster (wrong unit) |

### Intuition: Why Methods Work

Understanding *why* a method reduces MDE helps you judge when it will (and won't) work:

| Method | Intuition | When It Works Best | When It Fails |
|--------|-----------|-------------------|---------------|
| **CUPED** | "Subtract out predictable noise" — if you know a user typically converts at 2%, deviations from that baseline are more informative than raw conversions | High correlation between pre/post behaviour (ρ > 0.5) | New users, behaviour shifts, long experiments |
| **Interleaving** | "Same user, same context, different treatment" — within-user comparison eliminates individual differences | Ranking systems where both variants can be shown simultaneously | Non-ranking experiments, user-level treatments |
| **Switchback** | "Same market, different times" — temporal comparison controls for market-level confounds | Supply-side experiments, marketplace dynamics | Strong time-of-day effects, carryover effects |
| **Stratification** | "Compare like with like" — reduce variance by grouping similar units | Known heterogeneous segments (e.g., new vs returning users) | Continuous covariates, many small strata |
| **Sequential** | "Stop early when you know" — don't waste samples after decision is clear | Experiments with clear winners/losers | Subtle effects, need precise estimates |

### Practical Gotchas and Risks

**These issues are rarely discussed in papers but frequently cause problems in practice:**

#### Sample Ratio Mismatch (SRM) Risks

| Method | SRM Risk | Why | Mitigation |
|--------|----------|-----|------------|
| **CUPED** | Low | Post-hoc adjustment, doesn't affect assignment | Monitor raw assignment ratios |
| **CUPAC** | Medium | ML model errors can correlate with treatment | Validate model on holdout, check residual balance |
| **Stratification** | Medium | Stratification bugs can cause imbalance | Verify stratum sizes match expectations |
| **Interleaving** | High | Position bias, presentation order effects | Use debiased interleaving (Section 4.4) |
| **Switchback** | Medium | Time-based confounds can masquerade as SRM | Validate with placebo experiments |
| **Sequential** | Low | Stopping rules don't affect assignment | Use valid stopping boundaries |

#### Computational and Operational Costs

| Method | Training Cost | Inference Cost | Operational Complexity |
|--------|--------------|----------------|----------------------|
| **CUPED** | None | O(n) simple regression | Low — can run post-hoc |
| **CUPAC** | High (ML model training) | O(n) model inference | Medium — need ML pipeline |
| **Stratification** | None | O(n) grouping | Low — need stratum definitions |
| **Interleaving** | None | O(n) per query | High — need real-time blending |
| **Switchback** | None | O(T) time periods | High — need treatment switching infra |
| **Sequential** | None | O(k) per analysis | Medium — need monitoring dashboard |

#### Common Failure Modes

1. **CUPED with behaviour shifts:** If a major product change occurs between pre-period and experiment, correlation drops and CUPED provides little benefit. *Mitigation:* Use shorter pre-periods, monitor correlation stability.

2. **Interleaving with position bias:** Naive interleaving systematically favours items shown in certain positions. *Mitigation:* Use Team Draft or debiased methods (Section 4.4).

3. **Switchback with carryover:** If treatment effects persist after switching, estimates are biased. *Mitigation:* Use longer periods, model carryover explicitly.

4. **Sequential testing with peeking:** Looking at results without proper boundaries inflates false positive rates. *Mitigation:* Use mSPRT or always-valid inference, not ad-hoc peeking.

5. **CUPAC model staleness:** ML models trained on old data may not predict current behaviour well. *Mitigation:* Retrain regularly, monitor prediction quality.

### Ratio Metrics: Special Considerations

**Ratio metrics (CTR, conversion rate, revenue per session) are harder to reduce variance for than additive metrics.** Here's why and what to do:

#### Why Ratio Metrics Are Harder

| Issue | Explanation | Impact on MDE |
|-------|-------------|---------------|
| **Denominator variance** | Both numerator and denominator vary, increasing total variance | 20-50% higher MDE vs additive |
| **Heavy tails** | Revenue metrics often have extreme values | Inflated variance, unstable estimates |
| **Correlation structure** | Numerator and denominator are correlated | Standard CUPED less effective |

#### Recommended Approaches for Ratio Metrics

| Metric Type | Recommended Method | Why | Expected Gain |
|-------------|-------------------|-----|---------------|
| **CTR (clicks/impressions)** | Delta method + CUPED on linearised metric | Handles ratio structure properly | 20-40% |
| **Conversion rate** | CUPED on pre-period conversion rate | High temporal correlation | 30-50% |
| **Revenue per user** | Winsorisation + CUPED | Controls heavy tails | 40-60% |
| **Revenue per session** | Stratify by session count + CUPED | Reduces denominator variance | 30-50% |

#### Delta Method for Ratio Metrics

For ratio metric $R = Y/X$, the linearised version is:
$\tilde{R} = Y - \hat{R} \cdot X$

Where $\hat{R}$ is the overall ratio. Apply CUPED to $\tilde{R}$ rather than $R$ directly.

**Practical tip:** Many experimentation platforms (Statsig, Eppo, internal tools) now support delta method automatically. Check your platform's documentation.

### Method Stacking: What Combines Well

| Base Method | Can Add | Expected Combined Gain | Cannot Add |
|-------------|---------|----------------------|------------|
| **CUPED** | Stratification, Sequential | 40-70% total | Another CUPED variant |
| **Interleaving** | CUPED (for secondary metrics) | 60-95% total | Switchback |
| **Switchback** | CUPED (within-period) | 30-50% total | Interleaving |
| **Sequential** | Any variance reduction | Faster + smaller MDE | Another sequential method |

### Decision Flowchart for Practitioners

```
START: What are you testing?
│
├─► Ranking/Recommendation algorithm?
│   └─► YES → Use INTERLEAVING (50-90% MDE reduction)
│             Add CUPED for secondary metrics
│
├─► Marketplace/pricing experiment?
│   └─► YES → Is budget interference a concern?
│             ├─► YES → Use BUDGET-SPLIT or SWITCHBACK
│             └─► NO  → Use CLUSTER RANDOMISATION
│
├─► Standard UI/feature experiment?
│   └─► YES → Do you have pre-experiment data?
│             ├─► YES → Use CUPED (start here, 30-50% reduction)
│             │         Consider CUPAC if you have ML infra
│             └─► NO  → Use STRATIFICATION (10-30% reduction)
│
└─► Need faster decisions?
    └─► YES → Add SEQUENTIAL TESTING to any of the above
              Use mSPRT for continuous monitoring
```

---

## 1. Introduction

### 1.1 Background

Online controlled experiments (A/B tests) are the gold standard for causal inference in digital platforms. However, detecting small treatment effects requires either large sample sizes or long experiment durations, both of which are costly. The **baseline MDE formula** is:

$$MDE_{baseline} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2}{n}}$$

Where:
- $\alpha$ = significance level (Type I error rate)
- $\beta$ = Type II error rate (1 - power)
- $\sigma^2$ = variance of the outcome metric
- $n$ = sample size per treatment arm

This formula serves as our **reference equation** throughout this review. Each technique modifies one or more components to achieve MDE reduction.

Reducing MDE can be achieved through three primary mechanisms:
1. **Variance Reduction** - Decreasing $\sigma^2$
2. **Sample Size Increase** - Increasing $n$ effectively
3. **Statistical Efficiency** - Improving estimator precision

### 1.2 Scope

This review covers techniques from 25+ papers spanning:
- Covariate adjustment methods (CUPED, CUPAC, ML-based, Pre+In-Experiment Combined)
- Sequential testing approaches (GST, mSPRT, Always Valid Inference)
- Experimental design innovations (Switchback, Staggered Rollout, Budget-Split, Interleaving)
- Marketplace-specific methods (Cluster randomisation, Multiple randomisation)
- Adaptive methods (Multi-armed bandits, Adaptive experimental design)

---

## 2. Variance Reduction Techniques

All methods in this section reduce MDE by decreasing the variance term $\sigma^2$ in the baseline equation.


### 2.1 CUPED (Controlled-experiment Using Pre-Experiment Data)

**Source:** [Improving the sensitivity of online controlled experiments by utilising pre-experiment data](https://robotics.stanford.edu/~ronnyk/2013-02CUPEDImprovingSensitivityOfControlledExperiments.pdf)

**Core Idea:** Use pre-experiment covariate data to reduce variance through regression adjustment.

**MDE Equation Modification:**
$$MDE_{CUPED} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2(1-\rho^2)}{n}}$$

Where $\rho$ is the correlation between pre-experiment covariate $X$ and outcome $Y$. The variance reduction factor is $(1-\rho^2)$.

**Method:**
The CUPED estimator adjusts the outcome $Y$ using a pre-experiment covariate $X$:
$$\hat{Y}_{CUPED} = Y - \theta(X - \bar{X})$$
Where $\theta = \frac{Cov(Y, X)}{Var(X)}$ minimises variance.

**Key Findings:**
- 50%+ variance reduction achievable with highly correlated pre-experiment metrics
- Works best when pre-experiment behaviour strongly predicts post-experiment outcomes
- Simple to implement and widely adopted (Microsoft, Netflix, LinkedIn)

**Limitations:**
- Requires pre-experiment data availability
- Effectiveness depends on covariate-outcome correlation
- Single covariate may not capture all predictive information

---

### 2.2 CUPAC (Controlled-experiment Using Predictions As Covariates)

**Source:** 
[Control Using Predictions as Covariates in Switchback Experiments](https://www.researchgate.net/profile/Yixin-Tang-5/publication/345698207_Control_Using_Predictions_as_Covariates_in_Switchback_Experiments/links/5fab109b458515078107aa8b/Control-Using-Predictions-as-Covariates-in-Switchback-Experiments.pdf)
[Leveraging covariate adjustments at scale in online A/B testing](https://www.amazon.science/publications/leveraging-covariate-adjustments-at-scale-in-online-a-b-testing)

**Core Idea:** Use ML predictions as covariates instead of raw pre-experiment data.

**MDE Equation Modification:**
$$MDE_{CUPAC} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2(1-\rho_{ML}^2)}{n}}$$

Where $\rho_{ML}$ is the correlation between ML prediction $\hat{Y}_{ML}$ and actual outcome $Y$. Since ML models can achieve $\rho_{ML} > \rho_{single}$, CUPAC achieves greater MDE reduction than basic CUPED.


### Covariate Adjustment via the Imputation Operator

The following summary outlines the mathematical framework for leveraging pre-experiment data to increase the precision of A/B tests at scale.

#### 1. The Imputation Operator
The imputation operator $\hat{f}_t$ fills in missing counterfactuals by combining observed outcomes $Y_n$ with predictions from a model trained on covariates $z_n$.

$$\hat{Y}_n(t) = 
\begin{cases} 
Y_n & \text{if } J_n = t \\ 
\hat{f}(z_n; \hat{\theta}_t) & \text{if } J_n = 1-t 
\end{cases}$$

The efficiency gain is directly tied to the model's predictive power ($R^2$). 

**Variance Adjustment:**
$$\sigma^2_{\text{adj}} \approx \sigma^2_{\text{DIM}} \cdot (1 - R^2)$$

*For example, an $R^2$ of 0.5 allows an experiment to reach the same power in half the time.*

**Method:**

1.  **Fit:** Train $\hat{f}_0$ on the control group and $\hat{f}_1$ on the treatment group using pre-experiment covariates $z_n$.
2.  **Impute:** For every user $n$, compute the "completed" outcomes $(\hat{Y}_n(0), \hat{Y}_n(1))$.
3.  **Estimate:** Calculate the Average Treatment Effect (ATE):
    $$\widehat{ATE} = \frac{1}{N} \sum_{n=1}^{N} \left( \hat{Y}_n(1) - \hat{Y}_n(0) \right)$$

**Key Findings:**
- 10-30% additional variance reduction over basic CUPED
- Particularly effective for complex metrics, imbalanced data
- Requires ML infrastructure investment

Different ML models offer different tradeoffs:

| Model | Mechanism | Strength | Scaling Challenge |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Ordinary Least Squares | Simple and fast | Overfits with high-dimensional $z$ |
| **Lasso (L1)** | Sparsity Penalty | Automatic feature selection | Tuning $\lambda$ is computationally expensive |
| **PCR** | PCA + Regression | Handles collinearity | Requires pre-computing principal components |


**Limitations:**
- Requires ML infrastructure and model maintenance
- Model must be trained on pre-experiment data only to avoid bias
- Computational overhead for prediction generation

---

### 2.3 Theoretical Foundations for Multi-Covariate Adjustment

**Source:** 
[From Augmentation to Decomposition: A New Look at CUPED (2023)](https://alexdeng.github.io/public/files/CUPED_2023_Metric_Decomp.pdf)

[Control Using Predictions as Covariates in Switchback
Experiments](https://www.researchgate.net/profile/Yixin-Tang-5/publication/345698207_Control_Using_Predictions_as_Covariates_in_Switchback_Experiments/links/5fab109b458515078107aa8b/Control-Using-Predictions-as-Covariates-in-Switchback-Experiments.pdf)

**Core Idea:** Provides theoretical framework for optimal covariate selection and multiple covariate extension. Moving beyond simple regression, CUPED (2023) is viewed as an **Efficiency Augmentation**. Given a raw treatment effect estimator $\hat{\Delta}$, we add a mean-zero term to reduce noise without introducing bias.

$$\hat{\Delta}_{aug} = \hat{\Delta} - \theta \hat{\Delta}_{null}$$

Where:
* $\hat{\Delta} = \bar{Y}_T - \bar{Y}_C$ (The observed difference)
* $\hat{\Delta}_{null}$ is a term where $E[\hat{\Delta}_{null}] = 0$ (The "Noise" term)
* $\theta = \frac{Cov(\hat{\Delta}, \hat{\Delta}_{null})}{Var(\hat{\Delta}_{null})}$ (The optimal scaling factor)

The "New Look" involves decomposing a current metric $Y$ into two distinct components:

$$Y = Y_{null} + Y_{residual}$$

* **$Y_{null}$ (Approximate Null Augmentation):** The portion of the metric "that would have happened anyway" (e.g., via counterfactual logging). It is highly correlated with $Y$ but independent of the treatment.
* **$Y_{residual}$:** The component actually sensitive to the treatment effect.


**MDE Equation Modification:**
With multiple covariates $X_1, ..., X_k$:
$$MDE_{multi} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2(1-R^2)}{n}}$$

Where $R^2$ is the coefficient of determination from regressing $Y$ on all covariates. This generalises the single-covariate case where $R^2 = \rho^2$.

In this framework, $\rho$ is typically much higher than in traditional CUPED because $Y_{null}$ is measured **during** the experiment rather than using historical data.

**Method:**
1. Decompose outcome into signal (treatment effect) and noise (individual variation)
2. Use optimal linear combination of multiple covariates
3. Achieve semiparametric efficiency bound

**Key Findings:**
- Provides theoretical guidance on covariate selection, splits
- Multiple covariates can achieve higher $R^2$ than single covariate
- Connections to semiparametric efficiency theory

**Limitations:**
- Requires careful covariate engineering
- Diminishing returns with additional covariates
- Overfitting risk with many covariates

| Feature | Deng et al. (Airbnb) | Masoero et al. (Amazon/DoorDash) |
| :--- | :--- | :--- |
| **Primary Method** | Metric Decomposition (ANA) | Cross-fitted G-Computation |
| **Philosophy** | "Subtract the Noise" | "Impute the Missing Counterfactual" |
| **Covariate Source** | In-experiment counterfactuals | High-dimensional pre-experiment features |
| **Model Type** | Linear/Taylor Expansion | Non-linear Machine Learning (XGBoost) |
| **Best Use Case** | Ranking and Recommendation engines | Large-scale General Experiment Platforms |
| **Scaling Mechanism**| Domain-specific logging | Automated Cross-fitting (DML) |

---

### 2.4 Stratification

**Source:** [Efficient experimentation: A review of four methods to reduce the costs of A/B testing](https://ideas.repec.org/a/aza/ama000/y2025v11i1p61-71.html)

**Core Idea:** Divide population into homogeneous strata and randomise within strata.

**MDE Equation Modification:**
$$MDE_{strat} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2_{within}}{n}}$$

Where $\sigma^2_{within} < \sigma^2$ because between-strata variance is removed:
$$\sigma^2_{within} = \sigma^2 - \sigma^2_{between}$$

**Method:**
1. Define strata based on pre-experiment characteristics
2. Randomise treatment assignment within each stratum
3. Estimate treatment effect as weighted average across strata

**Key Findings:**
- Reduces variance by removing between-strata variation
- Complementary to CUPED (can be combined for multiplicative gains)
- Requires careful stratum definition

**Limitations:**
- Requires pre-experiment stratification variables
- Too many strata can reduce power
- Stratum imbalance can complicate analysis

---

### 2.5 Temporal Stratification for Non-Stationary A/B Tests

**Source:** [Non-stationary A/B tests](https://www.amazon.science/publications/non-stationary-a-b-tests) (addresses time-varying treatment effects)

**Core Idea:** Account for non-stationarity in treatment effects over time by stratifying data into time buckets. Instead of just averaging all Group A and all Group B results, the data is partitioned into temporal strata, with the number of strata growing adaptively as more data arrives.

**MDE Equation Modification:**
Standard stratification removes between-strata variance. For temporal stratification with $K$ time buckets:
$$MDE_{temporal} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2_{within-time}}{n}}$$

Where $\sigma^2_{within-time} = \sigma^2 - \sigma^2_{between-time}$, and $\sigma^2_{between-time}$ captures variance due to time-varying treatment effects and temporal trends.

The key innovation is using a **sample-dependent number of strata** $K(n)$:
- As sample size $n$ grows, the number of time buckets $K(n)$ increases
- Finer buckets capture more temporal variation
- Optimal $K(n)$ balances bias (too few buckets) vs. variance (too many buckets)

**Variance Reduction Mechanism:**
$$\sigma^2_{within-time} = \sigma^2 \cdot (1 - \eta^2_{temporal})$$

Where $\eta^2_{temporal} = \sigma^2_{between-time}/\sigma^2$ is the fraction of variance explained by temporal effects. Typical values: 10-30% variance reduction in non-stationary environments.

**Method:**
1. Partition experiment duration into $K(n)$ time buckets
2. Compute treatment effect within each bucket
3. Aggregate using inverse-variance weighting across buckets
4. Adaptively refine bucket granularity as data accumulates

**Key Findings:**
- Reduces bias from time-varying effects
- Improves variance estimation accuracy
- Particularly relevant for long-running experiments with temporal trends
- Adaptive bucket sizing outperforms fixed stratification

**Limitations:**
- Requires modelling assumptions about temporal dynamics
- More complex analysis than standard stratification
- May require longer experiments to estimate dynamics
- Computational overhead for adaptive bucket selection

---

### 2.6 Efficient Semiparametric Estimation Under Covariate Adaptive Randomisation

**Source:** 
- [Covariate Adjustment in Randomised Trials](https://covariateadjustment.github.io/)
- [Efficient Semiparametric Estimation of Average Treatment Effects Under Covariate Adaptive Randomisation](https://www.amazon.science/publications/efficient-semiparametric-estimation-of-average-treatment-effects-under-covariate-adaptive-randomisation) by Rafi et al.

**Core Idea:** Establishes theoretical efficiency bounds for covariate adjustment methods.

**MDE Equation Modification:**
The semiparametric efficiency bound gives the minimum achievable variance:
$$MDE_{optimal} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2_{eff}}{n}}$$

Where $\sigma^2_{eff}$ is the semiparametric efficient variance, representing the theoretical lower bound.

**How the Semiparametric Efficient Variance is Calculated:**

The efficient variance $\sigma^2_{eff}$ is derived from the **efficient influence function** in semiparametric theory. For the Average Treatment Effect (ATE) $\tau = E[Y(1)] - E[Y(0)]$:

1. **Efficient Influence Function:** The efficient influence function for the ATE is:
   $$\psi(Y, W, X) = \frac{W(Y - \mu_1(X))}{e(X)} - \frac{(1-W)(Y - \mu_0(X))}{1-e(X)} + \mu_1(X) - \mu_0(X) - \tau$$
   
   Where:
   - $W$ = treatment indicator
   - $\mu_1(X) = E[Y|W=1, X]$ = conditional mean under treatment
   - $\mu_0(X) = E[Y|W=0, X]$ = conditional mean under control
   - $e(X) = P(W=1|X)$ = propensity score

2. **Efficient Variance Formula:** The semiparametric efficient variance is the variance of the influence function:
   $$\sigma^2_{eff} = E[\psi^2] = E\left[\frac{Var(Y|W=1,X)}{e(X)}\right] + E\left[\frac{Var(Y|W=0,X)}{1-e(X)}\right] + Var(\tau(X))$$
   
   For a randomised experiment with balanced assignment $e(X) = 0.5$:
   $$\sigma^2_{eff} = 2 \cdot E[Var(Y|W,X)] + Var(\tau(X))$$

3. **Practical Interpretation:** The efficient variance has three components:
   - **Residual variance:** $E[Var(Y|W,X)]$ - variance of $Y$ not explained by covariates
   - **Treatment effect heterogeneity:** $Var(\tau(X))$ - variance of conditional treatment effects
   - **Propensity weighting:** adjustment for unequal treatment probabilities

4. **Achieving the Bound:** Estimators that achieve this bound include:
   - **AIPW (Augmented Inverse Propensity Weighting):** Combines outcome regression with propensity weighting
   - **TMLE (Targeted Maximum Likelihood Estimation):** Uses targeted updates to achieve efficiency
   - **Regression adjustment:** OLS with correctly specified covariates

5. **Connection to CUPED:** When $\mu_1(X) - \mu_0(X) = \tau$ (constant treatment effect) and using linear regression:
   $$\sigma^2_{eff} = \sigma^2(1 - R^2)$$
   This shows CUPED achieves the semiparametric bound under homogeneous treatment effects.

**Method:**
1. Derive efficiency bounds for various randomisation schemes
2. Construct estimators achieving these bounds (AIPW, TMLE)
3. Show robustness to model misspecification (doubly robust property)

**Key Findings:**
- Provides benchmark for evaluating variance reduction methods
- Justifies use of regression adjustment in randomised experiments
- AIPW estimators are "doubly robust": consistent if either outcome model OR propensity model is correct
- Extends to complex randomisation schemes (stratified, covariate-adaptive)

**Limitations:**
- Theoretical framework; practical implementation may not achieve bounds exactly
- Requires estimation of nuisance functions ($\mu_0$, $\mu_1$, $e$)
- Computational complexity for some estimators
- Finite-sample performance may differ from asymptotic theory

---

### 2.7 Variance Reduction Combining Pre-Experiment and In-Experiment Data

**Source:** [Variance reduction combining pre-experiment and in-experiment data](https://arxiv.org/abs/2410.09027)

**Core Idea:** Extend CUPED by leveraging both pre-experiment covariates AND in-experiment (concurrent) control group data. It extends CUPED in a CUPAC-like manner (ML model predicting Y), then takes residuals unexplained by the model and performs a regression using in-experiment covariates.

**MDE Equation Modification:**
$$MDE_{combined} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2(1-R^2_{combined})}{n}}$$

Where $R^2_{combined} > R^2_{CUPED}$ because:
- Pre-experiment data captures stable user characteristics
- In-experiment data captures temporal/environmental factors

**Method:**
The combined estimator uses both pre-experiment covariate $X_{pre}$ and in-experiment control data $X_{in}$:

$$\hat{Y}_{combined} = Y - \theta_1(X_{pre} - \bar{X}_{pre}) - \theta_2(X_{in} - \bar{X}_{in})$$

**Key Findings:**
- 30-60% variance reduction (vs 20-50% for CUPED alone)
- Particularly effective in volatile environments with temporal shocks
- Achieves semiparametric efficiency bound under certain conditions

**Limitations:**
- Requires careful modelling of correlation structure
- In-experiment adjustment can introduce bias if not properly implemented
- More complex to implement and validate; likely less stable for large scale prod environments

**Related:** For sparse and delayed outcomes specifically, see Section 2.8 which addresses targeted variance reduction when most users have zero/null metric values.

| Method | Data Used | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- |
| **Naive (Diff-in-Means)** | None | Simple, unbiased. | High variance; requires large samples. |
| **CUPED** | Linear Pre-experiment | Fast, easy to implement. | Limited by weak historical correlation. |
| **CUPAC** | ML Pre-experiment | Better at capturing non-linear patterns. | Still only uses "old" data. |
| **combining pre-experiment and in-experiment** | **Pre + In-experiment** | **Highest variance reduction.** | Requires careful selection of in-experiment variables. |

---

### 2.8 Variance Reduction for Sparse and Delayed Outcomes

**Source:** [Variance Reduction Using In-Experiment Data: Efficient and Targeted Online Measurement for Sparse and Delayed Outcomes](https://dl.acm.org/doi/10.1145/3580305.3599928) 

**Core Idea:** Standard variance reduction methods (CUPED, CUPAC) struggle with **sparse outcomes** (where most users have zero values, e.g., purchases, conversions) and **delayed outcomes** (metrics that take time to materialise, e.g., subscription renewals, long-term retention). This paper introduces targeted variance reduction using in-experiment surrogate metrics that are correlated with the sparse/delayed primary outcome.

**The Sparse/Delayed Outcome Challenge:**
- **Sparse outcomes:** In e-commerce, only 2-5% of users convert. The outcome $Y$ is mostly zeros, making pre-experiment covariates weakly predictive.
- **Delayed outcomes:** Subscription renewal happens 30+ days after treatment. Pre-experiment data is stale by the time the outcome is observed.
- **Standard CUPED fails:** When $\rho_{pre,Y}$ is low due to sparsity or delay, variance reduction is minimal.

**MDE Equation Modification:**
$MDE_{sparse} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2(1-\rho^2_{surrogate})}{n}}$

Where $\rho_{surrogate}$ is the correlation between an in-experiment surrogate metric $S$ and the sparse/delayed outcome $Y$. The key insight is that $\rho_{surrogate} \gg \rho_{pre}$ because:
- Surrogates are measured during the experiment (temporally proximate)
- Surrogates can be chosen to be leading indicators of the outcome

**Method:**
1. **Identify surrogate metrics:** Find in-experiment metrics $S$ that are:
   - Correlated with the primary outcome $Y$
   - Observable earlier or more frequently than $Y$
   - Examples: page views → purchases, engagement → retention, clicks → conversions

2. **Targeted adjustment:** Apply CUPED-style adjustment using the surrogate:
   $\hat{Y}_{adj} = Y - \theta(S - \bar{S})$
   Where $\theta = \frac{Cov(Y, S)}{Var(S)}$

3. **Efficient estimation:** Use cross-fitting to avoid overfitting when selecting surrogates from many candidates

**Surrogate Selection Criteria:**
| Criterion | Description | Example |
|-----------|-------------|---------|
| **Temporal proximity** | Surrogate observed before/during outcome window | Clicks (day 1) → Purchase (day 7) |
| **Causal pathway** | Surrogate on causal path to outcome | Engagement → Retention |
| **High correlation** | Strong $\rho_{S,Y}$ | Add-to-cart → Purchase |
| **Low sparsity** | Surrogate less sparse than outcome | Page views (dense) → Conversions (sparse) |

**Key Findings:**
- 40-70% variance reduction for sparse outcomes (vs 10-20% with standard CUPED)
- Enables earlier experiment decisions by using leading indicators
- Particularly effective for conversion metrics in ads and e-commerce
- Can reduce experiment duration by 50%+ for delayed outcomes

**Limitations:**
- Requires domain knowledge to identify good surrogates
- Surrogate-outcome relationship may vary across segments
- Risk of surrogate metric gaming if used for decisions
- Treatment may affect surrogate-outcome correlation (violation of surrogacy assumption)

**When to Use:**
- Primary metric has >90% zero values (sparse)
- Outcome takes >7 days to observe (delayed)
- Pre-experiment CUPED achieves <20% variance reduction
- Leading indicator metrics are available

**Connection to Section 2.7:** While Section 2.7 combines pre-experiment and in-experiment data for general variance reduction, this approach specifically targets the sparse/delayed outcome problem by using in-experiment *surrogate* metrics rather than in-experiment *control* data.

---

### 2.9 Variance Reduction Methods: Comparison Table

| Method | Section | MDE Modification | Typical Variance Reduction | Complexity | Data Requirements | Best Use Case |
|--------|---------|------------------|---------------------------|------------|-------------------|---------------|
| **CUPED** | 2.1 | $\sigma^2 \rightarrow \sigma^2(1-\rho^2)$ | 20-50% | Low | Pre-experiment metric | General A/B tests |
| **CUPAC** | 2.2 | $\sigma^2 \rightarrow \sigma^2(1-\rho_{ML}^2)$ | 30-60% | Medium | ML infrastructure | Complex metrics |
| **Multi-Covariate** | 2.3 | $\sigma^2 \rightarrow \sigma^2(1-R^2)$ | 30-60% | Medium | Multiple covariates | Rich feature sets |
| **Stratification** | 2.4 | $\sigma^2 \rightarrow \sigma^2_{within}$ | 10-30% | Low | Stratification vars | Heterogeneous populations |
| **Temporal Stratification** | 2.5 | $\sigma^2 \rightarrow \sigma^2(1-\eta^2_{temporal})$ | 10-30% | Medium | Time-series data | Non-stationary environments |
| **Semiparametric** | 2.6 | $\sigma^2 \rightarrow \sigma^2_{eff}$ | Theoretical bound | High | Depends on estimator | Theoretical benchmark |
| **Pre+In Combined** | 2.7 | $\sigma^2 \rightarrow \sigma^2(1-R^2_{combined})$ | 30-60% | Medium | Pre + concurrent control | Volatile environments |
| **Sparse/Delayed Surrogates** | 2.8 | $\sigma^2 \rightarrow \sigma^2(1-\rho^2_{surrogate})$ | 40-70% | Medium | Surrogate metrics | Sparse/delayed outcomes |

**Combining Variance Reduction Methods:**

Several variance reduction methods can be combined, but with diminishing returns due to overlapping variance components:

1. **CUPED + Stratification:** Stratification operates at the *randomisation* stage while CUPED operates at the *estimation* stage, so they can be combined.

**Combined MDE Equation (Stratification + Covariate Adjustment):**
   $$MDE_{combined} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2 \cdot (1-\eta^2_{strat}) \cdot (1-R^2_{adj})}{n}}$$
   Where $\eta^2_{strat} = \sigma^2_{between}/\sigma^2$ is variance explained by strata, and $R^2_{adj}$ is variance explained by covariate adjustment.

2. **CUPAC vs. Multi-Covariate CUPED vs. Pre+In:** These are *alternatives*, not complements—they all perform covariate adjustment at the estimation stage. Choose one.

3. **Diminishing Returns:** The semiparametric bound $\sigma^2_{eff}$ is the theoretical floor. Gains are sub-multiplicative because methods capture overlapping variance.

4. **Temporal Stratification + CUPED:** Can be combined—temporal stratification handles time-varying effects while CUPED handles individual-level variance. The combined effect is approximately multiplicative:
   $\sigma^2_{combined} \approx \sigma^2 \cdot (1-\eta^2_{temporal}) \cdot (1-\rho^2)$

**Practical guidance:** Start with CUPED (low complexity, high impact), add stratification if population is heterogeneous, add temporal stratification for long-running experiments with non-stationarity, upgrade to CUPAC or Pre+In only if substantial residual variance remains.


---


## 3. Sequential Testing Methods

All methods in this section reduce MDE by enabling early stopping, effectively reducing required sample size $n$.

### 3.1 Group Sequential Testing (GST)

**Source:** O'Brien & Fleming (1979), Pocock (1977), Lan & DeMets (1983)

**Core Idea:** Perform interim analyses at pre-specified times with adjusted significance thresholds.

**MDE Equation Modification:**
For a fixed-horizon test with sample size $n$, GST allows early stopping. The expected sample size under the alternative is:
$$E[n_{GST}] = n \cdot ASN_{ratio}$$

Where **ASN (Average Sample Number)** is the expected number of samples before the test terminates. $ASN_{ratio} = E[n_{GST}]/n_{fixed}$ is typically 0.5-0.7 when a true effect exists, meaning GST requires fewer samples on average. The effective MDE becomes:
$$MDE_{GST} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2}{n/ASN_{ratio}}}$$

**Method:**
1. Pre-specify number of looks $K$ and timing
2. Use spending function to allocate Type I error across looks
3. Stop early if effect is significant or futile

**Key Findings:**
- Can reduce expected sample size by 30-50% under true effect
- Requires pre-planning of analysis schedule
- Well-established statistical theory

**Limitations:**
- Inflexible timing of analyses
- May not be optimal for continuous monitoring
- Requires commitment to analysis schedule

---

### 3.2 Always Valid Inference: Continuous Monitoring of A/B Tests

**Source:** [Always Valid Inference: Continuous Monitoring of A/B Tests](https://pubsonline.informs.org/doi/10.1287/opre.2021.2135)

**Core Idea:** Construct confidence sequences that remain valid under continuous monitoring (no p-hacking).

**MDE Equation Modification:**
The confidence sequence width at time $t$ is wider than fixed-horizon, but early stopping reduces expected sample size:
$$MDE_{AVI} \approx (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2 \cdot c_{AVI}}{E[n_{stop}]}}$$

Where $c_{AVI} \approx 1.2-1.3$ is the penalty for continuous monitoring, but $E[n_{stop}] < n_{fixed}$.

**Key Parameters Explained:**

1. **α (Significance Level):** The Type I error rate—probability of falsely rejecting the null hypothesis when no true effect exists. Typically α = 0.05 (5%). This is the same α from the baseline MDE formula.

2. **c_AVI (Always Valid Inference Penalty Factor):** The "price" paid for continuous monitoring validity, typically **1.2-1.3**. This means confidence intervals are 20-30% wider than fixed-horizon tests.

   *Where does c_AVI come from?* The confidence sequence uses `log(2/α)` instead of `z²_{α/2}` and includes a `(1 + 1/t)` correction term. For a fixed-horizon test at sample size n:
   - Fixed-horizon CI half-width: $z_{\alpha/2} \cdot \sigma/\sqrt{n}$
   - Always-valid CI half-width: $\sqrt{2\sigma^2(1 + 1/n)\log(2/\alpha)/n}$
   
   The ratio of these widths gives c_AVI. For α = 0.05 and large n:
   $c_{AVI} \approx \sqrt{\frac{2\log(40)}{z_{0.025}^2}} \approx \sqrt{\frac{2 \times 3.69}{3.84}} \approx 1.39$
   
   In practice, optimised mixture constructions achieve c_AVI ≈ 1.2-1.3.

3. **E[n_stop] (Expected Stopping Sample Size):** The expected number of observations before the experiment terminates.

   *How is E[n_stop] calculated?* It's the expectation over the stopping time distribution:
   $E[n_{stop}] = \sum_{t=1}^{\infty} t \cdot P(\text{stop at } t) = \sum_{t=1}^{\infty} P(\text{not stopped by } t)$
   
   Under the **null hypothesis** (no effect): E[n_stop] = ∞ (never stop, or stop at max sample size)
   
   Under the **alternative hypothesis** (true effect δ exists): The stopping time depends on how quickly evidence accumulates. For a confidence sequence that excludes 0 when |μ̂_t| > boundary_t:
   
   $P(\text{stop by } t | \delta) = P\left(\left|\hat{\mu}_t - 0\right| > \sqrt{\frac{2\sigma^2(1+1/t)\log(2/\alpha)}{t}} \Big| \mu = \delta\right)$
   
   Since $\hat{\mu}_t \sim N(\delta, \sigma^2/t)$ under the alternative, this becomes:
   $P(\text{stop by } t | \delta) = 1 - \Phi\left(\frac{boundary_t - \delta}{\sigma/\sqrt{t}}\right) + \Phi\left(\frac{-boundary_t - \delta}{\sigma/\sqrt{t}}\right)$
   
   The expected stopping time is then:
   $E[n_{stop}|\delta] = \sum_{t=1}^{n_{max}} \left[1 - P(\text{stop by } t | \delta)\right]$
   
   **Practical approximation:** For effect size δ and variance σ², E[n_stop] scales roughly as:
   $E[n_{stop}] \approx \frac{c \cdot \sigma^2 \cdot \log(1/\alpha)}{\delta^2}$
   
   where c is a constant depending on the specific confidence sequence construction (typically c ≈ 4-8).

**Method:**
Uses mixture martingale approach to create confidence sequences:
$$CS_t = \left[\hat{\mu}_t \pm \sqrt{\frac{2\sigma^2(1 + 1/t)\log(2/\alpha)}{t}}\right]$$

The key insight is that this boundary shrinks as $O(1/\sqrt{t})$ but with a $\sqrt{\log(2/\alpha)}$ factor instead of $z_{\alpha/2}$, ensuring validity at all stopping times.

**Net Efficiency Trade-off:**
- **Cost:** Wider CI by factor c_AVI ≈ 1.2-1.3
- **Benefit:** Earlier stopping when effect exists, so E[n_stop] < n_fixed
- **Break-even:** When $\frac{c_{AVI}}{1} < \frac{n_{fixed}}{E[n_{stop}]}$, i.e., when early stopping savings exceed the width penalty

For moderate-to-large effects, the early stopping benefit typically dominates, yielding net MDE reduction.

**Key Findings:**
- Enables truly continuous monitoring
- Valid at any stopping time (not just pre-specified)
- Controls Type I error under arbitrary peeking

**Limitations:**
- ~20-30% wider confidence intervals than fixed-horizon
- Requires variance estimation
- Trade-off between validity and efficiency

---

### 3.3 Mixture Sequential Probability Ratio Test (mSPRT)

**Source:** Popularised by Optimizely, Amplitude; based on [Robbins (1974)](https://www.jstor.org/stable/2285511)

**Core Idea:** Sequential test using mixture of likelihood ratios over effect sizes.

**MDE Equation Modification:**
The mSPRT uses a mixing distribution $\pi(\theta)$ over effect sizes:
$$MDE_{mSPRT} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2}{E[n_{stop}]}}$$

Where $E[n_{stop}]$ is typically 20-40% less than fixed-horizon $n$ when effect exists.

**Method:**
Compute mixture likelihood ratio:
$$\Lambda_n = \int L_n(\theta) \pi(\theta) d\theta$$
Stop when $\Lambda_n > 1/\alpha$ or $\Lambda_n < \alpha$.

**Key Terms Explained:**

1. **θ (theta):** The unknown treatment effect size (e.g., difference in means between treatment and control). This is what we're trying to detect.

2. **L_n(θ) - Likelihood Ratio:** The ratio of data probability under effect size θ versus the null (θ=0):
   $$L_n(\theta) = \frac{P(\text{data} | \theta)}{P(\text{data} | \theta = 0)}$$
   
   For normally distributed data with known variance$$ $\sigma^2$:
   $L_n(\theta) = \exp\left(\frac{n\theta\bar{X}}{\sigma^2} - \frac{n\theta^2}{2\sigma^2}\right)$
   
   where $\bar{X}$ is the observed sample mean difference.

3. **π(θ) - Mixing Distribution:** A prior distribution over possible effect sizes. Common choices:
   - **Normal:** $\pi(\theta) = N(0, \tau^2)$ - centred at zero with spread τ
   - **Uniform:** $\pi(\theta) = U[-\delta_{max}, \delta_{max}]$ - equal weight on an interval
   - **Point mass mixture:** discrete weights on specific effect sizes

4. **dθ:** Standard calculus notation indicating integration over all possible θ values.

5. **The Integral Λ_n:** Averages the likelihood ratio across all possible effect sizes, weighted by how plausible each is (according to π). This makes mSPRT robust—you don't need to guess the exact effect size in advance.

**Why use a mixture?** A standard SPRT requires specifying a single alternative θ₁. If you guess wrong, power suffers. The mixture approach hedges by considering a range of alternatives, providing robustness to effect size misspecification.

**Key Findings:**
- More powerful than fixed-horizon tests when effect exists
- Robust to effect size misspecification
- Industry standard for continuous experimentation

**Limitations:**
- Slightly lower power than GST at pre-specified times
- Requires choice of mixing distribution
- More complex implementation than fixed-horizon

---

### 3.4 Futility-Aware Early Termination

**Source:** [Know When to Fold: Futility-Aware Early Termination in Online Experiments](https://www.amazon.science/publications/know-when-to-fold-futility-aware-early-termination-in-online-experiments)

**Core Idea:** Stop experiments early when the treatment is unlikely to ever achieve statistical significance, saving resources for more promising experiments. While efficacy stopping (3.1-3.3) asks "Is the effect proven?", futility stopping asks "Is proving an effect still possible?"

**MDE Equation Modification:**
Futility analysis doesn't directly reduce MDE but reduces *wasted* sample size on experiments that won't succeed:
$$E[n_{futility}] = n_{max} \cdot P(\text{reach } n_{max}) + \sum_{k} n_k \cdot P(\text{stop at } k | \text{futile})$$

The expected sample size is reduced when futile experiments are terminated early, freeing resources for other experiments.

**Method:**
1. At each interim analysis, compute the **conditional power**—the probability of achieving significance given current data
2. If conditional power falls below a threshold (e.g., <10-20%), declare futility and stop
3. Alternatively, use **predictive probability**—the Bayesian probability that the final result will be significant

The paper moves beyond simple P-values and introduces more sophisticated ways to "fold" an experiment:

- Sequential Bayes Factors (SBF): Instead of a fixed-horizon P-value, they use a Bayesian approach that updates as data comes in. If the "evidence" for the null hypothesis (that there is no effect) becomes overwhelming, the experiment is flagged.

- Machine Learning Prediction: They developed a data-driven model that looks at the "trajectory" of an experiment. By comparing the first few days of a current test to thousands of historical experiments, the model can predict the probability that the test will eventually "turn green."

- Optimisation-Based Method: A framework that balances the risk of a "Type II error" (stopping an experiment that might have eventually won) against the "Utility" (the time and traffic saved by stopping it).

Traditional methods like Conditional Power and SBF are statistically sound, they can be overly conservative or sensitive to noise in short-duration experiments. The ML-based approach outperformed traditional methods in Amazon's environment because it could recognise patterns (like "Monday-effects" or high-variance metrics) that simple statistical formulae ignore. The Optimisation approach was the most effective for business leaders because it translated "statistical power" into "business dollars saved."

**Conditional Power Calculation:**
$$CP = P\left(Z_{final} > z_{\alpha/2} \mid Z_{current}, \theta = \hat{\theta}_{current}\right)$$

If the observed effect $\hat{\theta}_{current}$ is small or negative, CP will be low, triggering futility stopping.

**Futility Boundaries:**
- **Binding:** Once crossed, experiment must stop (more aggressive resource savings)
- **Non-binding:** Crossing is advisory; experiment can continue (preserves Type I error)

**Key Findings:**
- Can reduce average experiment duration by 20-40% for null/small effects
- Frees experimentation capacity for more promising treatments
- Particularly valuable when experimentation slots are limited
- Complements efficacy stopping for a complete sequential framework

**Limitations:**
- Risk of stopping experiments that would have eventually succeeded (Type II error increase)
- Requires careful threshold calibration
- Conditional power depends on assumed effect size under alternative
- May not be appropriate for exploratory experiments where learning is valuable even without significance

**When to Use Futility Stopping:**
- High volume of experiments competing for traffic
- Clear minimum effect size of practical interest
- Opportunity cost of continuing is high
- Confirmatory (not exploratory) experiments

---

### 3.5 Sequential Testing Methods: Comparison Table

| Method | Section | MDE Mechanism | Sample Size Reduction | Monitoring Flexibility | Planning Required | Best Use Case |
|--------|---------|---------------|----------------------|----------------------|-------------------|---------------|
| **GST** | 3.1 | Early stopping at $K$ looks | 30-50% | Pre-specified times only | High | Fixed monitoring schedule |
| **Always Valid** | 3.2 | Continuous stopping | 20-40% | Any time | Low | Automated decisions |
| **mSPRT** | 3.3 | Mixture likelihood ratio | 20-40% | Any time | Medium | Continuous monitoring |
| **Futility Stopping** | 3.4 | Stop when effect unlikely | 20-40% (for null effects) | At interim analyses | Medium | High experiment volume |

### 3.6 Comparison of Early Stopping Methods

| Method | Foundation | Decision Metric (Rule) | Futility Handling | Efficacy Handling | Low-Level Detail / Implementation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fixed Horizon** | Frequentist | p-value at end date | None (Must finish) | None | Baseline; ignores all interim data. |
| **Conditional Power (CP)** | Frequentist | Prob. of final significance | Fold if $CP < \text{Threshold}$ (e.g., 0.2) | Often used with $\alpha$-spending | Assumes future data follows current trend or the original "Minimum Detectable Effect." |
| **Bayesian Predictive Power (BPP)** | Bayesian | Posterior Prob. of success | Fold if $BPP < \text{Threshold}$ | Stop if $BPP > \text{Threshold}$ | Integrates out the uncertainty of the effect size using a prior (usually informed by history). |
| **Sequential Bayes Factor (SBF)** | Bayesian | Bayes Factor (BF) | Fold if $BF < 1/k$ | Stop if $BF > k$ | Robust to "peeking"; BF represents the likelihood of $H_1$ vs $H_0$. |
| **mSPRT** | Frequentist | Likelihood Ratio | Fold if $LR < \text{Lower Boundary}$ | Stop if $LR > \text{Upper Boundary}$ | Controls Type I error for continuous monitoring; used by Optimizely/Netflix. |
| **Alpha-Spending (O'Brien-Fleming)** | Frequentist | p-value (Adjusted) | Binding/Non-binding boundaries | Stop if $p < \alpha(t)$ | More conservative early on to avoid "false wins"; requires pre-planned interim looks. |
| **ML-Prediction (New)** | Machine Learning | Predicted Prob. of Success | Fold if $P(\text{Sig}) < \text{Threshold}$ | Not the primary focus | Uses historical meta-data and time-series trends to "guess" the final outcome. |
| **Utility Optimisation (New)** | Economic / RL | Expected Utility (Gain - Cost) | Fold if $E[U] < 0$ | Stop if $E[U] > \text{Threshold}$ | Balances "Business Value" vs "Wait Time." Unique because it incorporates the *cost of time*. |


---


## 4. Experimental Design Innovations

Methods in this section modify the experimental design to either reduce variance, increase effective sample size, or handle interference.


### 4.1 Switchback Experiments

**Source:** 
- [Design and Analysis of Switchback Experiments](https://arxiv.org/abs/2009.00148)
- [Unraveling the Interplay between Carryover Effects and Reward Autocorrelations in Switchback Experiments](https://openreview.net/forum?id=ZwcMZ443BF)
- [Robust and efficient multiple-unit switchback experimentation](https://www.arxiv.org/abs/2506.12654)
- [Data-Driven Switchback Experiments: Theoretical Tradeoffs and Empirical Bayes Designs](https://arxiv.org/abs/2306.05524)

**Core Idea:** Alternate treatment assignment over time within units.

**MDE Equation Modification:**
For a switchback design with $T$ periods:
$$MDE_{switchback} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2_{eff}}{T}}$$

Where effective variance accounts for carryover ($\gamma$) and autocorrelation ($\rho$):
$$\sigma^2_{eff} = \sigma^2\left(1 + 2\rho + \frac{\gamma^2}{\sigma^2}\right)$$

**Method:**
1. Divide time into periods of length $\Delta$
2. Randomly assign treatment to each period
3. Estimate effect from within-unit comparisons

**Key Findings:**

*Carryover-Autocorrelation Interplay (Wen et al. 2022):* [Unraveling the Interplay between Carryover Effects and Reward Autocorrelations in Switchback Experiments](https://openreview.net/forum?id=ZwcMZ443BF)
- Carryover effects and autocorrelation interact non-trivially
- Optimal period length depends on both factors
- Longer periods reduce carryover bias but increase variance

*Data-Driven Designs (Xiong et al. 2023):* [Data-Driven Switchback Experiments: Theoretical Tradeoffs and Empirical Bayes Designs](https://arxiv.org/pdf/2406.06768)
- Empirical Bayes approaches can optimise switchback patterns
- Use historical data to estimate carryover and autocorrelation
- Adaptive designs outperform fixed patterns by 20-40%

*Multiple-Unit Designs (Missault et al. 2023):* [Robust and efficient multiple-unit switchback experimentation](https://www.arxiv.org/abs/2506.12654)
- Randomise across both units and time
- More robust to model misspecification
- Better variance-bias trade-off

**Limitations:**
- Carryover effects can bias estimates
- Requires careful period length selection
- Not suitable for treatments with long-lasting effects

---

### 4.2 Staggered Rollout Designs

**Source:** [Optimal Experimental Design for Staggered Rollouts](https://arxiv.org/abs/1911.03764)

**Core Idea:** Roll out treatment to units at different times, using timing variation for identification. Uses an 'S-shape' rollout where most of the time is spent at equal allocation (this is common practice today).

The theory behind is an extension of diff in diff for staggered deployments (two-way fixed effects) using generalised least squares.

**MDE Equation Modification:**
With $K$ rollout waves and optimal timing:
$$MDE_{staggered} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2}{n \cdot DE}}$$

Where $DE$ is the design efficiency (typically 1.2-2.0 for optimal designs), effectively increasing sample size.

**Method:**
1. Randomly assign rollout times to units
2. Compare early vs. late adopters
3. Use difference-in-differences style estimation

Precision-Guided Adaptive Experimentation (PGAE) is the algo they use to estimate variance and compares treated with remaining untreated (if the variance is already small enough, termnates the experiment).

**Key Findings:**
- Can be more efficient than standard A/B when treatment effects vary over time (handling carryover effects better)
- Optimal designs depend on anticipated effect dynamics
- 20-50% variance reduction with proper design (or same with fewer units)

**Limitations:**
- Requires rollout flexibility
- Assumes parallel trends
- Complex analysis with time-varying effects



---

### 4.3 Interleaved Online Testing

**Source:** [Large-scale validation and analysis of interleaved search evaluation](https://www.cs.cornell.edu/~tj/publications/chapelle_etal_12a.pdf)

**Core Idea:** Present both treatment and control simultaneously to the same user, using preference signals to detect differences. The user sees a blended list; if they click more on results from Ranker A than Ranker B, Ranker A earns a "win" for that query.

**MDE Equation Modification:**
In standard A/B testing:
$$MDE_{AB} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2}{n}}$$

In interleaved testing, each user provides a paired comparison:
$$MDE_{interleaved} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{\sigma^2_{paired}}{n}}$$

Where $\sigma^2_{paired} \ll 2\sigma^2$ because within-user variation is eliminated. Typical MDE reduction: **50-90%**.

**Method:**
1. For each user query, interleave results from treatment and control systems
2. Observe user preference (clicks, engagement) between interleaved results
3. Use paired statistical test (e.g., sign test, paired t-test)

| Algorithm | Core Mechanism | Key Strength | Known Weakness |
| :--- | :--- | :--- | :--- |
| **Balanced Interleaving** | Ensures that at any rank $k$, the top $k$ items in $I$ contain an equal number of items from top of $A$ and $B$. | Deterministic and easy to implement. | Sensitive to "rank bias"; can be "gamed" if one ranker has a very similar but slightly worse list. |
| **Team Draft Interleaving** | Mimics captains picking teams. Each ranker takes turns picking their highest-ranked item not yet in $I$. | Highly intuitive and robust against most biases. | Can occasionally lead to sub-optimal lists if one ranker has much lower quality documents. |
| **Probabilistic Interleaving** | Items are selected based on a probability distribution (often a softmax) over the remaining items in each list. | Eliminates deterministic bias and allows for better theoretical "unbiased" properties. | Higher variance than Team Draft; requires more data to reach statistical significance. |
| **Optimised / ML-based** | Uses historical click data to weight the "credit" assigned to a click based on its position. | Highest sensitivity (requires the least amount of data). | More complex to implement and requires existing baseline data to train. |


#### Summary of Click Weighting and Credit Assignment

The paper's click weighting logic centres on the "Skip-Above" heuristic and the concept of "Responsibility." Rather than treating every click as a uniform vote, the authors assign credit by analysing a click in the context of the documents  the user bypassed. A click on a document $d_i$ at rank $i$ is only considered a strong preference signal if the user explicitly skipped documents at ranks $j < i$. This filters out position bias, where users naturally click the top  result regardless of quality. Mathematically, the preference $S$ for Ranker $A$ over Ranker $B$ for a set of clicks $C$ is calculated by aggregating the relative ranks assigned by each algorithm:

$$S(A, B) = \sum_{d \in C} \text{sgn}(\text{rank}_B(d) - \text{rank}_A(d))$$

In more advanced probabilistic models, the weight is refined using the probability that a specific ranker was "responsible" for the clicked document. This approach ensures that if both rankers suggested the same document  at the same rank, the click is neutralised (assigned zero weight) because it provides no discriminative information. 
The credit $W$ assigned to Ranker $A$ for a click on document $d$ in the interleaved list $I$ is defined as the  difference in the expected number of times each ranker would have placed that document at that position:

$$W(A|d) = P(d \in I | \text{Ranker } A) - P(d \in I | \text{Ranker } B)$$

This weighting mechanism is what allows interleaving to detect subtle differences in ranking quality with significantly smaller sample sizes than traditional A/B testing.

**Key Findings:**
- 10-100x more sensitive than traditional A/B tests for ranking systems
- click weighting based on rank
- Particularly effective for search, recommendations, ad ranking
- Each user serves as their own control

**Limitations:**
- Only applicable to ranking/recommendation systems where interleaving is possible
- Cannot measure absolute metrics (only relative preferences)
- Presentation bias can affect results

---

### 4.4 Debiased Balanced Interleaving

**Source:** [Debiased balanced interleaving at Amazon Search](https://www.amazon.science/publications/debiased-balanced-interleaving-at-amazon-search)

Balanced Interleaving (BI)—the industry standard for years—has a "fidelity" problem. If Ranker A and Ranker B are very similar (e.g., they only differ by one item at the bottom of the list), the merging process can accidentally favour one ranker over the other due to the way it fills positions. This bias is particularly problematic in "mature" search engines

**Core Idea:** Address presentation bias in interleaved experiments through debiasing techniques. For every item in the merged list, they calculate the probability that the item would have appeared at that specific position under different random interleaving outcomes. When a user clicks an item, the credit is weighted by the inverse of this probability ($1/p$).

**MDE Equation Modification:**
Standard interleaving has presentation bias. Debiased interleaving corrects this:
$$MDE_{debiased} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{\sigma^2_{paired,debiased}}{n}}$$

Where $\sigma^2_{paired,debiased}$ accounts for position bias correction.

**Method:**
1. Use balanced interleaving to ensure fair position allocation
2. Apply debiasing corrections for position effects
3. Estimate unbiased preference signals

**Key Findings:**
- Reduces bias from position effects in interleaved experiments
- Maintains sensitivity advantages of interleaving
- More accurate treatment effect estimates

**Limitations:**
- More complex implementation than standard interleaving
- Requires position bias modelling
- May slightly increase variance due to debiasing

---

### 4.5 Interleaved Online Testing in Large-Scale Systems

Interleaving is great at picking winners, but it’s historically been bad at telling you how much better a winner is and how to handle dozens of competitors at once.

**Source:** [Interleaved online testing in large-scale systems](https://www.amazon.science/publications/interleaved-online-testing-in-large-scale-systems)

**Core Idea:** Extend interleaved testing methodology to large-scale production systems with practical considerations for implementation at scale.

**MDE Equation Modification:**
Building on standard interleaved testing:
$$MDE_{large-scale} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{\sigma^2_{paired} \cdot c_{scale}}{n}}$$

Where $c_{scale}$ accounts for system-level factors (latency constraints, cache effects, load balancing) that may slightly inflate variance compared to idealized interleaving. Typically $c_{scale} \approx 1.0-1.2$.

**Method:**
1. Implement interleaving infrastructure that handles production traffic at scale
2. Account for system-level effects (caching, latency, load distribution)
3. Design credit assignment mechanisms for complex multi-stage systems
4. Handle edge cases: timeouts, partial responses, system failures

Use a Bradley-Terry model (a statistical model often used to rank sports teams based on game outcomes) to create a unified leaderboard of all candidate algorithms.

Apply False Discovery Rate (FDR) controls to ensure that a "lucky" win doesn't result in a bad algorithm being promoted.

**Key Challenges Addressed:**
- **Latency constraints:** Interleaving must not significantly increase response time
- **Cache coherence:** Treatment/control may have different cache behaviour
- **Load balancing:** Ensure fair comparison under varying system load
- **Multi-stage systems:** Credit assignment when multiple components contribute to outcome

**Key Findings:**
- Interleaving remains highly sensitive (50-90% MDE reduction) even at scale
- System-level effects can be mitigated with careful engineering
- Practical implementation requires infrastructure investment
- Works well for search, recommendations, and ad ranking at production scale

**Limitations:**
- Requires significant engineering investment for production deployment
- System-level effects may reduce sensitivity compared to idealized experiments
- Not all metrics are amenable to interleaved measurement
- Debugging and monitoring more complex than standard A/B tests

---

### 4.6 Adaptive Experimental Design and Counterfactual Inference

Traditional A/B tests are great for inference (figuring out why something works) but are expensive because they send 50% of traffic to a potentially inferior version. Adaptive designs (like Bandits) are great for payoff (quickly moving traffic to the winner) but often "break" standard statistical tests.

In the real world, user behaviour changes (e.g., weekends vs. weekdays). If an adaptive algorithm naively moves traffic based on early, unrepresentative data, it can create a feedback loop that leads to Simpson’s Paradox, where the algorithm "convinces itself" a bad treatment is good, simply because it was tested during a high-traffic or high-converting time period

**Source:** [Adaptive experimental design and counterfactual inference](https://arxiv.org/abs/2210.14369)

**Core Idea:** Use adaptive allocation and counterfactual reasoning (based on cumulative gain) to improve experimental efficiency. 

**MDE Equation Modification:**
Adaptive designs can achieve:
$$MDE_{adaptive} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2}{n_{eff}}}$$

Where $n_{eff} > n$ through optimal allocation and counterfactual imputation.

**Method:**
1. Adaptively allocate samples based on observed outcomes
2. Use counterfactual inference to impute missing potential outcomes
3. Combine observed and imputed data for estimation
4. Relies on always valid stats 

**Key Findings:**
- Can improve efficiency over fixed allocation
- Counterfactual inference provides additional information
- Particularly useful when treatment effects are heterogeneous

**Limitations:**
- More complex implementation
- Requires careful handling of adaptive inference
- May introduce bias if not properly implemented

---

### 4.7 Bias Correction for Ranking Interference

**Source:** [A Bias Correction Approach for Interference in Ranking Experiments](https://dl.acm.org/doi/abs/10.1287/mksc.2022.0046) (Marketing Science, 2023)

**Core Idea:** In ranking experiments (search, recommendations, ads), items compete for user attention within the same page. When a treatment changes one item's position, it affects the visibility and click probability of *all other items*—creating interference within the ranking. This paper provides a bias correction framework for estimating true item-level treatment effects.

**The Ranking Interference Problem:**
Consider testing a new ad creative:
- **Naive view:** Compare click rates between treatment (new creative) and control (old creative)
- **Reality:** If the new creative is more attractive, it "steals" clicks from other items on the page
- **Bias:** The treatment effect is overestimated because control items performed worse due to competition, not because the treatment is better

**Types of Ranking Interference:**

| Interference Type | Mechanism | Example |
|------------------|-----------|---------|
| **Position effects** | Higher positions get more attention | Item moved from position 3→1 gets more clicks |
| **Attention competition** | Attractive items draw attention from neighbours | Eye-catching ad reduces clicks on adjacent items |
| **Budget effects** | Users have limited click/purchase budget | Extra click on treatment item = fewer clicks elsewhere |
| **Substitution** | Similar items compete for same need | Promoting one product cannibalises similar products |

**MDE Equation Modification:**
The naive estimator has interference bias:
$\hat{\tau}_{naive} = \tau_{true} + \underbrace{\tau_{interference}}_{\text{bias from item competition}}$

The corrected estimator accounts for position and attention effects:
$\hat{\tau}_{corrected} = \hat{\tau}_{naive} - \hat{\tau}_{interference}$

Where $\hat{\tau}_{interference}$ is estimated using a structural model of user attention and click behaviour.

**Method:**
1. **Model user attention:** Specify how attention is allocated across positions (e.g., cascade model, position-based model)
2. **Estimate attention parameters:** Use historical click data to calibrate the model
3. **Compute counterfactual:** What would other items' performance be if treatment item weren't present?
4. **Correct treatment effect:** Subtract the interference component from naive estimate

**Attention Models for Bias Correction:**

| Model | Assumption | Best For |
|-------|------------|----------|
| **Position-based** | Click probability depends only on position | Simple rankings with clear position effects |
| **Cascade** | Users scan top-down, stop after satisfying click | Search results, sequential browsing |
| **Attention allocation** | Fixed attention budget distributed across items | Ad displays, recommendation carousels |
| **Examination hypothesis** | Clicks = P(examine) × P(click\|examine) | Separating position from relevance effects |

**Key Findings:**
- Naive ranking experiments can overestimate treatment effects by 20-50%
- Bias is larger when treatment items are highly attractive
- Bias correction enables accurate item-level causal inference
- Particularly important for ad effectiveness measurement

**Limitations:**
- Requires specifying and estimating attention model
- Model misspecification can introduce new biases
- Computationally intensive for large catalogues
- May not capture all interference mechanisms (e.g., cross-session effects)

**Connection to Other Methods:**
- **Debiased Interleaving (4.4):** Addresses presentation bias in A/B comparisons; this addresses item-level interference
- **Interleaved Testing (4.3):** Reduces MDE through paired comparisons; this corrects bias in effect estimates
- **Marketplace Interference (5.1):** Addresses user-level interference; this addresses item-level interference within a single user's view

**When to Use:**
- Measuring individual item/ad treatment effects
- Items compete for attention on the same page
- Treatment is expected to significantly change item attractiveness
- Accurate causal attribution is required (e.g., ad billing, content creator payments)

---

### 4.8 Experimental Design Methods: Comparison Table

| Method | Section | MDE Mechanism | MDE Reduction | Complexity | Best Use Case |
|--------|---------|---------------|---------------|------------|---------------|
| **Switchback** | 4.1 | Within-unit comparison | 20-40% | High | Marketplace/interference |
| **Staggered Rollout** | 4.2 | Timing variation | 20-50% | Medium | Gradual launches |
| **Interleaved Testing** | 4.3 | Paired comparison | 50-90% | Medium | Ranking/recommendation |
| **Debiased Interleaving** | 4.4 | Bias-corrected pairing | 50-90% | Medium-High | Search systems |
| **Large-Scale Interleaving** | 4.5 | Paired comparison at scale | 50-90% | High | Production ranking systems |
| **Adaptive Design** | 4.6 | Optimal allocation | Varies | High | Heterogeneous effects |
| **Ranking Interference Correction** | 4.7 | Item-level bias correction | Bias correction | High | Ad/item-level attribution |

**Interleaving Methods: Detailed Comparison**

The three interleaving approaches (4.3, 4.4, 4.5) represent an evolution from foundational methodology to production-ready systems:

| Aspect | Standard Interleaving (4.3) | Debiased Interleaving (4.4) | Large-Scale Interleaving (4.5) |
|--------|----------------------------|----------------------------|-------------------------------|
| **Primary Focus** | Foundational methodology | Bias correction | Production deployment |
| **Problem Solved** | Reduce MDE via paired comparisons | Fix "fidelity" bias when rankers are similar | Scale to production with many candidates |
| **Key Innovation** | Skip-Above heuristic, credit by rank | Inverse probability weighting (1/p) | Bradley-Terry model, FDR controls |
| **Comparison Type** | Binary (A vs B) | Binary (A vs B, unbiased) | Multi-way (A vs B vs C vs ...) |
| **Output** | Which ranker wins | Which ranker wins (bias-corrected) | Ranked leaderboard + effect magnitudes |
| **Best For** | General ranking experiments | Mature systems with similar rankers | Large-scale production, many algorithms |
| **Complexity** | Medium | Medium-High | High |
| **Infrastructure** | Basic interleaving | + probability computation | + Bradley-Terry, FDR, latency handling |

**When to Use Each:**
- **Standard (4.4):** Starting point for any ranking experiment; use when rankers are substantially different
- **Debiased (4.5):** When rankers are similar (e.g., A/B testing incremental improvements to mature search)
- **Large-Scale (4.6):** When comparing many algorithms simultaneously in production with strict latency/reliability requirements

---


## 5. Marketplace-Specific Methods

Methods in this section address the unique challenges of two-sided marketplaces where interference between units inflates variance.

**Two Approaches to Interference:**

Marketplace experimentation requires dealing with interference—when treating one unit affects outcomes for others. There are two fundamentally different goals:

| Goal | Approach | Question Answered | Sections |
|------|----------|-------------------|----------|
| **Handle Interference** | Choose the right experimental design | "What is the total treatment effect if we launch to everyone?" | 5.1 |
| **Decompose Interference** | Use multiple randomisation layers | "How much of the effect is direct vs. spillover?" | 5.2 |

- **Section 5.1 (Handling):** Focuses on *eliminating or containing* interference to get an unbiased estimate of the total treatment effect. Methods include cluster randomisation, synthetic control, and budget-split designs.

- **Section 5.2 (Decomposing):** Focuses on *understanding* interference by separating direct effects (treatment → treated user) from indirect effects (treatment → other users via spillovers). Uses multiple randomisation layers.

Choose **5.1 methods** when you need a launch decision. Choose **5.2 methods** when you need to understand *why* the treatment works (or doesn't).

---

### 5.1 Handling Interference: Experimental Design in Marketplaces

**Source:** [Experimental Design in Marketplaces](https://projecteuclid.org/journals/statistical-science/advance-publication/Experimental-Design-in-Marketplaces/10.1214/23-STS883.short)

**Core Idea:** To make "launch/no-launch" decisions correctly, they must move away from "naive" A/B testing. By using the right randomisation layer and accounting for market dynamics (like price and inventory), platforms can estimate the Total Treatment Effect—what would happen if the feature were rolled out to 100% of the market—rather than just the "within-experiment" difference.

**MDE Equation Modification:**
With interference, the naive estimator has inflated variance:
$$MDE_{naive} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2(\sigma^2 + \sigma^2_{interference})}{n}}$$

Proper marketplace designs reduce or eliminate $\sigma^2_{interference}$:
$$MDE_{marketplace} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2_{design}}{n_{eff}}}$$

**Method:**
- Cluster randomisation at market/region level
- Synthetic control for counterfactual construction
- Hybrid approaches combining multiple techniques

#### Cluster Randomisation

**Core Idea:** Randomise at a higher level (markets, regions, time periods) where interference is contained within clusters.

**MDE Equation for Cluster Randomisation:**
$$MDE_{cluster} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2(1 + (m-1)\rho_{ICC})}{n}}$$

Where:
- $m$ = average cluster size (users per market/region)
- $\rho_{ICC}$ = intra-cluster correlation (how similar outcomes are within a cluster)
- The term $(1 + (m-1)\rho_{ICC})$ is the **design effect**—the variance inflation from clustering

**Key Trade-off:** Cluster randomisation eliminates interference bias but increases variance because:
- Effective sample size is $n_{eff} = n / (1 + (m-1)\rho_{ICC})$
- With high ICC or large clusters, you may need 2-10x more data

**Optimal Cluster Design:**
- More clusters with fewer units each → lower variance but harder logistics
- Fewer clusters with more units → higher variance but simpler implementation
- Rule of thumb: Aim for $\geq 20$ clusters per arm for reliable inference

#### Causal Clustering for Network Interference

**Source:** [Causal clustering: design of cluster experiments under network interference](https://arxiv.org/abs/2310.14983) by Viviano (2023)

**Core Idea:** Standard cluster randomisation assumes clusters are pre-defined (e.g., geographic regions). But when you have network data showing how units are connected, you can *optimally construct* clusters to minimise cross-cluster interference. This paper provides algorithms for "causal clustering"—forming clusters that maximise within-cluster connections and minimise between-cluster connections.

**The Cluster Design Problem:**
Traditional approach: Use existing boundaries (cities, regions, stores)
- Problem: These boundaries may not align with actual interference patterns
- Example: Users in adjacent postcodes may interact heavily; users in same city may not

Causal clustering approach: Use network data to form optimal clusters
- Advantage: Clusters are designed to contain interference
- Example: Cluster users who frequently interact (friends, co-workers, neighbours)

**MDE Equation Modification:**
With optimally designed clusters, the design effect is reduced:
$MDE_{causal-cluster} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2(1 + (m-1)\rho_{ICC}^{opt})}{n}}$

Where $\rho_{ICC}^{opt} < \rho_{ICC}^{arbitrary}$ because optimal clustering groups similar/connected units together, reducing within-cluster variance relative to between-cluster variance.

**Method:**
1. **Construct network:** Build graph of unit connections (social ties, transactions, geographic proximity)
2. **Define interference model:** Specify how treatment spills over through network edges
3. **Optimise clustering:** Use graph partitioning algorithms to minimise cross-cluster edges
4. **Randomise at cluster level:** Assign entire clusters to treatment/control
5. **Estimate with interference-robust methods:** Account for residual cross-cluster spillovers

**Clustering Algorithms:**

| Algorithm | Approach | Best For |
|-----------|----------|----------|
| **Spectral clustering** | Eigenvectors of graph Laplacian | Dense networks with community structure |
| **Modularity optimisation** | Maximise within-cluster edges | Social networks |
| **Geographic + network** | Combine spatial and network data | Location-based services |
| **Balanced partitioning** | Equal-sized clusters | When cluster size matters for power |

**Key Trade-offs:**

| Design Choice | Fewer, Larger Clusters | More, Smaller Clusters |
|---------------|----------------------|----------------------|
| **Interference containment** | Better (more edges within) | Worse (more edges across) |
| **Statistical power** | Lower (fewer clusters) | Higher (more clusters) |
| **Logistics** | Simpler | More complex |
| **Optimal for** | Strong network effects | Weak network effects |

**Key Findings:**
- Optimal clustering can reduce design effect by 30-50% vs arbitrary clustering
- Particularly valuable when network structure is known but doesn't align with natural boundaries
- Enables cluster experiments in settings where geographic clustering is inappropriate
- Provides theoretical guarantees on bias-variance trade-off

**Limitations:**
- Requires network data (may not be available or complete)
- Computational cost for large networks
- Clusters may not be interpretable or actionable
- Network structure may change over time

**When to Use Causal Clustering:**
- Network data is available (social graph, transaction network, communication patterns)
- Natural boundaries don't align with interference patterns
- Strong network effects are expected
- Sufficient computational resources for optimisation

**Connection to Other Methods:**
- **Standard Cluster Randomisation:** Causal clustering optimises the cluster formation step
- **Network-Aware Bandits (5.3):** Both use network structure; this for design, bandits for allocation
- **Two-Sided Platforms:** Can combine with demand/supply-side randomisation

#### Synthetic Control Methods

**Core Idea:** When you can only treat a few large units (e.g., entire markets), construct a "synthetic" control by weighting untreated units to match the treated unit's pre-treatment trajectory.

**Method:**
1. **Pre-treatment matching:** Find weights $w_j$ for control units $j$ such that:
   $$\sum_j w_j X_j^{pre} \approx X_{treated}^{pre}$$
   where $X^{pre}$ are pre-treatment outcomes/covariates

2. **Counterfactual construction:** The synthetic control outcome is:
   $$Y_{synthetic}(t) = \sum_j w_j Y_j(t)$$

3. **Treatment effect estimation:**
   $$\hat{\tau}(t) = Y_{treated}(t) - Y_{synthetic}(t)$$

**MDE for Synthetic Control:**
$$MDE_{synth} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{\sigma^2_{pre-match} + \sigma^2_{post}}{T_{post}}}$$

Where:
- $\sigma^2_{pre-match}$ = residual variance from pre-treatment fit
- $T_{post}$ = number of post-treatment periods
- Better pre-treatment fit → lower MDE

**Key Assumptions:**
- **Parallel trends:** Absent treatment, treated and synthetic control would follow same trajectory
- **No anticipation:** Treatment doesn't affect pre-treatment outcomes
- **SUTVA at cluster level:** No interference between clusters

**Synthetic Control vs. Difference-in-Differences:**

| Aspect | Synthetic Control | Difference-in-Differences |
|--------|------------------|--------------------------|
| **Control construction** | Weighted combination | Simple average or single unit |
| **Pre-treatment fit** | Optimised weights | Assumes parallel trends |
| **Best for** | Few treated units, good donor pool | Many treated units, clear parallel trends |
| **Inference** | Permutation-based | Standard errors |

#### Budget-Split Design

**Source:** [Trustworthy Online Marketplace Experimentation with Budget-split Design](https://arxiv.org/abs/2012.08724)

**Core Idea:** Split advertiser budgets rather than users to handle marketplace interference. Creates isolated "worlds" preventing budget cross-talk through cannibalisation effects. This is a specific application of cluster randomisation where the "cluster" is an advertiser's budget.

**MDE Equation Modification:**
Budget-split eliminates interference:
$$MDE_{budget-split} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2_{budget}}{n_{impressions}}}$$

Where $\sigma^2_{budget}$ is typically much smaller than $\sigma^2(1 + (n_c-1)\rho_{ICC})$ from standard cluster randomisation.

**Method:**
1. Split each advertiser's budget into treatment and control portions
2. Run separate auctions for each budget portion
3. Estimate treatment effect from budget-level comparison

**Key Findings:**
- Eliminates interference from budget competition
- Provides unbiased estimates in ad marketplace settings
- 30-50% MDE reduction vs. standard cluster randomisation

**Limitations:**
- Requires budget-level tracking infrastructure
- May not capture all marketplace dynamics (e.g., cross-advertiser effects)
- Assumes budget is the primary interference mechanism
- Specific to ad marketplaces with budget constraints

**Budget-Split vs. Switchback Selection:**

The choice depends on **Budget Density**—whether a campaign has enough spend for stable pacing when partitioned.

$$B_{min} = N_{min} \times avg\_CPC$$

Where $N_{min}$ is minimum daily events (typically 50-100) for pacing convergence.

| Budget Density | Recommended Design | Rationale |
|----------------|-------------------|-----------|
| **High** (Budget/2 > B_min) | Budget-Split | High power, zero temporal bias |
| **Medium** | Biased Split (90/10) | Meets B_min for larger arm |
| **Low** | Switchback | Preserves market integrity |

**Switch to Switchback when:** Pacing shows >20% hourly variance, arms fail to spend allocation, or >30% of budget consumed in single auction.

#### Hybrid Approaches

Many platforms combine methods:
1. **Cluster + CUPED:** Cluster randomisation for interference, CUPED for within-cluster variance reduction
2. **Synthetic Control + Matching:** Match on pre-treatment covariates, then apply synthetic control
3. **Switchback within Clusters:** Time-based randomisation within geographic clusters

**Key Findings:**
- No single best approach for all marketplaces
- Design choice depends on interference structure
- Hybrid approaches often optimal

**Limitations:**
- Cluster randomisation increases variance due to fewer effective units (design effect)
- Synthetic control requires parallel trends assumption and good donor pool
- Complex to implement and validate
- May need domain expertise to define appropriate clusters
- Trade-off: bias reduction vs. variance inflation

#### Experimenting in Equilibrium

**Source:** [Experimenting in Equilibrium](https://arxiv.org/abs/1903.02124) by Wager & Xu (2021)

**Core Idea:** Standard A/B tests assume that treating some users doesn't affect the outcomes of control users (SUTVA). In marketplaces, this assumption fails because the experiment itself shifts market equilibrium—prices, supply, and demand adjust in response to the treatment. This paper provides a framework for estimating treatment effects when the experiment perturbs market equilibrium.

**The Equilibrium Problem:**
Consider a ride-sharing experiment testing a new pricing algorithm:
- **Partial equilibrium view:** Treatment users see different prices → different behaviour
- **General equilibrium reality:** Changed demand from treatment users → drivers reallocate → wait times change for *everyone* (including control)

The naive A/B test estimate captures only the partial equilibrium effect, but the full rollout would produce a different (general equilibrium) effect.

**MDE Equation Modification:**
The paper shows that equilibrium effects introduce bias rather than variance:
$\hat{\tau}_{naive} = \tau_{partial} + \underbrace{\tau_{equilibrium}}_{\text{bias from equilibrium shift}}$

To estimate the true full-rollout effect $\tau_{GE}$:
$MDE_{equilibrium} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2}{n}} + \text{equilibrium correction}$

**Method:**
1. **Model the market:** Specify supply/demand curves and equilibrium conditions
2. **Estimate structural parameters:** Use experimental variation to identify demand/supply elasticities
3. **Simulate counterfactual equilibrium:** Compute what would happen under full rollout
4. **Correct the treatment effect:** Adjust naive estimate for equilibrium shift

**Key Insight - When Equilibrium Effects Matter:**

| Market Characteristic | Equilibrium Effect | Example |
|----------------------|-------------------|---------|
| **Inelastic supply** | Large | Ride-sharing (fixed drivers) |
| **Elastic supply** | Small | E-commerce (infinite inventory) |
| **Thick market** | Small | Large ad marketplace |
| **Thin market** | Large | Local services marketplace |
| **Price-setting treatment** | Large | Dynamic pricing experiments |
| **Non-price treatment** | Usually small | UI changes |

**Practical Guidance:**
- **Small treatment fraction:** Equilibrium effects scale with treatment fraction; smaller experiments have smaller bias
- **Short duration:** Less time for market to re-equilibrate
- **Structural estimation:** When equilibrium effects are large, use structural models to extrapolate

**Key Findings:**
- Naive A/B tests can be severely biased in marketplace settings
- Bias direction depends on market structure (can over- or under-estimate)
- Structural modelling enables extrapolation from partial to general equilibrium
- Particularly relevant for pricing, matching, and allocation experiments

**Limitations:**
- Requires economic modelling expertise
- Structural assumptions may be wrong
- Computationally intensive for complex markets
- May not be necessary for non-price treatments or thick markets

**When to Use Equilibrium Correction:**
- Treatment affects prices or allocation mechanisms
- Market has constrained supply (drivers, inventory, ad slots)
- Treatment fraction is large (>10%)
- Full rollout decision is high-stakes

#### Two-Sided Platform Experimentation

**Source:** [Experimental Design in Two-Sided Platforms: An Analysis of Bias](https://arxiv.org/abs/2002.05670) by Johari, Li, Liskovich, Weintraub (2022)

**Core Idea:** In two-sided platforms (e.g., ride-sharing, e-commerce, ad marketplaces), experimenters must choose *which side* to randomise. This paper provides a rigorous analysis of the bias that arises from different randomisation strategies and guidance on optimal experimental design.

**The Two-Sided Randomisation Problem:**
Consider a ride-sharing platform testing a new matching algorithm:
- **Randomise riders:** Some riders get new algorithm, others get old
- **Randomise drivers:** Some drivers use new algorithm, others use old
- **Randomise both:** Complex factorial design

Each choice leads to different bias patterns because treating one side affects outcomes on the other side through the matching process.

**Bias Analysis by Randomisation Strategy:**

| Strategy | Bias Source | Bias Direction | When to Use |
|----------|-------------|----------------|-------------|
| **Demand-side (buyers/riders)** | Supply reallocation to treatment | Overestimates treatment effect | Treatment primarily affects demand |
| **Supply-side (sellers/drivers)** | Demand reallocation to treatment | Overestimates treatment effect | Treatment primarily affects supply |
| **Two-sided (both)** | Reduced but not eliminated | Smaller bias | Treatment affects both sides |
| **Market-level cluster** | No within-market interference | Unbiased but high variance | Gold standard when feasible |

**MDE Equation Modification:**
The paper shows that one-sided randomisation introduces bias proportional to market thickness:
$\hat{\tau}_{one-sided} = \tau_{true} + \underbrace{\frac{\tau_{true}}{M}}_{\text{interference bias}}$

Where $M$ is a measure of market thickness (number of potential matches). In thin markets, bias can be substantial.

For two-sided randomisation with treatment fractions $p_D$ (demand) and $p_S$ (supply):
$\text{Bias} \propto p_D \cdot p_S \cdot \tau_{true}$

**Method:**
1. **Characterise the platform:** Identify demand side, supply side, and matching mechanism
2. **Assess market thickness:** Thin markets have larger interference bias
3. **Choose randomisation strategy:**
   - If treatment affects one side primarily → randomise that side
   - If treatment affects both sides → consider two-sided or cluster randomisation
4. **Estimate and correct bias:** Use structural models or design-based corrections

**Key Insight - Optimal Randomisation Side:**

| Treatment Type | Recommended Randomisation | Rationale |
|---------------|--------------------------|-----------|
| Buyer-facing UI change | Demand-side | Treatment only directly affects buyers |
| Seller onboarding improvement | Supply-side | Treatment only directly affects sellers |
| Matching algorithm change | Two-sided or cluster | Affects both sides through matching |
| Pricing change | Market-level cluster | Strong equilibrium effects |
| Commission rate change | Supply-side | Primarily affects seller behaviour |

**Practical Guidance:**
- **Thin markets:** Bias is larger; prefer cluster randomisation or smaller treatment fractions
- **Thick markets:** One-sided randomisation may be acceptable; bias is smaller
- **Symmetric treatments:** Two-sided randomisation reduces bias
- **Asymmetric treatments:** Randomise the side most affected by treatment

**Key Findings:**
- One-sided randomisation systematically overestimates treatment effects
- Bias magnitude depends on market thickness and treatment fraction
- Two-sided randomisation reduces but doesn't eliminate bias
- Market-level cluster randomisation eliminates bias but increases variance

**Limitations:**
- Requires understanding of platform matching mechanism
- Market thickness may vary across segments/times
- Two-sided randomisation is operationally complex
- Cluster randomisation may not be feasible for global platforms

**Connection to Other Methods:**
- **Equilibrium Correction (Wager & Xu):** Provides structural approach to correct bias
- **Budget-Split:** A form of supply-side randomisation for ad platforms
- **Switchback:** Time-based alternative when spatial clustering isn't feasible

### 5.2 Decomposing Interference: Multiple Randomisation Designs

**Source:** 
- [Measuring direct and Indirect Impacts in a Multi-Sided Marketplace](https://drive.google.com/file/d/1auP6JpB0DxIQRCIDazum5uFlm9BtTkiS/view)
- [Multiple Randomisation Designs: Estimation and Inference with Interference](https://arxiv.org/abs/2112.13495)

**Core Idea:** Use multiple layers of randomisation to identify direct and indirect effects.

If you give a 20% discount to a rider in Uber, they are more likely to book a ride (direct effect).

Because that rider booked a ride, there is one fewer driver available for everyone else. This might increase wait times for the "Control" group, making the discount look more successful than it actually is by "stealing" supply from the control (indirect effect).

**MDE Equation Modification:**
For direct effect estimation with two-stage randomisation:
$$MDE_{direct} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2_{direct}}{n_1 \cdot n_2}}$$

Where $n_1$ = number of clusters, $n_2$ = individuals per cluster.

**Method:**
1. First randomisation: Assign clusters to treatment intensity
2. Second randomisation: Assign individuals within clusters
3. Estimate direct effects, spillovers, and equilibrium effects separately

**Key Findings:**
- Can decompose total effect into components
- Enables understanding of interference mechanisms
- More informative than single randomisation

**Limitations:**
- More complex designs may increase variance for any single effect
- Requires larger sample sizes for precise estimation
- Trade-off between precision and identification

---

### 5.3 Multi-Armed Bandits with Network Interference

In a network of $N$ units with $A$ possible treatments, the total number of configurations is $A^N$. This "curse of dimensionality" makes traditional bandit algorithms (which require exploring every option) impossible to use.

**Source:** [Multi-Armed Bandits with Network Interference](https://arxiv.org/abs/2405.18621)

**Core Idea:** To make the problem tractable, the authors assume **Sparse Network Interference**:
* The reward of unit $i$ is only influenced by its own treatment and the treatments of its immediate neighbors in a graph.
* The maximum degree of this graph ($d$) is much smaller than $N$.

The paper uses **Discrete Fourier Analysis** to decompose the reward function. By viewing treatment assignments as inputs to a function, they show that:
* Local interference implies that the reward function has a **sparse representation** in the Fourier domain.
* The problem can be solved by estimating a limited number of "Fourier coefficients" rather than trying every possible treatment combination.

The paper introduces algorithms for two scenarios:
* **Known Network:** Uses an "Explore-Then-Commit" strategy to estimate coefficients.
* **Unknown Network:** Employs **Lasso-based regression** to simultaneously identify the interference structure (who affects whom) and the optimal treatment strategy.

**MDE Equation Modification:**
Standard bandit regret bound:
$$R_T = O(\sqrt{KT\log T})$$

With network interference, network-aware bandits achieve:
$$R_T^{network} = O(\sqrt{KT\log T \cdot (1 + \rho_{network})})$$

**Method:**
1. Model interference structure (network graph)
2. Modify reward estimates to account for spillovers
3. Adapt exploration strategy to network topology


Treatments are mapped to a discrete space (e.g., $x_i \in \{+1, -1\}$ for binary actions). The Fourier basis functions $\phi_S(x)$ are defined for every subset of units $S \subseteq \{1, \dots, N\}$ as the product of their treatments:

$$\phi_S(x) = \prod_{i \in S} x_i$$

The total reward $f(x)$ is represented as a weighted sum of these basis functions. The weights $\hat{f}(S)$ are the **Fourier coefficients**:

$$f(x) = \sum_{S \subseteq \{1, \dots, N\}} \hat{f}(S) \phi_S(x)$$

Under the **Network Interference** assumption, the complexity collapses:
* **Locality:** If unit $i$ is only influenced by its neighbors, then $\hat{f}(S) = 0$ for any set $S$ that exceeds the neighborhood radius.
* **Dimensionality Reduction:** Instead of $2^N$ coefficients, we only need to estimate coefficients for small, local subsets $S$, making the learning process polynomial rather than exponential.


| Fourier Component | Terminology | Interpretation |
| :--- | :--- | :--- |
| $\hat{f}(\emptyset)$ | Degree 0 | The **baseline reward** (intercept) when no treatments are applied. |
| $\hat{f}(\{i\})$ | Degree 1 | The **Direct Effect**: How treating unit $i$ changes its own outcome. |
| $\hat{f}(\{i, j\})$ | Degree 2 | The **Interference Effect**: How the interaction between $i$ and $j$ affects the outcome. |
| Higher Orders | Degree $> 2$ | Complex, multi-unit spillover effects (usually assumed to be zero in sparse networks). |


**Key Findings:**
- Ignoring interference leads to suboptimal policies
- Network-aware bandits can improve regret by 30-50%
- Applicable to ad targeting with social effects

**Limitations:**
- Requires knowledge of network structure
- Computational complexity scales with network size
- May not capture all interference mechanisms

#### Adversarial Bandits with Interference

**Source:** [Multi-Armed Bandits with Interference: Bridging Causal Inference and Adversarial Bandits](https://arxiv.org/abs/2402.01845) by Jia, Frazier, Kallus (2024)

**Core Idea:** While the sparse network approach above assumes a known/learnable interference structure, this paper takes an **adversarial** perspective—handling worst-case interference without assuming a specific network topology. It bridges causal inference (potential outcomes framework) with adversarial bandits (no distributional assumptions on rewards).

**Key Distinction from Network Approach:**

| Aspect | Sparse Network (above) | Adversarial (this paper) |
|--------|------------------------|--------------------------|
| **Interference assumption** | Known sparse graph structure | Arbitrary, worst-case |
| **Reward model** | Fourier decomposition | Potential outcomes |
| **Regret benchmark** | Stochastic optimal policy | Best fixed policy in hindsight |
| **Best for** | Social networks with known structure | Unknown/adversarial interference |

**The Causal Inference Bridge:**
The paper frames interference using the **potential outcomes** framework from causal inference:
- Each unit $i$ has potential outcomes $Y_i(a_1, ..., a_N)$ depending on *all* treatment assignments
- Standard bandits assume $Y_i(a_i)$—no interference (SUTVA)
- This paper relaxes SUTVA while maintaining tractable regret bounds

**Method:**
1. **Exposure mapping:** Define how interference propagates (e.g., fraction of treated neighbours)
2. **Robust policy learning:** Optimise for worst-case interference patterns
3. **Regret decomposition:** Separate regret from treatment selection vs. interference estimation

**Regret Bound:**
$R_T^{adversarial} = O(\sqrt{T \cdot K \cdot \log(K)} + T \cdot \epsilon_{interference})$

Where $\epsilon_{interference}$ captures the "price of interference"—additional regret due to spillovers that cannot be eliminated.

**Key Findings:**
- Provides regret guarantees even under adversarial interference
- No need to know or estimate network structure
- Connects bandit literature to causal inference literature
- Applicable when interference patterns are unknown or adversarial

**Limitations:**
- More conservative bounds than network-aware approach when structure is known
- May be overly pessimistic if interference is actually benign
- Computational complexity for large action spaces

**When to Use Each Approach:**

| Scenario | Recommended Approach |
|----------|---------------------|
| Known social network structure | Sparse Network (Fourier) |
| Unknown interference patterns | Adversarial |
| Competitive marketplace dynamics | Adversarial |
| Stable, learnable spillovers | Sparse Network |
| Worst-case guarantees needed | Adversarial |


---

### 5.4 Marketplace Methods: Comparison Table

| Method | Section | MDE Mechanism | Interference Handling | Complexity | Best Use Case |
|--------|---------|---------------|----------------------|------------|---------------|
| **Cluster Randomisation** | 5.1 | Reduce within-cluster interference | Good | Medium | Geographic markets |
| **Causal Clustering** | 5.1 | Optimal cluster formation | Very Good | High | Network-based interference |
| **Synthetic Control** | 5.1 | Weighted counterfactual construction | Good | Medium-High | Few treated units |
| **Budget-Split** | 5.1 | Eliminates budget interference | Excellent | High | Ad marketplaces |
| **Equilibrium Correction** | 5.1 | Structural modelling | Excellent | Very High | Pricing/allocation experiments |
| **Two-Sided Randomisation** | 5.1 | Optimal side selection | Good | Medium | Two-sided platforms |
| **Multiple Randomisation** | 5.2 | Separate direct/indirect effects | Excellent | High | Effect decomposition |
| **Network-Aware Bandits** | 5.3 | Model spillovers (known structure) | Good | High | Social networks |
| **Adversarial Bandits** | 5.3 | Worst-case interference | Good | High | Unknown interference |

**Marketplace Design Selection Guide:**

| Scenario | Recommended Method | Why |
|----------|-------------------|-----|
| Ad marketplace with budget competition | Budget-Split | Eliminates cannibalisation |
| Few large markets to treat | Synthetic Control | Constructs counterfactual from donor pool |
| Many small markets | Cluster Randomisation | Standard approach, sufficient power |
| Network data available, strong spillovers | Causal Clustering | Optimises cluster boundaries for interference |
| Pricing/allocation experiments | Equilibrium Correction | Accounts for market re-equilibration |
| Two-sided platform (ride-sharing, e-commerce) | Two-Sided Randomisation | Reduces interference bias |
| Need to understand spillovers | Multiple Randomisation | Decomposes direct/indirect effects |
| Social network effects (known graph) | Network-Aware Bandits | Models interference structure |
| Unknown/adversarial interference | Adversarial Bandits | Robust worst-case guarantees |

---


## 6. Cross-Experiment Learning

### 6.1 Learning Across Experiments and Time

**Source:** [Learning Across Experiments and Time: Tackling Heterogeneity in A/B Testing](https://www.amazon.science/publications/learning-across-experiments-and-time-tackling-heterogeneity-in-a-b-testing)

**Core Idea:** Use information from past experiments to improve current ones. While CUPED uses pre-experiment *user data*, this approach uses pre-experiment *experiment results*—leveraging the historical database of A/B tests to reduce variance and improve power.

**MDE Equation Modification:**
With informative prior from historical experiments:
$$MDE_{meta} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2}{n + n_{prior}}}$$

Where $n_{prior}$ is the effective sample size contributed by historical data. This can yield 20-40% MDE reduction.

**The Heterogeneity Challenge:**
Treatment effects vary across:
- **Time:** Seasonality, trends, novelty effects
- **Segments:** User types, geographies, platforms
- **Experiments:** Different features, contexts

Naive pooling ignores this heterogeneity; the paper addresses how to learn *despite* it.

**Method:**
1. **Meta-analysis:** Pool estimates across similar past experiments
2. **Hierarchical models:** Share information through partial pooling
   $$\theta_i \sim N(\mu, \tau^2)$$ where $\mu$ is the population mean effect and $\tau^2$ captures between-experiment variance
3. **Bayesian updating:** Use posterior from past experiments as prior for current
4. **Time-series modelling:** Account for temporal dynamics in effect sizes

**Effective Prior Sample Size:**
The information from $K$ historical experiments with average sample size $\bar{n}$ contributes:
$$n_{prior} \approx \frac{K \cdot \bar{n}}{1 + \frac{\sigma^2}{\tau^2}}$$

When between-experiment heterogeneity ($\tau^2$) is low relative to within-experiment variance ($\sigma^2$), historical data is highly informative.

**Key Findings:**
- Historical data can reduce MDE by 20-40%
- Heterogeneity must be explicitly modeled (not ignored)
- Works best for mature platforms with many similar experiments
- Requires experiment database infrastructure

**Limitations:**
- Must handle concept drift (effects change over time)
- Privacy considerations for data sharing across teams
- Requires substantial historical data (dozens of similar experiments)
- Risk of bias if historical experiments are systematically different
- Computational complexity for hierarchical models

**When to Use:**
- Mature experimentation platform with rich history
- Running similar experiments repeatedly (e.g., UI tweaks, pricing tests)
- Effect sizes are relatively stable across time/segments
- Experimentation velocity is high (many experiments per quarter)

---

## 7. Master Comparison of All Methods

### 7.1 Summary Table: All Methods by MDE Reduction Mechanism

| Method | Section | Category | MDE Equation Change | Typical MDE Reduction | Complexity | Data Requirements |
|--------|---------|----------|---------------------|----------------------|------------|-------------------|
| **CUPED** | 2.1 | Variance Reduction | $\sigma^2 \rightarrow \sigma^2(1-\rho^2)$ | 20-50% | Low | Pre-experiment data |
| **CUPAC** | 2.2 | Variance Reduction | $\sigma^2 \rightarrow \sigma^2(1-\rho_{ML}^2)$ | 30-60% | Medium | ML infrastructure |
| **Multi-Covariate** | 2.3 | Variance Reduction | $\sigma^2 \rightarrow \sigma^2(1-R^2)$ | 30-60% | Medium | Multiple covariates |
| **Stratification** | 2.4 | Variance Reduction | $\sigma^2 \rightarrow \sigma^2_{within}$ | 10-30% | Low | Stratification vars |
| **Temporal Stratification** | 2.5 | Variance Reduction | $\sigma^2 \rightarrow \sigma^2(1-\eta^2_{temporal})$ | 10-30% | Medium | Time-series data |
| **Semiparametric** | 2.6 | Variance Reduction | $\sigma^2 \rightarrow \sigma^2_{eff}$ | Theoretical bound | High | Depends on estimator |
| **Pre+In Combined** | 2.7 | Variance Reduction | $\sigma^2 \rightarrow \sigma^2(1-R^2_{combined})$ | 30-60% | Medium | Pre + in-experiment |
| **Sparse/Delayed Surrogates** | 2.8 | Variance Reduction | $\sigma^2 \rightarrow \sigma^2(1-\rho^2_{surrogate})$ | 40-70% | Medium | Surrogate metrics |
| **GST** | 3.1 | Sequential Testing | $n \rightarrow n \cdot ASN_{ratio}$ | 30-50% (sample) | Medium | Pre-planned schedule |
| **Always Valid** | 3.2 | Sequential Testing | Early stopping any time | 20-40% (sample) | Medium | Continuous data |
| **mSPRT** | 3.3 | Sequential Testing | $n \rightarrow E[n_{stop}]$ | 20-40% (sample) | Medium | Continuous data |
| **Futility Stopping** | 3.4 | Sequential Testing | Stop when effect unlikely | 20-40% (for null) | Medium | Interim analyses |
| **Switchback** | 4.1 | Design Innovation | Within-unit comparison | 20-40% | High | Time-series data |
| **Staggered Rollout** | 4.2 | Design Innovation | $n \rightarrow n \cdot DE$ | 20-50% | Medium | Rollout flexibility |
| **Interleaved Testing** | 4.3 | Design Innovation | $2\sigma^2 \rightarrow \sigma^2_{paired}$ | 50-90% | Medium | Ranking systems |
| **Debiased Interleaving** | 4.4 | Design Innovation | Bias-corrected pairing | 50-90% | Medium-High | Search systems |
| **Large-Scale Interleaving** | 4.5 | Design Innovation | Paired comparison at scale | 50-90% | High | Production ranking |
| **Adaptive Design** | 4.6 | Design Innovation | Optimal allocation | Varies | High | Heterogeneous effects |
| **Ranking Interference Correction** | 4.7 | Design Innovation | Item-level bias correction | Bias correction | High | Ad/item attribution |
| **Cluster Randomisation** | 5.1 | Marketplace | Reduce interference | Varies | Medium | Geographic markets |
| **Synthetic Control** | 5.1 | Marketplace | Weighted counterfactual | Varies | Medium-High | Few treated units |
| **Budget-Split** | 5.1 | Marketplace | Eliminates interference | 30-50% vs cluster | High | Ad marketplaces |
| **Equilibrium Correction** | 5.1 | Marketplace | Structural modelling | Bias correction | Very High | Pricing experiments |
| **Two-Sided Randomisation** | 5.1 | Marketplace | Optimal side selection | Bias reduction | Medium | Two-sided platforms |
| **Multiple Randomisation** | 5.2 | Marketplace | Separate effects | Enables identification | High | Multi-level data |
| **Network-Aware Bandits** | 5.3 | Marketplace | Model spillovers (known) | Varies | High | Social networks |
| **Adversarial Bandits** | 5.3 | Marketplace | Worst-case interference | Varies | High | Unknown interference |
| **Cross-Experiment** | 6.1 | Learning | $n \rightarrow n + n_{prior}$ | 20-40% | High | Historical experiments |

### 7.2 Method Selection Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MDE Reduction Decision Tree                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. What type of system are you testing?                                    │
│     ├─ Ranking/Recommendation → Consider INTERLEAVED TESTING (50-90% MDE↓)  │
│     │   ├─ Similar rankers? → Debiased Interleaving (4.4)                   │
│     │   └─ Many algorithms? → Large-Scale Interleaving (4.5)                │
│     └─ Other → Continue to step 2                                           │
│                                                                             │
│  2. Is your primary metric sparse or delayed?                               │
│     ├─ Yes (>90% zeros or >7 day delay) → Surrogate-based VR (2.8, 40-70%)  │
│     └─ No  → Continue to step 3                                             │
│                                                                             │
│  3. Do you have pre-experiment data?                                        │
│     ├─ Yes → Start with CUPED/CUPAC (20-60% MDE↓)                           │
│     │   └─ Also have in-experiment control? → Pre+In Combined (2.7)         │
│     └─ No  → Consider stratification (10-30% MDE↓)                          │
│                                                                             │
│  4. Is there interference between units?                                    │
│     ├─ Yes → Consider:                                                      │
│     │        ├─ Budget-split (ad marketplaces, 5.1)                         │
│     │        ├─ Switchback (time-based interference, 4.1)                   │
│     │        ├─ Cluster randomisation (geographic, 5.1)                     │
│     │        ├─ Multiple randomisation (effect decomposition, 5.2)          │
│     │        └─ Network-aware bandits (social effects, 5.3)                 │
│     └─ No  → Standard randomisation + variance reduction                    │
│                                                                             │
│  5. Do you need continuous monitoring?                                      │
│     ├─ Yes → mSPRT (3.3) or Always Valid Inference (3.2)                    │
│     │   └─ High experiment volume? → Add Futility Stopping (3.4)            │
│     └─ No  → GST (3.1) or fixed-horizon                                     │
│                                                                             │
│  6. Are treatment effects non-stationary over time?                         │
│     ├─ Yes → Temporal Stratification (2.5)                                  │
│     └─ No  → Standard analysis                                              │
│                                                                             │
│  7. Do you have historical experiment data?                                 │
│     ├─ Yes → Leverage cross-experiment learning (6.1, 20-40% MDE↓)          │
│     └─ No  → Build experiment database for future                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Combinability Matrix

| Method 1 | Method 2 | Combinable? | Expected Combined Benefit |
|----------|----------|-------------|---------------------------|
| CUPED | Stratification | ✓ | Multiplicative (40-70% total) |
| CUPED | Temporal Stratification | ✓ | Multiplicative (handles time + individual variance) |
| CUPED | GST | ✓ | Apply CUPED at each look |
| CUPED | Switchback | ✓ | Use pre-period as covariate |
| CUPED | Interleaved | ✓ | Both reduce variance |
| Sparse/Delayed Surrogates | Sequential | ✓ | Surrogate VR + early stopping |
| Pre+In Combined | Stratification | ✓ | Apply within strata |
| Pre+In Combined | Sequential | ✓ | Apply at each analysis |
| Stratification | Temporal Stratification | ~ | Overlapping variance components |
| Stratification | GST | ✓ | Stratify then sequential |
| GST | mSPRT | ✗ | Choose one approach |
| GST | Futility Stopping | ✓ | Efficacy + futility boundaries |
| mSPRT | Futility Stopping | ✓ | Add futility to continuous monitoring |
| Switchback | Budget-Split | ~ | Depends on context |
| Interleaved | CUPED | ✓ | Additional variance reduction |
| Cross-Experiment | CUPED | ✓ | Prior + covariate adjustment |
| Cross-Experiment | Sequential | ✓ | Informative prior + early stopping |

---

## 8. Practical Recommendations for Ads A/B Testing

### 8.1 Quick Wins (Low Effort, High Impact)

1. **Implement CUPED with pre-experiment ad engagement**
   - Use 7-14 day pre-period
   - Expected: 30-50% MDE reduction
   - **Gotcha:** Verify correlation stability—if ρ < 0.3, benefit is minimal
   - **Gotcha:** For ratio metrics (CTR), use delta method linearisation first

2. **Consider Interleaved Testing for ranking changes**
   - If testing ad ranking algorithms
   - Expected: 50-90% MDE reduction
   - **Gotcha:** Only works for ranking/recommendation systems—not applicable to UI changes
   - **Gotcha:** Use debiased interleaving to avoid position bias (see Section 4.4)

3. **Stratify by advertiser size/vertical**
   - Reduces between-group variance
   - Expected: 10-20% additional reduction
   - **Gotcha:** Too many strata (>20) can increase variance; use coarse groupings

### 8.2 Medium-Term Investments

1. **Build CUPAC infrastructure**
   - ML models for outcome prediction
   - Expected: 10-30% over basic CUPED
   - **Gotcha:** Model training cost is non-trivial—budget 2-4 weeks engineering time
   - **Gotcha:** Stale models degrade performance; plan for weekly/monthly retraining
   - **Gotcha:** Validate model predictions don't correlate with treatment assignment (SRM risk)

2. **Implement sequential testing**
   - mSPRT for continuous monitoring
   - Reduces time-to-decision by 30-50%
   - **Gotcha:** Ad-hoc peeking without proper boundaries inflates false positives to 20-30%
   - **Gotcha:** Sequential methods trade off precision for speed—final estimates are wider

3. **Develop switchback capability**
   - For marketplace-level experiments
   - Use data-driven design optimisation
   - **Gotcha:** Carryover effects bias estimates—use periods ≥2x expected carryover duration
   - **Gotcha:** Time-of-day and day-of-week effects require careful period design

### 8.3 Long-Term Platform Capabilities

1. **Budget-split design for ad experiments**
   - Eliminates marketplace interference
   - Gold standard for ad platform experiments
   - **Gotcha:** Requires significant infrastructure investment (budget isolation, shadow auctions)
   - **Gotcha:** May not be feasible for all experiment types (e.g., creative testing)

### 8.4 Common Pitfalls to Avoid

| Pitfall | Why It Happens | How to Avoid |
|---------|---------------|--------------|
| **Using CUPED for new users** | No pre-experiment data exists | Use stratification or accept higher MDE |
| **Interleaving for non-ranking experiments** | Misunderstanding method scope | Reserve for ranking/recommendation only |
| **Ignoring ratio metric structure** | Treating CTR like additive metric | Apply delta method before CUPED |
| **Peeking without sequential boundaries** | Impatience, pressure for results | Implement proper mSPRT or GST |
| **CUPAC with stale models** | Neglecting model maintenance | Schedule regular retraining |
| **Switchback with short periods** | Underestimating carryover | Use periods ≥2x carryover duration |
| **Over-stratification** | Trying to control too many variables | Limit to 5-10 meaningful strata |
| **Combining incompatible methods** | Not understanding method assumptions | Consult combinability matrix (7.3) |

### 8.5 Metric-Specific Recommendations

| Metric Type | Primary Challenge | Recommended Approach | Expected MDE Reduction |
|-------------|------------------|---------------------|----------------------|
| **CTR** | Ratio metric, denominator variance | Delta method + CUPED | 20-40% |
| **Conversion rate** | Binary outcome, low base rate | Stratify by propensity + CUPED | 30-50% |
| **Revenue** | Heavy tails, extreme values | Winsorise at 99th percentile + CUPED | 40-60% |
| **Revenue per session** | Ratio + heavy tails | Winsorise + delta method + CUPED | 30-50% |
| **Time on site** | Right-skewed distribution | Log transform + CUPED | 30-50% |
| **Ad impressions** | Count data, overdispersion | Stratify by user activity + CUPED | 20-40% |

### 8.6 Long-Term Platform Capabilities (continued)

2. **Cross-experiment learning system**
   - Meta-analysis infrastructure
   - Bayesian priors from historical data

3. **Automated experiment design selection**
   - Choose optimal method based on context
   - Adaptive sample size determination

---

## 9. Open Research Questions

1. **Optimal combination of methods:** How to best combine variance reduction, sequential testing, and design innovations?

2. **Interference in ad auctions:** Better models for budget-mediated interference

3. **Long-term effects:** Methods for detecting delayed treatment effects with reduced MDE

4. **Heterogeneous treatment effects:** Variance reduction for subgroup analyses

5. **Privacy-preserving methods:** MDE reduction under differential privacy constraints

6. **Interleaving extensions:** Applying interleaved testing beyond ranking systems

---

## Appendix A: Mathematical Details

### A.1 Baseline MDE Formula Derivation

For a two-sample t-test with equal sample sizes $n$ per arm:
$$MDE = (z_{1-\alpha/2} + z_{1-\beta}) \cdot SE(\hat{\tau})$$

Where $SE(\hat{\tau}) = \sqrt{\frac{\sigma^2_T}{n} + \frac{\sigma^2_C}{n}} = \sqrt{\frac{2\sigma^2}{n}}$ (assuming equal variances).

### A.2 CUPED Variance Reduction

Given outcome $Y$ and covariate $X$:
$$\hat{Y}_{adj} = Y - \theta(X - E[X])$$

The variance-minimising $\theta$ is:
$$\theta^* = \frac{Cov(Y, X)}{Var(X)}$$

This gives:
$$Var(\hat{Y}_{adj}) = Var(Y) - \frac{Cov(Y,X)^2}{Var(X)} = Var(Y)(1 - \rho_{XY}^2)$$

### A.2 Interleaved Testing Variance

In interleaved testing, each user provides a paired comparison. Let $D_i = Y_i^T - Y_i^C$ be the preference for user $i$:
$$Var(\bar{D}) = \frac{Var(D)}{n} = \frac{Var(Y^T) + Var(Y^C) - 2Cov(Y^T, Y^C)}{n}$$

Since $Cov(Y^T, Y^C)$ is typically large (same user), $Var(D) \ll 2\sigma^2$.

### A.3 Switchback Variance

For a switchback design with $T$ periods and treatment indicator $W_t$:
$$\hat{\tau} = \frac{1}{T_1}\sum_{t:W_t=1} Y_t - \frac{1}{T_0}\sum_{t:W_t=0} Y_t$$

With carryover effect $\gamma$ and autocorrelation $\rho$:
$$Var(\hat{\tau}) \approx \frac{2\sigma^2}{T}\left(1 + 2\rho + \frac{\gamma^2}{\sigma^2}\right)$$

---

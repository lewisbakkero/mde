# Literature Review: Techniques to Reduce Minimum Detectable Effect (MDE) in Ads A/B Test Platforms

## Executive Summary

This literature review synthesizes research on methods to reduce the Minimum Detectable Effect (MDE) in online experimentation platforms, with particular focus on advertising contexts. MDE represents the smallest treatment effect that an experiment can reliably detect given its statistical power, sample size, and significance level. Reducing MDE enables platforms to detect smaller but meaningful effects, accelerating decision-making and improving experimentation efficiency.

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
- Marketplace-specific methods (Cluster randomization, Multiple randomization)
- Adaptive methods (Multi-armed bandits, Adaptive experimental design)

---

## 2. Variance Reduction Techniques

All methods in this section reduce MDE by decreasing the variance term $\sigma^2$ in the baseline equation.


### 2.1 CUPED (Controlled-experiment Using Pre-Experiment Data)

**Source:** [Improving the sensitivity of online controlled experiments by utilizing pre-experiment data](https://robotics.stanford.edu/~ronnyk/2013-02CUPEDImprovingSensitivityOfControlledExperiments.pdf)

**Core Idea:** Use pre-experiment covariate data to reduce variance through regression adjustment.

**MDE Equation Modification:**
$$MDE_{CUPED} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2(1-\rho^2)}{n}}$$

Where $\rho$ is the correlation between pre-experiment covariate $X$ and outcome $Y$. The variance reduction factor is $(1-\rho^2)$.

**Method:**
The CUPED estimator adjusts the outcome $Y$ using a pre-experiment covariate $X$:
$$\hat{Y}_{CUPED} = Y - \theta(X - \bar{X})$$
Where $\theta = \frac{Cov(Y, X)}{Var(X)}$ minimizes variance.

**Key Findings:**
- 50%+ variance reduction achievable with highly correlated pre-experiment metrics
- Works best when pre-experiment behavior strongly predicts post-experiment outcomes
- Simple to implement and widely adopted (Microsoft, Netflix, LinkedIn)

**Limitations:**
- Requires pre-experiment data availability
- Effectiveness depends on covariate-outcome correlation
- Single covariate may not capture all predictive information

---

### 2.2 CUPAC (Controlled-experiment Using Predictions As Covariates)

**Source:** [Leveraging covariate adjustments at scale in online A/B testing](https://www.amazon.science/publications/leveraging-covariate-adjustments-at-scale-in-online-a-b-testing)

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

Where $R^2$ is the coefficient of determination from regressing $Y$ on all covariates. This generalizes the single-covariate case where $R^2 = \rho^2$.

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

**Core Idea:** Divide population into homogeneous strata and randomize within strata.

**MDE Equation Modification:**
$$MDE_{strat} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2_{within}}{n}}$$

Where $\sigma^2_{within} < \sigma^2$ because between-strata variance is removed:
$$\sigma^2_{within} = \sigma^2 - \sigma^2_{between}$$

**Method:**
1. Define strata based on pre-experiment characteristics
2. Randomize treatment assignment within each stratum
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
- Requires modeling assumptions about temporal dynamics
- More complex analysis than standard stratification
- May require longer experiments to estimate dynamics
- Computational overhead for adaptive bucket selection

---

### 2.6 Efficient Semiparametric Estimation Under Covariate Adaptive Randomization

**Source:** [Covariate Adjustment in Randomized Trials](https://covariateadjustment.github.io/)

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
   
   For a randomized experiment with balanced assignment $e(X) = 0.5$:
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
1. Derive efficiency bounds for various randomization schemes
2. Construct estimators achieving these bounds (AIPW, TMLE)
3. Show robustness to model misspecification (doubly robust property)

**Key Findings:**
- Provides benchmark for evaluating variance reduction methods
- Justifies use of regression adjustment in randomized experiments
- AIPW estimators are "doubly robust": consistent if either outcome model OR propensity model is correct
- Extends to complex randomization schemes (stratified, covariate-adaptive)

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
- Requires careful modeling of correlation structure
- In-experiment adjustment can introduce bias if not properly implemented
- More complex to implement and validate; likely less stable for large scale prod environments

| Method | Data Used | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- |
| **Naive (Diff-in-Means)** | None | Simple, unbiased. | High variance; requires large samples. |
| **CUPED** | Linear Pre-experiment | Fast, easy to implement. | Limited by weak historical correlation. |
| **CUPAC** | ML Pre-experiment | Better at capturing non-linear patterns. | Still only uses "old" data. |
| **combining pre-experiment and in-experiment** | **Pre + In-experiment** | **Highest variance reduction.** | Requires careful selection of in-experiment variables. |

---

### 2.8 Variance Reduction Methods: Comparison Table

| Method | MDE Modification | Typical Variance Reduction | Complexity | Data Requirements | Best Use Case |
|--------|------------------|---------------------------|------------|-------------------|---------------|
| **CUPED** | $\sigma^2 \rightarrow \sigma^2(1-\rho^2)$ | 20-50% | Low | Pre-experiment metric | General A/B tests |
| **CUPAC** | $\sigma^2 \rightarrow \sigma^2(1-\rho_{ML}^2)$ | 30-60% | Medium | ML infrastructure | Complex metrics |
| **Multi-Covariate** | $\sigma^2 \rightarrow \sigma^2(1-R^2)$ | 30-60% | Medium | Multiple covariates | Rich feature sets |
| **Stratification** | $\sigma^2 \rightarrow \sigma^2_{within}$ | 10-30% | Low | Stratification vars | Heterogeneous populations |
| **Temporal Stratification** | $\sigma^2 \rightarrow \sigma^2(1-\eta^2_{temporal})$ | 10-30% | Medium | Time-series data | Non-stationary environments |
| **Semiparametric** | $\sigma^2 \rightarrow \sigma^2_{eff}$ | Theoretical bound | High | Depends on estimator | Theoretical benchmark |
| **Pre+In Combined** | $\sigma^2 \rightarrow \sigma^2(1-R^2_{combined})$ | 30-60% | Medium | Pre + concurrent control | Volatile environments |

**Combining Variance Reduction Methods:**

Several variance reduction methods can be combined, but with diminishing returns due to overlapping variance components:

1. **CUPED + Stratification:** Stratification operates at the *randomization* stage while CUPED operates at the *estimation* stage, so they can be combined.

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
   
   In practice, optimized mixture constructions achieve c_AVI ≈ 1.2-1.3.

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

**Source:** Popularized by Optimizely, Amplitude; based on [Robbins (1974)](https://www.jstor.org/stable/2285511)

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
   - **Normal:** $\pi(\theta) = N(0, \tau^2)$ - centered at zero with spread τ
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

### 3.4 Sequential Testing Methods: Comparison Table

| Method | MDE Mechanism | Sample Size Reduction | Monitoring Flexibility | Planning Required | Best Use Case |
|--------|---------------|----------------------|----------------------|-------------------|---------------|
| **GST** | Early stopping at $K$ looks | 30-50% | Pre-specified times only | High | Fixed monitoring schedule |
| **Always Valid** | Continuous stopping | 20-40% | Any time | Low | Automated decisions |
| **mSPRT** | Mixture likelihood ratio | 20-40% | Any time | Medium | Continuous monitoring |

---


## 4. Experimental Design Innovations

Methods in this section modify the experimental design to either reduce variance, increase effective sample size, or handle interference.

### 4.1 Budget-Split Design for Marketplace Experimentation

**Source:** [Trustworthy Online Marketplace Experimentation with Budget-split Design](https://arxiv.org/abs/2012.08724)

**Core Idea:** Split advertiser budgets rather than users to handle marketplace interference.

**MDE Equation Modification:**
In standard cluster randomization with $m$ clusters:
$$MDE_{cluster} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2(1 + (n_c-1)\rho_{ICC})}{n}}$$

Budget-split eliminates interference:
$$MDE_{budget-split} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2_{budget}}{n_{impressions}}}$$

Where $\sigma^2_{budget}$ is typically much smaller than $\sigma^2(1 + (n_c-1)\rho_{ICC})$.

**Method:**
1. Split each advertiser's budget into treatment and control portions
2. Run separate auctions for each budget portion
3. Estimate treatment effect from budget-level comparison

**Key Findings:**
- Eliminates interference from budget competition
- Provides unbiased estimates in marketplace settings
- 30-50% MDE reduction vs. cluster randomization

**Limitations:**
- Requires budget-level tracking infrastructure
- May not capture all marketplace dynamics
- Assumes budget is the primary interference mechanism

---

### 4.2 Switchback Experiments

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

*Carryover-Autocorrelation Interplay (Hu et al. 2022):*
- Carryover effects and autocorrelation interact non-trivially
- Optimal period length depends on both factors
- Longer periods reduce carryover bias but increase variance

*Data-Driven Designs (Xiong et al. 2023):*
- Empirical Bayes approaches can optimize switchback patterns
- Use historical data to estimate carryover and autocorrelation
- Adaptive designs outperform fixed patterns by 20-40%

*Multiple-Unit Designs (Ye et al. 2023):*
- Randomize across both units and time
- More robust to model misspecification
- Better variance-bias trade-off

**Limitations:**
- Carryover effects can bias estimates
- Requires careful period length selection
- Not suitable for treatments with long-lasting effects

---

### 4.3 Staggered Rollout Designs

**Source:** [Optimal Experimental Design for Staggered Rollouts](https://arxiv.org/abs/1911.03764)

**Core Idea:** Roll out treatment to units at different times, using timing variation for identification.

**MDE Equation Modification:**
With $K$ rollout waves and optimal timing:
$$MDE_{staggered} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2}{n \cdot DE}}$$

Where $DE$ is the design efficiency (typically 1.2-2.0 for optimal designs), effectively increasing sample size.

**Method:**
1. Randomly assign rollout times to units
2. Compare early vs. late adopters
3. Use difference-in-differences style estimation

**Key Findings:**
- Can be more efficient than standard A/B when treatment effects vary over time
- Optimal designs depend on anticipated effect dynamics
- 20-50% variance reduction with proper design

**Limitations:**
- Requires rollout flexibility
- Assumes parallel trends
- Complex analysis with time-varying effects

---

### 4.4 Interleaved Online Testing

**Source:** Chapelle et al. (2012) - "Large-scale validation and analysis of interleaved search evaluation"

**Core Idea:** Present both treatment and control simultaneously to the same user, using preference signals to detect differences.

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

**Key Findings:**
- 10-100x more sensitive than traditional A/B tests for ranking systems
- Particularly effective for search, recommendations, ad ranking
- Each user serves as their own control

**Limitations:**
- Only applicable to ranking/recommendation systems where interleaving is possible
- Cannot measure absolute metrics (only relative preferences)
- Presentation bias can affect results

---

### 4.5 Debiased Balanced Interleaving

**Source:** Debiased balanced interleaving at Amazon Search

**Core Idea:** Address presentation bias in interleaved experiments through debiasing techniques.

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
- Requires position bias modeling
- May slightly increase variance due to debiasing

---

### 4.6 Adaptive Experimental Design and Counterfactual Inference

**Source:** Adaptive experimental design and counterfactual inference

**Core Idea:** Use adaptive allocation and counterfactual reasoning to improve experimental efficiency.

**MDE Equation Modification:**
Adaptive designs can achieve:
$$MDE_{adaptive} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2}{n_{eff}}}$$

Where $n_{eff} > n$ through optimal allocation and counterfactual imputation.

**Method:**
1. Adaptively allocate samples based on observed outcomes
2. Use counterfactual inference to impute missing potential outcomes
3. Combine observed and imputed data for estimation

**Key Findings:**
- Can improve efficiency over fixed allocation
- Counterfactual inference provides additional information
- Particularly useful when treatment effects are heterogeneous

**Limitations:**
- More complex implementation
- Requires careful handling of adaptive inference
- May introduce bias if not properly implemented

---

### 4.7 Experimental Design Methods: Comparison Table

| Method | MDE Mechanism | MDE Reduction | Complexity | Best Use Case |
|--------|---------------|---------------|------------|---------------|
| **Budget-Split** | Eliminates interference | 30-50% vs cluster | High | Ad marketplaces |
| **Switchback** | Within-unit comparison | 20-40% | High | Marketplace/interference |
| **Data-Driven Switchback** | Optimized period design | 20-40% over fixed | High | When historical data available |
| **Staggered Rollout** | Timing variation | 20-50% | Medium | Gradual launches |
| **Interleaved Testing** | Paired comparison | 50-90% | Medium | Ranking/recommendation |
| **Debiased Interleaving** | Bias-corrected pairing | 50-90% | Medium-High | Search systems |
| **Adaptive Design** | Optimal allocation | Varies | High | Heterogeneous effects |

---


## 5. Marketplace-Specific Methods

Methods in this section address the unique challenges of two-sided marketplaces where interference between units inflates variance.

### 5.1 Experimental Design in Marketplaces

**Source:** "Experimental Design in Marketplaces"

**Core Idea:** Design experiments that account for two-sided interference and equilibrium effects.

**MDE Equation Modification:**
With interference, the naive estimator has inflated variance:
$$MDE_{naive} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2(\sigma^2 + \sigma^2_{interference})}{n}}$$

Proper marketplace designs reduce or eliminate $\sigma^2_{interference}$:
$$MDE_{marketplace} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2_{design}}{n_{eff}}}$$

**Method:**
- Cluster randomization at market/region level
- Synthetic control for counterfactual construction
- Hybrid approaches combining multiple techniques

**Key Findings:**
- No single best approach for all marketplaces
- Design choice depends on interference structure
- Hybrid approaches often optimal

**Limitations:**
- Cluster randomization increases variance due to fewer units
- Synthetic control requires parallel trends assumption
- Complex to implement and validate

---

### 5.2 Multiple Randomization Designs

**Source:** 
- "Measuring direct and Indirect Impacts in a Multi-Sided Marketplace"
- "Multiple Randomization Designs: Estimation and Inference with Interference"

**Core Idea:** Use multiple layers of randomization to identify direct and indirect effects.

**MDE Equation Modification:**
For direct effect estimation with two-stage randomization:
$$MDE_{direct} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2_{direct}}{n_1 \cdot n_2}}$$

Where $n_1$ = number of clusters, $n_2$ = individuals per cluster.

**Method:**
1. First randomization: Assign clusters to treatment intensity
2. Second randomization: Assign individuals within clusters
3. Estimate direct effects, spillovers, and equilibrium effects separately

**Key Findings:**
- Can decompose total effect into components
- Enables understanding of interference mechanisms
- More informative than single randomization

**Limitations:**
- More complex designs may increase variance for any single effect
- Requires larger sample sizes for precise estimation
- Trade-off between precision and identification

---

### 5.3 Multi-Armed Bandits with Network Interference

**Source:** "Multi-Armed Bandits with Network Interference"

**Core Idea:** Adapt bandit algorithms to handle interference between units.

**MDE Equation Modification:**
Standard bandit regret bound:
$$R_T = O(\sqrt{KT\log T})$$

With network interference, network-aware bandits achieve:
$$R_T^{network} = O(\sqrt{KT\log T \cdot (1 + \rho_{network})})$$

**Method:**
1. Model interference structure (network graph)
2. Modify reward estimates to account for spillovers
3. Adapt exploration strategy to network topology

**Key Findings:**
- Ignoring interference leads to suboptimal policies
- Network-aware bandits can improve regret by 30-50%
- Applicable to ad targeting with social effects

**Limitations:**
- Requires knowledge of network structure
- Computational complexity scales with network size
- May not capture all interference mechanisms

---

### 5.4 Marketplace Methods: Comparison Table

| Method | MDE Mechanism | Interference Handling | Complexity | Best Use Case |
|--------|---------------|----------------------|------------|---------------|
| **Cluster Randomization** | Reduce within-cluster interference | Good | Medium | Geographic markets |
| **Multiple Randomization** | Separate direct/indirect effects | Excellent | High | Effect decomposition |
| **Network-Aware Bandits** | Model spillovers | Good | High | Social networks |

---

## 6. Adaptive and Learning Methods

### 6.1 Multi-Armed Bandits for A/B Testing

**Source:** "Efficient experimentation: A review of four methods"

**Core Idea:** Adaptively allocate traffic to better-performing treatments.

**MDE Equation Modification:**
Bandits don't directly reduce MDE but reduce regret (opportunity cost):
$$Regret_{AB} = n \cdot \Delta / 2$$
$$Regret_{bandit} = O(\sqrt{n \cdot K \cdot \log n})$$

**Method:**
- Thompson Sampling: Bayesian approach with posterior sampling
- UCB (Upper Confidence Bound): Optimistic exploration
- Epsilon-greedy: Random exploration with exploitation

**Key Findings:**
- Bandits reduce opportunity cost during experimentation
- But complicate statistical inference
- Hybrid approaches emerging (explore-then-commit)

**Limitations:**
- Inference is more complex than standard A/B
- May not achieve same statistical power
- Implementation complexity

---

### 6.2 Learning Across Experiments and Time

**Source:** "Learning Across Experiments and Time: Tackling Heterogeneity in A/B Testing"

**Core Idea:** Use information from past experiments to improve current ones.

**MDE Equation Modification:**
With informative prior from historical experiments:
$$MDE_{meta} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2}{n + n_{prior}}}$$

Where $n_{prior}$ is the effective sample size from historical data.

**Method:**
1. Meta-analysis: Pool estimates across experiments
2. Hierarchical models: Share information through priors
3. Transfer learning: Use past data to reduce variance

**Key Findings:**
- Historical data can significantly reduce MDE (20-40%)
- Heterogeneity across experiments must be modeled
- Requires experiment database infrastructure

**Limitations:**
- Must handle concept drift
- Privacy considerations for data sharing
- Requires substantial historical data

---

### 6.3 Adaptive Methods: Comparison Table

| Method | MDE Mechanism | Primary Benefit | Complexity | Best Use Case |
|--------|---------------|-----------------|------------|---------------|
| **Thompson Sampling** | Adaptive allocation | Reduced regret | Medium | Optimization focus |
| **UCB** | Optimistic exploration | Reduced regret | Medium | Unknown effect sizes |
| **Cross-Experiment Learning** | Informative priors | 20-40% MDE reduction | High | Mature platforms |

---


## 7. Case Studies

### 7.1 Netflix Case Studies

**Source:** "Improving the Sensitivity of Online Controlled Experiments: Case Studies at Netflix"

**Key Implementations:**

1. **CUPED at Scale:**
   - 40-50% variance reduction on engagement metrics
   - Pre-experiment viewing history as covariate
   - Automated pipeline for covariate selection

2. **Stratification:**
   - Device type, region, tenure as strata
   - 10-20% additional variance reduction
   - Combined with CUPED for multiplicative gains

3. **Metric Selection:**
   - Chose metrics with lower inherent variance
   - Developed proxy metrics for long-term outcomes
   - Balanced sensitivity with business relevance

**Lessons Learned:**
- Infrastructure investment pays off
- Combination of methods most effective
- Metric engineering as important as statistical methods

---

## 8. Master Comparison of All Methods

### 8.1 Summary Table: All Methods by MDE Reduction Mechanism

| Method | Category | MDE Equation Change | Typical MDE Reduction | Complexity | Data Requirements |
|--------|----------|---------------------|----------------------|------------|-------------------|
| **CUPED** | Variance Reduction | $\sigma^2 \rightarrow \sigma^2(1-\rho^2)$ | 20-50% | Low | Pre-experiment data |
| **CUPAC** | Variance Reduction | $\sigma^2 \rightarrow \sigma^2(1-\rho_{ML}^2)$ | 30-60% | Medium | ML infrastructure |
| **Multi-Covariate** | Variance Reduction | $\sigma^2 \rightarrow \sigma^2(1-R^2)$ | 30-60% | Medium | Multiple covariates |
| **Stratification** | Variance Reduction | $\sigma^2 \rightarrow \sigma^2_{within}$ | 10-30% | Low | Stratification vars |
| **Pre+In Combined** | Variance Reduction | $\sigma^2 \rightarrow \sigma^2(1-R^2_{combined})$ | 30-60% | Medium | Pre + in-experiment |
| **GST** | Sequential Testing | $n \rightarrow n \cdot ASN_{ratio}$ | 30-50% (sample) | Medium | Pre-planned schedule |
| **Always Valid** | Sequential Testing | Early stopping any time | 20-40% (sample) | Medium | Continuous data |
| **mSPRT** | Sequential Testing | $n \rightarrow E[n_{stop}]$ | 20-40% (sample) | Medium | Continuous data |
| **Non-Stationary** | Sequential Testing | Time-varying adjustment | Varies | Medium | Temporal modeling |
| **Budget-Split** | Design Innovation | Eliminates interference | 30-50% vs cluster | High | Budget data |
| **Switchback** | Design Innovation | Within-unit comparison | 20-40% | High | Time-series data |
| **Data-Driven Switchback** | Design Innovation | Optimized periods | 20-40% over fixed | High | Historical data |
| **Staggered Rollout** | Design Innovation | $n \rightarrow n \cdot DE$ | 20-50% | Medium | Rollout flexibility |
| **Interleaved Testing** | Design Innovation | $2\sigma^2 \rightarrow \sigma^2_{paired}$ | 50-90% | Medium | Ranking systems |
| **Debiased Interleaving** | Design Innovation | Bias-corrected pairing | 50-90% | Medium-High | Search systems |
| **Adaptive Design** | Design Innovation | Optimal allocation | Varies | High | Heterogentic effects |
| **Multiple Randomization** | Marketplace | Separate effects | Enables identification | High | Multi-level data |
| **Cross-Experiment** | Learning | $n \rightarrow n + n_{prior}$ | 20-40% | High | Historical experiments |

### 8.2 Method Selection Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MDE Reduction Decision Tree                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. What type of system are you testing?                                    │
│     ├─ Ranking/Recommendation → Consider INTERLEAVED TESTING (50-90% MDE↓)  │
│     └─ Other → Continue to step 2                                           │
│                                                                             │
│  2. Do you have pre-experiment data?                                        │
│     ├─ Yes → Start with CUPED/CUPAC (20-60% MDE↓)                           │
│     └─ No  → Consider stratification (10-30% MDE↓)                          │
│                                                                             │
│  3. Is there interference between units?                                    │
│     ├─ Yes → Consider:                                                      │
│     │        ├─ Budget-split (ad marketplaces)                              │
│     │        ├─ Switchback (time-based interference)                        │
│     │        ├─ Cluster randomization (geographic)                          │
│     │        └─ Multiple randomization (effect decomposition)               │
│     └─ No  → Standard randomization + variance reduction                    │
│                                                                             │
│  4. Do you need continuous monitoring?                                      │
│     ├─ Yes → mSPRT or Always Valid Inference                                │
│     └─ No  → GST or fixed-horizon                                           │
│                                                                             │
│  5. Do you have historical experiment data?                                 │
│     ├─ Yes → Leverage cross-experiment learning (20-40% MDE↓)               │
│     └─ No  → Build experiment database for future                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Combinability Matrix

| Method 1 | Method 2 | Combinable? | Expected Combined Benefit |
|----------|----------|-------------|---------------------------|
| CUPED | Stratification | ✓ | Multiplicative (40-70% total) |
| CUPED | GST | ✓ | Apply CUPED at each look |
| CUPED | Switchback | ✓ | Use pre-period as covariate |
| CUPED | Interleaved | ✓ | Both reduce variance |
| Pre+In Combined | Stratification | ✓ | Apply within strata |
| Pre+In Combined | Sequential | ✓ | Apply at each analysis |
| Stratification | GST | ✓ | Stratify then sequential |
| GST | mSPRT | ✗ | Choose one approach |
| Switchback | Budget-Split | ~ | Depends on context |
| Interleaved | CUPED | ✓ | Additional variance reduction |

---

## 9. Practical Recommendations for Ads A/B Testing

### 9.1 Quick Wins (Low Effort, High Impact)

1. **Implement CUPED with pre-experiment ad engagement**
   - Use 7-14 day pre-period
   - Expected: 30-50% MDE reduction

2. **Consider Interleaved Testing for ranking changes**
   - If testing ad ranking algorithms
   - Expected: 50-90% MDE reduction

3. **Stratify by advertiser size/vertical**
   - Reduces between-group variance
   - Expected: 10-20% additional reduction

### 9.2 Medium-Term Investments

1. **Build CUPAC infrastructure**
   - ML models for outcome prediction
   - Expected: 10-30% over basic CUPED

2. **Implement sequential testing**
   - mSPRT for continuous monitoring
   - Reduces time-to-decision by 30-50%

3. **Develop switchback capability**
   - For marketplace-level experiments
   - Use data-driven design optimization

### 9.3 Long-Term Platform Capabilities

1. **Budget-split design for ad experiments**
   - Eliminates marketplace interference
   - Gold standard for ad platform experiments

2. **Cross-experiment learning system**
   - Meta-analysis infrastructure
   - Bayesian priors from historical data

3. **Automated experiment design selection**
   - Choose optimal method based on context
   - Adaptive sample size determination

---

## 10. Open Research Questions

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

The variance-minimizing $\theta$ is:
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

# Search Queries

Queries are grouped by review and topic. Run each query against arxiv, Semantic Scholar, and Google Scholar, filtered by the date window from Step 1 of the runbook.

## Maintenance

Add a new query when:
- A new term of art emerges in the field
- A method gets renamed or a new abbreviation becomes standard
- You discover a paper you should have found but didn't

Remove or narrow a query when it consistently returns irrelevant results. Track the rejection reasons in `seen-papers.json` — if the same query produces mostly rejects, it's too broad.

## MDE Reduction Review

### Variance Reduction / Covariate Adjustment

- `CUPED variance reduction A/B test`
- `covariate adjustment online experiment`
- `CUPAC prediction as covariate`
- `machine learning variance reduction randomized experiment`
- `augmented inverse probability weighting experiments`
- `semiparametric efficient estimator randomized trial`
- `deep learning covariate adjustment experiments`
- `regression adjustment online controlled experiments`

### Sequential Testing

- `always valid inference A/B test`
- `sequential testing optional stopping`
- `mSPRT mixture sequential probability ratio test`
- `group sequential test online experiment`
- `anytime valid confidence sequence`
- `safe testing e-values experimentation`

### Experimental Design / Interference

- `switchback experiment design marketplace`
- `cluster randomized experiment interference`
- `staggered rollout difference-in-differences`
- `budget split design advertising experiment`
- `two-sided platform experiment design`
- `multiple randomization experiment`
- `network interference experimental design`
- `first-price pacing equilibrium experiment`
- `ACURT ad experiment clustering`

### Interleaving

- `interleaved ranking evaluation`
- `debiased interleaving search ranking`
- `interleaving online experiment large scale`
- `team draft interleaving`

### Ratio Metrics / Heavy Tails / Robust

- `ratio metric variance reduction delta method`
- `heavy tailed metric A/B test robust`
- `winsorization online experiment`
- `extreme value theory experimentation`

### Platform / Meta Topics

- `winners curse effect size bias experimentation`
- `variance estimation A/B test validation`
- `cross-experiment learning hierarchical Bayesian`
- `heterogeneous treatment effect experimentation`
- `variance reduction non-stationary treatment effect`

## Prediction of Intervention Performance Review

### Ad Measurement / Incrementality

- `incrementality measurement advertising experiment`
- `ghost ads counterfactual exposure`
- `predicted incrementality experimentation PIE`
- `lift test calibration advertising`
- `ad attribution causal inference`
- `multi-touch attribution causal`
- `observational ad measurement bias`
- `last-click attribution bias causal`

### Observational / Quasi-Experimental

- `propensity score matching advertising`
- `double machine learning advertising`
- `synthetic control method causal`
- `Bayesian structural time series causal impact`
- `geo experiment ad measurement`
- `CausalImpact Google Brodersen`

### Surrogate / Long-Term Effects

- `surrogate index long-term treatment effect`
- `auto-surrogate A/B test long-term`
- `short-term proxy long-term outcome`
- `persistent confounding long-term causal`

### Cross-Experiment Prediction / Meta-Learning

- `cross-experiment meta-learning treatment effect`
- `pooling experiments treatment effect prediction`
- `prediction powered inference causal`
- `in-context learning causal effect estimation`
- `prior-data fitted network causal`
- `transfer learning heterogeneous treatment effect`

### MMM / Marketing Mix

- `Bayesian marketing mix model calibration`
- `marketing mix model lift test`
- `hierarchical Bayesian media mix`
- `geo-level media mix model`

### Transportability / External Validity

- `transportability causal inference Pearl Bareinboim`
- `external validity randomized trial generalization`
- `sample average treatment effect population`
- `data fusion causal inference`
- `experimental observational data combination`
- `shrinkage estimator RCT observational`
- `target trial emulation observational`
- `experimental grounding hidden confounding`

## Author-Centric Watchlists

Papers by these researchers are often relevant; check new publications by each at least annually.

### MDE Reduction Review

- Ron Kohavi
- Alex Deng
- Susan Athey, Guido Imbens
- Michael Jordan, Lihua Lei
- Dean Eckles, Eytan Bakshy
- Tatiana Xifara, Aleksander Fabijan
- Ramesh Johari
- Nikolaos Vasiloglou, Stefan Wager
- Amit Gandhi, Christian Kroer
- Alexander D'Amour

### Prediction Review

- Brett Gordon, Florian Zettelmeyer
- Garrett Johnson, Randall Lewis
- Susan Athey, Guido Imbens, Raj Chetty
- Judea Pearl, Elias Bareinboim
- Stefan Wager, Susan Athey
- Nathan Kallus, Uri Shalit
- Eva Ascarza, Ayelet Israeli
- Miguel Hernán, James Robins (epidemiology/causal inference foundations)
- Jake Soloff, Alex Franks (sample-to-population)
- Brad Sturt, Davide Viviano

## Venue-Specific Checks

These venues occasionally publish relevant work not well-indexed by generic queries. Check table-of-contents or proceedings after each new issue/conference.

### For both reviews

- **Marketing Science** (INFORMS)
- **Journal of Marketing Research**
- **KDD** (applied data science track, especially industry papers)
- **WSDM**
- **WWW** / **The Web Conference**
- **NeurIPS** (causal inference, experimental design tracks)
- **ICML**

### Primarily for MDE Reduction

- **CODE@MIT** (Conference on Digital Experimentation)
- **KDD applied track**
- **AAAI**

### Primarily for Prediction review

- **Quantitative Marketing and Economics**
- **Econometrica**
- **American Economic Review**
- **NBER working papers** (working papers often appear here first)
- **Annals of Applied Statistics**

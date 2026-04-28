# Section Entry Template

Use this template when drafting a new method entry for either review. Match the existing style in the target review — the structure below mirrors what's used in §2–§7 of both reviews.

---

### X.Y [Method Name]

**Source:** Author, Author, Author ([Year]). "[Full Title]." *[Venue]*, Volume(Issue), Pages. [DOI or arXiv link].

#### Problem

One or two paragraphs. What is the method trying to solve? Why is this problem hard? How does existing work fall short?

Keep this concrete. Use a worked example if the problem is abstract.

#### Method

One to three paragraphs plus an optional enumerated list. How does the method work?

If there's a central insight, state it in bold:

> **Key insight:** [One sentence.]

If the method has a pipeline structure, enumerate the steps:

1. **Step name:** What happens, one sentence
2. **Step name:** What happens, one sentence
3. ...

If relevant, include a short equation — formatted in inline LaTeX — for the core estimator.

#### Assumptions

Table format. Every entry should have all four columns filled.

| Assumption | Formal Statement | What Happens If Violated | How to Check |
|------------|------------------|-------------------------|--------------|
| [Name] | [Math or precise prose] | [Consequence for the estimate] | [Diagnostic or test] |
| ... | ... | ... | ... |

Followed by:

*When it works:* One paragraph. Concrete circumstances where the method performs well.

*When it fails:* One paragraph. Concrete circumstances where the method breaks down.

#### Key Findings

Two or three bullets covering the empirical or theoretical results. Include specific numbers where available.

- Finding 1 (with number or specific claim)
- Finding 2
- Finding 3

Optionally, a small comparison or result table:

| Setting | Metric | Value |
|---------|--------|-------|
| ... | ... | ... |

#### Limitations

Enumerated list. Three to seven items. Be honest — the point is to set expectations.

1. **[Limitation name]:** Why this is a limitation and when it matters
2. **[Limitation name]:** ...
3. ...

#### How This Relates to [Nearby Method]

Optional but recommended when the method is close to an existing entry. Use a comparison table:

| Dimension | [This method] | [Nearby method] |
|-----------|---------------|-----------------|
| Unit of analysis | | |
| Data requirements | | |
| Identification strategy | | |
| Output | | |

One paragraph explaining the high-level distinction.

---

## Placement Checklist

After drafting the entry, before applying to the review file, verify:

- [ ] Section number fits the existing hierarchy (e.g., if adding to §4, is it 4.5? 4.6? does existing 4.5 get renumbered?)
- [ ] A row has been drafted for the **Master Comparison Table** in the same review
- [ ] A row has been drafted for the **Common Comparison Framework** table (Prediction review) or the §7 summary table (MDE review)
- [ ] A row has been drafted for the **Practitioner's Decision Guide / Quick Reference** table if applicable
- [ ] The method is added to the **Appendix B** assumptions reference with method-specific rows
- [ ] The reference is added to **Appendix C** in the appropriate category
- [ ] The **glossary (Appendix A)** has an entry for any new acronym or term of art
- [ ] Cross-references to related sections use the correct §X.Y notation
- [ ] The method is mentioned in the **Method Combinability Matrix** or relevant stacking discussion

## Style Notes

Match the voice of the existing review:

- **Opinionated, not encyclopedic.** State when the method doesn't apply. Don't hedge where the empirical evidence is clear.
- **Concrete over abstract.** If you can use a dollar amount, a percentage, or a named company, do.
- **Tables over lists where structure matters.** Assumption tables, comparison tables, findings tables.
- **Cite specific numbers from papers.** "R² = 0.88" beats "high predictive accuracy."
- **Admit limitations explicitly.** Every method has failure modes; list them.

## Length Guidelines

- Short entry (well-known method, brief treatment): 300–500 words
- Standard entry: 600–900 words
- Central method (extensive treatment with many tables): 1000–1500 words
- Only go beyond 1500 words if the method genuinely warrants it (PIE is the outlier example in the current reviews)

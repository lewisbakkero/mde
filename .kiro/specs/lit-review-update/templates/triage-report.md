# Triage Report — YYYY-QN

**Run date:** YYYY-MM-DD
**Search window:** YYYY-MM-DD to YYYY-MM-DD
**Total candidates found:** N
**New candidates (not in seen-papers.json):** M
**Candidates by disposition:** accepted: A, deferred: D, rejected: R

---

## Executive Summary

One paragraph. What's the headline? Did any genuinely significant paper land? Is the field shifting?

Example: "Three substantial papers this quarter, all on experimental grounding variants. One clear accept (Smith et al., 2026) extending Kallus et al. with honest confidence intervals — propose adding to §4.5 of the Prediction review. Two deferrals pending full read. Field trend: growing attention to uncertainty quantification in data-fusion methods; expect more in this area next quarter."

---

## Accepts (Proposed for Inclusion)

### 1. [Author et al., Year] — [Title]

- **Source:** arXiv ID / DOI / URL
- **Venue:** [venue or "preprint"]
- **Proposed section:** §X.Y of [MDE Reduction | Prediction] review
- **Why this paper:** One sentence on the contribution
- **Why accept:** How it passes the inclusion criteria
- **Suggested framing:** One paragraph sketch of the review entry

### 2. [next accepted paper]

...

---

## Deferrals (Borderline — Need Reviewer Decision)

### 1. [Author et al., Year] — [Title]

- **Source:** arXiv ID / DOI / URL
- **Why defer:** Specific criterion on the borderline
- **What the reviewer needs to decide:** e.g., "Is the empirical validation on a single dataset enough given the method's novelty?"
- **Implication of accepting:** Would require adding a new subsection to §4
- **Implication of rejecting:** Lose coverage of [X]; verify that existing entries cover similar ground

### 2. [next deferred paper]

...

---

## Rejects (Logged for Record)

Short entries. The point is traceability, not advocacy.

| # | Author, Year | Title | Reject reason |
|---|--------------|-------|---------------|
| 1 | | | |
| 2 | | | |

---

## Health Check Findings

### Stale Methods

Methods in the reviews whose most recent cited paper is >24 months old. For each, either (a) confirm the method is stable and no action is needed, (b) flag for targeted search next quarter, or (c) note that the method should be deprecated.

| Method | Review | Most recent citation | Action |
|--------|--------|---------------------|--------|
| | | | |

### Missing Cross-References

Methods present in the main body but missing from the master comparison table, decision flowchart, or appendix.

| Method | Missing from | Action |
|--------|--------------|--------|
| | | |

### Claims That May Need Updating

Phrases like "state of the art," "best known," or specific numerical benchmarks that may have been superseded.

| Section | Current claim | Concern | Action |
|---------|---------------|---------|--------|
| | | | |

### Dead Links

References whose URLs no longer resolve.

| Reference | URL status | Replacement candidate |
|-----------|-----------|-----------------------|
| | | |

---

## Proposed Changes Summary

Once the reviewer confirms accepts, this becomes the changelog for the edit commit.

### New entries

- §X.Y (review): Author et al., Year — brief description

### Modified entries

- §X.Y (review): update to reflect new related work

### Structural changes (if any)

- Added new subsection §X.Y
- Renumbered §X.Z to §X.Z+1

---

## Reviewer Decisions (Fill In During Human Review)

For each accept/defer/reject, the reviewer marks final disposition here. This section is the handoff from triage to drafting.

| Paper | Proposed | Final | Notes |
|-------|----------|-------|-------|
| Paper 1 | accept | | |
| Paper 2 | defer | | |
| Paper 3 | reject | | |

---

## Notes for Next Quarter

Anything that should be remembered for the next run:
- Queries that underperformed
- Gaps in coverage noticed during triage
- Authors whose next paper to watch for
- Topics trending in the field

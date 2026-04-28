# Runbook: Quarterly Literature Review Update

Follow these steps in order. Each step has a clear stopping condition.

## Prerequisites

- Both review files present in repo root
- `.kiro/specs/lit-review-update/seen-papers.json` exists and is readable
- `.kiro/specs/lit-review-update/search-queries.md` and `inclusion-criteria.md` are current

## Step 1 — Establish the Date Window

Look at `seen-papers.json`. Find the most recent `evaluated_at` timestamp. The search window is:

- **Start:** max(most recent `evaluated_at`, six months ago)
- **End:** today

If this is the first run: start window = 18 months ago.

Record the window in the triage report header.

## Step 2 — Execute Searches

For each query group in `search-queries.md`, run the listed queries against:

1. **arxiv** — via `remote_web_search` or arxiv API. Filter by submission date in window.
2. **Semantic Scholar** — via `webFetch` on `https://api.semanticscholar.org/graph/v1/paper/search?query=...&fields=title,abstract,authors,year,venue,externalIds&limit=20`
3. **Google Scholar** — via `remote_web_search`. No API; scrape search results. Lower precedence than the above.
4. **Venue-specific searches** where relevant (Marketing Science, JMR, KDD, NeurIPS, ICML, WSDM, WWW).

Deduplicate results within each query group (same arxiv ID or DOI).

## Step 3 — Deduplicate Against seen-papers.json

For each candidate from Step 2:

- Look up by `arxiv_id`, `doi`, or normalized title + first author
- If present in `seen-papers.json`: skip (already evaluated)
- If not present: add to the "new candidates" list

## Step 4 — Triage

Load `inclusion-criteria.md`. For each new candidate, classify as:

- **accept** — clearly in scope, propose for inclusion
- **defer** — borderline, flag for manual review, no draft yet
- **reject** — out of scope, record the reason

Reasons for rejection are critical — they prevent re-evaluation of the same paper next quarter. Common reject reasons:

- Not about measurement / intervention effects
- Duplicate of an included paper (same method, different name)
- Methodology paper without empirical validation on a relevant domain
- Technical note without novel methodological contribution

## Step 5 — Health Check on Existing Reviews

Independent of new papers, run these checks:

1. **Staleness check:** For each method in both reviews, find the most recent cited paper. Flag any method whose most recent reference is >24 months old.
2. **Orphan check:** For each method, verify it appears in (a) its primary section, (b) the master comparison table, (c) the decision flowchart or quick-reference table, (d) any appendix referring to it. Flag missing cross-references.
3. **Superseded-claim check:** Search for any claim of the form "state of the art," "best known," or specific numeric benchmarks. Flag for verification against newer literature.
4. **Dead-link check:** Resolve every URL in the references section. Flag any that return 404 or that have moved.

Record findings in the triage report under "Health Check."

## Step 6 — Write the Triage Report

Use `.kiro/specs/lit-review-update/templates/triage-report.md` as the template.

Save as `.kiro/specs/lit-review-update/triage-YYYY-QN.md` where YYYY-QN is the quarter (e.g., `triage-2026-Q2.md`).

The triage report is **the deliverable of the automated portion**. Stop here and wait for human review.

## Step 7 — Human Review

You (the repo owner) read the triage report and make decisions:

- For each **accept**: confirm or downgrade to defer/reject
- For each **defer**: promote to accept or downgrade to reject (force a decision)
- For each **reject**: confirm or promote to accept

Annotate the triage report with final decisions.

## Step 8 — Draft Section Additions

For each confirmed accept, use `.kiro/specs/lit-review-update/templates/section-entry.md` as the template.

The draft should include:
- Correct section placement (with justification)
- Problem / Method / Assumptions / Key Findings / Limitations structure
- Assumption table with "What Happens If Violated" column
- Cross-references to related methods
- Proposed additions to the master comparison table
- Proposed additions to the appropriate Appendix B sub-table and Appendix C reference list

Kiro drafts; you edit.

## Step 9 — Apply Edits to Reviews

Use `strReplace` with unique anchors to add the new content. Never `fsWrite` on the review files themselves.

After each addition, verify:
1. The new section appears in the table of contents (if one exists)
2. It is listed in the master comparison table
3. It appears in any quick-reference or decision-guide table where applicable
4. It is cross-referenced from related sections
5. Appendix B (assumptions) has a row for the new method
6. Appendix C (references) has the citation

## Step 10 — Update seen-papers.json

For every candidate evaluated in this run — accepted, deferred, or rejected — add an entry to `seen-papers.json` with:

- `arxiv_id` or `doi` (whichever is canonical)
- `title`
- `authors` (array)
- `year`
- `venue`
- `evaluated_at` (ISO date)
- `disposition` (accepted | deferred | rejected)
- `disposition_reason` (string)
- `review` (one of: "mde" | "prediction" | "both" | "none")
- `section` (section number in the relevant review, if accepted)

Commit the updated file.

## Step 11 — Final Commit

Single commit with message:

```
chore(lit-review): quarterly update YYYY-QN

- Added N new methods (list briefly)
- Updated M existing methods (list briefly)
- Health check findings: list
- Deferred K papers for future review

seen-papers.json: +P entries
```

Do not push. The user decides when to push.

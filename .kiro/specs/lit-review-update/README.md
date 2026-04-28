# Literature Review Update Workflow

Semi-automated process for keeping the two literature reviews in this repo current:

- `MDE-Reduction-Literature-Review.md` — methods to reduce MDE in A/B tests
- `Predicting-Intervention-Performance-Literature-Review.md` — methods to estimate causal effects without running an A/B test

## Philosophy

**The human stays in the loop on relevance and framing.** The automation handles the tedious parts: searching, deduplicating, triage. You stay in charge of which papers matter, where they fit, and how they're characterized.

Target cadence: **once per quarter.** More often wastes time; less often risks large synthesis debt.

Target time per update: **30–60 minutes**, excluding drafting time for papers that make the cut.

## The Files in This Directory

| File | Purpose |
|------|---------|
| `runbook.md` | Step-by-step procedure for a single update run |
| `search-queries.md` | Codified search queries, grouped by review and topic |
| `inclusion-criteria.md` | Rules for deciding what's relevant (and what isn't) |
| `seen-papers.json` | Tracking file — every paper that's been evaluated, with a disposition |
| `templates/triage-report.md` | Template for the output of a single run |
| `templates/section-entry.md` | Template for drafting a new method entry in the reviews |

## How to Trigger an Update

Open Kiro with this repo. Ask:

> "Run the quarterly literature review update per `.kiro/specs/lit-review-update/runbook.md`."

Kiro will execute the runbook, produce a triage report at `.kiro/specs/lit-review-update/triage-YYYY-QN.md`, and propose section additions. You review, accept/reject/defer, and Kiro drafts the edits.

## When to Adjust This Workflow

- **Search queries feel stale** → edit `search-queries.md`. Add new keywords as the field evolves (e.g., a new term of art emerges).
- **Too many false positives in triage** → tighten `inclusion-criteria.md`.
- **Too many missed papers** → loosen inclusion criteria, add search queries, or add more venues.
- **`seen-papers.json` gets large (>1000 entries)** → split by review, or archive pre-YYYY entries to a separate file.

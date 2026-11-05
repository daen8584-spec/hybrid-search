# 0001. Record architecture decisions

- Status: Accepted
- Date: 2026-11-05

## Context

Retrieval systems involve trade-offs between quality, latency, operational cost and simplicity that are not visible in the code. Without a record, the reasons behind a choice are lost and the choice is either preserved out of caution or reversed without understanding its purpose.

## Decision

Significant decisions are recorded as Architecture Decision Records in `docs/adr`, each with its context, the decision and its consequences. A record is never rewritten after it is accepted; a later decision supersedes it with a new record.

## Consequences

Reviewers can evaluate a design choice against its stated assumptions, and a change in those assumptions (a larger corpus, a different latency budget) points directly to the decisions that should be revisited.

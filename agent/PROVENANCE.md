# Candidate provenance and scope

2026-10-07: v8 duty-log candidate, based on the separately frozen v7 recovery candidate (platform revision 2cd4d2f5-5a61-4efb-bf57-f6b88d855923), itself based on v5.

Changes relative to v7:
- New duty.py interprets actual protocol-supplied public observation-request text using the configured LLM. It turns validated, time-zoned, source-cited facts into scheduled repairs and non-fault calibration waits.
- agent.py ingests notes at every decision, respects operating boundaries, and records reports.
- llm.py bounds response bytes and total request reading, rejects expired successful calls, and uses structured MiniMax output for the new extraction stage only.
- supervisor.py sends its model-derived log messages through the existing single-line secret-redaction logger.

Unchanged numerical planning, sky tracker, geometry, catalogue, pointing and science-score parameters. No uploaded card data, private truth, precomputed actions, model answers or credentials. Public notes can contain errors and codewords; unknown/invalid extraction falls back to the independent rule policy. The implementation does not claim complete understanding of arbitrary duty prose or guarantee hidden-card performance.

Validation is tracked outside the submission in the local worklog. A synthetic repeated-fault/calibration scenario tests mechanism, not a reconstruction of private evaluation truth. Official evaluation still required before candidate selection.

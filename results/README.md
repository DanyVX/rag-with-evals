# Results

This directory is reserved for reproducible experiment outputs.

No benchmark numbers are pre-filled. The project specification requires real corpus runs, confidence intervals, injection testing, human-verified evaluation, and judge-vs-human validation. Fabricated placeholder scores would defeat the entire point of this repository.

Each published result set should include:

- the corpus checksum manifest
- evaluation dataset checksum
- config hash
- random seed
- question count
- synthetic and human-verified results separately
- bootstrap confidence intervals
- failed-call and judge_error counts
- latency / token / cost totals
- prompt-injection attack success rate
- judge agreement and Cohen's kappa

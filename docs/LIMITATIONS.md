# Limitations

- Synthetic questions can leak source wording and favor lexical retrieval. Results must be reported separately from the human-verified subset.
- LLM judges are noisy and may prefer verbosity, familiar model styles, or option position. Judge agreement must be validated against human labels and reported with Cohen's kappa.
- The initial study uses one primary English documentation corpus plus one PDF layout-stress corpus. Results should not be generalized to every domain or language.
- PDF extraction is imperfect. Scanned pages without a text layer are skipped with warnings rather than silently embedded.
- Retrieval cannot reliably answer questions requiring aggregation over many independent chunks unless an explicit aggregation strategy is added.
- Prompt-injection mitigations reduce risk but cannot prove that retrieved untrusted text is harmless.
- Provider seed support is inconsistent. Determinism is guaranteed in CI through recorded/mock fixtures, not through every remote API.

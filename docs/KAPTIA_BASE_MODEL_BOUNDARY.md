# Kaptia / Base-model responsibility boundary

## Decision

The base language model (currently configured as Qwen) is a replaceable text-generation engine. Kaptia owns the application behavior around it. Product branding does not imply that the underlying foundation weights were trained from scratch.

## Remove duplicated model work

- **Removed:** an LLM call that asked the base model to emit a JSON task plan before normal answer generation.
- **Replaced with:** `scripts/kaptia_orchestrator.py`, a deterministic, bounded rules-based classifier for task length, output format, explicit file intent, and research intent.
- **Kept:** language generation by the base model, because language generation is the engine's core job.
- **Not yet removed:** semantic self-review. It still uses the same model, so it is not independent verification. Replacing it requires deterministic validators for objective properties plus external/source checks where applicable; simply deleting it would reduce existing quality checks.

## Responsibility split

| Capability | Owner |
|---|---|
| Tokenization, next-token prediction, base language generation | Base model and tokenizer |
| Task routing, workflow budgets, fallback behavior | Kaptia orchestrator |
| Tool allowlist, argument validation, execution permission | Kaptia runtime code |
| Local knowledge retrieval, source metadata, memory lifecycle | Kaptia knowledge/memory layer |
| Product constitution and policy enforcement | Kaptia policy layer; prompts are guidance, code guards enforce objective limits |
| Output format validation, artifact writing and hash/read-back checks | Kaptia delivery layer |
| Benchmark cases, regression tests, release gates | Kaptia evaluation layer |

## Limits of the new deterministic planner

It is intentionally a small rules engine, not a second language model. Keyword routing can misclassify nuanced requests. The original request remains the source of truth, and the writer receives it unchanged. Extend the rule set only when a regression case demonstrates the need; do not pretend the classifier understands intent like a model does.

## Verification policy

- Unit tests must prove planning does not invoke the base model.
- CI passing proves only the checks run in CI.
- No claims of improved answer quality until a held-out benchmark and human review show a measured improvement.
- Do not call same-model self-review an independent judge or fact-checker.
- Keep base model ID, tokenizer, revision, license and adapter provenance visible in technical manifests.

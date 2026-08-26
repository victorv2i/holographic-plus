# Local evidence verifier

Automatic extraction stays fail-closed until an owner installs an independent
evidence verifier. That is the correct default: a false `verified` writes a
hallucination into canonical memory. This document makes the opt-in decision
measurable and one command.

The extractor model (`qwen3:30b`) is disqualified. The verifier must be a
different local model.

## Default

Off. No `extraction.evidence_verifier` block is written by `enfold init` or
`enfold setup`. Unconfigured extraction dead-letters proposals as
`proposal_support_unverified` and writes no facts.

## One command to enable

Probe the recommended local model, then write the verifier block. Extraction
mode is not changed.

```bash
python -m enfold.evidence_verifier enable --config ~/.config/enfold/server.json
```

Equivalent with the measured recommendation spelled out:

```bash
python -m enfold.evidence_verifier enable \
  --config /absolute/path/to/server.json \
  --model qwen3.8:27b \
  --endpoint http://127.0.0.1:11434/api/chat
```

The command fails closed and does not write if:

- the model is the extractor (`qwen3:30b`)
- the endpoint is not a loopback `/api/chat` URL
- Ollama does not list the model on `/api/tags`

Restart the daemon after a successful write so the worker loads the new import.

The JSON that enable writes, and that you may add by hand instead:

```json
{
  "extraction": {
    "mode": "disabled",
    "evidence_verifier": {
      "import": "enfold.evidence_verifier:LocalOllamaEvidenceVerifier",
      "model": "qwen3.8:27b",
      "timeout_seconds": 30,
      "prefilter": true,
      "endpoint": "http://127.0.0.1:11434/api/chat"
    }
  }
}
```

`mode` stays whatever you already configured. Enable never turns extraction on.

## Measure before you decide

Hand-labeled fixture: `memory_eval/fixtures/verifier_cases.jsonl` (84 pairs).
Categories: exact support, partial support, plausible-but-unsupported inference,
subject swap, negation flip, number change, temporal drift, and prompt
injection embedded in the excerpt. Labels were written by hand, not by the
models under test.

```bash
python -m enfold.evidence_verifier eval
```

This always scores the deterministic prefilter. Live models are skipped, not
failed, when Ollama is absent or a tag is missing, so CI stays green.

`false_verify_rate` is false `verified` among unsupported cases. That is the
only error that can corrupt memory. A false `needs_review` is friction.

## Recommendation

Measured 2026-08-24 on this machine against the 84-case fixture, timeout 60s,
production pipeline (cheap prefilter may only reject; model errors fail closed
to `needs_review`). Latency is mean wall time per fixture case, including
prefilter rejects.

| Configuration | Precision | Recall | False verify rate | Latency per case |
| --- | ---: | ---: | ---: | ---: |
| prefilter only | n/a (never verifies) | 0.00 | 0.00 | 0.03 ms |
| qwen2.5:3b-instruct | 0.74 | 1.00 | 0.109 | 130 ms |
| qwen3.8:27b | 0.95 | 1.00 | 0.016 | 924 ms |

Recommended configuration: `qwen3.8:27b` with `prefilter=true` and a 30s
timeout, enabled only after the probe above succeeds.

Why this one:

- False verify is the only corrupting error. The 3b model verified 7
  unsupported claims (6 partial-support conjunctions, 1 tense flip). The 27b
  model verified 1 unsupported claim (the same past-versus-future tense flip:
  "Nia joined Harbor Studio on 3 May 2023" versus "Nia will join...").
- That 1.6% residual is not zero. Keep the default off if even one tense miss
  is unacceptable. It is still an order of magnitude safer than the 3b.
- Both models recalled every exact-support case. The 27b rejected every
  partial-support, subject-swap, negation, number, inference, and injection
  case.
- Injection cases never returned `verified` on any configuration. The cheap
  prefilter rejects instruction-shaped excerpts before a model is called.
- The 27b costs about 0.9s per proposal after warmup, versus 0.13s for the 3b
  and 18s on a cold first load. That is acceptable for a write gate.
- `qwen3:30b` remains disqualified on independence grounds. Embedding tags
  (`bge-m3`, `qwen3-embedding:*`, `snowflake-arctic-embed2`, `embeddinggemma`)
  cannot answer the chat verifier schema.

Keep the default off. This recommendation is a measured starting point, not an
automatic activation.

## Fail-closed contract

Any timeout, transport error, or unparseable model output is `needs_review`.
Lexical containment is not entailment. An instruction-shaped excerpt, including
prompt-injection attempts, never reaches the model and never returns
`verified`.

## Related

- Extraction activation remains a separate decision. See
  [`STAGING_ACTIVATION.md`](STAGING_ACTIVATION.md) and
  [`SERVER_DEPLOYMENT.md`](SERVER_DEPLOYMENT.md).
- Capture still requires `--verifier-ready` or an explicit unreviewed ack.

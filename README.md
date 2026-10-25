# TDS 2026-09 GA3

One deployable service answering Q2, Q3 and Q4, plus the Q6/Q7 media deliverables.

| Q | What to paste into the exam |
|---|---|
| 2 — A2A invoice agent | `<BASE>/a2a/` (card is served at `<BASE>/.well-known/agent-card.json`) |
| 3 — Safe AI mailroom | `<BASE>/mailroom` |
| 4 — Incident triage | `<BASE>` (routes live under `/v2/incidents`) |
| 6 — 3D face (GLB) | `https://raw.githubusercontent.com/solomonanujk/tds-2026-09-ga3/main/assets/katharine-hepburn-head.glb` |
| 7 — Morph video | `https://raw.githubusercontent.com/solomonanujk/tds-2026-09-ga3/main/assets/cori-to-gaddafi.mp4` |

`raw.githubusercontent.com` serves both assets over HTTPS with `access-control-allow-origin: *`.

## Run and deploy

```bash
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Deploy the folder as one web service (`render.yaml`, `Procfile` and `Dockerfile` are all here).
Health check: `GET /healthz` reports whether a model is configured.

### Environment

| Variable | Default | Purpose |
|---|---|---|
| `LLM_API_KEY` (or `AIPIPE_TOKEN`) | — | OpenAI-compatible key. **Without it the agents still answer**, using their deterministic analysers. |
| `LLM_BASE_URL` | `https://aipipe.org/openrouter/v1` | OpenAI-compatible base URL |
| `LLM_MODEL` | `openai/gpt-4.1-nano` | any cheap or free model |
| `GA3_DATA_DIR` / `GA3_DB_PATH` | `/tmp` | SQLite state file |
| `A2A_BASE_URL` | derived from the request | force the exact base URL advertised in the agent card |
| `GA3_DEBUG_TOKEN` | unset | when set, `GET /debug/requests?token=…` returns the last grader bodies, for tuning between Checks |

## Layout

```
main.py                 FastAPI app: mounts the three agents
ga3/canon.py            canonical key-sorted compact JSON + SHA-256
ga3/store.py            durable SQLite state: atomic claims, replay, conflict detection
ga3/llm.py              OpenAI-compatible client + content-addressed decision cache
ga3/invoice.py          Q2 A2A surface (card, message:send, tasks, cancel)
ga3/invoice_policy.py   Q2 decisions: five actions, facts, three decisive refs
ga3/mailroom.py         Q3 propose/commit, Ed25519 receipt verification
ga3/mailroom_policy.py  Q3 decisions and the frozen target/payload builders
ga3/incident.py         Q4 run state machine and OTLP assembly
ga3/planner.py          Q4 root cause, diagnostics and effect planning
ga3/trace.py            OTLP span helpers
tools/                  one-off builders for the Q6 GLB and Q7 morph
tests/                  simulated graders for Q2, Q3 and Q4
```

## Tests

Each file is a simulated grader that exercises the contract end to end, including
the failure paths (conflicts, replays, bad signatures, races, isolation):

```bash
python tests/test_invoice.py     # Q2
python tests/test_mailroom.py    # Q3
python tests/test_incident.py    # Q4
```

They run against the deterministic analysers, so they pass with no model key configured.

## Design notes

* **Protocol, storage and reasoning are separate layers.** Polls, replays, lists,
  conflicts, cancellations and receipt continuations never reach a model.
* **Decisions are cached by canonical content**, never by batch/evaluation/task id,
  so later Checks reuse the stable cores and only fresh audit cases cost tokens.
  Q4 is the exception by design: every run makes the planning call its trace reports.
* **Model output is never trusted as-is.** Q2 validates the action, facts and
  evidence against the package; Q3 rebuilds target and payload from the frozen
  schemas and downgrades an outbound send that lacks a trusted, recipient-scoped
  approval; Q4 validates against `allowedRootCauses`, the transcript's evidence ids
  and each tool's `inputSchema`.
* **Nothing sensitive is exported.** Q4 keeps transcripts, prompts, tool arguments
  and the `sensitive` block out of spans and error bodies.

## Q6 / Q7 how they were made

* `tools/build_face_glb.py` sculpts a head from an ellipsoid (brow, sockets, nose,
  cheeks, lips, chin, jaw taper, occipital bulge) with a relief term taken from the
  photograph's own shading, and reprojects the photograph into a spherical texture.
  `tools/check_glb.py` validates the GLB and rasterises it from five angles.
* `tools/face_morph.py` does a Delaunay piecewise-affine morph between hand-placed
  corresponding landmarks, then encodes under the 100,000-byte limit.

# Documentation index

Which document answers which question. `README.md` (repository root) is the user guide, `AGENTS.md`
the working notes for agents, `TODO.md` the short list of what runs, what is next and what waits
for a decision.

| question | document |
|---|---|
| How do I install, train, evaluate, run the LoRA rank sweep? | `../README.md` |
| Where is the code, which rules must a change respect? | `../AGENTS.md` |
| What is running / next / waiting for the user, what may be deleted? | `../TODO.md` |
| What was wrong in the upstream code, and what does the default path do about it (E-numbers, evidence)? | `errors.md` |
| Why is a step this fast, what is measured and what is left (O-numbers, F-findings, timing protocol)? | `optimization-backlog.md` |
| What does the FP16 selection prefix change, and what did it cost / save? | `fp16-prefix.md` |
| How accurate must the selection forward be (arms R / F / P / H, learning runs, recommendation)? | `selection-precision.md` |
| Where does the time of a whole job go (start-up, tokenisation, evaluation, saves) and what was cached? | `startup-overhead.md` |

Measurement scripts live in `../scripts/diagnostics/`; each has a header saying what it measured and
which document holds the result. Raw logs, npz files and run directories are never committed: they
stay under `$COLM_ARTIFACT_ROOT/artifacts/CoLM/` and are referred to from the documents by path.

# [MRV2] Make V2 the default, with a V1 fallback flag

Part of #906. Plan: [the flip](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#flip)

## Summary

Stage 1 of the flip. Switch the default to the V2 runner so every CI suite runs against it, and keep V1 one flag away while we flush out issues. This PR stays small so it's easy to revert: it changes the default and adds the fallback, and nothing else. Deleting V1 and moving the shared helpers wait for stage 2 (the remove-V1 issue).

## Scope (WP-FLIP-1)

- [ ] Point the platform's default `worker_cls` at `SpyreWorkerV2`
- [ ] Add one fallback flag in `envs.py` (for example `SPYRE_USE_V1_MODEL_RUNNER=1`), handled in `TorchSpyrePlatform.check_and_update_config`. It points `worker_cls` back at `TorchSpyreWorker` and keeps the engine-wide `VLLM_USE_V2_MODEL_RUNNER` at 0, which the scheduler protocol needs. This is the Wave 2 bring-up switch with the default reversed; users never set the two values themselves
- [ ] Apply the V1-only rejections and the `VLLM_USE_V2_MODEL_RUNNER=0` pin only when the fallback flag is set
- [ ] `test_layout.py`: allow the platform's import of the package. "Nothing outside `spyre/` imports it" and "V1 files aren't edited" end here
- [ ] Keep one small V1 smoke test (a small greedy decoder with the fallback flag set) until stage 2
- [ ] Document the fallback flag in the user docs and the release notes, including that it is best-effort (V1 isn't covered by CI beyond the smoke test) and that seeded sampling output changes on V2 (plan decision 3)

## Acceptance criteria

- [ ] The full CI suite passes on hardware with V2 as the default
- [ ] The V1 smoke test passes with the fallback flag set
- [ ] With the flag set, a run uses `TorchSpyreWorker` and the V1 runner end to end, with `VLLM_USE_V2_MODEL_RUNNER=0` (unit test of `check_and_update_config`)
- [ ] The fallback flag is documented
- [ ] No V1 file is deleted or moved

## Out of scope

Deleting V1 code, moving shared helpers, folding the worker into `spyre_worker.py`, removing V1-only tests: all stage 2.

## Dependencies

- Depends on: end-to-end bring-up; kernel step 2 (upstream replacements bound and triton-cpu dropped), because Power and Z need it.
- Must land on a vLLM pin that still ships the V1 runner, or the fallback flag has nothing to fall back to.
- Blocks: stage 2 (remove V1), after the validation period.

If a release is cut during the validation period, it ships V2 as the default with the fallback flag.

Size: 2–3 days, plus the validation period.

# [MRV2] Make V2 the default, with a V1 fallback flag

Part of #906. Plan: [the flip](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#flip)

## Summary

Stage 1 of the flip. Switch the default to the V2 runner so every CI suite runs against it, and keep V1 one flag away while we flush out issues. This PR stays small so it's easy to revert: it changes the default and adds the fallback, and nothing else. Deleting V1 and moving the shared helpers wait for stage 2 (the remove-V1 issue).

## Scope (WP-FLIP-1)

- [ ] Point the platform's default `worker_cls` at `SpyreWorkerV2`
- [ ] Flip the default of `SPYRE_MODEL_RUNNER` to `v2`. `SPYRE_MODEL_RUNNER=v1` is the fallback: it points `worker_cls` back at `TorchSpyreWorker`, keeps the engine-wide `VLLM_USE_V2_MODEL_RUNNER` at 0 (the scheduler protocol needs it) and skips the Triton gate. Users never set those values themselves
- [ ] Apply the V1-only rejections and the `VLLM_USE_V2_MODEL_RUNNER=0` pin only on `v1`; the MRV2 rejections from bring-up apply on `v2`
- [ ] `test_layout.py`: allow the platform's import of the package. "Nothing outside `spyre/` imports it" and "V1 files aren't edited" end here
- [ ] Keep one small V1 smoke test (a small greedy decoder with `SPYRE_MODEL_RUNNER=v1`) until stage 2
- [ ] Document the fallback flag in the user docs and the release notes, including that it is best-effort (V1 isn't covered by CI beyond the smoke test) and that seeded sampling output changes on V2 (plan decision 3)

## Acceptance criteria

- [ ] The full CI suite passes on hardware with V2 as the default
- [ ] The V1 smoke test passes with `SPYRE_MODEL_RUNNER=v1`
- [ ] With `SPYRE_MODEL_RUNNER=v1`, a run uses `TorchSpyreWorker` and the V1 runner end to end, with `VLLM_USE_V2_MODEL_RUNNER=0` and no Triton gate (unit test of `check_and_update_config` and a fresh-process test of `register()`)
- [ ] The fallback is documented
- [ ] No V1 file is deleted or moved

## Out of scope

Deleting V1 code, moving shared helpers, folding the worker into `spyre_worker.py`, removing V1-only tests: all stage 2.

## Dependencies

- Depends on: end-to-end bring-up; the upstream-kernels issue (replacements bound and triton-cpu dropped), because Power and Z need it.
- Must land on a vLLM pin that still ships the V1 runner, or the fallback flag has nothing to fall back to.
- Blocks: stage 2 (remove V1), after the validation period.

If a release is cut during the validation period, it ships V2 as the default with the fallback flag.

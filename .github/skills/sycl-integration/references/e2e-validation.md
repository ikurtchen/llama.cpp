# End-to-End Workload Validation (the L3 gate)

L3 is the level that licenses the sentence *"<project> runs on Intel GPU"*. It is earned by running
the project's **real entrypoint** on a **real workload** and comparing the result to a **reference**
within a tolerance the project itself defines.

> **Liveness is not correctness.** "It ran for 200 steps without crashing" is not a gate. A workload
> that silently produces wrong numbers is worse than one that crashes, because it ships.

---

## 1. Choose the reference

In priority order — take the first one that exists:

| # | Reference | When it applies | Comparison |
|---|---|---|---|
| 1 | **The project's own reference artifact** | training projects ship debug/reference state files; libraries ship golden outputs | exactly as the project compares it, with the project's own tolerance |
| 2 | **The project's own test/example expected output** | `examples/` with documented expected results | as documented |
| 3 | **The CUDA backend on a capable host** | an NVIDIA GPU is reachable | same seed, same config, same steps; compare outputs — this is also the only honest basis for any CUDA-vs-SYCL statement |
| 4 | **A CPU/framework reference run** | PyTorch/TF/NumPy can express the same computation | run on CPU at equal-or-higher precision |
| 5 | **A task metric with a published target** | inference/accuracy workloads (perplexity, recall@k, mAP, BLEU) | must land within the published/observed band |
| 6 | **Self-consistency only** | nothing above exists | **weakest** — record it as such; it caps the claim, and the gap goes in the residual package |

Record which one was used in `readiness.json` `gates.L3.reference`. A reader's first question is
"compared to what?", and levels 1–4 answer it; level 6 does not, and pretending otherwise is exactly
the failure this framework exists to prevent.

---

## 2. Workload recipes by shape

### Training run
- Run **N steps** (enough that divergence would show — typically ≥ 20–50) from a fixed seed on the
  project's own reference config.
- Compare the **loss curve**, not just the final loss: per-step `|Δloss|` against the reference,
  plus the loss at step 0 (which validates initialisation and the forward path alone).
- Suggested gate: step-0 loss matches to ~1e-4 relative; per-step `|Δloss|` under a threshold the
  project already uses; the curve is monotone-comparable (no divergence).
- Also check the gradient norm if the project prints it — it catches a broken backward kernel far
  earlier than the loss does.
- Determinism: fix the seed, disable dropout/data shuffling or seed them identically, and pin the
  data order. Non-associative reductions still differ run-to-run; the tolerance must cover that, and
  it must be the project's tolerance, not one invented to fit.

### Inference / generation
- Fixed prompt(s) + **greedy decoding** (temperature 0) → token-exact or near-exact comparison.
- If sampling is unavoidable, compare distributional statistics (perplexity on a fixed corpus),
  never a single sampled string.
- Also assert the output is not degenerate (all-same token, empty, NaN) — a broken norm kernel
  produces plausible-looking garbage.

### Library / query workload
- Run the project's own benchmark or example with its own correctness check
  (recall@k, exact-match, checksum) at the shapes the project documents.
- Compare against the CPU implementation the library already ships, if it has one.

### Simulation / rendering
- Fixed seed + fixed step count → compare conserved quantities (energy, mass, momentum) and a
  checksum of the final state; image outputs compare with the project's own PSNR/SSIM threshold.

### Graph / recommender training
- Loss curve + AUC after a fixed number of iterations, from the project's own model config, through
  the project's own parser (not a hand-built pipeline — see archetype F).

---

## 3. Run it as a gate

Every resolved e2e case is a gate — **all** must pass (config `benchmark.e2e_cases` if non-empty,
else `project.json` `e2e_cases`):

```bash
.sycl/scripts/evidence.sh record e2e baseline --label baseline \
    --reference "loss curve vs <project reference artifact>, 50 steps, seed 1337" \
    --tolerance "|Δloss| < 1e-3 per step; step-0 rel err < 1e-4" \
    -- "<the project's real entrypoint command>"

.sycl/scripts/evidence.sh record e2e case-<name> --label case-<name> \
    --reference "…" --tolerance "…" -- "<case command>"
```

Store the run ids in `profile/e2e.baseline.json` (top-level `evidence`, `cases[].evidence`) **and**
on `readiness.json` `gates.L3.evidence`.

**Make the harness print the comparison.** The record's value comes from the log containing
`step 12: sycl 4.2718  ref 4.2716  Δ 2.0e-4  (tol 1e-3)  OK`. A log that says only `done` proves the
process exited 0 and nothing else. If the project's driver does not print a comparison, add the
comparison to the driver (behind a `--check` flag) rather than eyeballing it.

---

## 4. When it fails

1. **Bisect to a kernel.** Dump per-layer/per-op activations for one step from both backends and find
   the first divergence. That kernel goes back to `migrate` (the workflow's `integrate → migrate`
   back-edge).
2. **Never loosen the workload tolerance to pass.** If the project's own tolerance is exceeded, the
   port is wrong, not the tolerance. The one legitimate adjustment is a documented, *quantified*
   allowance for reduction-order non-associativity — and it must be justified numerically
   (`atol·sqrt(K)` reasoning), recorded, and stated in the report.
3. **Suspect the host path too.** At L3 the failure is as likely in the runtime layer (a stream
   ordering bug, a pointer freed on the wrong context, an uninitialised buffer) as in a kernel. A
   result that is *sometimes* wrong is almost always synchronisation, not math.
4. **NaN/Inf on the first backward step** usually means a reduction or norm kernel; **slow drift over
   many steps** usually means an optimizer/accumulation dtype mismatch.

---

## 5. From L3 to L4

At `done`, re-run the same e2e command post-optimization with `--label final` and the **same env
fingerprint**. `achieved_level` becomes L4 only when both baseline and final e2e records exist and
share that fingerprint (check E7). Report the speedup as *SYCL baseline → SYCL final*; it is not a
CUDA comparison and must not be phrased as one.

If a CUDA-vs-SYCL statement is wanted, it needs its own A/B: the same workload, same steps, same
seed, on both devices, both captured as records, with both device names, drivers and clock states
stated. Without that, no cross-vendor claim is permitted at any level.

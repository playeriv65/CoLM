# CoLM: the paper against the code (algorithm and implementation audit)

Audit of "Mini-batch Coresets for Memory-efficient Language Model Training on Data Mixtures"
(Nguyen, Yang, Anand, Yang, Mirzasoleiman, ICLR 2025, arXiv 2407.19580, read in the v4 HTML: Sec. 3-5,
Alg. 1 = Appendix B, Appendix C) against the upstream code (commit `a6257b0`, `git show a6257b0:<path>`,
called "upstream") and this repository (`zelin-li-refactor`, "ours"). Two questions: does the code do what
the paper says (part 1), and does the premise of the method hold in this implementation (part 2)?
Nothing here changes a default or the algorithm; proposals are listed at the end for the user to decide.

Upstream file references: `T` = `colm/train/subset_trainer_distributed.py`, `F` = `colm/train/facility_location.py`,
`U` = `colm/train/utils.py`, `S` = `scripts/run_math_efficient.sh`, `B` = `colm/scripts/train/base_training_args.sh`.
Ours: `select.py`, `zo.py`, `features.py`, `facility_location.py`, `batching.py` are in `colm/selection/`.
Verdicts: **matches**, **deviates** (code differs from the paper text), **ambiguous** (the paper does not say).

## Part 1: correctness matrix

| # | element | paper (quote) | upstream | ours | verdict and why it matters |
|---|---|---|---|---|---|
| 1a | estimate | Eq. 6 (Sec. 4.3): "ĝ^vp_{i,t} = (L_i(θ_t + ε z_vp) − L_i(θ_t − ε z_vp)) / 2ε · z_vp"; ε is not given a value | `T:2193-2206` (+ε, −2ε, `(loss1 − loss2)/(2·mezo_eps)`), `mezo_eps=1e-3` in `S` | `zo.py:66-71` `Perturbation.projected_grad`, `features.py:163-188`; `mezo_eps=1e-3` | **matches**. Numerics: fp16 prefix + fp32 tail 2, fp32 last layer (docs/fp16-prefix.md); the finite difference agrees with the exact directional derivative to 1.7e-3 (median, part 2) |
| 1b | perturbed parameters | "we sample random perturbations for parameters corresponding to the last (LoRA) V-projection", "327K (matrix B) when using LoRA with rank 128" | `last_layers=['31.self_attn.v_proj']` + `'.lora_B'` (`train.py:176`), `assert len(named_parameters_to_optim) == 1` (`T:2186`) | `zo.py:18-39` `zo_parameters`: layer 31, `v_proj.lora_B` (2560 x 128) | **matches** |
| 1c | one forward | "can be calculated very efficiently in just one forward pass ... we first make a forward pass to get the activations X_{L-1} of the penultimate layer ... Then, we perturb the last-layer parameters twice" | prefix once, `deepcopy(past_key_values)`, two last-layer passes (`T:2189-2203`) | `zo.py` `LastLayerSplit`: prefix once, last layer replayed twice with `functional_call` | **matches** (one prefix, two last-layer passes; it is not "one forward" in FLOPs, the prefix dominates) |
| 1d | dropout | not stated | `model.eval()` in the ZO forwards (`T:2239,2254`) | eval mode for selection | **ambiguous**, consistent between the two codes |
| 2a | distribution and dimension | "z ∈ R^d ... z ~ N(0, I_d)"; `z_vp ∈ R^{d_vp}` | `torch.normal(0, 1, size=param.shape)` in the parameter dtype (`T:2218`), d_vp = 327,680 | `zo.py:55-64`, private CUDA generator | **matches** |
| 2b | how often z is drawn | "one can follow (Malladi et al., 2023) to use a fix seed to generate the same perturbation z_vp multiple times" (to save memory); Eq. 5/6 carry no step or example index on z; nothing says z is shared between steps or examples | `self.zo_random_seed = np.random.randint(1e9)` ONCE in `__init__` (`T:226`); every estimate calls `torch.manual_seed(self.zo_random_seed)` (`T:2212`): ONE z for the whole run, all examples, all steps, all ranks | `zo.py:48-64` `Perturbation.z()` cached from `zo_seed`, drawn once per trainer (`trainers.py:212`) | **ambiguous** in the paper, **one z per run** in both codes. The paper's own text says that SPSA "provides a rank-1 reconstruction of the gradient"; with one z shared by all examples every feature is `g_i · z`, a scalar times the same vector: the selection lives on a line (part 2). MeZO (the cited source of the trick) draws a fresh z per step |
| 2c | quality of the estimate | "smoother than the actual gradient calculated with backpropation" (sic); ĝ ≈ z z^T g (Eq. 5) | same | same | **ambiguous**: unbiased over the draw of z only; a fixed z never averages out. cos(ĝ, g) ≈ 1/sqrt(d) = 0.0017 (part 2) |
| 3 | dimension reduction | "for every big source, we sparsify ... by a mask vector M^t_q, which has 1 for the top h parameters with the largest magnitudes"; "h = 2560" (Table 4), "as small as 0.7%" | `select_masking` `T:1265-1312`: per source, importance `abs(mean over the source's examples of the Adam-transformed feature)`, `argsort(descending)[:zo_dim]`, `zo_dim=2560` | `select.py:205-240` `_mask`/`_rank`, same | **matches**. The paper does not say whether the magnitude is that of the mean or the mean of magnitudes; the code takes |mean|. With one z all coordinates have (nearly) the same |mean|: the mask is decided by ties (part 2) |
| 4a | metric | Eq. 8: l1 distance ("preferable to Euclidean distance in high dimensions") | `S: FACILITY_SELECT=l1` (`B` has `cosine`, overridden); `F:12-43` `similarity` l1 = `max(dists) − dists` | `facility_location.py:12-34`, `facility_similarity=l1` default | **matches** |
| 4b | objective | Eq. 2/4/8: `argmax_{S,|S|≤b_q} Σ_i max_{s∈S}[C − ‖f_i − f_s‖]`, "C is a big constant" | submodlib `FacilityLocationFunction(dense, separate_rep=False)` `F:103` | `facility_location.py:99-112` | **matches**; C only shifts the objective: greedy picks are identical for C = max dist and C = 1000 max dist (check below) |
| 4c | algorithm | "found by maximizing a monotone submodular function via the greedy algorithm" | `optimizer="LazyGreedy"`, `stopIfZeroGain=False` (`F:104`) | same (`facility_location.py:108-112`) | **matches**; verified against a brute-force greedy (below) |
| 4d | budget per source | `b_q = (b − |S_s|)·|V^t_q| / (|M^L_t| − |S_s|)` | `F:69-79` `floor(size/n·B)` then +1 to the smallest counts (`U:129-146`) | `facility_location.py:37-62`, `class_budgets` | **matches**, rounding is not specified by the paper. Identical to upstream on 140 of 140 random pools (floor); the non-default `ceil` start differs from upstream in 43 of 140 (ours re-sorts after every decrement). A float `floor(size/n·B)` can undercount when `size·B/n` is an integer (B = n only: never reached, pool 128, B ≤ 64) |
| 5a | small sources | "we regard small sources as those with less than |V|/c examples"; "we include all of their examples from the large batch" (Sec. 4.1); Appendix C: "any data sources whose sizes are below the average count" | `S: KEEP_SOURCES="0_1_3_5_7_8_9_10_11_13"`, `T:1878-1892` excluded from selection, always trained | `select.py:100-113`, same default | **matches**: with the 14 MathInstruct sources (sorted names, sizes from `data/MathInstruct.jsonl`, mean 18,717) exactly the 10 sources below the mean are `0,1,3,5,7,8,9,10,11,13`; they are also the 10 smallest. 27% of a random pool is kept (about 35 of 128), 29 are selected from the four big sources (2, 4, 6, 12). (`use_small_sources` in `get_training_dataset.py` subsets the data by the same rule: unrelated to this) |
| 5b | mini-batch | Alg. 1: "Sample batch M_t ⊂ D"; bs = 64 selected from 128 (Sec. 5.2), 4 A40, gradient accumulation 8 | pool = 4 GPUs x 4 x GAS 8 = 128, gathered to rank 0, selection over the whole 128 (`T:1852-1873`) | `trainers.py:321-367`, same pool | **matches** |
| 5c | source-wise | "selecting subsets separately yields a slightly better performance" | `source_wise_selection=proportional` (`S`) | default `proportional` | **matches** |
| 6 | weights | "we assign uniform weights to all the selected examples" (Sec. 4.1); Table 2 lists "Weighted medoids" as the worse ablation | `SELECTION_METHOD=submodlib`: cluster sizes are computed (`F:106-117`) but only used for `weightedsubmodlib` (`T:855`); default trains unweighted | `select.py:145-154`: weights forced to 1.0 unless `weightedsubmodlib`; the efficient path rejects it (`training_arguments.py:_validate`) | **matches** (CRAIG cluster weights are not the paper's method). Cluster-size weights: the reference loop of upstream = ours (`tests/test_facility_location.py`, 140 of 140) |
| 7a | Adam transform | Eq. 3/4: "we normalize every gradient dimension by the exponential average of its historical values"; "we calculate the historical terms m, v only based on the big groups' gradients"; Eq. 3 writes `m_t = (β1 m_{t−1} + (1−β1) g_t) / (1−β1^t)` | `T:1895-1914`: standard Adam moments (stored uncorrected), corrected with `global_step + 1` when used; kept sources removed BEFORE the transform | `select.py:182-203` `_adam` | **matches** the description; checked against a float64 recursion of the formulas (max difference 1.7e-16). **Ambiguous**: Eq. 3 puts the bias correction inside the recursion (a different, inflated history); the code follows torch Adam |
| 7b | history | "historical terms m, v only based on the big groups' gradients" | `T:1944-1950`: `prev_m = mean over the SELECTED subset of m_t`, `prev_v = mean of v_t` (mean of squares of per-example estimates) | `select.py:141-144` | **ambiguous**: the paper does not say selected or all candidates, nor whether v is the mean of squares (code) or the square of the mean (real Adam). The history lives in the coordinates of the one z, so it is coherent only because z is fixed |
| 7c | first step | not stated | zero history, `m_hat/sqrt(v_hat) = sign(g z)` | same | first step: every feature is ±1 (verified 3e-6), the selection depends on sign(g_i) alone |
| 8a | frequency, ratio, schedule | "for t = 1..T: sample M_t ... θ ← θ − η∇L_{S_t}(θ)" (Alg. 1); 1K iterations; lr 2e-5; "cosine scheduler with a 3% warm-up period" (App. C); LoRA r 128, α 512, dropout 0.05; Q,K,V + two FC layers (Phi) | selection every step; `S`: `GAS=8 DEVICE_BS=4 RATIO=0.5`, `MAX_STEPS=5` (a placeholder), `B`: `lr_scheduler_type linear`, `warmup_ratio 0.03`, `learning_rate 2e-5`, LoRA 128/512/0.05, targets `q k v fc1 fc2` | `training_arguments.py`: `max_steps=1024`, `learning_rate 2e-5`, `warmup_steps=0.03`, `lr_scheduler_type` left at the HF default = **linear**; LoRA and targets as the paper | **deviates** (both codes, equally): the schedule is linear, the paper says cosine. Untested whether it matters; `--lr_scheduler_type cosine` is one flag |
| 8b | truncation | "We set a maximum sequence length of 512" | `max_seq_length 512` (E4b: 8.6% of the examples cut) | no truncation, the context window only (`docs/errors.md` E4b) | **deviates by design**; documented |
| 9a | training loss | θ ← θ − η∇L_{S_t}(θ) (normalisation not stated) | per micro-batch of 2 examples the HF token mean, ÷ 8 (accelerate accumulation), DDP mean over 4 ranks (`T:2011-2012`); the logged loss is also ÷ ratio (E5) | `batching.py:66-74`: ONE token mean over all label tokens of the 64 selected examples (transformers-5 accumulation semantics, `docs/optimization-backlog.md` O8) | **deviates** from upstream, mildly (part 2 quantifies it); ambiguous in the paper. Not listed in `errors.md` |
| 9b | selection loss | L_i(θ), the loss of example i | per-sample loss divided by the padded micro-batch width (E2, a bug) | own-length token mean (E2 fixed) | **matches** the paper after E2 |
| 10a | algorithm box | Alg. 1 lines 4-14 | `T:1852-2012` | `select.py`, `trainers.py` | **matches**: keep small sources, per big source a proportional budget, ZO gradient, normalised, mask, l1 facility location, union, one update on S_t. The ZO estimate is computed for the whole pool at once, not per source, which is equivalent because z is shared |
| 10b | SuperGLUE sources | Sec. 5.4: "we warm up the model for 20 iterations ..., and then cluster the model's hidden states ... update the clustering four times" | no clustering code; sources come from a pre-made `load-superglue-*` file (`sample.data['source']`) | same (`colm/data/superglue.py:71`) | **paper has it, code does not**: Table 7 "clustering during fine-tuning" cannot be reproduced from the released code. Out of scope of the MathInstruct recipe; not verified |
| 10c | theory and pre-training | Theorems 4.1-4.3, Appendix F (Llama-60M pre-training) | none | none | not checked (no code to check; the theorems are statements about a k-medoid model of the gradients) |

### Implementation checks (CPU, `scripts/diagnostics/check_selection_algorithms.py`)

Result file `paper-review-20260929/check_selection_algorithms.json`. Random inputs, seed 0.

* Facility location, submodlib `LazyGreedy` against a brute-force numpy greedy on the same similarity matrix
  (100 problems, n 20-60, d 2-200, budget 2-12): the same picks in the same order in 74 (l1) / 89 (euclidean)
  of 100; **every one of the other 26 / 11 diverges at an exact tie** (two examples of a mutually nearest pair
  have exactly the same gain) and the objective is equal to float64 rounding (relative 0.0). With `cosine`
  (non-default) the objective is not the plain facility location on the signed similarities (submodlib differs
  by up to 6.7%, 0 ties).
* The greedy objective against the exact optimum by enumeration (n = 12, budget 2-4, 20 sets): 0.956 at worst,
  0.994 on average (guarantee 1 − 1/e = 0.632). The picks are identical for C = max dist and C = 1000 max dist.
* The similarity matrix (l1 / euclidean as `max − distance`, cosine) against a direct loop: 6e-6 at most.
* Per-source budgets and cluster weights against the upstream functions (`git show a6257b0`, executed
  without their heavy imports): budgets equal on 140 of 140 (floor), picks and cluster sizes equal on 140 of
  140 random source compositions (including real MathInstruct pool compositions with the kept sources set
  aside); no source ever gets more than its size; the `ceil` start differs (see 4d).
* The Adam transform of `CoresetSelector._adam` against a float64 recursion of the formulas over 20 steps:
  1.7e-16.

## Part 2: does the premise hold?

The user's remark: using ONE random direction z is strange. It is, and this part measures how strange. The
paper's Eq. 5/6 is one z per estimate (as in SPSA / MeZO, where z is fresh at every step); nothing in the text
says that z is shared by the examples of a batch or fixed for the whole run, but with any shared z the
per-example features `ĝ_i = (z·g_i) z` are collinear.

### Set-up

* Model state: the 100-step Phi-2 LoRA r128 adapter of `layer-signal-20260928` (fixed: no training happens
  in these tests, "steps" are consecutive pools, the Adam moments of the selector evolve). 16 pools of 32 real
  MathInstruct examples from its pool file (512 examples; four consecutive pools form one 4-GPU selection pool
  of 128) plus 24 pools of 128 sampled uniformly from MathInstruct (seed 20260929, 3,072 examples): 3,584 examples,
  16 pools of 32 for the geometry and 28 pools of 128 for the selection tests.
* True gradient: for every example `g_i = dL_i/dB` of the perturbed parameter (last-layer v_proj LoRA-B,
  327,680 values), autograd through the last layer, final norm, LM head and the own-length token-mean loss, on
  the library's prefix (fp16, fp32 tail 2, promoted to fp32), MATH attention backward
  (`scripts/diagnostics/measure_true_gradients.py`, GPU 2, peak 13.6 GB, 100 s + 543 s).
  Validation: `g_i · z0` (z0 = the recipe's fixed direction, `zo_seed 534895718`) against the library's
  finite-difference estimate (`MezoEfficient.extract`) on all 512 examples: median relative difference 1.7e-3,
  correlation 0.999999; the library estimate with the pool packed in reverse order differs from the forward
  packing by 4.4e-3 (median), 2 sign flips in 512 (the repeat noise of the fixed-z selection).
* Selection: the library's own `CoresetSelector` (Adam transform with its moments, per-source top-2560
  mask, source-wise facility location, l1, 64 of 128 with the 10 kept sources), one chain per variant with its
  own moments, from step 0 like a real run; only the features differ. "Oracle" = the same selector fed the true
  gradients. Overlap = picks of the 4 big sources (29.8 per step; 9.68 = 32.5% expected for two independent
  random selections with the same quotas); skill = (overlap - random) / (picks - random), 0 = chance, 1 = same picks.
  Scripts: `analyze_z_geometry.py` (CPU), results `paper-review-20260929/analysis.json`, `analysis_steps.json`.
* Not done: whole-model gradients (only the ZO parameter, the paper's proxy), a changing model, learning runs.

### What the true gradient looks like (T1a, 16 pools of 32, all pairs)

| quantity | value |
|---|---|
| mean pairwise cosine between examples | 0.0018 (std over pools 0.0022): the gradients are nearly orthogonal |
| cosine of an example with the pool mean | 0.16 |
| coefficient of variation of the norms | 0.49 (median ‖g‖ 0.23, range 0.04-1.0) |
| energy of the top 1 / 3 / 10 singular directions of a pool of 32 | 15% / 32% / 67%; participation ratio of the centred gradients 15.8 |
| Spearman between l2 and l1 distances | 0.996 |
| Spearman between l2 and cosine distance | 0.06 |
| `Spearman(‖g_i‖, loss_i)` (candidates of the pools of 128) | 0.81 |

Distances are dominated by norms (`‖g_i − g_k‖² ≈ ‖g_i‖² + ‖g_k‖²`); there are no dense, low-dimensional
clusters for medoids to sit in, and the cosine geometry is noise.

### What one z keeps (T1b, T1c, T3)

* Fraction of a gradient kept by the projection on z: `|z·g_i| / (‖z‖ ‖g_i‖)` = **0.0016** for z0, 0.0014 for fresh z,
  the value for a random direction is 0.8/sqrt(d) = 0.0014: about `1/d = 3e-6` of the squared norm.
* Sample mean of m directional estimates `(1/m) Σ_j (g·z_j) z_j` against the true gradient (T3, 64 examples,
  cosine, theory `sqrt(m/(m+d))` in brackets): m = 1: 0.0015 (0.0017); 16: 0.0068 (0.0070); 256: 0.0281
  (0.0279); 4096: 0.1111 (0.1111). The estimator is unbiased over z but its direction is nearly orthogonal to g
  until m is of the order of d; with m = 1 it is a random direction times a scalar.
* Distances of m random projections against the true l2 distances (all pairs of the 16 pools, Spearman /
  Pearson; fresh draws: mean over draws, in brackets 5-95% of the Spearman):

| projection | Spearman | Pearson |
|---|---:|---:|
| m = 1, the recipe's z0 | 0.39 (0.26-0.54, over pools) | 0.46 |
| m = 1, fresh z (3200 draws x pools) | 0.34 (0.13-0.55) | 0.38 |
| m = 1, a different z for every example (control: the features are not comparable) | 0.36 (0.21-0.56) | 0.37 |
| m = 4 | 0.65 (0.48-0.80) | 0.67 |
| m = 16 | 0.87 (0.80-0.93) | 0.88 |
| m = 64 | 0.96 (0.94-0.98) | 0.97 |
| m = 256 | 0.99 | 0.99 |

  The shared z is **not better than an independent z per example**: what one direction keeps of the
  distances is the norm of the example (its `|z·g_i|` grows with `‖g_i‖`; Spearman of `|s_i|` with `‖g_i‖` 0.53, with
  the loss 0.42), not the direction. The projection on the pool-mean gradient (a directional quantity) is
  uncorrelated with `s_i = z0·g_i` (Spearman -0.03).

### The features as the code builds them (T1b)

* With one z all features of a step are `g_i z`, and after the Adam transform (history proportional to z and
  z²) `sign(z_j) h(g_i)`: the centred feature matrix of a source has **rank 1** (share of the first singular
  value 1.000 for z0, for the finite-difference estimate and for fresh z without Adam; 0.996 for fresh z with
  Adam). The l1 distance is `2560 |h(g_i) − h(g_k)|`, facility location is a 1-D k-medoids on a scalar
  (the paper's rank-1 remark, applied across examples). At step 0 the update is the sign of g, every feature is ±1
  and the selection sees sign(g_i) only (verified, 3e-6). The coordinate mask is decided by ties: all
  |mean| are equal up to rounding; the selected coordinates have mean |z_j| 1.09 against 0.80 for a random
  coordinate (the 1e-8 in Adam's denominator favours large |z_j|). Cosine distance on these features would be
  degenerate.
* Correlation of the facility-location distances (within source, pipeline features, l1) with those of the
  oracle over 28 steps (Pearson / Spearman): z0 0.37 / 0.36, fresh z 0.25 / 0.25 (the moments of the previous
  z are unrelated to the new one), fresh z without Adam 0.40 / 0.39, the per-example-z control 0.50 / 0.48 (the
  oracle's distances are themselves dominated by the norms), m = 4: 0.56 / 0.56, 16: 0.74 / 0.70, 64: 0.82 / 0.78, 256: 0.85 / 0.81.

### Selection agreement (T1d, T2; 28 steps of pools of 128)

| variant (features into the same selector) | picks in common with the oracle (of 29.8) | skill vs oracle | skill vs "raw l2, all coordinates" |
|---|---:|---:|---:|
| random within candidates | 9.7 | 0 | 0 |
| **z0, exact directional derivative (the recipe's z)** | 10.4 | 0.04 | 0.07 |
| z0, the library's finite difference (in order / reversed packing) | 10.8 / 10.4 | 0.05 / 0.04 | 0.09 / 0.06 |
| fresh z every step (two independent draws) | 10.0 / 10.0 | 0.01 / 0.01 | 0.02 / 0.05 |
| fresh z, no Adam | 10.8 | 0.06 | 0.09 |
| m = 4 directions, features = the m derivatives | 11.6 | 0.10 | 0.13 |
| m = 16 (fresh every step / fixed) | 14.7 / 14.3 | 0.25 / 0.22 | 0.32 / 0.30 |
| m = 64 | 17.2 | 0.37 | 0.56 |
| m = 256 | 17.8 | 0.39 | 0.79 |
| a different z per example (control) | 10.7 | 0.05 | 0.18 |
| oracle without Adam (raw gradient, mask, l1) | 17.2 | 0.37 | 0.72 |
| oracle, raw l2 on all 327,680 coordinates | 18.2 | 0.42 | 1.00 |
| oracle, cosine on all coordinates | 9.7 | 0.01 | 0.00 |

  (skill: mean over the 28 steps; standard error 0.01-0.03.)

* The recipe's selection agrees with the oracle at chance. Two independent z pick sets that agree at chance
  (33.4% of the picks for z0 against fresh, and for fresh against another fresh; chance 32.5%): the selection is a
  function of the draw of z, not of the data. The same z0 with fp32 repeat noise agrees 76-81% with itself.
* **T2**: a fresh z per step is not better than the fixed one (skill 0.01-0.06 against 0.04). Repeating the fixed-z
  chain over 24 independent z (28 steps each): skill 0.037 +- 0.019 (5-95%: 0.013-0.065), the recipe's z0 gives
  0.030 (42nd percentile); the gradient-matching error is the same for all of them (0.964 +- 0.006; z0: 0.966).
  Nothing about z0 is special, and nothing about the draw matters: every z is at chance.
* The oracle is itself fragile: the same selector on the raw gradients (no Adam) agrees on 57% of the picks with the
  Adam version, with cosine on 32% (chance). The geometry of the selection depends on the design of the transform
  as much as on the estimate. Against the cleanest reference (raw l2, all coordinates) m = 256 projections through
  the same Adam + mask + l1 pipeline agree with skill 0.79 (m = 64: 0.56), z0 with 0.07.
* Which examples do the picks favour? The oracle's picks overlap the highest-norm examples of their source
  (15.9 of 29.8, chance 9.7) and the highest-loss ones less (12.8); z0's picks 11.6 and 10.5.

### Does the coreset match the large-batch gradient (the premise of Sec. 3)? (T1e)

Relative error `‖mean g over the 64 - mean g over the 128‖ / ‖mean g over the 128‖` on the ZO parameter, 28 pools
(standard error about 0.01):

| selection of the 64 | uniform weights (the paper's) | cluster-size weights (CRAIG) |
|---|---:|---:|
| random 64 of 128 | 0.914 | |
| random within candidates (kept sources always, per-source quotas) | 0.967 | 1.178 (stratified weights) |
| oracle (true gradients, code selector) | 0.979 | 1.255 |
| z0 (the recipe) | 0.963 | 1.326 |
| m = 64 / 256 | 0.976 / 0.984 | 1.271 / 1.454 |
| lowest / highest gradient norm within source | 1.004 / 0.994 | |
| the 35 kept examples alone | 1.758 | |

No selection, not even the oracle, is closer to the pool's gradient than random: the mean gradient of a pool is
a small common component under near-orthogonal per-example gradients, and the mean of any 64 examples deviates
from it by about its own size. With one z or with the true gradients, the coreset selection here does not
reduce the deviation from the large batch; what selection changes (composition of sources) is decided before
the features. This is measured on the last-layer proxy, not on the whole-model gradient.

### Training-loss weighting (item 9a)

Step gradient on the ZO parameter of the library's global token mean against upstream's mean of per-micro-batch
(2 examples) token means, on random selections of 64 (28 pools x 20): cosine 0.86, relative difference 0.51; the
largest example weight is 6.3x uniform in the library (a long chain-of-thought example carries its token share) and
1.9x upstream (a share of at most 1/2 inside its pair); total variation between the two weightings 0.26. The
paper says the selected examples get "uniform weights"; neither code weights the examples equally (upstream
within pairs only).

### What this means

* **Bugs**: none found in the selection machinery: facility location (up to exact-gain tie-breaks), similarity,
  budgets, the Adam transform and cluster weights are correct against independent references and equal to
  upstream (default path). Minor: the `ceil` budget rounding differs from upstream (non-default); a float `floor`
  in the budget could misround where `size·B/n` is an integer (unreachable at B < n).
* **Design, not a coding error**: one z per run (paper: ambiguous, MeZO: fresh per step) makes the feature a
  1-D quantity that carries the example's gradient norm and no direction. With it the l1 distance, the top-2560
  mask, the cosine variant and most of Sec. 4.2-4.3 have no effect beyond a monotone transform of a scalar. A fresh z
  per step does not repair it (every single direction is at chance); the estimate needs m of about 64-256 directions
  to reproduce the geometry (Spearman 0.96-0.99) and still agrees with the code's own Adam-transformed oracle
  only at skill 0.37-0.39, because that oracle differs from the raw geometry (its skill against the raw l2
  selection is 0.42).
* The premise "small mini-batches that match the gradient of the large batch" is not realised in this space, for
  the oracle either: the per-example gradients of the last-layer LoRA-B are nearly orthogonal (cosine 0.002), so
  a medoid represents only itself and the norm. By the paper's own Table 2 the largest gain comes
  from keeping the small sources (+3% average accuracy), then Adam normalisation (+1.5%), per-source selection
  (+0.5%) and uniform weights (+0.4%), against a reported standard deviation of 0.5-0.9 on the average column.
  Table 5 reports the sparsified MeZO estimate above the sparsified true gradient (average 56.6 +- 0.9 against
  54.7 +- 0.3; in-domain 51.9 against 51.0, out-of-domain 61.4 against 58.3). A random 1-D projection of that
  gradient cannot carry more geometry than the gradient, so if that gap is real it must come from what a signed
  norm-like scalar selects (hypothesis, not tested: e.g. a different balance of high-norm examples than the
  geometry-based picks: the oracle's picks lean to high-norm examples, z0's much less), or from noise. This
  matches `docs/selection-precision.md`: at 300 steps F is not distinguishable from random selection with the
  selector's structure. Not tested here: any effect of the selection on 1024-step accuracy.
* **Deviations from the paper text that are unrelated to z**: linear instead of cosine learning-rate schedule (both
  codes); loss weighting (above); no clustering of hidden states for SuperGLUE (code missing).

### Proposals (nothing applied; the user decides)

| # | change | what it costs | expected effect |
|---|---|---|---|
| P1 | Use the exact last-layer gradient (autograd through the replayed last layer, per example) as the feature instead of `g_i z` (the paper's Table 5 row "sparsified actual grad" scored below sparsified MeZO, 54.7 against 56.6 on average: its own data argue against it, so decide with a learning run) | rough timing on a shared GPU, pool of 32 (7 packs, 8.7k tokens): prefix 470 ms; the library's m = 1 estimate 135 ms; exact gradient 303 ms (one forward, 32 backwards through layer and head); memory: one layer's activations, the head logits at the label positions | features carry direction, but the picks then depend on the transform (Adam oracle against raw oracle: 57% of the picks in common) |
| P2 | Keep the forward-only estimate but with m directions per step (the same z_1..z_m for all examples) and the m derivatives as the feature | same timing run: m = 1 135 ms, 4: 538 ms, 16: 2.6 s, 64: 14 s (every direction replays the last layer and the LM head at the label positions twice): m = 64 costs 30x the prefix | Spearman 0.96 with the true distances at m = 64; not affordable at m = 64, useless at m <= 4 |
| P3 | If the single z is kept, say so in the documentation: the feature is a signed gradient-norm score, and `zo_dim`, `facility_similarity` and `mezo_topk` do nothing there | none | honesty of the ablations |
| P4 | Loss weighting: choose explicitly between the global token mean (now), upstream's per-micro-batch means, and a per-example mean ("uniform weights" of the paper) | a function in `batching.py`; a learning comparison to decide | removes a 6x weight on long examples if uniform is intended |
| P5 | `--lr_scheduler_type cosine` to follow the paper | one flag; the effect is untested | matches App. C |
| P6 | Budget rounding: integer arithmetic for `floor`, the upstream `ceil` order | a few lines | none in the default path |

### Unverified

Whole-model gradients (only the ZO parameter, the paper's proxy); a changing model along a run (one 100-step
adapter, static); any downstream accuracy effect (the 300-step learning runs of `docs/selection-precision.md`
found none); the paper's theorems; the SuperGLUE clustering; the unpublished parts of the authors' pipeline;
the timings (one run on a card shared with other jobs, no repeats; ratios only). The oracle is the code's own
selector on true gradients, one of several defensible references (the table shows how much they disagree).
Bulky outputs (`G_pools.npy`, `G_extra.npy`, meta, z0, JSONs, logs): `/mnt/data2/zelin4593/artifacts/CoLM/paper-review-20260929/`.

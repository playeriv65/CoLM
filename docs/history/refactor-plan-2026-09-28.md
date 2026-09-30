> **Historical design notes of the refactor (2026-09-28).** Kept as written. The `legacy` switch and
> several options mentioned below were removed afterwards, so parts of this text no longer describe
> the code. The current state is in `README.md`, `AGENTS.md` and `docs/errors.md`.

# CoLM refactor design (design only, no code changed)

Repo: `/mnt/data/zelin4593/research_pro/CoLM`, branch `zelin-li-refactor` at `eea53ab` (public repo, no secrets).
Read for this design: AGENTS.md, TODO.md, README.md, docs/optimization-backlog.md, every file in `colm/`, `math_eval/`, `superglue_eval/`, `tests/`, `configs/`, `scripts/`, plus the installed transformers 5.17.0 / peft 0.21.0 / accelerate 1.15.0 / torch 2.13.0 sources under `.venv/lib/python3.12/site-packages`.
Line numbers below are for `eea53ab`; **branch `task/exact-opts` will move them** (it edits `trainers.py`, `custom_phi.py`, collation), so every reference is re-resolved after that merge (step S0).

Scope decision (from the user): keep **every selection unit** (`rep`, `mezo`, `masked_grad`, `completion_length`, `length_loss_weighted`, efficient MeZO) and **every trainer** (`CustomTrainer`, `SubsetTrainer`, `SubsetTrainerEfficient`). They stay as classes/options; what goes is the hack around them.

Legend for the "class" column: **EXACT** = bitwise identical on CPU fp32 (indices, weights, features, losses, LoRA weights) and identical indices on GPU; **FLOAT** = identical maths, different float rounding (must still give identical selected indices in the harness); **B#** = behaviour change, listed in section 1.4, ships behind a named option whose default is the legacy behaviour until you decide.

---------------------------------------------------------------------------------------------------

## 0. Ground rules for the whole refactor

1. **Equivalence first.** A frozen reference (git tag `pre-refactor` on the merged `task/exact-opts` head) produces golden files (section 3, S0). Nothing is deleted or rewritten before the golden that covers it exists.
2. Every step lands as its own commit on a `task/refactor` worktree, `ruff format/check` + `CUDA_VISIBLE_DEVICES="" pytest -q` green, goldens unchanged unless the step is a B# step.
3. Public entry points stay: `python -m colm.train.train <config.json>`, `scripts/run.sh`, JSON key names (moved between dataclasses, not renamed, unless you approve Q4), `indices/iter*_{full,sampling,selected}_indices.pt` file layout, the `step_timing` JSONL tree node names (the summariser and `docs/optimization-backlog.md` depend on them).
4. No new hard-coded strings: enum-like options become `Literal[...]`/`StrEnum`, model-specific facts move to `configs/model_profiles/*.json`, batch keys to one constants module.

---------------------------------------------------------------------------------------------------

## 1. Inventory

### 1.1 Hacks and smells (file:line at eea53ab)

**A. Trainer / HF-loop integration (`colm/train/trainers.py`)**

| ID | Where | Why it exists | Replacement (API, location) | Class |
|---|---|---|---|---|
| A1 | 293-295, 363-365 (`{}` placeholder micro-batches, `if not inputs`) | The stock loop derives `do_sync_step` from a running `step` counter modulo `gradient_accumulation_steps` (`transformers/trainer.py:1821-1823`) and the DDP `no_sync` context from `i == len(batch_samples) - 1` (`:1833-1841`), so a step that trains fewer micro-batches than `gas` must be padded. `SubsetTrainer` only (Efficient returns exactly `gas` micro-batches). | Keep the mechanism (it is the only stock-loop-compatible one; the alternatives change what `gradient_accumulation_steps` means or need a copy of `_run_epoch`, which AGENTS.md forbids) but isolate it: `PlaceholderBatch` sentinel dataclass + `MicroBatchPlan.pad_to(gas)` in `colm/selection/plan.py`, `isinstance` check instead of truthiness, one docstring explaining the loop arithmetic with the trainer.py line refs, and a test that asserts the sync step. | EXACT |
| A2 | 291-317 (`_training_step` = copy of `Trainer.training_step`) | Needs `/ n_trained` instead of `/ gas`, per-example weight, logged loss `/ small_batch_ratio`. | Stock `Trainer.training_step` (`trainer.py:1975-2046`) already divides by `self.current_gradient_accumulation_steps` (`:2035-2037`) and handles `n_gpu>1`, empty-cache, `optimizer.train()`. Override becomes: `self.current_gradient_accumulation_steps = plan.n_trained; return super().training_step(...) / log_divisor`. The weight moves into `compute_loss` (`trainer.py:2048`, documented as the override point). Fallback if you dislike writing HF's private attribute (it is only assigned in `_run_epoch`, `:1813`): keep a 12-line step; a test pins the scaling either way. | EXACT (verify bitwise) |
| A3 | 166-182 overrides of private `_maybe_log_save_evaluate`, `_clip_grad_norm`, `_track_num_input_tokens`, public `floating_point_ops` **only to time them** | Step timer sections around HF internals. | Timing via `TrainerCallback` events: `on_step_begin/on_pre_optimizer_step/on_optimizer_step/on_substep_end/on_step_end` (`transformers/trainer_callback.py:371-392`) give the interval boundaries; residual is reported as `hf_loop_other`. `floating_point_ops` is a documented override point (`trainer.py:4064`) and returns 0; token tracking is already a no-op when `include_num_input_tokens_seen == "no"` (`trainer.py:2587-2588`). Removes 4 private-API overrides. Timing tree node names for selection/train phases stay; `backward` becomes `train - forward - prepare_inputs`. | EXACT (timing only) |
| A4 | 184-193 `_model_inputs` filters batch keys through private `_signature_columns` / `_set_signature_columns_if_needed`, plus `modify_forward` branch | The collator emits extra keys (`sources`, `indices`, `weights`, `completion_lengths`) the model forward rejects. | Collator emits tensors for the model and **one** reserved key `colm_meta` (dict of `[B]` tensors); `compute_loss` pops it. No signature introspection. | EXACT |
| A5 | 107 `model_accepts_loss_kwargs = False` | Restores 4.43 loss scaling. | Keep. It is exactly what the `compute_loss` docstring prescribes (`trainer.py:2073`). Add a comment naming the HF contract. | - |
| A6 | 315-317 logged loss `/ small_batch_ratio` | Upstream curve comparability. | Named option `log_loss_divisor: "small_batch_ratio" \| "none"` (default legacy). | EXACT |
| A7 | 241 `zo_random_seed = np.random.randint(1000000000)` | Relies on global numpy state after `set_seed`; never logged. | Config `zo_seed: int \| None`; `None` derives `np.random.RandomState(seed).randint(1_000_000_000)`. Measured: this equals the legacy draw (`np.random.seed(s); np.random.randint(...)` is the same MT19937 stream), valid as long as nothing consumes numpy RNG between `set_seed` and trainer init (it does not; the harness asserts the value). The seed is logged at trainer init. | EXACT |
| A8 | 110-115 `self.dtype`; 352, 397 | Used for scalar features and weights. | Explicit `Features.dtype` / `weights: float32` (they are cast `.to(self.dtype)` after building in float32, so weights are fp16-rounded under AMP: kept, see B4). | EXACT |
| A9 | 341-354, 436-438 python-type dispatch (`isinstance(rep, (int, float))`, `all(isinstance(r, int)...)`) | Scalar units return python numbers, vector units tensors. | `Features(values: Tensor[n, d], valid: BoolTensor[n], kind)`; `kind` in {`vector`, `int_scalar`, `float_scalar`} reproduces the three dtype paths (long, AMP dtype, float32). | EXACT |
| A10 | 344 `torch.norm(rep).item()` per example | Validity check = one CUDA sync per example. | Vectorised `valid = isfinite.all(1) & (values.norm(dim=1) != 0)` in the pipeline; one sync per pool. Same predicate (empty / NaN / zero norm; note `isnan` only, so `inf` stays valid as today). | EXACT |
| A11 | 258-262, 773-779 `_check_args` via `assert` + `SubsetTrainerEfficient(SubsetTrainer)` overriding 5 methods | Efficient reuses SubsetTrainer plumbing by inheritance; asserts vanish under `python -O`. | Both derive from `CoresetTrainer`; they differ only by injected policy objects (extractor set, `BudgetPolicy`, `MicroBatchPlan`, `drop_invalid`); validation is `SelectionArguments.validate(trainer_kind)` raising `ValueError`. | EXACT |
| A12 | 335-645 (about 310 lines) selection algorithm inside the Trainer, reading `self.args`, `self.optimizer`, `self.state`, `self.named_parameters_to_optim`; tests monkeypatch `trainer._select_on_main` | Grew organically. | `colm/selection/` package (section 2). `CoresetSelector` takes plain tensors and returns `Selection`; unit-testable without a Trainer. | EXACT |
| A13 | 298, 742, 871, 881 `model.eval()` / `model.train()` per call (module walk, measured 28+33+27 ms/step in the backlog) | Dropout must be off for ZO forwards. | One `selection_mode(model)` context manager around the whole selection (eval on enter; the stock `training_step` calls `model.train()` at the first real micro-batch, `trainer.py:2004`). Overlaps the exact-opts host-overhead item: **reconcile at S0**. | EXACT |
| A14 | 146-149, 204-212, 490-501, 570-576 `save_indices` (`_micro_step * world_size + rank` arithmetic, O(N^2) `extract_and_save_original_indices`) | Debug artefact writer inside the selection code. | `IndexRecorder` (observer called with `Selection` + pool meta; registered from the trainer only when `save_indices`). Same file names and payloads; O(N) lookup. | EXACT |
| A15 | about 60 `t.section` / `t.fine` blocks woven through the logic | Step timing. | Stage runner wraps each stage in `timer.section(stage.name)` (names kept); fine timers inside the model forward are attached as forward hooks by a profiler helper (`register_forward_pre_hook/forward_hook`, same mechanism `custom_phi.py:31-44` already uses for the last layer); a few explicit `timer.fine` remain in the GEMM-sized spots. | EXACT |
| A16 | 533-551 `_adam_update` reads `self.optimizer.state[...]["exp_avg"]`; 596-600 `select_masking` reads model weights | Cross-object reach-ins. | `MomentSource` protocol (`OwnMoments` = the running mean over the selected subset, `OptimizerMoments` reads torch `Adam/AdamW` state) and `FeatureExtractor.coordinate_prior()` (`mezo_selection=weight`). Stages receive what they need as arguments. | EXACT |
| A17 | 537-551 silent zero moments when the real optimizer has no `exp_avg` yet **or is not Adam** | Cold start. | Cold start (empty state) stays zeros; a non-Adam optimizer raises at bind time. | B8 (only non-Adam optimizers) |
| A18 | 525 `clip_last` divides by literal `32` | 32-layer assumption. | `model.config.num_hidden_layers`. | EXACT for 32-layer models |
| A19 | 439-441 `v` uses the **untransformed** squares while `m` uses the transformed features | Upstream. | Kept; the Adam stage takes `(transformed, raw_squared)` explicitly and the docstring says so. | EXACT |
| A20 | 465 `torch.randperm` (global torch RNG), 623 `np.random.choice` (global numpy RNG) for `mezo_topk=random/sampling` | Global RNG use. | Dedicated `torch.Generator`/`np.random.default_rng` seeded from `selection_seed`. Only non-default top-k modes (default is `largest`). | B9 |
| A21 | 151-163 `log()` override adding `step_time_s`, `select_time_s`, `peak_mem_gb` | Extra metrics. | Keep: `Trainer.log` is a documented override point (`trainer.py:4029`). State moves into a small `StepMeter`. | - |
| A22 | 239, 362 `_num_train_microbatches`, `_micro_step` mutable trainer attributes shared between hooks | Hidden coupling between `get_batch_samples` and `training_step`. | Immutable `MicroBatchPlan` stored on the trainer for the current update (`self._plan`). | EXACT |

**B. Zeroth-order path and model surgery (`custom_phi.py`, `trainers.py`)**

| ID | Where | Why | Replacement | Class |
|---|---|---|---|---|
| C1 | `custom_phi.py:16-137` re-implements the PhiModel forward (embed, position ids, mask, dropout, rotary, layer loop) and the head+loss; `isinstance(PhiForCausalLM)` gate | Need the hidden state before the last layer and to replay the last layer twice. Re-implementation must track every transformers change (mask API already moved once, packing/attention backends are being added to it right now). | `colm/models/split.py::LastLayerSplit`, architecture-agnostic: prefix = the **model's own decoder forward** with a `register_forward_pre_hook(..., with_kwargs=True)` on `layers[-1]` (`torch/nn/modules/module.py:1624`) that stores `(args, kwargs)` and raises a private `_PrefixDone`; replay = `torch.func.functional_call(last_layer, overrides, args, kwargs)` (`torch/_functorch/functional_call.py:13`) followed by the final norm. Because the real forward builds mask, rotary, dropout and packed-input handling, attention backends and packing (F5/F6/O7) are inherited for free. Exception-safe: `capture_outputs` resets its state in `finally` (`transformers/utils/output_capturing.py:285-290`) and hooks are removed in `finally`. Probe on tiny Phi **and** Llama (CPU, this session): replayed last layer + final norm equals the full decoder output **bitwise**; `functional_call` with a shifted weight changes the output and leaves the parameter untouched. Bind-time `verify()` runs a 2-example dummy batch and raises if replay != full forward. | EXACT (probe: bitwise) |
| C1b | `custom_phi.py:31-44, 100-110` sub-block timing hooks | Fine timing. | Profiler helper (A15). | - |
| C1c | Final norm / lm_head lookup | Phi calls it `final_layernorm` (`modeling_phi.py:390`), Llama `norm`; `PreTrainedModel.get_decoder()` raises for Phi (`modeling_utils.py:2251-2262`: only looks for `language_model/text_model/decoder/text_decoder`). | Use `base_model` (`modeling_utils.py:1472`, `getattr(self, base_model_prefix, self)`) for the decoder, `get_output_embeddings()` (`:1055`) for the head, and `layers_attr` / `final_norm_attr` from the model profile with an auto-detect (unique norm-class child of the decoder) and the bind-time self-test as safety net. Selection forward must call the **inner** decoder module, not the accelerate-wrapped outer `forward`, to stay fp32 like today (F3). | EXACT |
| C2 | `trainers.py:723-738` `param.data = param.data + s * z * eps`, then `+eps, -2eps, +eps` | MeZO in-place perturb/restore. | Out-of-place: `Perturbation.shifted(sign)` returns `{name: theta + ...}` overrides fed to `functional_call`; parameters are never written. To stay bitwise on the loss: `plus = theta + (1*z)*eps`, `minus = plus + (-2*z)*eps` (same arithmetic as today). Legacy in-place restore leaves `theta` off by up to 7.5e-9 per estimate in a fp32 probe (`((th+z e)-2 z e)+z e != th`, measured this session); out-of-place removes that drift. | EXACT per estimate; drift removal is FLOAT over steps (B2) |
| C3 | `trainers.py:674, 727, 853` `torch.manual_seed(zo_random_seed)` on the **global** RNG | Regenerate the same z. | `torch.Generator(device).manual_seed(seed)` + `torch.normal(..., generator=g)` (`torch/_torch_docs.py:8416`). Probe: CPU stream is identical to the reseeded global stream (`torch.equal`); CUDA must be checked on GPU in S9. Side effect to be aware of: today every estimate resets the global torch RNG, so the training dropout stream restarts from the same state every step. | z: EXACT. Training RNG stream: **B1** (option `legacy_global_rng`, default on until you decide) |
| C4 | `training_arguments.py:5-11, 129-148, 197-203` substring match `layers.31.self_attn.v_proj` (`LAST_LAYER_GROUPS`, `ATTENTION_MODULES`, `last_layer_index=31`), then train.py:217 appends `.lora_B` to the args | Phi names, 32 layers, args mutated as data flow. | `ZOTarget(layer=-1 -> num_hidden_layers-1, modules=[...], param_suffix="lora_B" if lora else "")` resolved once by anchored regex over `named_parameters()`; groups (`qkv_proj`, `fc`) live in the model profile. Resolved names are logged. `param_suffix=""` keeps the full-weight (weight+bias) case for `lora=false`. | EXACT |
| C5 | z drawn 4x per micro-batch (perturb+, -2, restore, feature) | Consequence of C2/C3. | Under `direction = fixed_seed` (F1: z is constant for the whole run) draw z once at bind and cache; `per_step` policy (seed + step) is a named option for research, not needed by any current config. | EXACT |
| C6 | `trainers.py:763-766` "efficient MeZO perturbs exactly one tensor" | `custom_phi` design. | Lifted: any set of last-layer parameters works with `functional_call` (z per tensor drawn sequentially from one generator in `named_parameters()` order, as today). Optional generalisation; the assert can stay for v1. | EXACT |
| C7 | `custom_phi.py:123-133` per-sample loss = `per_token.mean(dim=1)` over the padded width (F2) | Upstream. | Named option `zo_loss_normalizer: "padded_length" \| "valid_tokens"` (default legacy = D1 in the backlog). | EXACT for default |
| C8 | `custom_phi.py:115-116` full `[B,T,V]` `logits.float()` | - | Not touched here: exact-opts O5 (label-only lm_head) lands first; `LastLayerSplit.suffix_hidden()` returns the final-normed hidden so the head/label gather stays a separate, replaceable step. | - |

**C. Data (`colm/data/*`, `colm/train/utils.py`)**

| ID | Where | Why | Replacement | Class |
|---|---|---|---|---|
| D1 | `get_training_dataset.py:363-404, 559-585, 623-657` tokenisation inside the collator on every step (main process when `dataloader_num_workers=0`) | Upstream design. | `datasets.Dataset` (already a dependency) + `.filter(len(output)>0)` + `.with_transform(tokenize)` (lazy, runs in dataloader workers, deterministic per string, so token ids are identical). Prompt formatting and `sorted(set(source))` -> int mapping become dataset columns. | EXACT (golden on batches) |
| D2 | `:537-585` vs `:587-657` two near-identical collators (+ dead `naive__call__`) | Source/no-source variants. | One `SelectionCollator(DataCollatorForSeq2Seq)` (`transformers/data/data_collator.py:489`): pads `input_ids/labels/attention_mask` with `label_pad_token_id=-100`, integer meta columns pass through as `[B]` tensors (probe: `sources/indices/completion_lengths` survive as int64 `[B]`), moved under `colm_meta`. **Must set `tokenizer.padding_side="right"` explicitly**: the stock collator follows the tokenizer (Llama tokenizers default to left), while `rep` reads the last real token as `attention_mask.sum-1` and the padded-mean F2 divisor assumes right padding. | EXACT for phi-2 (mask dtype int vs bool: B6 note) |
| D3 | `:406-535` `SupervisedDataset`: Python loops over 262k rows, dead `get_super_class`, `naive__getitem__`, unused `weights` column | - | Replaced by D1. | EXACT |
| D4 | `:62-217` `load_raw_dataset`: hard-coded `subset_selection = "use_small_sources"` (`:95`), five branches of which four are dead, hard-coded "smallest 10 sources" (`:183`), `torch.load` of pickled index files (`:76,84`), asserts | Sampling experiments. Only reachable when `percentage != 1.0`. | Delete (Q2) or replace by explicit `subset_indices: list[int]` from a JSON/npy file. | EXACT for `percentage=1.0` |
| D5 | `:219-360, 660-745` + `train.py:302-305`: LESS-format path (`encode_data`, `encode_with_*`, `concat_messages`, llama2-chat variant with a stray `print`) | Inherited from LESS. Unreachable for MathInstruct (`"instruction"` column). | Delete unless you train on messages-format data (Q2). | - |
| D6 | `data_arguments.py` `max_seq_length` only used by D5; MathInstruct truncation is `tokenizer.model_max_length` (set from `model_args.model_max_length`, `train.py:160`) | Two knobs for one thing. | One field `max_length`. | B10 (the ignored knob disappears; value identical) |
| D7 | `get_training_dataset.py:23-30` vs `data/utils.py:538-544` two different `temp_seed` (one sets the global torch seed and never restores it) | Copy-paste. | One helper, restoring both numpy and torch state. | EXACT (only used when D4 branches run) |
| D8 | `train/utils.py:8-34` `collate_fn` re-collation of single-example slices + `trainers.py:94-97` `_split_examples` | Re-batch selected examples; each row keeps the padding of its original batch. | `ExamplePool` stores rows (padded, as sliced today, plus `length`); `SelectionCollator.collate(examples, padding="original")` reproduces `pad_sequence` behaviour. `padding="tight"` (trim rows to real length first) is a named option: same maths, fewer padded tokens (45% padding in training per the backlog), different shapes (dropout masks/float rounding). | original: EXACT; tight: B3 |
| D9 | `training_arguments.py:89-92` `keep_sources: "0_1_3_5_7_8_9_10_11_13"` string; `train.py:309-318` parses it **only if** the collator class is the sourced one, else silently `[]`; sources are ints (`trainers.py:417, 453-455`) | Encoding by dataset order. | `keep_sources: list[str]` of source **names**, resolved to ids at dataset build (fail fast if a name is missing). Names for the current default (indices into the sorted source list, computed from `data/MathInstruct.jsonl` this session): `data/CoT/MATH_train.json, data/CoT/TheoremQA.json, data/CoT/college_math.json, data/CoT/gsm_train.json, data/CoT/number_comparison.json, data/PoT/MATH_train.json, data/PoT/TheoremQA.json, data/PoT/aqua_rat_filtered.json, data/PoT/gsm_gpt4.json, data/PoT/numglue.json`. A test asserts they are the 10 smallest sources. | EXACT |
| D10 | `print()` in data and `facility_location.py:52` | - | `logging`. | - |
| D11 | SuperGLUE training path: `train.py:226-286` (branch on `"superglue" in train_files[0]`, task name parsed from the file string), duplicated `convert_superglue_to_hf(_source)` (`get_training_dataset.py:759-930`), monkey-patched `model.forward` (`train.py:258-263`, `data/utils.py:167-255`), `modify_forward`, `NondiffCollator`, `DataCollatorWithPaddingAndNesting`, `ICLCollator` | MeZO-repo tasks. Not tested, not used by any config; `superglue_eval/` (evaluation) needs only `tasks.py`, `templates.py`, `encode_prompt`. | Q1. Default proposal: delete the training path and the monkey-patch, keep evaluation-side modules under `colm/data/superglue/`. If kept: one convert function with a `with_source` flag, and the option-length loss via a `compute_loss` override instead of patching `model.forward`. | - |

**D. Entry point and configuration (`train.py`, `*_arguments.py`)**

| ID | Where | Why | Replacement | Class |
|---|---|---|---|---|
| E1 | `train.py:107, 113` `"phi-2" in model_name_or_path`, `"Llama" in ...`; same globs in `math_eval/eval_pretrained.sh:11`, `eval_finetuned.sh:29` | Per-model recipe. | `configs/model_profiles/{phi,llama,default}.json` keyed by `config.model_type` (LoRA target modules, precision recipe, `layers_attr`, `final_norm_attr`, `zo_groups`, dropout fields). The current mapping is preserved: phi -> targets `[q,k,v,fc1,fc2]` + fp16 AMP over fp32 weights; llama -> `[q,k,v,o]` + fp16 AMP/fp32; everything else -> `[q,k,v,o]` + bf16. peft's own default for phi is `[q_proj, v_proj, fc1, fc2]` (`peft/utils/constants.py:98`, no `k_proj`), so the explicit list cannot be dropped. Shell scripts read the dtype from the profile via `python -m colm.models.profile <path> --field eval_dtype`. | EXACT for the three current cases (test table) |
| E2 | `train.py:112-125` mutates `training_args.fp16/bf16` and the private `mixed_precision` **after** `TrainingArguments.__post_init__` (`training_args.py:1582-1586` computes it there) | Precision depends on the model. | Resolve the model profile before building `TrainingArguments`: parse `ModelArguments` first, merge profile defaults into the raw config dict (explicit user values win), then `parser.parse_dict`. No post-hoc mutation. | EXACT |
| E3 | `choices` in `metadata` (`training_arguments.py:57-92`, `model_arguments.py`) | Documentation. | **Not enforced for JSON configs** (measured: `data_selection_unit="typo"` is accepted through `parse_dict`). Use `Literal[...]` types (`transformers/hf_argparser.py:192` turns them into argparse choices) plus `__post_init__` validation that also runs for JSON. | B10 (typos now fail fast) |
| E4 | `train.py:98-102` accepts either one `.json` argument or CLI flags; `README`/`run.sh` document `run.sh <cfg.json> ... --extra_flag value` | Convenience. | **Broken today**: measured `ValueError: Some specified arguments are not used by the HfArgumentParser: ['configs/...json']` (`hf_argparser.py:354`). New `load.py`: read the JSON, reject unknown keys, feed it as parser defaults, then parse the remaining CLI flags on top. | B10 (bug fix) |
| E5 | `train.py:208-212` `embed_tokens/lm_head .weight.data = ....float()` (peft issue 341 workaround) | fp16 GradScaler cannot unscale fp16 trainable parameters. | For phi-2 the base weights are already fp32 (`torch_dtype none` -> fp32, `train.py:45-52`), so these two lines are no-ops there. Replace by `get_peft_model(..., autocast_adapter_dtype=True)` (default, `peft/mapping_func.py:110`; it upcasts fp16/bf16 adapter weights to fp32, `tuners_utils.py:2705-2735`) and a single "trainable parameters are fp32" loop that covers the `modules_to_save` copies (which `cast_adapter_dtype` does not touch, it only visits `BaseTunerLayer`s). Harness compares the full `(name, dtype, shape, requires_grad)` table. | EXACT for phi-2 recipe; bf16 recipe: embedding/head base weights no longer upcast (B11) |
| E6 | `train.py:172-181` `enable_dropout=false` asserts `PhiConfig`, sets only `resid_pdrop` | O9/D3 experiments. | Profile key `dropout_fields` (phi: `resid_pdrop`; `embd_pdrop` and `attention_dropout` are already 0.0 in phi-2's config) applied to the config before `from_pretrained`; `lora_dropout=0` stays. | EXACT |
| E7 | `train.py:182-186, 352-356` FSDP-config-driven checkpointing and `pytorch_model_fsdp.bin` clean-up | Upstream FSDP recipe. | `TrainingArguments.gradient_checkpointing` / `gradient_checkpointing_kwargs` are native. Keep the file clean-up as a callback only if FSDP is still used (Q6 candidate). | EXACT |
| E8 | `train.py` `main()` is 260 lines; `tests/test_trainers.py:_args` and `tests/dist_worker.py` duplicate its post-processing (parse `keep_sources`, append `.lora_B`) | No library entry. | `colm/train/main.py` composes `parse_run_config`, `build_tokenizer`, `build_model`, `build_data`, `build_trainer`; tests call the same builders. `colm/train/train.py` stays as a 3-line shim for `python -m`. | EXACT |
| E9 | `training_arguments.py` carries about 30 selection fields inside HF `TrainingArguments`; trainers read `self.args.<anything>` | Convenience. | `SelectionArguments`, `ZOArguments` dataclasses passed to the trainer constructor. `HfArgumentParser` accepts several dataclasses, and field names stay unique, so **flat JSON keys keep working**. | EXACT |
| E10 | `train.py:223` `print(model)`; unused fields `train_dataset_names`, `analysis_mode/analysis_dataset` (only feeds the SuperGLUE branch), `checkpoint_path` naming | - | Log at debug, drop unused fields. | - |
| E11 | `data_selection_method` in {`submodlib`, `weightedsubmodlib`, `none`}; `efficient_mezo` + `data_selection_unit=mezo` select the trainer (`train.py:320-325`); `"weighted" in method` string test (`trainers.py:503`); `"grad" in unit` substring test (`:533, 477`) | Historical string encoding. | `selection.method: coreset \| none`, `selection.weighting: uniform \| cluster_size`, `selection.unit: {rep, mezo, mezo_efficient, masked_grad, completion_length, length_loss_weighted}`; trainer chosen by a table lookup; `MomentSource` is a property of the extractor instead of `"grad" in name`. | EXACT (old keys accepted as aliases if Q4 says keep) |

**E. Facility location / utils**

| ID | Where | Replacement | Class |
|---|---|---|---|
| F1 | `facility_location.py:6, 38` torchmetrics only for `pairwise_cosine_similarity` | `F.normalize(X) @ F.normalize(X).T` (same ops); drop the dependency. | FLOAT-identical in practice; harness on cosine goldens |
| F2 | `:52` `print`, `:47, 57` unused `elapsed`, `:109` docstring lists `order_mg/_sz` outputs (returns 2), `np.append` on a list initial value in a loop (`:124-131, 160-161`) | Lists + one `np.concatenate`; logging. | EXACT |
| F3 | `:153-158` O(N^2) Python loop for cluster sizes (`np.where(orders == i)`, `np.argmax(S[i, orders])`) | `S[:, orders].argmax(1)` (first max, identical tie-break) + `np.bincount`. | EXACT (test with tie-heavy integer data) |
| F4 | `train/utils.py:43-83` budget helpers (`increase_array_to_threshold_v2` uses the **global** numpy RNG, used only for `source_wise_selection=balanced`) | Move to `selection/budgets.py`; RNG from `selection_seed` only for `balanced`. | B9 for `balanced` only |

### 1.2 Confirmed dead code (delete in S1, after the goldens exist)

Verified by an AST/text reference count across `colm/ math_eval/ superglue_eval/ tests/ scripts/`:

- `colm/data/get_training_dataset.py`: `SupervisedDataset.get_super_class`, `SupervisedDataset.naive__getitem__`, `DataCollatorForSupervisedDataset*.naive__call__`, the `weights` field of dataset items and collator output, the four unreachable branches of `load_raw_dataset` (`random`, `balanced_longest_selection`, `longest_sourcewise_selection`, `longest_selection`).
- `colm/data/utils.py`: `ICLCollator`, `SIGUSR1Callback`, `count_time`, `write_predictions_to_file`, `write_metrics_to_file`, `EnhancedJSONEncoder` (only used by the two writers), duplicate `Prediction` (the live one is in `superglue_eval/eval_superglue.py:68`), duplicate `temp_seed`.
- `colm/train/training_arguments.py`: `train_dataset_names` (0 uses), `analysis_mode`/`analysis_dataset`/`modify_forward`/`non_diff`/`only_train_option`/`max_new_tokens` if Q1 = delete the SuperGLUE training path.
- `colm/train/utils.py`: `collate_fn` after D8 (`convert_to_ordered_range` and the budget helpers move to `selection/`, not deleted).
- `colm/train/trainers.py`: `per_source=False` parameter of `select_masking` (never passed), `_check_args` asserts (replaced by validation), `_micro_step` bookkeeping outside `IndexRecorder`.
- `colm/train/custom_phi.py`: whole file after S10.
- Conditional on Q2: `encode_data`, `get_encode_function`, `encode_with_prompt_completion_format`, `encode_with_messages_format(_with_llama2_chat)`, `concat_messages`, `DataArguments.max_seq_length/subset_index_files/percentage/sample_data_seed/hf_datasets_cache_dir`, `train.py:302-305` column removal.
- Docs: `AGENTS.md`/README lines describing the deleted paths.

### 1.3 Upstream behaviours kept on purpose, made explicit and named

| Behaviour (AGENTS.md / backlog) | Option (default = legacy) |
|---|---|
| Logged loss divided by `small_batch_ratio` | `log_loss_divisor = "small_batch_ratio"` |
| Same z every step and rank (F1) | `zo.direction = "fixed_seed"` (other value: `per_step`) |
| Per-sample ZO loss over padded width (F2 / D1) | `zo.loss_normalizer = "padded_length"` (other: `valid_tokens`) |
| fp32 selection prefix, no autocast (F3 / D2) | `zo.selection_autocast = false` |
| fp32 weights + fp16 AMP for phi-2 | model profile `precision` (unchanged) |
| v of the Adam feature from untransformed squares (A19) | documented, not an option |
| Invalid features dropped before gather in `SubsetTrainer`, kept in Efficient | `drop_invalid` property of the trainer kind, documented |
| Training rows keep original padding (D8) | `data.recollate_padding = "original"` (other: `tight`) |
| Global torch RNG reset by every estimate (C3) | `legacy_global_rng = true` (replays `manual_seed(seed)` + the z draw so the global stream is left in the legacy state; used by the equivalence tests) |
| Scalar features stored in the AMP dtype (A8) | `selection.scalar_feature_dtype = "amp"` (other: `float32`) |
| `rep` / `length_loss_weighted` extract features with dropout **on** (no `model.eval()` in `trainers.py:652-665, 706-721`) | `selection.feature_mode = "train"` (other: `eval`) |

### 1.4 Behaviour changes for you to decide (not applied silently)

- **B1** Global torch RNG no longer reset every step: training dropout masks (LoRA `0.05`, phi `resid_pdrop 0.1`) differ from old runs; selected indices unaffected. Legacy flag keeps bitwise reproducibility of old trajectories.
- **B2** Perturbation without in-place restore: per-estimate features bitwise equal, but the LoRA-B tensor no longer picks up the ~1e-9 per-step drift. Multi-step trajectories differ at rounding level.
- **B3** `recollate_padding = "tight"`: same maths, fewer padded tokens; dropout masks and float rounding differ. Curve check needed.
- **B4** Scalar-feature dtype: under AMP the length features are built in fp16 and squared in fp16 (`trainers.py:352, 439-441`), which overflows to `inf` for values > 255 (`length * loss / 10` can reach that). `float32` fixes it and can change selections of `length_loss_weighted`.
- **B5** `feature_mode = "eval"` for `rep` / `length_loss_weighted` (dropout off during feature extraction; today features are noisy).
- **B6** Attention mask built from lengths instead of `input_ids.ne(pad_token_id)`: identical for phi-2 (its tokenizer has no pad token, so `<pad>` is added by `add_padding_to_tokenizer`) and any tokenizer with a dedicated pad; differs for `pad == eos` tokenizers (the EOS input position is masked today).
- **B7** Adam moments (`prev_m_t`, `prev_v_t`) are not in checkpoints, so a resumed run restarts them: save them in `on_save` (resume-only fix).
- **B8** Non-Adam optimizer with `masked_grad` raises instead of silently using zero moments.
- **B9** Non-default random paths (`mezo_topk=random/sampling`, `source_wise_selection=balanced`) use a dedicated seeded generator instead of the global RNG: results differ from old runs (default configs unaffected).
- **B10** Fail-fast config: JSON typos raise; `max_seq_length` disappears; `run.sh <cfg> --flag v` works (README already claims it).
- **B11** bf16 recipe: `embed_tokens`/`lm_head` base weights are no longer upcast to fp32 (memory drop); phi-2/Llama fp16 recipe unaffected.

### 1.5 API verification (installed versions)

| Mechanism | Verified at | Note |
|---|---|---|
| `Trainer.get_batch_samples`, `training_step`, `compute_loss`, `log`, `floating_point_ops`, `_prepare_inputs` | `transformers/trainer.py:2207, 1975, 2048, 4029, 4064, 2304` | `training_step` divides by `current_gradient_accumulation_steps` (`:2035-2037`), assigned `len(batch_samples)` at `:1813` |
| Loop sync arithmetic | `trainer.py:1821-1841` | Reason for A1 |
| `TrainerCallback` step events | `transformers/trainer_callback.py:371-392` | timing, index recorder, moment checkpointing |
| `DataCollatorForSeq2Seq`, `DataCollatorWithFlattening` | `transformers/data/data_collator.py:489, 1366` | probe: extra int columns pass through; flattening kept for the packing work |
| `Literal` -> argparse choices; unknown key handling | `transformers/hf_argparser.py:192, 354` | JSON values are **not** validated (measured) |
| Decoder access / head | `modeling_utils.py:1472 (base_model), 1055 (get_output_embeddings), 2251 (get_decoder, raises for Phi)` | use `base_model` |
| Phi loop honours `layers[: num_hidden_layers]` | `models/phi/modeling_phi.py:379`, final norm `:390` | irrelevant with the hook design, relevant for the rejected "truncate layers" alternative |
| Output capture exception safety and per-layer hidden states | `utils/output_capturing.py:285-290` (`finally`), `:271-277` (`output_hidden_states` may be a list of layer indices) | list form is an alternative for the penultimate state only (no mask/rotary kwargs) |
| Pre-hook with kwargs | `torch/nn/modules/module.py:1624` (`with_kwargs`) | probe passes on tiny Phi and Llama |
| `functional_call` | `torch/_functorch/functional_call.py:13` | probe passes under `inference_mode` |
| Generator-based normal | `torch/_torch_docs.py:8416` | CPU stream == reseeded global (probe) |
| peft dtype handling / defaults / layer selection | `peft/mapping_func.py:110`, `peft/tuners/tuners_utils.py:2705-2735`, `peft/utils/constants.py:98`, `peft/tuners/lora/config.py:722` | `layers_to_transform` exists but LoRA is on all layers here, only ZO is last-layer |
| accelerate collectives | `accelerate/utils/operations.py:505 (gather_object), 675 (broadcast_object_list)`, `accelerate/state.py:350 (use_distributed)` | `gather_object` is all-gather; rank-0-only `gather_object(dst=0)` stays a torch.distributed call behind `PoolExchange` |
| `mixed_precision` computed in post-init | `training_args.py:1582-1586` | E2 |

### 1.6 Documentation errors found

- `docs/optimization-backlog.md` F4 says the 10 smallest MathInstruct sources are 36.9% of the examples. Measured from `data/MathInstruct.jsonl`: 70,615 of 262,039 = **26.9%** (the four selected sources add to 73.1%, which matches the quoted percentages). O1's "~37% of the selection forward" should read ~27%.

---------------------------------------------------------------------------------------------------

## 2. Target architecture

### 2.1 Layout

```
colm/
  config/
    args.py          ModelArguments, DataArguments, SelectionArguments, ZOArguments, TrainingArguments (HF subclass, only HF fields + profiling)
    profiles.py      ModelProfile dataclass; loads configs/model_profiles/<model_type>.json
    load.py          parse_run_config(argv): JSON as defaults + CLI overrides, unknown keys rejected, profile merged BEFORE TrainingArguments
  data/
    schema.py        batch keys (`input_ids`, `labels`, `attention_mask`, META="colm_meta")
    mathinstruct.py  build_dataset(): datasets.Dataset, filter, with_transform tokenisation, source registry (names <-> ids)
    collate.py       SelectionCollator(DataCollatorForSeq2Seq); ExamplePool rows; recollate(padding=original|tight)
    prompts.py       prompt templates shared with math_eval
    superglue/       tasks.py, templates.py, encode.py   (evaluation side; training side per Q1)
  models/
    build.py         tokenizer/model/peft construction from ModelProfile (dtype, dropout, embedding resize, autocast_adapter_dtype)
    zo_target.py     ZOTarget: layer index + module group -> named parameters
    split.py         LastLayerSplit (prefix via pre-hook, replay via functional_call, bind-time verify)
  selection/
    types.py         Features, PoolMeta, Selection
    extractors/      base.py (FeatureExtractor, registry) rep.py length.py mezo.py mezo_efficient.py masked_grad.py
    zo.py            Direction (fixed seed, cached), Perturbation (out-of-place), projected_grad
    stages.py        KeepSources, FeatureTransform, AdamStage(+MomentSource), CoordinateMask
    facility_location.py, budgets.py
    selector.py      CoresetSelector: ordered stages, rank-0 logic, explicit state
    plan.py          BudgetPolicy, MicroBatchPlan (PerExamplePlan, RepackPlan), PlaceholderBatch
    exchange.py      PoolExchange: gather / broadcast (single-process no-ops)
  train/
    trainers.py      CustomTrainer, CoresetTrainer base, SubsetTrainer, SubsetTrainerEfficient (each about 60 lines)
    callbacks.py     IndexRecorder, SelectionStateCheckpoint (B7)
    main.py          composes the builders; train.py = shim
  profiling/step_timing.py   moved unchanged (StepTimer, TransferCensus, StepTimingCallback, summariser)
configs/model_profiles/{phi,llama,default}.json
tests/  equivalence/ (goldens + generator), unit tests per module
```

### 2.2 Interfaces (signatures, not implementation)

```python
# selection/types.py
@dataclass
class Features:
    values: Tensor          # [n, d]; scalar units use d = 1
    valid: BoolTensor       # [n]; False rows are dropped by drop_invalid trainers
    kind: Literal["vector", "int_scalar", "float_scalar"]   # reproduces the three legacy dtype paths

@dataclass
class Selection:
    indices: list[int]      # into the gathered pool; order = kept sources first, then facility location order
    weights: list[float]

# selection/extractors/base.py
class FeatureExtractor(Protocol):
    batched: ClassVar[bool]                 # extract() accepts n >= 1 examples per call
    moment_source: ClassVar[MomentKind]     # OWN | OPTIMIZER
    def bind(self, ctx: BindContext) -> None            # model, ZOTarget, args; fail-fast checks here
    def extract(self, batch: dict, *, need: BoolTensor | None = None) -> Features
    def coordinate_prior(self) -> Tensor | None         # for mezo_selection = weight
# `need` is the hook for exact-opts O1: rows with need=False (kept sources) are never read
# by the pipeline; the extractor may skip their ZO forward.

# selection/selector.py
class CoresetSelector:
    def __init__(self, stages: Sequence[Stage], fl: FacilityLocationConfig, weighting: Weighting): ...
    def select(self, feats: Features, meta: PoolMeta, budget: int, step: int) -> Selection
    def state_dict(self) / load_state_dict(self)         # Adam moments (B7)

# selection/plan.py
class MicroBatchPlan:
    batches: list[dict | PlaceholderBatch]; n_trained: int
    @classmethod build(cls, pool: ExamplePool, mine: Selection, policy: BudgetPolicy, collator) -> "MicroBatchPlan"

# selection/exchange.py
class PoolExchange:                                     # wraps PartialState().use_distributed + torch.distributed
    def gather(self, local: LocalPool) -> GatheredPool  # features only on rank 0, examples on all ranks
    def broadcast(self, selection: Selection | None) -> Selection
```

Design rules: extractors know the model, not the selector; stages know tensors, not the trainer; the trainer knows both only through `CoresetPipeline.__call__(pool_batches, step) -> MicroBatchPlan`. A registry dict (`@register("mezo")`) maps `selection.unit` to a class; no entry points, no plugin system.

### 2.3 Data flow of one optimizer step

```
HF loop (stock) --get_batch_samples(epoch_iterator, gas)--> CoresetTrainer.get_batch_samples
  1 sample pool     super().get_batch_samples(): gas micro-batches x bs examples (device tensors + colm_meta)
  2 features        per micro-batch: extractor.extract(batch[, need])  ->  Features            [timer: selection/features]
                    SubsetTrainer: bs=1 per call; Efficient: batched [bs, numel]
                    invalid rows dropped (SubsetTrainer only) with one vectorised check
  3 local pool      Features.cat + ExamplePool rows moved to CPU once                          [selection/examples_to_cpu]
  4 gather          PoolExchange.gather: features -> rank 0, examples -> all ranks            [selection/gather]
  5 select (rank 0) CoresetSelector.select:                                                    [selection/main/<stage>]
                      KeepSources -> FeatureTransform -> Adam(MomentSource) -> CoordinateMask -> FacilityLocation(per source) -> index map + weights
  6 broadcast       Selection to all ranks; each rank takes its slice                          [selection/scatter]
  7 plan            PerExamplePlan (SubsetTrainer: one example per micro-batch, weight, placeholders first)
                    RepackPlan (Efficient: chunks of int(bs*ratio), SelectionCollator)         [selection/recollate]
  8 train           stock loop: training_step per micro-batch; compute_loss pops colm_meta and applies the weight;
                    current_gradient_accumulation_steps = plan.n_trained; log divisor via log_loss_divisor
  9 observers       IndexRecorder(selection, pool meta) if save_indices; timer stages close against the step clock
```

### 2.4 Mapping of every existing unit and trainer

| Old unit / trainer | New class | batched | model forward | scalar dtype path | moments | drop invalid | Notes |
|---|---|---|---|---|---|---|---|
| `rep` (`trainers.py:652-665`) | `RepExtractor` | no | inner decoder forward, `hidden_states[-1]` at last real token; drops the useless `labels=input_ids` (loss was computed and discarded) | vector | own | yes | `feature_mode` option (B5); EXACT feature (loss unused) |
| `mezo`, non-efficient (`:667-687`) | `MezoExtractor` | no | two full forwards via `functional_call(model, shifted)` under eval + `inference_mode` | vector | own | yes | z from `Direction`; token-mean loss (no F2 issue at bs=1) |
| `masked_grad` (`:689-703`) | `GradExtractor` | no | forward + `torch.autograd.grad(loss / new_accumulation_steps, params)` in train mode | vector | **optimizer** (`OptimizerMoments`) | yes | keeps the `/ new_accumulation_steps` scale so it matches the optimizer's units |
| `completion_length` (`:704-705`) | `CompletionLengthExtractor` | no | none (`needs_model = False`) | int | own | 0 dropped | |
| `length_loss_weighted` (`:706-721`) | `LengthLossExtractor` | no | forward under `no_grad`, NaN loss -> 0 | float (AMP legacy, B4) | own | yes | the variable named `completion_lengths` is really the padded token count; renamed, value unchanged |
| efficient MeZO (`:831-866`, `custom_phi.py`) | `MezoEfficientExtractor` | yes | `LastLayerSplit.prefix` once, `suffix` twice with `Perturbation` overrides; per-sample loss per `zo_loss_normalizer` | vector (fp32) | own | **no** | features `[B, numel]`, `mezo_transform` must be `none` (validated) |
| `CustomTrainer` | `CustomTrainer` | - | - | - | - | - | only `save_indices` and `assert_finite_grad_norm` (the latter as a `TrainerCallback` on `on_pre_optimizer_step`) |
| `SubsetTrainer` | `SubsetTrainer(CoresetTrainer)` | | | | | | policy: `PerExamplePlan`, `BudgetPolicy(k = int(gas*bs*ratio))`, `drop_invalid=True`, weights by `selection.weighting` |
| `SubsetTrainerEfficient` | `SubsetTrainerEfficient(CoresetTrainer)` | | | | | | policy: `RepackPlan`, `BudgetPolicy(k = n_batches * int(bs*ratio))`, `drop_invalid=False`, `weighting=uniform` (validated) |

Selection stages map one-to-one to the old `_select_on_main`: keep-sources scan (`:410-433`), `_transform_reps` (`:509-528`), `_adam_update` (`:530-567`), `select_masking`/`_rank_coordinates` (`:578-633`, `mezo_selection` split into two explicit knobs: `feature_weighting` (`weight_grad` multiplies the feature by the parameter, an extractor concern) and `mask_importance` (`weight` ranks coordinates by |parameter|, a mask concern)), `select_data` (`:635-645`), index/weight mapping and the `max_samples <= 0` branch (`:470-507`).

### 2.5 Distributed and RNG notes

- `PoolExchange` keeps today's semantics: barrier-free gather of features to rank 0 and an all-gather of examples (any rank may train another rank's example), broadcast of `(indices, weights)`. Only the implementation is swapped: `accelerate.utils.gather_object`/`broadcast_object_list` where they fit, `torch.distributed.gather_object(dst=0)` for rank-0-only features, everything gated by `PartialState().use_distributed`. Exact-opts O3 (scalars instead of `[32, 327680]`) plugs into `Features` (extractor returns `g_i` + a regenerable direction) without touching the selector.
- RNG streams after the refactor: z from a private generator; dropout from the global torch RNG (B1); numpy global stream untouched by the selection except through the explicit `selection_seed` generators (B9).

---------------------------------------------------------------------------------------------------

## 3. Ordered implementation steps and how each is verified

**Verification vocabulary.** *Golden* = small files produced once by the frozen reference (`pre-refactor` tag, run in its own worktree with the same venv), compared bitwise on CPU fp32 by `torch.equal`/`np.array_equal`. *Live differential* = the same test run against the frozen worktree on the spot when a golden would be awkward. GPU checks need a GPU you have reserved for this period (I will ask you which; nothing runs on an unreserved card). CPU model = the existing tiny random Phi + char tokenizer (`tests/conftest.py`), plus tiny Llama and Qwen2 configs added for the architecture tests.

**S0 (prerequisite, size M) Freeze and capture goldens.**
- Wait for `task/exact-opts` to merge; create tag `pre-refactor` on that head and a worktree `task/refactor` from it. Reconcile A13/A3/O1/O5/O6 with this design (the other branch's semantics are the reference, its code shape is what we replace).
- `tests/equivalence/generate.py` (run only in the frozen tree) writes `tests/equivalence/golden/*.npz|json`, each a few KB:
  1. collated batches of the mixture fixture (fixed sampler seed): `input_ids, labels, attention_mask, sources, indices, completion_lengths`;
  2. config resolution table: for `microsoft/phi-2`, a Llama-2 path, a zephyr path: resolved `lora_target_modules`, `fp16/bf16`, dtype, ZO parameter names, `output_dir`;
  3. model table `(name, dtype, shape, requires_grad)` after build, with and without embedding resize;
  4. facility location: `X`, strategy, metric, `per_class_start`, orders + weights (random and tie-heavy integer data);
  5. `_select_on_main` cases: recorded `(all_reps, sources, budget, keep_sources, step, prev moments)` and outputs `(indices, weights, m, v)` over a 6-step sequence, grid over `mezo_transform x mezo_topk(largest, smallest, largest_smallest) x mezo_selection x mezo_optim x facility_similarity x source_wise_selection`;
  6. features per unit on a fixed batch (`rep, mezo, masked_grad, completion_length, length_loss_weighted`, efficient `[B, numel]`);
  7. 3-step trajectories for `CustomTrainer`, each `SubsetTrainer` unit, `SubsetTrainerEfficient`: per-step loss, selected indices/weights, LoRA weight checksums, global torch RNG state after step 1 (`torch.get_rng_state()` hash);
  8. 2-rank gloo trajectory (extends `tests/test_distributed.py`).
- GPU golden (needs your card): frozen code, `configs/timing_phi2_efficient.json` shape with `save_indices`, 20 steps: per-step `selected` indices, loss, step time; this is the GPU reference for S9-S11.
- Verification: generator run twice gives byte-identical files (determinism check); existing `pytest -q` green at the tag.

**S1 (S) Delete confirmed dead code (section 1.2, unconditional part).** Verification: `test_imports.py` (imports every module), whole suite, goldens untouched; a grep test lists none of the deleted symbols.

**S2 (M) Config layer: dataclass split, `Literal` validation, JSON+CLI loader, model profiles, `keep_sources` names, `zo_seed`.**
- Verification: golden 2 (config table) bitwise; new tests: JSON typo raises, `run.sh cfg --max_steps 3` parses (regression test for E4), profile table for phi/llama/other reproduces train.py:107-121 decisions, `keep_sources` names equal the 10 smallest sources on the real jsonl (skipped if `data/` absent), `zo_seed` equals the legacy draw, old `output_dir` name string identical.

**S3 (M) `models/build.py`: peft/dtype/dropout/embedding-resize via profile.**
- Verification: golden 3 model table identical (tiny Phi CPU, resize case, `enable_dropout=false`); phi-2 fp32/fp16 recipe table on CPU meta-level (dtype list only, no forward); `print(model)` gone.

**S4 (M) Data: `datasets` + `with_transform` tokenisation, `SelectionCollator`, `ExamplePool`, `colm_meta`.**
- Verification: golden 1 bitwise on batches (`attention_mask` compared after `.bool()`), dataset length after the empty-output filter equals the old one (262,039 rows on the real file; fixture count on CPU), sampler order identical for a fixed seed, `padding_side` asserted `right` for a tokenizer whose default is left, re-collation with `padding="original"` equals `collate_fn` output bitwise on random selections. Trainers still use the legacy path via a thin adapter, so all trainer tests stay green.

**S5 (S) Facility location and budgets.** Verification: golden 4 bitwise, including tie-heavy inputs (F3 tie-break), cosine golden after removing torchmetrics (`torch.equal`, fallback `assert_close(rtol=1e-6)` and identical orders), NaN handling test kept.

**S6 (M) `selection/stages.py` + `CoresetSelector`, called from the legacy trainer through an adapter.**
- Verification: golden 5 grid bitwise including the multi-step Adam state carry (`OwnMoments`), the `max_samples <= 0` branch (more kept than budget), `keep_sources` with an empty candidate set, `masked_grad` `OptimizerMoments` with a real `AdamW` state and the cold-start zeros, `clip_last` for a 32-layer config.

**S7 (S) `Features`, vectorised validity, `PoolExchange`.**
- Verification: existing 2-rank gloo test plus golden 8 (indices and LoRA checksums equal rank by rank); validity predicate test with `nan`, `inf`, all-zero, empty rows against the old branchy predicate; single-process no-op path.

**S8 (M) Extractors without ZO: `rep`, `completion_length`, `length_loss_weighted`, `masked_grad`.**
- Verification: golden 6 features bitwise per unit in both modes where the legacy behaviour is train-mode dropout (seeded); scalar dtype path test (`int_scalar` -> long, `float_scalar` -> AMP dtype under a simulated fp16 flag); `masked_grad` scaling `/ new_accumulation_steps`; trainer trajectories golden 7 for these units.

**S9 (M) ZO core: `Direction`, `Perturbation`, `MezoExtractor`, `legacy_global_rng`.**
- Verification: (a) z from `torch.Generator` equals reseeded-global z on CPU (`torch.equal`) **and on CUDA** (GPU check, one `normal` call of 327,680 floats and of a two-tensor target); (b) feature bitwise vs golden 6 for `mezo`, with `plus = theta + (1*z)*eps`, `minus = plus + (-2*z)*eps`; (c) parameters bitwise unchanged after `extract` (stronger than the old `atol=1e-6` test in `tests/test_trainers.py:test_mezo_perturbation_is_restored`); (d) with `legacy_global_rng=true` the global torch RNG hash after a step equals golden 7 and 3-step trajectories match bitwise; with it off the loss curve and indices are compared and the difference is reported, not asserted (B1/B2).

**S10 (L) `LastLayerSplit` + `MezoEfficientExtractor`; delete `custom_phi.py`.**
- Verification: (a) architecture test on tiny Phi, Llama, Qwen2: replay == full decoder output bitwise, `verify()` raises when a wrong `final_norm_attr` is configured; (b) per-sample loss equals golden 6 and equals the old `forward_final_layer(per_sample_loss=True)`; (c) padded vs packed inputs (`use_cache=False`, F5 test carried over from exact-opts) give the same per-segment loss; (d) `lora` off (full weights) target case; (e) **GPU, phi-2**: features for one real batch vs the frozen tree (bitwise expected because the decoder is the same fp32 module; tolerance 1e-6 relative if not), then 20-step selected indices equal the GPU golden, then step time within +-2% of the frozen tree (one run, >=100 steps for the final gate, per the timing rule; the split adds one dict lookup and a hook, and removes four z generations per micro-batch).

**S11 (L) Trainer thinning.**
- `CoresetTrainer` (composition of pipeline objects), `PlaceholderBatch`, stock `training_step` with `current_gradient_accumulation_steps`, weight in `compute_loss`, `colm_meta` pop, callbacks for timing/index recording/finite-grad-norm, drop the four private-HF overrides, `selection_mode` context.
- Verification: golden 7 and 8 bitwise on CPU (all three trainers, every unit, 2 ranks); loss scaling test (`gas=4, ratio=0.5`: grads equal the old `_training_step` bitwise); DDP sync test asserting `sync_gradients` is true exactly on the last real micro-batch; `step_timing` closure test (`test_step_timing.py`) still closes within its tolerance, with node names unchanged except `backward` derivation; **GPU**: 20-step trace equals the GPU golden, then the 130-step timing config and the summariser (`python -m colm.train.step_timing`) show no regression beyond +-2%.

**S12 (M) `main.py` builders; SuperGLUE training path per Q1; W&B env; remove arg mutation.**
- Verification: `test_train_main.py` (both trainers end to end), config golden, a test that `TrainingArguments` fields are not modified after construction, `wandb` not imported (existing test), README command lines executed once in a dry-run config.

**S13 (M) Behaviour-change switches B1-B5, B7, B9 implemented as options (default legacy), each with its dedicated test; then one commit per decision you make.**
- Verification per switch: legacy value reproduces golden bitwise; new value passes its own property test (e.g. B2: parameter never written; B3: same per-example loss as `original` up to `assert_close`, fewer padded tokens; B4: no `inf` in squares for lengths of 500; B7: resume reproduces the uninterrupted selection sequence).
- Curve checks for B1/B3 use a learning-curve comparison (not timing, so several seeds apply): report mean +- spread of eval loss at the same steps.

**S14 (S) Docs.** AGENTS.md code map, README (config table, new layout), TODO.md, `docs/optimization-backlog.md` (fix F4 26.9%/O1 27%, item statuses), eval shell scripts read dtype from the profile.

Ordering logic: S1-S5 remove risk without touching the training loop; S6-S8 are pure selection logic behind an adapter; S9-S10 touch the model path (the risky part, GPU-verified); S11 rewires the trainers only after every piece has its own golden.

---------------------------------------------------------------------------------------------------

## 4. Risks and open questions

**Risks**
1. **Moving target.** `task/exact-opts` rewrites the files this design replaces. Golden must be captured from its merged head, and its O1/O5/O6/O7 features must be re-expressed through `need`, `suffix_hidden`, packed collation and the model's own attention backend. If it lands late, S1-S5 can proceed on the untouched modules (`data/`, `config/`, `facility_location`, `budgets`) but S6+ wait.
2. **Hook-based split across architectures.** Verified bitwise on tiny Phi and Llama here; real Qwen/Gemma-style models with embedding scaling or per-layer attention types are covered by the bind-time `verify()` but not run. Gradient-checkpointing and PEFT wrappers are irrelevant under `inference_mode`, but any model whose last layer is not `base_model.layers[-1]` needs a profile entry.
3. **CUDA generator parity** (z from `torch.Generator("cuda")` vs reseeded default generator) is verified only on CPU so far; S9 checks it on GPU first and falls back to drawing z on CPU and copying if the streams differ (changes nothing semantically, costs one H2D per run because z is cached).
4. **`current_gradient_accumulation_steps`** is a private HF attribute; a unit test pins the loss scaling so a transformers upgrade fails loudly instead of silently changing the gradient scale.

**Questions for you (need a decision, everything else is decided above)**
- **Q1** SuperGLUE *training* path (`train.py:226-286`, monkey-patched `forward`, `NondiffCollator`, duplicated converters, untested): delete it and keep only `superglue_eval` (my recommendation), or keep and port it to a `compute_loss` override?
- **Q2** MathInstruct-only data: may I delete the LESS/messages path (`encode_data` ...), `load_raw_dataset` sub-sampling branches, `subset_index_files` (pickled index files), `percentage`? If you sample subsets, tell me which mode is still needed and it becomes an explicit `subset_indices` file.
- **Q3** Behaviour changes B1-B5, B7, B9: default them to the new behaviour after the equivalence run, or keep legacy defaults? My recommendation: B1, B2, B7 yes (bug-like, no effect on selection), B4 yes (fp16 overflow), B3 and B5 only after a curve comparison because they change training numerics.
- **Q4** Config compatibility: keep every current JSON key and value (aliases for `efficient_mezo`, `data_selection_method`, `mezo_selection`, underscore-separated `keep_sources`) so existing configs/logs/README commands work unchanged, or rename to the cleaner names in section 1.1 E11/D9 in one breaking commit? Recommendation: aliases for one release, then remove.
- **Q5** Golden policy: tag + generated golden files in-tree (recommended, keeps the repo free of legacy code) versus keeping a frozen copy of the old trainer under `tests/` as a live oracle until the refactor ends. Also: which GPU may I use for the S0/S10/S11 phi-2 checks?

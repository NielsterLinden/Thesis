# Physics Bias Experiments

## Overview

9 experiment configs testing physics-informed attention biases for 4t vs background classification.
~93 full jobs (+ up to 28 follow-up jobs for single-seed experiments that warrant re-runs).

**Key config references:**
- Seed key: `classifier.trainer.seed` (default 42, sweep: `42,123,314`)
- SM mode: `classifier.model.bias_config.sm_interaction.mode` — `binary | fixed_coupling | running_coupling`
- Lorentz features: `classifier.model.bias_config.lorentz_scalar.features` — e.g. `[m2, deltaR]`
- Type-pair init: `classifier.model.bias_config.typepair_kinematic.init_from_physics` — `none | binary | fixed_coupling`
- Type-pair freeze: `classifier.model.bias_config.typepair_kinematic.freeze_table` — `true | false`
- Pre-encoder: `classifier.model.nodewise_mass.enabled`, `classifier.model.mia_blocks.enabled`
- Global MET: `classifier.globals.include_met`, `classifier.model.bias_config.global_conditioned.mode`

---

## Experiment Table

| # | File | Description | Jobs | Seeds |
|---|------|-------------|------|-------|
| 1 | `baseline.yaml` | No bias, 3 sizes × 3 seeds | 9 | 3 |
| 2 | `part.yaml` | ParT lorentz_scalar (m2+deltaR), 3 sizes × 3 seeds | 9 | 3 |
| 3 | `single_module_sweep.yaml` | Each novel module alone: typepair, sm (YAML) + nodewise_mass, mia_blocks (CLI) | 12 | 3 |
| 4 | `lorentz_features.yaml` | 7 feature combos, 1 seed | 7 | 1* |
| 5 | `sm_progression.yaml` | SM binary → fixed → running, 3 seeds | 9 | 3 |
| 6 | `module_combinations.yaml` | 4 attention-bias combos (YAML) + 3 pre-encoder combos (CLI) | 21 | 3 |
| 7 | `data_scaling.yaml` | sm + combo at 3 sizes, 1 seed | 6 | 1* |
| 8 | `interpretability.yaml` | typepair init×freeze (YAML) + sm comparison (CLI) | 14 | 2 |
| 9 | `met_treatment.yaml` | 4-way MET isolation sweep | 4 | 1* |
| **Total** | | | **91** | |

\* Single-seed: results are exploratory. See per-experiment caveats below.

**Follow-up triggers:**
- Exp 4: re-run top-2 and bottom-2 feature configs with 3 seeds (+8 jobs)
- Exp 7: re-run with 3 seeds if scaling behavior is strong (+12 jobs)
- Exp 9: re-run improving configs (3 or 4) with 3 seeds (+4–8 jobs)

---

## Condor Submit — Test Runs (1 epoch, 500 samples, WandB off)

Run all test commands before the weekend to catch shape/config errors.
Each test job takes ~1 min. Total: 91 test jobs.

### Exp 1 — baseline

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/baseline experiment.name=4tbg_physics_baseline_test classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Exp 2 — part (ParT reference)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/part experiment.name=4tbg_physics_part_test classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Exp 3 — single_module_sweep

**3a — typepair + sm (6 jobs):**

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/single_module_sweep experiment.name=4tbg_physics_single_module_test classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

**3b — nodewise_mass alone (3 jobs):**

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/single_module_sweep experiment.name=4tbg_physics_single_module_test classifier.model.attention_biases=none classifier.model.nodewise_mass.enabled=true classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

**3c — mia_blocks alone (3 jobs):**

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/single_module_sweep experiment.name=4tbg_physics_single_module_test classifier.model.attention_biases=none classifier.model.mia_blocks.enabled=true classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Exp 4 — lorentz_features

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/lorentz_features experiment.name=4tbg_physics_lorentz_features_test classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Exp 5 — sm_progression

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/sm_progression experiment.name=4tbg_physics_sm_progression_test classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Exp 6 — module_combinations

**6a — pure attention_biases rows (12 jobs):**

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/module_combinations experiment.name=4tbg_physics_combinations_test classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

**6b — lorentz + nodewise_mass (3 jobs):**

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/module_combinations experiment.name=4tbg_physics_combinations_test classifier.model.attention_biases=lorentz_scalar classifier.model.nodewise_mass.enabled=true classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

**6c — lorentz + sm_interaction + mia_blocks (3 jobs):**

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/module_combinations experiment.name=4tbg_physics_combinations_test classifier.model.attention_biases=lorentz_scalar+sm_interaction classifier.model.mia_blocks.enabled=true classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

**6d — lorentz + typepair + sm + nodewise_mass (3 jobs):**

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/module_combinations experiment.name=4tbg_physics_combinations_test classifier.model.attention_biases=lorentz_scalar+typepair_kinematic+sm_interaction classifier.model.nodewise_mass.enabled=true classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Exp 7 — data_scaling

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/data_scaling experiment.name=4tbg_physics_data_scaling_test classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Exp 8 — interpretability

**8a — typepair init × freeze variants (12 jobs):**

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/interpretability experiment.name=4tbg_physics_interpretability_test classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

**8b — sm_interaction comparison (2 jobs):**

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/interpretability experiment.name=4tbg_physics_interpretability_test classifier.model.attention_biases=sm_interaction classifier.model.bias_config.sm_interaction.mode=running_coupling classifier.trainer.seed=42,123 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Exp 9 — met_treatment

**9a — pairwise control vs MET-tokens (2 jobs):**

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/met_treatment experiment.name=4tbg_physics_met_treatment_test classifier.globals.include_met=false,true classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

**9b — MET attention modulation modes (2 jobs):**

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/met_treatment experiment.name=4tbg_physics_met_treatment_test classifier.globals.include_met=true classifier.model.attention_biases=lorentz_scalar+global_conditioned classifier.model.bias_config.global_conditioned.mode=global_scale,met_direction classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

---

## Condor Submit — Full Weekend Runs

Remove `_test` suffix, remove `data.limit_samples`, and change `epochs` back to 50 (already set in each YAML). Use these commands for the full runs.

### Exp 1 — baseline (9 jobs)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/baseline --multirun'
```

### Exp 2 — part (9 jobs)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/part --multirun'
```

### Exp 3a — single_module (typepair+sm, 6 jobs)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/single_module_sweep --multirun'
```

### Exp 3b — single_module (nodewise_mass, 3 jobs)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/single_module_sweep classifier.model.attention_biases=none classifier.model.nodewise_mass.enabled=true classifier.trainer.seed=42,123,314 --multirun'
```

### Exp 3c — single_module (mia_blocks, 3 jobs)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/single_module_sweep classifier.model.attention_biases=none classifier.model.mia_blocks.enabled=true classifier.trainer.seed=42,123,314 --multirun'
```

### Exp 4 — lorentz_features (7 jobs)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/lorentz_features --multirun'
```

### Exp 5 — sm_progression (9 jobs)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/sm_progression --multirun'
```

### Exp 6a — module_combinations (pure attention_biases, 12 jobs)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/module_combinations --multirun'
```

### Exp 6b — module_combinations (lorentz+nodewise_mass, 3 jobs)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/module_combinations classifier.model.attention_biases=lorentz_scalar classifier.model.nodewise_mass.enabled=true classifier.trainer.seed=42,123,314 --multirun'
```

### Exp 6c — module_combinations (lorentz+sm+mia_blocks, 3 jobs)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/module_combinations classifier.model.attention_biases=lorentz_scalar+sm_interaction classifier.model.mia_blocks.enabled=true classifier.trainer.seed=42,123,314 --multirun'
```

### Exp 6d — module_combinations (lorentz+typepair+sm+nodewise_mass, 3 jobs)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/module_combinations classifier.model.attention_biases=lorentz_scalar+typepair_kinematic+sm_interaction classifier.model.nodewise_mass.enabled=true classifier.trainer.seed=42,123,314 --multirun'
```

### Exp 7 — data_scaling (6 jobs)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/data_scaling --multirun'
```

### Exp 8a — interpretability typepair variants (12 jobs)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/interpretability --multirun'
```

### Exp 8b — interpretability sm_interaction comparison (2 jobs)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/interpretability classifier.model.attention_biases=sm_interaction classifier.model.bias_config.sm_interaction.mode=running_coupling classifier.trainer.seed=42,123 --multirun'
```

### Exp 9a — met_treatment control vs MET-tokens (2 jobs)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/met_treatment classifier.globals.include_met=false,true --multirun'
```

### Exp 9b — met_treatment MET modulation modes (2 jobs)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/met_treatment classifier.globals.include_met=true classifier.model.attention_biases=lorentz_scalar+global_conditioned classifier.model.bias_config.global_conditioned.mode=global_scale,met_direction --multirun'
```

---

## Follow-up Re-runs (trigger-based)

### Exp 4 — lorentz_features top-2 and bottom-2 with 3 seeds

After initial results, identify top-2 and bottom-2 performing feature configs. Re-run each with seeds 42, 123, 314:

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/lorentz_features classifier.model.bias_config.lorentz_scalar.features=[m2,deltaR],[log_kt,z,deltaR,log_m2] classifier.trainer.seed=42,123,314 --multirun'
```

(Replace feature lists with actual top-2/bottom-2 after inspection.)

### Exp 7 — data_scaling with 3 seeds (if scaling behavior is strong)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/data_scaling classifier.trainer.seed=42,123,314 --multirun'
```

### Exp 9 — met_treatment improving configs with 3 seeds

If config 3 or 4 shows improvement over config 2:

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=bias_experiments/met_treatment classifier.globals.include_met=true classifier.model.attention_biases=lorentz_scalar+global_conditioned classifier.model.bias_config.global_conditioned.mode=met_direction classifier.trainer.seed=42,123,314 --multirun'
```

(Replace `mode` with whichever config improved.)

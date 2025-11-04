<!-- a85d9b57-85e1-4c84-a8b9-06498377e3b4 f100ad12-c528-4e79-81e6-2987164a9bca -->
# Codebase Structure Refactor & Documentation Overhaul

## Phase 1: Structural Analysis & Refactoring

### Current Issues Identified

1. **phase1/general split is unclear**: `general/` only contains test MLP code; split is by project phase rather than functionality
2. **Facts system is scattered**: Code split between `utils/facts_builder.py`, `plots/io_utils.py`, and `reports/utils/read_facts.py`
3. **Naming ambiguities**:

   - `plots/` handles training-time monitoring but name doesn't convey this
   - `reports/experiments/` confusing since "experiments" usually means training sweeps

4. **Not optimized for extensibility**: Adding new models/loops requires understanding arbitrary phase1/general distinction
5. **Namespace risk**: Current plan would flatten to top-level causing import collisions

### Proposed New Structure

**Keep `thesis_ml/` as root package** to avoid import collisions and maintain clean namespace:

```
src/thesis_ml/
├── cli/                (NEW - clean entry points)
│   ├── train/          (was train/)
│   │   ├── __main__.py
│   │   └── __init__.py (DISPATCH)
│   └── reports/        (was reports/__main__.py)
│       ├── __main__.py
│       └── __init__.py
├── training_loops/     (NEW - consolidate all loops)
│   ├── autoencoder.py
│   ├── gan_autoencoder.py
│   ├── diffusion_autoencoder.py
│   └── simple_mlp.py
├── architectures/      (NEW - replaces phase1/general split)
│   ├── autoencoder/    (was phase1/autoenc)
│   │   ├── assembly.py
│   │   ├── encoders/
│   │   ├── decoders/
│   │   ├── bottlenecks/
│   │   └── losses/
│   └── simple/         (was general/models)
│       └── mlp.py
├── data/               (unchanged)
├── facts/              (NEW - complete facts system)
│   ├── builders.py     (from utils/facts_builder.py)
│   ├── writers.py      (from plots/io_utils.py)
│   └── readers.py      (from reports/utils/read_facts.py)
├── monitoring/         (RENAMED from plots/)
│   ├── orchestrator.py
│   ├── io_utils.py     (figure utilities only)
│   └── families/
├── reports/            (post-training analysis)
│   ├── analyses/       (RENAMED from experiments/)
│   ├── inference/
│   ├── plots/
│   └── utils/          (minus read_facts.py)
└── utils/              (general utilities only)
```

**Console Scripts** (add to `pyproject.toml`):

```toml
[project.scripts]
thesis-train = "thesis_ml.cli.train:main"
thesis-report = "thesis_ml.cli.reports:main"
```

Then users can run: `thesis-train` or `thesis-report` directly.

### Refactoring Steps

**Step 0: Create import rewrite script**

- Create `scripts/refactor_imports.py` to automate import updates
- This will update all imports throughout codebase in one pass

**Step 1: Create new directory structure**

- Create `cli/train/` and `cli/reports/`
- Create `training_loops/`, `architectures/`, `facts/`, `monitoring/`
- Create `reports/analyses/`
- Add `__init__.py` to all new packages

**Step 2: Consolidate facts system FIRST**

- Create `facts/builders.py` - move `build_event_payload` from `utils/facts_builder.py`
- Create `facts/writers.py` - move `append_jsonl_event`, `append_scalars_csv` from `plots/io_utils.py`
- Create `facts/readers.py` - move content from `reports/utils/read_facts.py`
- Add `facts/__init__.py` with clean exports
- Update imports in training loops (use rewrite script)

**Step 3: Move training loops**

- `phase1/train/ae_loop.py` → `training_loops/autoencoder.py`
- `phase1/train/gan_ae_loop.py` → `training_loops/gan_autoencoder.py`
- `phase1/train/diffusion_ae_loop.py` → `training_loops/diffusion_autoencoder.py`
- `general/train/test_mlp_loop.py` → `training_loops/simple_mlp.py`
- Update imports in each file (use rewrite script)

**Step 4: Move architectures**

- `phase1/autoenc/` → `architectures/autoencoder/`
- `general/models/` → `architectures/simple/`
- Keep `DESIGN.md` in `architectures/autoencoder/`
- Update all imports (use rewrite script)

**Step 5: Rename monitoring**

- `plots/` → `monitoring/`
- Update `io_utils.py` to only contain figure-related utilities (facts writers already moved)
- Update all imports (use rewrite script)

**Step 6: Rename reports/experiments**

- `reports/experiments/` → `reports/analyses/`
- Remove `read_facts.py` from `reports/utils/` (already moved to facts/)
- Update imports (use rewrite script)

**Step 7: Reorganize CLI entry points**

- Move `train/` → `cli/train/`
- Move `reports/__main__.py` → `cli/reports/__main__.py`
- Update `cli/train/__init__.py` DISPATCH to import from `training_loops/`
- Ensure both can still be called via `python -m thesis_ml.cli.train` etc.

**Step 8: Update pyproject.toml**

- Add console scripts for `thesis-train` and `thesis-report`
- Verify package structure is correct

**Step 9: Run import rewrite script**

- Execute `python scripts/refactor_imports.py`
- Manually review changes
- Fix any edge cases the script missed

**Step 10: Delete old directories**

- Remove `phase1/` directory
- Remove `general/` directory
- Remove `utils/facts_builder.py` (moved to facts/)
- Remove `reports/utils/read_facts.py` (moved to facts/)

**Step 11: Validation**

- Run full test suite: `pytest`
- Test training: `thesis-train` or `python -m thesis_ml.cli.train`
- Test reports: `thesis-report` or `python -m thesis_ml.cli.reports`
- Verify all imports resolve correctly
- Run smoke test on real data

**Step 12: Update config paths if needed**

- Check if any config files reference old module paths
- Update Hydra config_path references in decorators if needed

## Phase 2: Documentation Overhaul

### Documents to Create (from scratch)

**1. Main README.md**

- Project overview and purpose
- Quick start (installation + first run)
- Directory structure explanation
- Key concepts: training loops, facts, monitoring, reports
- Links to detailed guides

**2. ARCHITECTURE.md**

- System architecture overview with ASCII diagram
- Component responsibilities
- Data flow: training → facts → monitoring → reports
- Design decisions and rationale
- Extension points for future development

**3. TRAINING_GUIDE.md**

Two-by-two quadrant structure:

**USING EXISTING CODE**

- Running training locally
- Running on HPC (Stoomboot)
- CLI commands and overrides
- Monitoring training progress

**CREATING NEW CODE**

- Adding a new architecture (encoder/decoder/bottleneck)
- Adding a new training loop
- Registering in dispatcher
- Creating corresponding configs

**4. REPORTS_GUIDE.md**

Two-by-two quadrant structure:

**USING EXISTING REPORTS**

- Running reports from sweep directories
- Available report types
- Configuring inference
- Reading outputs

**CREATING NEW REPORTS**

- Report structure and template
- Reading facts with `facts.readers`
- Creating custom plots
- Adding inference capabilities
- Registering new report types

**5. FACTS_SYSTEM.md**

- What are facts and why they matter
- Fact types: events (JSONL) vs scalars (CSV)
- Event schema and payload structure
- Writing facts in training loops
- Reading facts in reports
- Adding new fact types

**6. CONFIGS_GUIDE.md**

- Hydra configuration system explained
- Config composition and defaults
- Environment switching (local vs stoomboot)
- Experiment configs and sweeps
- Override patterns and examples
- Config group organization

**7. HPC_GUIDE.md**

- Stoomboot cluster overview
- Directory structure on HPC
- Job submission (train.sub, report.sub)
- Monitoring jobs (condor_q, logs)
- Data synchronization
- Common issues and solutions

**8. DEVELOPMENT_GUIDE.md**

- Setting up development environment
- Installing in editable mode
- Running tests
- Code organization principles
- Adding dependencies
- Pre-commit hooks

### Documents to Remove/Archive

**Remove these outdated docs:**

- `REFactor_NOTES.md`
- `MULTIRUN_FIX_SUMMARY.md`
- `DEBUG_MULTIRUN_PROMPT.md`
- `REPORTS_REFACTOR_SUMMARY.md`
- `REPORTS_IMPORT_CHECK.md`
- `remove_QUICK_REPORT_COMMANDS.md`

**Consolidate into new guides:**

- `CLI_USAGE.md` → fold into TRAINING_GUIDE.md
- `CONFIG_SUMMARY.md` → fold into CONFIGS_GUIDE.md
- `VALIDATION_COMMANDS.md` → fold into DEVELOPMENT_GUIDE.md
- `REPORT_GUIDE.md` → fold into REPORTS_GUIDE.md

## Phase 3: Final Presentation

**Create CODEBASE_OVERVIEW.md**:

- ASCII architecture diagram showing all components
- Directory tree with annotations
- Component interaction flowchart
- Key files and their purposes
- Example end-to-end workflows:
  - Training a model locally
  - Running a sweep on HPC
  - Generating a comparison report
  - Adding a new architecture
- Quick reference: most common commands
- Troubleshooting guide

**Summary document for presentation**:

- Before/after structure comparison
- What changed and why
- Benefits for future development
- How to navigate the new structure
- Next steps and future enhancements

### To-dos

- [ ] Complete structural analysis and finalize refactoring decisions
- [ ] Create new directory structure (training_loops/, architectures/, facts/, monitoring/, reports/analyses/)
- [ ] Move and update training loops to training_loops/
- [ ] Move phase1/autoenc to architectures/autoencoder/ and general/models to architectures/simple/
- [ ] Create facts/ module and move facts-related code from utils and plots
- [ ] Rename plots/ to monitoring/ and update imports
- [ ] Rename reports/experiments/ to reports/analyses/
- [ ] Update train/__init__.py and all entry points with new imports
- [ ] Systematically update all imports throughout codebase
- [ ] Delete old directories (phase1/, general/) and obsolete files
- [ ] Run tests and validate training/reports still work
- [ ] Write new README.md from scratch
- [ ] Write ARCHITECTURE.md explaining system design
- [ ] Write TRAINING_GUIDE.md with 2x2 quadrants (using/creating × training/configs)
- [ ] Write REPORTS_GUIDE.md with 2x2 quadrants (using/creating × analysis/inference)
- [ ] Write FACTS_SYSTEM.md documenting the facts architecture
- [ ] Write CONFIGS_GUIDE.md explaining Hydra setup and patterns
- [ ] Write HPC_GUIDE.md for stoomboot usage
- [ ] Write DEVELOPMENT_GUIDE.md for contributors
- [ ] Remove/archive obsolete documentation files
- [ ] Create final CODEBASE_OVERVIEW.md with visual diagrams and complete system tour

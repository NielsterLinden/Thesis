# Stoomboot HPC Job Submission Guide

## Session Init (after SSH)

After `ssh stbc-i1.nikhef.nl`, run one command to activate conda, go to project, git pull, and set WandB:

```bash
source /project/atlas/users/nterlind/Thesis-Code/hpc/stoomboot/init_session.sh
```

**One-liner alias**: Add to `~/.bashrc` on the HPC:

```bash
alias thesis='source /project/atlas/users/nterlind/Thesis-Code/hpc/stoomboot/init_session.sh'
```

Then type `thesis` after SSH.

**WandB API key**: Create `hpc/stoomboot/.wandb_env` (gitignored) with `export WANDB_API_KEY="your_key"`, or put the key in `~/.wandb_api_key`. See `.wandb_env.example`.

## Prerequisites

1. SSH access to Stoomboot configured
2. Conda environment created at `/data/atlas/nterlind/venvs/thesis-ml`
3. Dataset uploaded to `/data/atlas/nterlind/datasets/4tops_splitted.h5`
4. Log directories created: `/data/atlas/nterlind/logs/{stdout,stderr}/`

## Quick Start

### Submit a Training Job (GPU)
```bash
cd /project/atlas/nterlind/Thesis-Code
condor_submit hpc/stoomboot/train.sub -append 'arguments = phase1/experiment=compare_globals_heads'
```

### Submit a Report Generation Job (CPU)
```bash
condor_submit hpc/stoomboot/report.sub -append 'arguments = --config-name compare_globals_heads +sweep_dir=/data/atlas/users/nterlind/outputs/experiments/exp_20251031-094244_compare_globals_heads'
```

### Submit with Hydra arguments
```bash
# Training: Add experiment or config overrides
condor_submit hpc/stoomboot/train.sub -append 'arguments = phase1/experiment=compare_globals_heads trainer.epochs=50'

# Reporting: Specify experiment directory
condor_submit hpc/stoomboot/report.sub -append 'arguments = --config-name compare_globals_heads +sweep_dir=/path/to/experiment'
```

### Legacy Jobs (deprecated, use train.sub/report.sub instead)
```bash
# Old CPU job (still works but deprecated)
condor_submit hpc/stoomboot/job_cpu.sub

# Old GPU job (still works but deprecated)
condor_submit hpc/stoomboot/job_gpu.sub
```

## Monitoring Jobs

```bash
# View your jobs
condor_q

# Detailed view
condor_q -better -nobatch

# Why isn't my job running?
condor_q -analyze <job_id>

# Follow job output live
condor_tail -f <job_id>

# View completed job
condor_history <job_id>
```

## Managing Jobs

```bash
# Remove specific job
condor_rm <job_id>

# Remove all your jobs
condor_rm <username>

# Hold a job
condor_hold <job_id>

# Release a held job
condor_release <job_id>
```

## Viewing Logs

```bash
# Stdout
cat /data/atlas/nterlind/logs/stdout/gpu_<ClusterId>.<ProcId>.out

# Stderr
cat /data/atlas/nterlind/logs/stderr/gpu_<ClusterId>.<ProcId>.err

# HTCondor log (job events)
cat /data/atlas/nterlind/logs/gpu_<ClusterId>.log
```

## Advanced: Parameter Sweeps

Create a custom submit file for sweeps:

**`hpc/stoomboot/sweep_example.sub`**
```condor
universe     = vanilla
executable   = hpc/stoomboot/run.sh
initialdir   = /project/atlas/nterlind/Thesis-Code

# Parameterized arguments
arguments    = trainer.epochs=$(epochs) phase1/latent_space=$(latent)

output       = /data/atlas/nterlind/logs/stdout/sweep_$(ClusterId).$(ProcId).out
error        = /data/atlas/nterlind/logs/stderr/sweep_$(ClusterId).$(ProcId).err
log          = /data/atlas/nterlind/logs/sweep_$(ClusterId).log

request_cpus = 4
request_memory = 8GB
request_disk = 20GB

requirements = (OpSysAndVer == "AlmaLinux9")
should_transfer_files = NO
stream_output = True
stream_error = True

# Queue multiple jobs with different parameters
queue epochs,latent from (
    10,vq
    10,none
    50,vq
    50,none
)
```

Submit:
```bash
condor_submit hpc/stoomboot/sweep_example.sub
```

## Verifying GPU Availability

```bash
# Check GPU nodes and their ClassAds
condor_status -constraint 'CUDACapability > 0' -af Name CUDAGlobalMemoryMb CudaDeviceName

# View all CUDA-related ClassAds
condor_status -long | grep -i cuda
```

## First Run Checklist

Before first submission:

```bash
# 1. Create log directories
mkdir -p /data/atlas/nterlind/logs/{stdout,stderr}

# 2. Verify GPU ClassAds (check the actual names used on Stoomboot)
condor_status -af Name CUDAGlobalMemoryMb CudaDeviceName | head

# 3. Dry-run submit file
condor_submit -dump hpc/stoomboot/job_gpu.sub | less

# 4. Test conda environment activation
cd /project/atlas/nterlind/Thesis-Code
eval "$(conda shell.bash hook)"
conda activate /data/atlas/nterlind/venvs/thesis-ml
python -c "import thesis_ml; import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 5. Test Hydra config selection
python -m thesis_ml.train env=stoomboot trainer.epochs=1 logging.save_artifacts=false
```

## Troubleshooting

### Job held (status H)
```bash
condor_q -hold
condor_q -analyze <job_id>
# Fix issue, then:
condor_release <job_id>
```

### Job not matching requirements
```bash
condor_q -better -analyze <job_id>
# Check if GPU requirement is too strict or if nodes are busy
```

### Output not appearing
- Check `stream_output = True` is in submit file
- Verify log directories exist and are writable
- Check job stderr for Python errors

### Import errors
```bash
# Verify package installation
conda activate /data/atlas/nterlind/venvs/thesis-ml
pip show thesis-ml
# Reinstall if needed:
pip install -e /project/atlas/nterlind/Thesis-Code
```

## Resource Guidelines

**CPU Jobs:**
- Light training (< 1M params): 4 CPUs, 8GB RAM
- Medium training: 8 CPUs, 16GB RAM
- Dataset preprocessing: 4-8 CPUs, 16-32GB RAM

**GPU Jobs:**
- Single GPU training: 8 CPUs, 32GB RAM, 1 GPU
- Large models: 16 CPUs, 64GB RAM, 1 GPU
- Ensure GPU memory requirement matches your model

**Disk:**
- Minimal (pre-loaded data): 10-20GB
- With caching: 50-100GB
- Include safety margin for outputs/checkpoints

## Where Outputs Go

With `env=stoomboot` selected:

- **Run outputs:** `/data/atlas/users/nterlind/outputs/runs/run_<timestamp>_<name>/`
- **Multirun sweeps:** `/data/atlas/users/nterlind/outputs/multiruns/exp_<timestamp>_<name>/`
- **Report outputs:** `/data/atlas/users/nterlind/outputs/reports/report_<timestamp>_<name>/`
- **Job logs:** `/data/atlas/users/nterlind/logs/`
- **Facts/plots:** Inside each run directory under `facts/` and `train_figures/`

Verify after job completes:
```bash
# Training outputs
tree /data/atlas/users/nterlind/outputs/runs/ | head -20
ls /data/atlas/users/nterlind/outputs/runs/run_*/.hydra/config.yaml

# Multirun metadata
ls /data/atlas/users/nterlind/outputs/multiruns/exp_*/multirun.yaml

# Reports
ls /data/atlas/users/nterlind/outputs/reports/report_*/manifest.yaml
ls /data/atlas/users/nterlind/outputs/reports/report_*/training/figures/
```

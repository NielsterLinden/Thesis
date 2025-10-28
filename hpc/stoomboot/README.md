# Stoomboot HPC Job Submission Guide

## Prerequisites

1. SSH access to Stoomboot configured
2. Conda environment created at `/data/atlas/nterlind/venvs/thesis-ml`
3. Dataset uploaded to `/data/atlas/nterlind/datasets/4tops_splitted.h5`
4. Log directories created: `/data/atlas/nterlind/logs/{stdout,stderr}/`

## Quick Start

### Submit a CPU job
```bash
cd /project/atlas/nterlind/Thesis-Code
condor_submit hpc/stoomboot/job_cpu.sub
```

### Submit a GPU job
```bash
condor_submit hpc/stoomboot/job_gpu.sub
```

### Submit with Hydra arguments
```bash
# Method 1: Edit arguments line in .sub file
# Method 2: Use -append
condor_submit hpc/stoomboot/job_gpu.sub -append 'arguments = trainer.epochs=100 phase1/latent_space=vq'
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

- **Run outputs:** `/data/atlas/nterlind/outputs/runs/run_<timestamp>_<name>/`
- **Experiment sweeps:** `/data/atlas/nterlind/outputs/experiments/exp_<timestamp>_<name>/<job_num>/`
- **Job logs:** `/data/atlas/nterlind/logs/`
- **Facts/plots:** Inside each run directory under `facts/` and `figures/`

Verify after job completes:
```bash
tree /data/atlas/nterlind/outputs/runs/ | head -20
ls /data/atlas/nterlind/outputs/runs/run_*/cfg.yaml
```

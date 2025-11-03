# Quick Reference: Running compare_globals_heads Report

## Interactive Session (Stoomboot)

```bash
cd /project/atlas/users/nterlind/Thesis-Code
eval "$(conda shell.bash hook)"
conda activate /data/atlas/users/nterlind/venvs/thesis-ml

python -m thesis_ml.reports.report_from_sweep \
  --config-name compare_globals_heads \
  inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_20251031-152750_compare_globals_heads \
  inference.enabled=true \
  env.output_root=/data/atlas/users/nterlind/outputs
```

## Condor Submission

```bash
cd /project/atlas/users/nterlind/Thesis-Code

condor_submit hpc/stoomboot/report.sub \
  -append 'arguments = --config-name compare_globals_heads inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_20251031-152750_compare_globals_heads inference.enabled=true env.output_root=/data/atlas/users/nterlind/outputs'
```

## Monitor Job

```bash
condor_q                    # Check status
condor_tail -f <job_id>     # Follow output
```

## Verify Results

```bash
# Find latest report
REPORT_DIR=$(ls -td /data/atlas/users/nterlind/outputs/reports/report_*_compare_globals_heads | head -1)

# Check outputs
ls $REPORT_DIR/training/figures/
ls $REPORT_DIR/inference/figures/
cat $REPORT_DIR/inference/summary.json
```

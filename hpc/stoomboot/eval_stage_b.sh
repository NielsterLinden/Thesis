#!/bin/bash
# Local helper: same environment as Condor (thesis_inference.sh).
set -euo pipefail
cd /project/atlas/users/nterlind/Thesis-Code
exec hpc/stoomboot/thesis_inference.sh "$@"

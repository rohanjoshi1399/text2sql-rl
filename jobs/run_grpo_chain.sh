#!/bin/bash
# Submit GRPO training with automatic resubmission for multi-8h-job training.
# Each job resumes from the latest checkpoint automatically.
#
# Usage: bash jobs/run_grpo_chain.sh [max_submissions]
#   default: 4 submissions = up to 32 hours total capacity

set -euo pipefail

MAX_SUBS=${1:-4}

echo "=== GRPO Job Chain ==="
echo "Submitting $MAX_SUBS chained 8-hour jobs..."

JOB_ID=$(sbatch --parsable jobs/run_grpo.sh)
echo "  Job 1/$MAX_SUBS: $JOB_ID (immediate)"

for i in $(seq 2 $MAX_SUBS); do
    JOB_ID=$(sbatch --parsable --dependency=afterany:$JOB_ID jobs/run_grpo.sh)
    echo "  Job $i/$MAX_SUBS: $JOB_ID (after $JOB_ID)"
done

echo ""
echo "Chain submitted. Each job will resume from the latest checkpoint."
echo "Monitor with: squeue -u \$USER"

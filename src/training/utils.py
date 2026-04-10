"""Training utilities — checkpoint management and Slurm signal handling."""

import os
import signal
import sys


def find_latest_checkpoint(output_dir: str) -> str | None:
    """
    Find the latest checkpoint-XXXX directory in output_dir.
    Returns the full path to the latest checkpoint, or None if no checkpoints exist.
    """
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        d
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    # Sort by step number (checkpoint-100, checkpoint-200, ...)
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(output_dir, checkpoints[-1])


def setup_slurm_signal_handler(trainer, output_dir: str):
    """
    Register handler for SIGUSR1/SIGTERM sent by Slurm before job kill.

    Slurm sends the configured signal (e.g. SIGUSR1) before the time limit.
    Use ``#SBATCH --signal=USR1@120`` in job scripts to get 120 seconds warning.
    The handler saves a numbered checkpoint and exits cleanly so the next
    chained job can resume via find_latest_checkpoint().
    """

    def _handler(signum, frame):
        sig_name = signal.Signals(signum).name
        step = trainer.state.global_step if hasattr(trainer, "state") else 0
        save_dir = os.path.join(output_dir, f"checkpoint-{step}")
        print(f"\n[SLURM] Received {sig_name} — saving checkpoint-{step} before exit...")
        trainer.save_model(save_dir)
        trainer.save_state()
        print(f"[SLURM] Checkpoint saved to {save_dir}. Exiting gracefully.")
        sys.exit(0)

    # SIGUSR1 only exists on Unix/Linux (cluster runs Linux)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _handler)
    signal.signal(signal.SIGTERM, _handler)

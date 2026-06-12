#!/usr/bin/env python3
"""Probe GPU usability for DDP training.

Each GPU is probed in a fresh subprocess that does exactly what a DDP rank
does at startup (`torch.cuda.set_device` + a small allocation), so one bad
device cannot poison the others and Exclusive_Process conflicts show up the
same way they do in training.

Usage (on the training node):
    python check_gpus.py            # probe every visible GPU
    python check_gpus.py 0 1        # probe specific GPU ids
"""
import subprocess
import sys


def probe(gpu_id: int) -> None:
    """Claim the device like a DDP rank would. Runs in a child process."""
    import torch
    torch.cuda.set_device(gpu_id)
    x = torch.zeros(8, device=f'cuda:{gpu_id}')
    x += 1
    torch.cuda.synchronize(gpu_id)
    print(f"OK | {torch.cuda.get_device_name(gpu_id)}")


def run_nvidia_smi_report() -> None:
    queries = [
        ("GPU state", ["nvidia-smi", "--query-gpu=index,name,compute_mode,mig.mode.current,memory.used,memory.total",
                       "--format=csv"]),
        ("Compute processes", ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                               "--format=csv"]),
    ]
    for title, cmd in queries:
        print(f"\n--- {title} ---")
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            print(out.stdout.strip() or out.stderr.strip() or "(no output)")
        except FileNotFoundError:
            print("nvidia-smi not found in PATH")
        except subprocess.TimeoutExpired:
            print("nvidia-smi timed out — driver may be wedged (check dmesg)")


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == '--probe':
        probe(int(sys.argv[2]))
        return

    import torch
    if not torch.cuda.is_available():
        print("torch.cuda.is_available() is False — no usable CUDA runtime at all.")
        run_nvidia_smi_report()
        sys.exit(1)

    count = torch.cuda.device_count()
    if len(sys.argv) > 1:
        gpu_ids = [int(a) for a in sys.argv[1:]]
    else:
        gpu_ids = list(range(count))

    print(f"Visible CUDA devices: {count}")
    failures = 0
    for gpu_id in gpu_ids:
        if gpu_id >= count:
            print(f"GPU {gpu_id}: FAIL | id out of range — only {count} device(s) visible "
                  f"(CUDA_VISIBLE_DEVICES may be restricting the job)")
            failures += 1
            continue
        result = subprocess.run(
            [sys.executable, __file__, '--probe', str(gpu_id)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            print(f"GPU {gpu_id}: {result.stdout.strip()}")
        else:
            failures += 1
            # Last traceback line carries the CUDA error string
            err_lines = [l for l in result.stderr.strip().splitlines() if l.strip()]
            reason = err_lines[-1] if err_lines else "unknown error"
            print(f"GPU {gpu_id}: FAIL | {reason}")

    run_nvidia_smi_report()

    if failures:
        print(f"\n{failures} GPU(s) unusable. Common causes:")
        print("  - compute_mode 'Exclusive_Process': another process already holds a context")
        print("    -> find it in the compute-process table above, or pick free GPUs in gpu_ids")
        print("  - mig.mode.current 'Enabled': plain device ids cannot be used on a MIG GPU")
        print("    -> ask the admin, or use a non-MIG GPU")
        print("  - GPU wedged (ECC/Xid errors): check `dmesg | grep -i xid`, needs reset")
        sys.exit(1)
    print("\nAll probed GPUs usable.")


if __name__ == '__main__':
    main()

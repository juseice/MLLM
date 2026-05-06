import itertools
import subprocess
import signal
import os
import sys

# For KGW
# gamma_list = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
# delta_list = [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# For SIR
delta_list = [0.1, 0.5, 1.0, 2.0]

gpus = [0, 1, 2, 3]
# params = list(itertools.product(gamma_list, delta_list))
params = list(itertools.product(delta_list))

procs = []

def kill_all():
    print("\nTerminating all child processes...")
    for p in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass

try:
    for i, (delta) in enumerate(params):
        gpu = gpus[i % len(gpus)]
        delta = delta[0]

        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu} "
            f"python keyword_watermark_01.py "
            f"--delta {delta}"
        )

        print(f"Launching δ={delta} on GPU {gpu}")
        procs.append(subprocess.Popen(cmd, shell=True))

    for p in procs:
        p.wait()

    print("All experiments completed")
except KeyboardInterrupt:
    kill_all()
    sys.exit(1)

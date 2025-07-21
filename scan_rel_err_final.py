#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# scan_rel_err_final.py  ·  Barrido Nivel-1 final (alpha -> rel.err)
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, re, csv, sys, pathlib, os
import numpy as np

SCRIPT = pathlib.Path(__file__).parent / "seion_dynamics_1d_stable.py"
if not SCRIPT.exists():
    sys.exit("❌  Falta 'seion_dynamics_1d_stable.py'")

# Parámetros de alta fidelidad para el barrido final
alphas   = np.linspace(0.01, 1.00, 100) # Evitar el punto de inestabilidad en alpha=0
rank     = "64"
dt_val   = "0.002"
lambda_r = "0.2"
results  = []

print(f"───────── Barrido alpha | rank={rank} | dt={dt_val} | lambda={lambda_r} ─────────")
for alpha in alphas:
    a_str = f"{alpha:.2f}"
    cmd = [
        sys.executable, str(SCRIPT), "--quick",
        "--rank", rank, "--alpha", a_str, "--dt", dt_val, "--lambda_rep", lambda_r
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    print("• Ejecutando:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(f"❌ Error alpha={a_str}")

    m = re.findall(r"Curv-Assoc rel\.err\s*=\s*([0-9.+-eE]+)", proc.stdout)
    if not m:
        print(proc.stdout)
        sys.exit(f"❌ No rel.err para alpha={a_str}")
    
    rel_err = float(m[-1])
    results.append((alpha, rel_err))
    print(f"  alpha={a_str} -> rel.err={rel_err:.3e}")

csv_path = pathlib.Path("rel_err_vs_alpha_final.csv")
with csv_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["alpha","rel.err"])
    writer.writerows(results)

print("\n✅ Barrido completado. CSV en", csv_path.absolute())
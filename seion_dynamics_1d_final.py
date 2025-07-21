#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# seion_dynamics_1d_stable.py · Motor Seiónico 1-D (CPU / GPU)
#
# Versión final estable:
#   - Añadido término repulsivo cuártico (-i*lambda*(phi†*phi)²*phi)
#     para prevenir el "blow-up" y asegurar estabilidad numérica.
#   - Todos los parámetros son configurables por CLI.
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, time, math, json, argparse
import numpy as np

try:
    import cupy as xp
    backend = "GPU (CuPy)"
except ImportError:
    xp = np
    backend = "CPU (NumPy)"

# --- Interfaz de Línea de Comandos (CLI) ---
parser = argparse.ArgumentParser(description="Motor Seiónico 1-D Estable")
parser.add_argument("--quick", action="store_true", help="T=0.1s para pruebas rápidas")
parser.add_argument("--alpha", type=float, default=float(os.getenv("SEION_ALPHA", 0.05)), help="Parámetro de ruptura de simetría del núcleo.")
parser.add_argument("--rank",  type=int,   default=int(os.getenv("SEION_RANK", 48)), help="Rango de la descomposición CP-SVD.")
parser.add_argument("--lambda_rep", type=float, default=float(os.getenv("SEION_LAMBDA", 0.1)), help="Constante del término repulsivo estabilizador.")
parser.add_argument("-N",      type=int,   default=512, help="Tamaño de la malla.")
parser.add_argument("--dt",    type=float, default=0.002, help="Paso de tiempo del integrador.")
parser.add_argument("--T",     type=float, default=None, help="Tiempo final de la simulación.")
cli = parser.parse_args()

# --- Parámetros de la Simulación ---
N          = cli.N
L          = 10.0
dt         = cli.dt
T_final    = 0.1 if cli.quick else (cli.T if cli.T is not None else 1.0)
rank_r     = cli.rank
alpha_symm = cli.alpha
lambda_rep = cli.lambda_rep
mu_const   = 1.0
sigma_k    = 0.3 * L

# --- Construcción del Núcleo K(x;y,z) ---
xs = xp.linspace(0, L, N, endpoint=False)
K  = xp.empty((N, N, N), dtype=xp.float64)
for x in range(N):
    dy, dz = xs[x] - xs[:, None], xs[x] - xs[None, :]
    K[x] = xp.exp(-(dy**2 + dz**2) / (2 * sigma_k**2))
K *= (1.0 + 0.05 * xp.sin(2 * np.pi * xs / L))[:, None, None]
iy, iz = xp.indices((N, N))
K[:, iy > iz] *= (1.0 + alpha_symm)
K[:, iy < iz] *= (1.0 - alpha_symm)
K /= (2 * math.pi * sigma_k**2)

# --- Descomposición Tensorial (CP-SVD) ---
def cp_svd(tensor, r):
    X = xp.reshape(tensor, (N, N * N))
    U, S, VT = xp.linalg.svd(X, full_matrices=False)
    U, S, VT = U[:, :r], S[:r], VT[:r].reshape(r, N, N)
    v = xp.zeros((r, N)); w = xp.zeros((r, N)); lam = S.copy()
    for k in range(r):
        q, s, rvec = xp.linalg.svd(VT[k], full_matrices=False)
        v[k]   = q[:, 0]  * math.sqrt(s[0])
        w[k]   = rvec[0] * math.sqrt(s[0])
        lam[k] *= s[0]
    return U, v, w, lam

print(f"# CP-SVD (rank={rank_r}) …")
U, V, W, Lambda = cp_svd(K, rank_r)
Lambda /= N

def star(A, B):
    return U @ (Lambda * (V @ A) * (W @ B))

# --- Operadores Algebraicos ---
comm  = lambda A,B: star(A,B) - star(B,A)
assoc = lambda A,B,C: star(star(A,B), C) - star(A, star(B, C))
curv  = lambda A,B,C: star(A, star(B, C)) - star(B, star(A, C)) - star(comm(A,B), C)
g     = lambda X,Y: xp.vdot(X, Y)

# --- Ecuación de Movimiento Establecida ---
rng = xp.random.default_rng(0)
phi = 0.1 * (rng.standard_normal(N, dtype=np.float64) + 1j * rng.standard_normal(N, dtype=np.float64))

def RHS_stable(P):
    phi_conj = xp.conj(P)
    density = star(phi_conj, P)
    term_cubic = star(density, P)
    # TÉRMINO REPULSIVO ESTABILIZADOR
    term_quartic = star(density**2, P)
    return 1j * term_cubic - 1j * mu_const * P - 1j * lambda_rep * term_quartic

# --- Lazo de Tiempo (RK4) ---
print(f"# Dynamics | {backend} | N={N}, rank={rank_r}, alpha={alpha_symm:.2f}, dt={dt}, lambda={lambda_rep}\n")
n_steps  = int(T_final / dt)
t_start  = time.perf_counter()
E0 = None; last_err = None

for step in range(1, n_steps + 1):
    k1 = RHS_stable(phi)
    k2 = RHS_stable(phi + 0.5*dt*k1)
    k3 = RHS_stable(phi + 0.5*dt*k2)
    k4 = RHS_stable(phi + dt*k3)
    phi = phi + dt*(k1 + 2*k2 + 2*k3 + k4)/6

    if step % 20 == 0 or step == n_steps:
        E = xp.real(g(phi, star(xp.conj(phi), phi)))
        if E0 is None: E0 = float(E)
        print(f"t={step*dt:5.3f} | normPhi={xp.linalg.norm(phi):.4e}  Energy={E:.4e}")

        R, A = curv(phi, phi, phi), assoc(phi, phi, phi)
        rel  = min(float(xp.linalg.norm(R - A)), float(xp.linalg.norm(R + A)))
        denom = float(xp.linalg.norm(R))+float(xp.linalg.norm(A))+1e-30
        last_err = rel/denom
        print(f"   Curv-Assoc rel.err = {last_err:.2e}")

elapsed = time.perf_counter() - t_start
print(f"\nSimulación completa en {elapsed:.2f} s")

# --- Resumen JSON ---
summary = {
    "rank": rank_r, "alpha": alpha_symm, "lambda": lambda_rep,
    "rel_err": last_err,
    "energy_drift": float(xp.real(g(phi, star(xp.conj(phi), phi)))) - E0,
    "runtime_sec": elapsed
}
print("JSON:", json.dumps(summary))
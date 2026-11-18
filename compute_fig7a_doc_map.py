"""
Compute the DOC_1 map of Fig. 7(a): maximum degree of coherence DOC_1 reached
during free propagation, as a function of electron kinetic energy and
interaction length L.

Configuration (paper, Fig. 7): light wavelength 1064 nm, field amplitude
E0 = 4 MV/m, optical phase velocity matched to 30 keV electrons.  Electrons of
other energies are phase-mismatched (d_phi = -d_zeta * v_ph / v).

For every kinetic energy the one-step transfer matrix is diagonalized once;
the states for all interaction lengths then follow from powers of its
eigenvalues (pinem_recoil.propagate_to).  For every state DOC_1 is maximized
over the free-propagation distance s = zeta_free/zeta_T in [0.75, 1.25]; since
DOC_1(s) has period 1/2 in s (quadratic dispersion), this is the maximum over
all free-propagation distances.

Usage (from the repository folder):

    python compute_fig7a_doc_map.py                 # all CPU cores
    python compute_fig7a_doc_map.py --workers 8     # limit the number of processes
    python compute_fig7a_doc_map.py --n-energy 41 --n-length 25   # quick, coarse test

Runtime of the full grid (401 x 241 points) is roughly 400 x (10-60 s) of
single-core time, i.e. several hours on a laptop or ~8 min on 64 cores.
Output (plain-text CSV, in data/fig7/):
    fig7a_DOC1_max.csv      max DOC_1; rows: L (um), columns: kinetic energy (eV)
    fig7a_zfree_at_max.csv  free-propagation distance of the maximum (units of z_T)
    fig7a_per_energy.csv    a, g and sideband truncation l_max for every energy
    fig7a_parameters.csv    fixed parameters of the calculation
"""

import os
# One BLAS thread per process: the parallelism is over electron energies.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import pinem_recoil as pr

WAVELENGTH = 1064e-9        # m
E0 = 4e6                    # V/m
E_KIN_PHASE_MATCHED = 30e3  # eV
TRUNCATION_MARGIN = 100     # sidebands added to the estimated populated range


def doc_column(E_kin_eV, L_values, d_zeta, n_s):
    """Max DOC_1 (and the free-propagation distance where it occurs) for one
    kinetic energy and all interaction lengths L_values (m)."""
    p = pr.electron_parameters(E_kin_eV, WAVELENGTH, E0)
    v_ph = pr.electron_parameters(E_KIN_PHASE_MATCHED, WAVELENGTH, E0)["v"]
    a, g = p["a"], p["g"]
    l_max = pr.l_max_estimate(a, g, v_ph / p["v"] - 1) + TRUNCATION_MARGIN
    d_phi = -d_zeta * v_ph / p["v"]
    n_steps = (L_values * p["zeta_per_m"] / d_zeta).astype(int)
    phi = pr.propagate_to(a, g, l_max, d_zeta, d_phi, n_steps)
    s = np.linspace(0.75, 1.25, n_s)
    doc_max = np.zeros(len(L_values))
    s_at_max = np.zeros(len(L_values))
    for k in range(len(L_values)):
        doc = pr.doc1_free_propagation(phi[:, k], s)
        doc_max[k], s_at_max[k] = doc.max(), s[doc.argmax()]
    return doc_max, s_at_max, l_max, a, g


def _job(args):
    j, E_kin, L_values, d_zeta, n_s = args
    t0 = time.time()
    res = doc_column(E_kin, L_values, d_zeta, n_s)
    return j, res, time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--e-min", type=float, default=29e3, help="min kinetic energy (eV)")
    ap.add_argument("--e-max", type=float, default=31e3, help="max kinetic energy (eV)")
    ap.add_argument("--n-energy", type=int, default=401)
    ap.add_argument("--l-min", type=float, default=20e-6, help="min interaction length (m)")
    ap.add_argument("--l-max", type=float, default=140e-6, help="max interaction length (m)")
    ap.add_argument("--n-length", type=int, default=241)
    ap.add_argument("--d-zeta", type=float, default=0.1, help="propagation step")
    ap.add_argument("--n-s", type=int, default=5001, help="free-propagation samples in [0.75, 1.25] z_T")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out-dir", default=str(Path("data") / "fig7"))
    args = ap.parse_args(argv)

    E_kin = np.linspace(args.e_min, args.e_max, args.n_energy)
    L = np.linspace(args.l_min, args.l_max, args.n_length)
    doc = np.zeros((len(L), len(E_kin)))
    s_max = np.zeros_like(doc)
    l_trunc = np.zeros(len(E_kin), dtype=int)
    a_vals, g_vals = np.zeros(len(E_kin)), np.zeros(len(E_kin))

    jobs = [(j, E, L, args.d_zeta, args.n_s) for j, E in enumerate(E_kin)]
    # largest truncations first for better load balancing
    jobs.sort(key=lambda jb: -abs(jb[1] - 30.4e3))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n_done, (j, (dmax, smax, lm, a, g), dt) in enumerate(pool.map(_job, jobs), 1):
            doc[:, j], s_max[:, j], l_trunc[j], a_vals[j], g_vals[j] = dmax, smax, lm, a, g
            print(f"[{n_done:4d}/{len(jobs)}] E_kin = {E_kin[j]:8.1f} eV  l_max = {lm:4d}  "
                  f"max DOC1 = {dmax.max():.3f}  ({dt:.0f} s, total {time.time() - t0:.0f} s)", flush=True)

    save_results(args.out_dir, E_kin, L, doc, s_max, l_trunc, a_vals, g_vals, args.d_zeta, args.n_s)


def save_results(out_dir, E_kin, L, doc, s_max, l_trunc, a_vals, g_vals, d_zeta, n_s):
    """Write the map and its metadata as CSV files to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pr.save_map(out / "fig7a_DOC1_max.csv", doc, E_kin, L * 1e6, "E_kin_eV", "L_um")
    pr.save_map(out / "fig7a_zfree_at_max.csv", s_max, E_kin, L * 1e6, "E_kin_eV", "L_um")
    pr.save_columns(out / "fig7a_per_energy.csv", E_kin_eV=E_kin, a=a_vals, g=g_vals, l_max=l_trunc)
    pr.save_parameters(out / "fig7a_parameters.csv", [
        ("wavelength_m", WAVELENGTH, "light wavelength"),
        ("E0_V_per_m", E0, "field amplitude"),
        ("E_kin_phase_matched_eV", E_KIN_PHASE_MATCHED, "kinetic energy of the phase-matched electron"),
        ("d_zeta", d_zeta, "propagation step"),
        ("d_phi", "-d_zeta*v_ph/v", "field-phase step per propagation step"),
        ("truncation_margin", TRUNCATION_MARGIN, "sidebands added to the estimated populated range"),
        ("zfree_min_over_zT", 0.75, "free-propagation window searched for the maximum of DOC_1"),
        ("zfree_max_over_zT", 1.25, "(DOC_1 is periodic with period z_T/2)"),
        ("n_zfree", n_s, "number of free-propagation samples in the window"),
    ])
    print("saved CSV files in", out)


if __name__ == "__main__":
    main()

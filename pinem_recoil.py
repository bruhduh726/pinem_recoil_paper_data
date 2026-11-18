"""
pinem_recoil.py -- numerical core shared by all figure notebooks.

Quantum dynamics of a free electron interacting with an evanescent optical
field of uniform amplitude and linearly varying phase, including recoil
(finite recoil parameter ``a``) and phase mismatch, followed by free
propagation and the evaluation of the degree of coherence DOC_1.

All quantities are dimensionless, as in the paper:

    zeta  = omega z / v                normalized propagation distance
    theta = omega t                    normalized time
    g     = e v E0 / (hbar omega^2)    coupling strength
    a     = 2 m_e v^2 gamma^3 / (hbar omega)
                                       recoil parameter (a -> infinity: no recoil)
    zeta_T = 2 pi a                    normalized Talbot distance

The electron wave function is expanded in energy sidebands,
phi(zeta, theta) = sum_l phi_l(zeta) exp(-i l theta), and the sideband
amplitudes phi_l are returned on the grid l = -l_max ... l_max.  Positive l
corresponds to energy gain (absorption of l photons).

Interaction (paper, Appendix "Solution for a uniform field").  The
interaction region is split into steps of length ``d_zeta``; within each step
the field is uniform and its phase advances by ``-d_phi`` from one step to the
next (``d_phi = -d_zeta`` is perfect phase matching for the l = 0 state,
``d_phi = -d_zeta * v_ph/v`` in general).  Within a step the second-order
equation psi'' = -(a^2/4) psi + a B psi is solved exactly through the
eigen-decomposition B = V diag(b) V^T, giving the transfer matrix

    [psi, psi'](zeta + d_zeta) = M [psi, psi'](zeta),
    M = [[A, S], [C, A]],  A = V cosh(lam d) V^T, S = V sinh(lam d)/lam V^T,
                           C = V lam sinh(lam d) V^T,  lam = sqrt(a b - a^2/4).

Here B is the real symmetric tridiagonal matrix with diagonal l and
off-diagonal elements -g.  (The sign of the coupling only fixes the origin of
the optical phase: g -> -g multiplies phi_l by (-1)^l, i.e. shifts omega t by
pi.  The sign used here is the one that produced the published figures.)

The last section contains small helpers that write and read the plain-text
CSV data files in data/ (comma-separated, decimal point, one header line).

Only numpy and scipy are required.
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal, eig, solve
from scipy.special import jv
from scipy.constants import c, e, hbar, m_e

__all__ = [
    "electron_parameters", "sideband_axis", "initial_state", "step_matrix",
    "propagate_steps", "propagate_to", "nonrecoil_analytic",
    "sideband_limits", "l_max_estimate", "doc1_free_propagation",
    "talbot_carpet", "doc_from_density", "max_doc1_after_free_propagation",
    "save_map", "load_map", "save_columns", "load_columns",
    "save_parameters", "load_parameters",
]


# --------------------------------------------------------------------------
# Physical <-> dimensionless parameters
# --------------------------------------------------------------------------
def electron_parameters(E_kin_eV, wavelength, E0):
    """Dimensionless parameters for an electron of kinetic energy ``E_kin_eV``
    (eV) in a field of vacuum wavelength ``wavelength`` (m) and amplitude
    ``E0`` (V/m).

    Returns a dict with v (m/s), gamma, omega (rad/s), a, g, zeta_T and
    z_T (Talbot distance, m), and zeta_per_m (= omega/v).
    """
    gamma = 1.0 + E_kin_eV * e / (m_e * c**2)
    v = c * np.sqrt(1.0 - 1.0 / gamma**2)
    omega = 2 * np.pi * c / wavelength
    a = 2 * m_e * v**2 * gamma**3 / (hbar * omega)
    g = e * v * E0 / (hbar * omega**2)
    zeta_T = 2 * np.pi * a
    return dict(v=v, gamma=gamma, omega=omega, a=a, g=g, zeta_T=zeta_T,
                z_T=zeta_T * v / omega, zeta_per_m=omega / v)


def sideband_axis(l_max):
    """Sideband orders l = -l_max ... l_max."""
    return np.arange(-l_max, l_max + 1)


# --------------------------------------------------------------------------
# Interaction: transfer-matrix propagation
# --------------------------------------------------------------------------
def initial_state(l_max, a, ell0=0):
    """Monochromatic electron in sideband ``ell0``: the 2(2 l_max + 1) vector
    [psi, psi'] with psi_l = delta_{l,ell0} and psi'_l = (i a/2) delta_{l,ell0}
    (forward-propagating wave)."""
    n = 2 * l_max + 1
    state = np.zeros(2 * n, dtype=np.complex128)
    state[l_max + ell0] = 1.0
    state[n + l_max + ell0] = 1j * a / 2
    return state


def step_matrix(a, g, l_max, d_zeta, d_phi):
    """Return (ME, E): the one-step transfer matrix ME = M @ E and the phase
    matrix E = diag(exp(-i l d_phi)) (block-diagonal on [psi, psi'])."""
    ell = sideband_axis(l_max).astype(float)
    b, V = eigh_tridiagonal(ell, np.full(2 * l_max, -float(g)))
    lam = np.sqrt(a * b - a**2 / 4 + 0j)
    ep, em = np.exp(lam * d_zeta), np.exp(-lam * d_zeta)
    cosh, sinh = 0.5 * (ep + em), 0.5 * (ep - em)
    A = (V * cosh) @ V.T
    S = (V * (sinh / lam)) @ V.T
    C = (V * (lam * sinh)) @ V.T
    M = np.block([[A, S], [C, A]])
    phase = np.exp(-1j * ell * d_phi)
    E = np.concatenate([phase, phase])          # diagonal of E
    return M * E[np.newaxis, :], E               # M @ diag(E)


def _output_amplitudes(state, E, a, zeta, l_max):
    """phi_l at distance zeta from the internal [psi, psi'] vector."""
    ell = sideband_axis(l_max)
    n = 2 * l_max + 1
    return (E[:n] * state[:n]) * np.exp(-1j * a * zeta / 2) \
        * np.exp(-1j * zeta * ell**2 / a)


def propagate_steps(a, g, l_max, d_zeta, d_phi, n_steps, ell0=0,
                    initial=None):
    """Step-by-step propagation through a uniform-amplitude field.

    Returns (zeta, phi): zeta = n d_zeta for n = 0 ... n_steps-1 and the
    complex sideband amplitudes phi with shape (2 l_max + 1, n_steps).
    ``initial`` (optional) is a [psi, psi'] vector overriding ``ell0``.
    """
    ME, E = step_matrix(a, g, l_max, d_zeta, d_phi)
    state = initial_state(l_max, a, ell0) if initial is None else initial
    n = 2 * l_max + 1
    phi = np.zeros((n, n_steps), dtype=np.complex128)
    phi[:, 0] = state[:n]
    zeta = np.arange(n_steps) * d_zeta
    for k in range(1, n_steps):
        state = ME @ state
        phi[:, k] = _output_amplitudes(state, E, a, zeta[k], l_max)
    return zeta, phi


def propagate_to(a, g, l_max, d_zeta, d_phi, n_steps_list, ell0=0):
    """Amplitudes after an arbitrary number of steps, using the
    eigen-decomposition of the one-step matrix (one diagonalization serves
    all distances).  Returns phi with shape (2 l_max + 1, len(n_steps_list)),
    column k corresponding to zeta = n_steps_list[k] * d_zeta."""
    ME, E = step_matrix(a, g, l_max, d_zeta, d_phi)
    w, U = eig(ME)
    coeff = solve(U, initial_state(l_max, a, ell0))
    n_steps_list = np.atleast_1d(n_steps_list)
    phi = np.zeros((2 * l_max + 1, len(n_steps_list)), dtype=np.complex128)
    for k, N in enumerate(n_steps_list):
        state = U @ (w**int(N) * coeff)
        phi[:, k] = _output_amplitudes(state, E, a, N * d_zeta, l_max)
    return phi


def nonrecoil_analytic(g, l_max, zeta):
    """Phase-matched nonrecoil limit: |phi_l(zeta)| = |J_l(2 g zeta)|.
    Returns an array of shape (2 l_max + 1, len(zeta))."""
    ell = sideband_axis(l_max)
    return jv(ell[:, None], 2 * g * np.asarray(zeta)[None, :])


# --------------------------------------------------------------------------
# Sideband bounds (paper, Eqs. "lmaxandlmin")
# --------------------------------------------------------------------------
def sideband_limits(a, g, delta_v):
    """Classical bounds (l_min, l_max) on the significantly populated
    sidebands for an initial velocity mismatch delta_v = v/v_ph - 1."""
    d = float(delta_v)
    crit = np.sqrt(16 * g / a)
    if abs(d) > crit:
        s = np.sign(d)
        l_min = a / 2 * (s * np.sqrt(d**2 - s * 16 * g / a) - d)
        l_max = a / 2 * (s * np.sqrt(d**2 + s * 16 * g / a) - d)
    else:
        l_max = a / 2 * (np.sqrt(np.heaviside(d, 0) * d**2 + 16 * g / a) - d)
        l_min = a / 2 * (-np.sqrt(np.heaviside(-d, 0) * d**2 + 16 * g / a) - d)
    return l_min, l_max


def l_max_estimate(a, g, delta_k):
    """Heuristic estimate of the largest populated |l| used to choose the
    numerical truncation (``delta_k`` = v_ph/v - 1).  Either the electron
    oscillates without reaching phase matching ("bounce", l ~ g zeta_bounce)
    or it reaches it and is limited by recoil."""
    zeta_bounce = 2 * np.pi / np.abs(delta_k + 1e-8)
    l_res = a / 2 * np.abs(delta_k + 1e-8)
    if 2 * g * zeta_bounce < l_res:
        l_est = g * zeta_bounce
    else:
        l_est = np.abs(l_res) + np.sqrt(l_res**2 + a * g * np.pi)
    return int(np.abs(l_est))


# --------------------------------------------------------------------------
# Free propagation, temporal profile and degree of coherence
# --------------------------------------------------------------------------
# After the interaction the sidebands propagate freely as
#   phi_l(zeta_free) = phi_l exp(i l zeta_free) exp(-i l^2 zeta_free / a).
# The first factor is a rigid shift of the time axis; we work in the frame
# co-moving with it, so only the recoil phase exp(-i 2 pi l^2 s) matters,
# with s = zeta_free / zeta_T the distance in units of the Talbot distance.

def doc1_free_propagation(phi, s):
    """DOC_1 of the sideband state ``phi`` after free propagation over
    s = zeta_free/zeta_T (scalar or array):

        DOC_1(s) = | sum_l phi_l phi*_{l-1} exp(-i 2 pi (2l-1) s) |^2

    (for a normalized state).  Returns an array with the shape of ``s``."""
    l_max = (len(phi) - 1) // 2
    ell = sideband_axis(l_max)[1:]
    pair = phi[1:] * np.conj(phi[:-1])
    s = np.asarray(s, dtype=float)
    out = np.empty(s.size)
    for i0 in range(0, s.size, 512):                    # chunk to save memory
        si = s.ravel()[i0:i0 + 512]
        out[i0:i0 + 512] = np.abs(
            np.exp(-2j * np.pi * np.outer(si, 2 * ell - 1)) @ pair)**2
    return out.reshape(s.shape)


def talbot_carpet(phi, s, n_theta):
    """Electron probability density |phi(s, theta)|^2 on theta = linspace(0,
    2 pi, n_theta) for each free-propagation distance s = zeta_free/zeta_T.

    phi(s, theta) = sum_l phi_l exp(-i l theta) exp(-i 2 pi l^2 s); each
    column is normalized such that sum(|phi|^2) * dtheta = 1.
    Returns (theta, density) with density of shape (n_theta, len(s))."""
    l_max = (len(phi) - 1) // 2
    ell = sideband_axis(l_max)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    s = np.asarray(s, dtype=float)
    coeffs = phi[None, :] * np.exp(-2j * np.pi * np.outer(s, ell**2))
    psi = np.exp(-1j * np.outer(theta, ell)) @ coeffs.T  # (n_theta, n_s)
    dens = np.abs(psi)**2
    dens /= dens.sum(axis=0, keepdims=True) * (theta[1] - theta[0])
    return theta, dens


def doc_from_density(density, m=1):
    """DOC_m of a sampled temporal profile on theta = linspace(0, 2 pi, n)
    (trapezoidal integration): |int rho e^{i m theta}|^2 / |int rho|^2.
    ``density`` may be 1D or (n_theta, n) with one profile per column."""
    rho = np.asarray(density)
    theta = np.linspace(0, 2 * np.pi, rho.shape[0])
    shape = (-1,) + (1,) * (rho.ndim - 1)
    num = np.trapezoid(rho * np.exp(1j * m * theta).reshape(shape), theta, axis=0)
    den = np.trapezoid(rho, theta, axis=0)
    return np.abs(num)**2 / np.abs(den)**2


def max_doc1_after_free_propagation(phi, s_min=0.75, s_max=1.25, n_s=2500):
    """Maximum of DOC_1 over free-propagation distances
    s = zeta_free/zeta_T in [s_min, s_max] (n_s samples)."""
    return doc1_free_propagation(phi, np.linspace(s_min, s_max, n_s)).max()


# --------------------------------------------------------------------------
# Plain-text (CSV) data files
# --------------------------------------------------------------------------
# Three kinds of files are used in data/ (comma-separated, decimal point):
#   map        2-D array.  First row: values of the column axis x; first
#              column: values of the row axis y; top-left cell "y \ x";
#              data[i, j] belongs to y[i], x[j].
#   columns    1-D arrays of equal length, one per column, names in the
#              header line.  Complex arrays are split into Re_ and Im_ columns.
#   parameters columns name, value, description (scalars of one figure).

def save_map(path, data, x, y, x_name, y_name, fmt="%.6g", axis_fmt="%.10g"):
    r"""Write the 2-D array ``data[i_y, i_x]`` with its axes ``x`` (columns)
    and ``y`` (rows) to a CSV file; the top-left cell is "y_name \ x_name"."""
    data = np.asarray(data)
    header = f"{y_name} \\ {x_name}," + ",".join(axis_fmt % v for v in np.asarray(x))
    np.savetxt(path, np.column_stack([np.asarray(y, dtype=float), data]), delimiter=",",
               fmt=[axis_fmt] + [fmt] * data.shape[1], header=header, comments="")


def load_map(path):
    """Read a file written by :func:`save_map`; returns ``(x, y, data)``."""
    with open(path, encoding="utf-8") as f:
        x = np.array([float(v) for v in f.readline().strip().split(",")[1:]])
    raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return _maybe_int(x), _maybe_int(raw[:, 0]), raw[:, 1:]


def _maybe_int(a):
    """Convert a float array to int if all its values are integers."""
    a = np.asarray(a)
    if (a.dtype.kind == "f" and a.size and np.all(np.isfinite(a))
            and np.all(a == np.round(a)) and np.abs(a).max() < 2**31):
        return a.astype(int)
    return a


def _fmt(v, fmt):
    if isinstance(v, (str, np.str_)):
        return str(v)
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return fmt % v


def save_columns(path, fmt="%.10g", **columns):
    """Write equal-length 1-D arrays as named columns of a CSV file.
    Complex arrays ``name`` are written as columns ``Re_name`` and ``Im_name``."""
    cols = {}
    for name, v in columns.items():
        v = np.asarray(v)
        if np.iscomplexobj(v):
            cols["Re_" + name], cols["Im_" + name] = v.real, v.imag
        else:
            cols[name] = v
    lengths = {k: len(v) for k, v in cols.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"columns of unequal length: {lengths}")
    lines = [",".join(cols)]
    lines += [",".join(_fmt(v, fmt) for v in row) for row in zip(*cols.values())]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def load_columns(path):
    """Read a file written by :func:`save_columns`; returns a dict of 1-D arrays
    (numeric columns as float, or int if all values are integers; other columns
    as str).  ``Re_x``/``Im_x`` pairs are joined into the complex array ``x``."""
    with open(path, encoding="utf-8") as f:
        names = f.readline().strip().split(",")
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=str, ndmin=2)
    out = {}
    for k, name in enumerate(names):
        try:
            out[name] = _maybe_int(raw[:, k].astype(float))
        except ValueError:
            out[name] = raw[:, k]
    for name in [n[3:] for n in names if n.startswith("Re_") and "Im_" + n[3:] in out]:
        out[name] = (out.pop("Re_" + name) + 1j * out.pop("Im_" + name)).astype(complex)
    return out


def save_parameters(path, rows, fmt="%.12g"):
    """Write the scalar parameters of a figure; ``rows`` is a list of
    ``(name, value, description)`` (no commas in name or description)."""
    for name, _, desc in rows:
        if "," in desc or "," in name:
            raise ValueError(f"no commas allowed in parameter name/description: {name}")
    save_columns(path, name=[r[0] for r in rows], value=[_fmt(r[1], fmt) for r in rows],
                 description=[r[2] for r in rows])


def load_parameters(path):
    """Read a file written by :func:`save_parameters`; returns ``{name: value}``
    (int, float, or str)."""
    with open(path, encoding="utf-8") as f:
        f.readline()
        rows = [line.rstrip("\r\n").split(",", 2) for line in f if line.strip()]
    out = {}
    for name, v, _ in rows:
        try:
            out[name] = int(v)
        except ValueError:
            try:
                out[name] = float(v)
            except ValueError:
                out[name] = v
    return out

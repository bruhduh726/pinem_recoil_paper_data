# Data files

All data are plain-text CSV files (comma-separated, decimal point, UTF-8, one header line), one folder per figure.
Every file is written by the notebook of the corresponding figure; the map of Fig. 7(a) is written by
`compute_fig7a_doc_map.py`. File names start with the figure number and the panel letter(s), e.g. `fig7d_density.csv`
holds the map shown in Fig. 7(d).

**Opening in Excel.** Use *Data → From Text/CSV* (delimiter: comma). Double-clicking a file can garble it in Excel
installations that use a decimal comma (e.g. Czech or Spanish regional settings).

**Reading in Python.** `pinem_recoil.load_map`, `load_columns` and `load_parameters` read the three kinds of files
below (any CSV reader works too, e.g. `numpy.loadtxt(file, delimiter=",", skiprows=1)` or `pandas.read_csv`).

## File types

* **Maps** (`*_P.csv`, `*_dE.csv`, `*_density.csv`, `*_DOC1.csv` with two axes, `fig7a_*`): a 2-D array.
  The first row holds the values of the horizontal axis x, the first column the values of the vertical axis y,
  and the top-left cell names them as `y \ x`. The cell in row i, column j is the value at (y_i, x_j).
  Values are written with 6 significant digits, axes with 10.
* **Column tables** (all other data files): one quantity per column, names in the header line.
  Complex amplitudes φ are split into columns `Re_phi` and `Im_phi` and written at full double precision.
  `nan` marks points where a quantity does not exist.
* **Parameters** (`*_parameters.csv`): columns `name, value, description` with every scalar parameter of the
  calculation. A unit, if any, is part of the name (`_m`, `_eV`, `_s`, `_V_per_m`, …); everything else is
  dimensionless.

## Notation

| Name | Meaning |
|---|---|
| `ell` | sideband order ℓ (energy change ℓħω; positive = gain) |
| `zeta` | normalized propagation distance ζ = ωz/v |
| `zeta_int` | interaction distance ζ (Fig. 8) |
| `P` | sideband probability P_ℓ = \|φ_ℓ\|² |
| `phi` | complex sideband amplitude φ_ℓ (`Re_phi`, `Im_phi`) |
| `a`, `g` | recoil parameter a = 2mₑv²γ³/ħω and coupling strength g = evE₀/ħω² |
| `d_zeta`, `d_phi` | propagation step and field-phase step per propagation step (`d_phi = −d_zeta`: phase matching) |
| `delta_v` | velocity mismatch δ_v = v/v_ph − 1 (fraction, not %) |
| `z_free_over_zT` | free-propagation distance in units of the Talbot distance, z_free/z_T = ζ_free/ζ_T, ζ_T = 2πa |
| `omega_t` | time ωt over one optical period |
| `density` | electron probability density \|φ(z_free, ωt)\|²; each column normalized to ∫ density d(ωt) = 1 |
| `DOC1` | degree of coherence DOC₁ |

## fig1/
* `fig1c_P.csv`, `fig1d_P.csv` — maps P_ℓ(ζ) of panels (c) (analytical J²_ℓ(2gζ), a → ∞) and (d) (a = 3000);
  rows `ell`, columns `zeta`.
* `fig1_parameters.csv`.

## fig2/
* `fig2a_dE.csv`, `fig2c_dE.csv`, `fig2e_dE.csv` — classical energy change ΔE/ħω; rows: time `t_fs` (fs), columns:
  initial phase offset `chi_rad` of each trajectory (the figure shows every second trajectory).
* `fig2b_P.csv`, `fig2d_P.csv`, `fig2f_P.csv` — quantum maps P_ℓ(ζ); rows `ell`, columns `zeta`.
* `fig2_rows.csv` — for each row of the figure: initial velocity `v0_over_vph`, velocity
  mismatch `delta_v` = v0/v_ph − 1 (0, −0.002, −0.005), `a`, `g`, truncation `l_max`.
* `fig2_parameters.csv`.

## fig3/
* `fig3_contours.csv` — column `phi_rad` (phase φ) and, for each ε₀/Ω², the upper and lower branches of δ_v
  (fraction; `nan` where the contour does not exist).
* `fig3bc_bounds.csv` — δ_v,0 (red line), δ_v,max and δ_v,min (red markers) of panels (b) and (c).
* `fig3_parameters.csv` — Ω²/ω² = 5.3 × 10⁻⁶ and the physical parameters it follows from.

## fig4/
* `fig4_P_a{a}_g{g}.csv` — 16 maps P_ℓ(ζ), one per panel (rows of the figure: a, columns: g); rows `ell`, columns `zeta`.
* `fig4_parameters.csv`.

## fig5/
* `fig5a_bounds.csv` — panel (a): `delta_v`, classical sideband bounds `l_max`, `l_min` (a = 10⁴, g = 0.1).
* `fig5b_P.csv` … `fig5e_P.csv` — maps P_ℓ(ζ) of panels (b)–(e); rows `ell`, columns `zeta`.
* `fig5_panels.csv` — `delta_v`, bounds `l_min`, `l_max` (horizontal lines) and truncation of panels (b)–(e).
* `fig5_parameters.csv` — includes `z_um_per_zeta` for the upper z axis (ħω = 2 eV).

## fig6/
* `fig6ab_P.csv` — map P(ζ) of panels (a) and (b) (b is a close-up of a); rows: simulation sideband index `ell_sim`
  (phase matching at ℓ_sim = 0), columns `zeta`. The published figure plots the vertical coordinate −ℓ_sim
  (labelled ℓ − ℓ_ph).
* `fig6c_P.csv` — the same for panel (c) (a → ∞).
* `fig6a_injected_sidebands.csv` — initial sidebands ℓ_sim of the nine electrons.
* `fig6_parameters.csv`.

## fig7/
* `fig7a_DOC1_max.csv` — panel (a): maximum of DOC₁ over free propagation; rows: interaction length `L_um` (µm),
  columns: kinetic energy `E_kin_eV` (eV).
* `fig7a_zfree_at_max.csv` — free-propagation distance (units of z_T) at which that maximum occurs.
* `fig7a_per_energy.csv` — `a`, `g` and sideband truncation `l_max` for every energy of the map.
* `fig7a_parameters.csv`.
* Blue cross (panels b–f) and green cross (panels g–k):
  * `fig7b_P.csv`, `fig7g_P.csv` — P_ℓ during the interaction; rows `ell`, columns `z_um` (µm, up to 1.2 L).
  * `fig7c_phi_at_L.csv`, `fig7h_phi_at_L.csv` — complex amplitudes φ_ℓ and P_ℓ at z = L.
  * `fig7d_density.csv`, `fig7i_density.csv` — density; rows `omega_t`, columns `z_free_over_zT`.
  * `fig7e_density_at_max.csv`, `fig7j_density_at_max.csv` — density at the maximum of DOC₁ (red line of d, i).
  * `fig7f_DOC1.csv`, `fig7k_DOC1.csv` — DOC₁ versus `z_free_over_zT`.
  * `fig7_blue_parameters.csv`, `fig7_green_parameters.csv` — energy, L, a, g, δ_v, z_T, maximum DOC₁, ….

## fig8/
* `fig8a_P.csv`, `fig8c_P.csv` — maps P_ℓ(ζ_int); rows `ell`, columns `zeta_int`.
* `fig8b_DOC1.csv`, `fig8d_DOC1.csv` — DOC₁ after the interaction distance ζ_int (columns `zeta_int`) followed by free
  propagation (rows `z_free_over_zT`).
* `fig8e_phi.csv`, `fig8h_phi.csv`, `fig8k_phi.csv` — state φ_ℓ after the interaction (ζ_int = 25, 250, 244).
* `fig8e_density.csv`, `fig8h_density.csv`, `fig8k_density.csv` — density; rows `omega_t`, columns `z_free_over_zT`.
* `fig8f_density_at_max.csv`, `fig8i_density_at_max.csv`, `fig8l_density_at_max.csv` — density at the maximum of DOC₁.
* `fig8g_DOC1.csv`, `fig8j_DOC1.csv`, `fig8m_DOC1.csv` — DOC₁ versus `z_free_over_zT`.
* `fig8_parameters.csv` — parameters of each panel group (suffix `_ab`, `_cd`, `_efg`, `_hij`, `_klm`).

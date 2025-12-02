#!/usr/bin/env python3
"""
Streamlit STD-NMR simulator GUI

Converted from the original Dash-based app.
"""

import numpy as np
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------
# Global constants / helpers  (same as your Dash app)
# ------------------------------------------------------------

PPM_MIN = 0.0
PPM_MAX = 10.0
N_POINTS = 6000
ppm_axis = np.linspace(PPM_MIN, PPM_MAX, N_POINTS)

DEFAULT_INO_PPM = np.array([
    4.047, 4.047,  # H1, H2
    3.609,         # H3
    3.520, 3.520,  # H4, H5
    3.264          # H6
])

LIGAND_NONEXCH_H = 6.0
PROTEIN_H_PER_KDA = 63.0

LIGAND_VISIBILITY_FACTOR = 9.0

SATURATION_K0 = 0.3
SAT_REF_DB = -50.0

CONTACT_NONLINEAR_GAMMA = 1.8

DEFAULT_NAMES = ["H1", "H2", "H3", "H4", "H5", "H6"]
DEFAULT_PROX = [1.0, 0.8, 0.8, 0.5, 0.2, 0.0]


def lorentzian(x, x0, intensity=1.0, lw=0.02):
    gamma = lw / 2.0
    return intensity * (gamma ** 2 / ((x - x0) ** 2 + gamma ** 2))


def tau_c_ns_from_MW(MW_kDa: float) -> float:
    return 0.6 * MW_kDa


def effective_T2_ms(MW_kDa: float) -> float:
    tau_c = tau_c_ns_from_MW(MW_kDa)
    tau_c = max(tau_c, 1.0)
    return 300.0 / tau_c


def effective_T1rho_ms(MW_kDa: float, t1rho_scale: float) -> float:
    return max(0.1, t1rho_scale * effective_T2_ms(MW_kDa))


def protein_linewidth_from_MW(MW_kDa: float) -> float:
    tau_c = tau_c_ns_from_MW(MW_kDa)
    tau_c_ref = tau_c_ns_from_MW(10.0)
    lw_ref = 0.02
    return lw_ref * (tau_c / tau_c_ref)


def approx_protein_nonexchangeable_protons(MW_kDa: float) -> float:
    return PROTEIN_H_PER_KDA * MW_kDa


def saturation_factor(power_dB: float, duration_s: float, k0: float = SATURATION_K0) -> float:
    duration_s = max(duration_s, 0.0)
    power_dB = float(power_dB)
    rel_db = power_dB - SAT_REF_DB
    rel_power = 10.0 ** (rel_db / 20.0)
    x = k0 * (rel_power ** 2) * duration_s
    if x <= 0.0:
        return 0.0
    sf = 1.0 - np.exp(-x)
    return float(np.clip(sf, 0.0, 1.0))


def simulate_ligand_spectrum(ppm_axis, peaks_ppm, linewidth=0.02):
    spec = np.zeros_like(ppm_axis)
    for shift in peaks_ppm:
        spec += lorentzian(ppm_axis, shift, intensity=1.0, lw=linewidth)
    return spec


def simulate_protein_spectrum(ppm_axis, MW_kDa, n_peaks=80, seed=1234):
    rng = np.random.default_rng(seed)
    lw = protein_linewidth_from_MW(MW_kDa)

    regions = [
        (0.0, 1.5, 0.35),
        (1.5, 3.0, 0.25),
        (3.0, 5.0, 0.20),
        (6.0, 8.5, 0.15),
        (8.5, 10.0, 0.05),
    ]
    weights = np.array([r[2] for r in regions])
    weights /= weights.sum()
    peaks_per_region = rng.multinomial(n_peaks, weights)

    spec = np.zeros_like(ppm_axis)

    for (ppm_lo, ppm_hi, _), n_reg_peaks in zip(regions, peaks_per_region):
        if n_reg_peaks == 0:
            continue
        positions = rng.uniform(ppm_lo, ppm_hi, n_reg_peaks)

        if ppm_hi <= 1.5:
            amp_min, amp_max = 0.6, 1.0
        elif ppm_hi <= 3.0:
            amp_min, amp_max = 0.4, 0.9
        elif ppm_hi <= 5.0:
            amp_min, amp_max = 0.3, 0.8
        elif ppm_hi <= 8.5:
            amp_min, amp_max = 0.2, 0.7
        else:
            amp_min, amp_max = 0.1, 0.6

        intensities = rng.uniform(amp_min, amp_max, n_reg_peaks)
        for pos, amp in zip(positions, intensities):
            spec += lorentzian(ppm_axis, pos, intensity=amp, lw=lw)

    envelope_centers = [1.0, 4.0, 7.5]
    envelope_lw = lw * 4.0 + 0.3
    for c in envelope_centers:
        spec += lorentzian(ppm_axis, c, intensity=1.0, lw=envelope_lw)

    return spec


def apply_T2_filter(spectrum, MW_kDa, T2_filter_ms):
    T2_ms = effective_T2_ms(MW_kDa)
    T2_ms = max(T2_ms, 1.0)
    return spectrum * np.exp(-T2_filter_ms / T2_ms)


def apply_T1rho_filter(spectrum, MW_kDa, spinlock_ms, t1rho_scale):
    T1rho_ms = effective_T1rho_ms(MW_kDa, t1rho_scale)
    return spectrum * np.exp(-spinlock_ms / T1rho_ms)


def simulate_STD(
    ppm_axis,
    ligand_peaks_ppm,
    MW_kDa=50.0,
    ligand_lw=0.02,
    protein_sat_eff=0.8,
    sat_power_dB=-50.0,
    sat_duration_s=2.0,
    T2_filter_ms=0.0,
    T1rho_ms=0.0,
    t1rho_scale=0.7,
    n_protein_peaks=80,
    protein_conc_uM=20.0,
    ligand_conc_uM=4000.0,
    proximity_weights=None,
):
    ligand_peaks_ppm = np.array(ligand_peaks_ppm, dtype=float)
    n_protons = len(ligand_peaks_ppm)

    protein_nH = approx_protein_nonexchangeable_protons(MW_kDa)
    ligand_nH = LIGAND_NONEXCH_H
    ligand_scale = ligand_conc_uM * ligand_nH * LIGAND_VISIBILITY_FACTOR
    protein_scale = protein_conc_uM * protein_nH

    ligand_ref_raw = simulate_ligand_spectrum(ppm_axis, ligand_peaks_ppm, linewidth=ligand_lw)
    ligand_ref = ligand_ref_raw * ligand_scale

    protein_raw = simulate_protein_spectrum(ppm_axis, MW_kDa=MW_kDa, n_peaks=n_protein_peaks)
    protein_raw *= protein_scale

    off_raw = ligand_ref + protein_raw

    protein_filt = apply_T2_filter(protein_raw, MW_kDa, T2_filter_ms)
    protein_filt = apply_T1rho_filter(protein_filt, MW_kDa, T1rho_ms, t1rho_scale)

    off_filt = ligand_ref + protein_filt

    sat_factor = saturation_factor(sat_power_dB, sat_duration_s, SATURATION_K0)
    eff_sat = np.clip(protein_sat_eff * sat_factor, 0.0, 1.0)

    if proximity_weights is None or len(proximity_weights) != n_protons:
        proximity_weights = [1.0, 0.8, 0.8, 0.5, 0.2, 0.0][:n_protons]
    base_w = np.array([float(max(0.0, min(1.0, w))) for w in proximity_weights], dtype=float)

    nonzero = base_w > 0
    if np.any(nonzero):
        flattened = np.where(nonzero, 1.0, 0.0)
        contact_factors = (1.0 - sat_factor) * base_w + sat_factor * flattened
        max_cf = contact_factors.max()
        if max_cf > 0:
            contact_factors /= max_cf
    else:
        contact_factors = base_w

    contact_nl = np.power(contact_factors, CONTACT_NONLINEAR_GAMMA)

    ligand_on_raw = np.zeros_like(ppm_axis)
    for shift, contact in zip(ligand_peaks_ppm, contact_nl):
        atten = 1.0 - eff_sat * contact
        atten = max(0.0, atten)
        ligand_on_raw += lorentzian(ppm_axis, shift, intensity=atten, lw=ligand_lw)

    ligand_on = ligand_on_raw * ligand_scale
    protein_on = protein_filt * (1.0 - eff_sat)
    on_filt = ligand_on + protein_on

    std_spec = off_filt - on_filt

    global_max = max(
        np.max(np.abs(ligand_ref)),
        np.max(np.abs(protein_raw)),
        np.max(np.abs(off_raw)),
        np.max(np.abs(off_filt)),
        np.max(np.abs(on_filt)),
        np.max(np.abs(std_spec)),
    )
    if global_max > 0:
        ligand_ref  /= global_max
        protein_raw /= global_max
        off_raw     /= global_max
        off_filt    /= global_max
        on_filt     /= global_max
        std_spec    /= global_max

    return (
        ligand_ref,
        protein_raw,
        off_raw,
        off_filt,
        on_filt,
        std_spec,
        eff_sat,
    )


def make_figure(
    ligand_ref,
    protein_unfiltered,
    off_raw,
    off_filt,
    on_filt,
    std_spec,
):
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            "Ligand only (scaled by conc × #H × visibility factor)",
            "Protein only (unfiltered, scaled by conc × #H)",
            "Ligand+Protein mixture (no saturation, no filters)",
            "Mixture (OFF vs ON, both with filters)",
            "STD difference (OFF_filtered − ON_filtered)",
        ),
    )

    fig.add_trace(go.Scatter(x=ppm_axis, y=ligand_ref, name="Ligand"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ppm_axis, y=protein_unfiltered, name="Protein (unfiltered)"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=ppm_axis, y=off_raw, name="Mixture (no sat, no filters)"),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=ppm_axis, y=off_filt, name="OFF (filtered)"),
                  row=4, col=1)
    fig.add_trace(go.Scatter(x=ppm_axis, y=on_filt, name="ON (filtered)",
                             line=dict(dash="dot")),
                  row=4, col=1)
    fig.add_trace(go.Scatter(x=ppm_axis, y=std_spec, name="STD"), row=5, col=1)

    fig.update_xaxes(autorange="reversed")

    fig.update_layout(
        height=1100,
        showlegend=True,
        margin=dict(l=60, r=20, t=60, b=40),
        xaxis5_title="Chemical shift (ppm)",
        yaxis_title="Intensity (arb. u., normalized)",
        plot_bgcolor="#000000",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#000000"),
    )

    for i in range(1, 6):
        fig["layout"][f"xaxis{i}"].update(
            showgrid=True,
            gridcolor="#444444",
            zeroline=False,
            color="#000000",
        )
        fig["layout"][f"yaxis{i}"].update(
            showgrid=True,
            gridcolor="#444444",
            zeroline=False,
            color="#000000",
        )

    return fig


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="STD-NMR Simulator",
        layout="wide",
    )

    st.title("STD-NMR Simulator: ligand + protein")
    st.markdown(
        "Interactively explore how protein size, saturation power, filter times, "
        "and ligand epitope mapping affect STD spectra."
    )

    # --- Top-level controls (two columns band) ---
    col1, col2, col3 = st.columns(3)

    with col1:
        MW = st.number_input("Protein size (kDa)", min_value=5.0, max_value=300.0,
                             value=50.0, step=5.0)
        sat = st.number_input("Protein saturation efficiency (0–1)",
                              min_value=0.0, max_value=1.0, value=0.9, step=0.05)
        sat_power_dB = st.number_input("Saturation power (dB, negative; -40 to -60)",
                                       min_value=-80.0, max_value=-20.0, value=-50.0, step=1.0)
        sat_duration_s = st.number_input("Saturation duration (s)",
                                         min_value=0.0, max_value=10.0, value=2.0, step=0.1)

    with col2:
        T2_ms = st.number_input("T2 filter time (ms)",
                                min_value=0.0, max_value=200.0, value=0.0, step=5.0)
        T1rho_ms = st.number_input("T1ρ spin-lock time (ms)",
                                   min_value=0.0, max_value=400.0, value=0.0, step=10.0)
        T1rho_scale = st.number_input("T1ρ/T2 scaling (larger = weaker T1ρ filter)",
                                      min_value=0.2, max_value=2.0, value=0.7, step=0.1)
        ligand_lw = st.number_input("Ligand linewidth (FWHM, ppm)",
                                    min_value=0.005, max_value=0.1, value=0.02, step=0.005)

    with col3:
        n_peaks = st.number_input("Number of protein peaks",
                                  min_value=5, max_value=200, value=80, step=5)
        protein_conc = st.number_input("Protein concentration (µM)",
                                       min_value=0.1, max_value=500.0, value=20.0, step=1.0)
        lp_ratio = st.number_input("Ligand:protein ratio (ligand / protein)",
                                   min_value=1.0, max_value=5000.0, value=200.0, step=10.0)

    st.markdown("### Ligand proton definitions")

    st.write(
        "For each ligand proton, set its ppm position (0–10) and proximity weight "
        "(0–1; 1 = very close to binding site, 0 = non-interacting)."
    )

    # --- Ligand proton sliders ---
    ppm_vals = []
    prox_vals = []

    proton_cols = st.columns(3)
    for i in range(6):
        col = proton_cols[i % 3]
        with col:
            st.markdown(f"**Proton {i+1} ({DEFAULT_NAMES[i]})**")
            ppm = st.number_input(
                f"ppm {i+1}",
                min_value=0.0,
                max_value=10.0,
                value=float(DEFAULT_INO_PPM[i]),
                step=0.001,
                key=f"ppm_{i+1}",
            )
            prox = st.slider(
                f"Proximity {i+1}",
                min_value=0.0,
                max_value=1.0,
                value=float(DEFAULT_PROX[i]),
                step=0.05,
                key=f"prox_{i+1}",
            )
            ppm_vals.append(ppm)
            prox_vals.append(prox)
            st.caption(f"Weight = {prox:.2f}")

    ligand_conc = protein_conc * lp_ratio

    (
        ligand_ref,
        protein_unfiltered,
        off_raw,
        off_filt,
        on_filt,
        std_spec,
        eff_sat,
    ) = simulate_STD(
        ppm_axis,
        ligand_peaks_ppm=ppm_vals,
        MW_kDa=MW,
        ligand_lw=ligand_lw,
        protein_sat_eff=sat,
        sat_power_dB=sat_power_dB,
        sat_duration_s=sat_duration_s,
        T2_filter_ms=T2_ms,
        T1rho_ms=T1rho_ms,
        t1rho_scale=T1rho_scale,
        n_protein_peaks=int(n_peaks),
        protein_conc_uM=protein_conc,
        ligand_conc_uM=ligand_conc,
        proximity_weights=prox_vals,
    )

    fig = make_figure(
        ligand_ref,
        protein_unfiltered,
        off_raw,
        off_filt,
        on_filt,
        std_spec,
    )

    st.markdown("---")
    st.subheader("Simulated spectra")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"**Ligand concentration:** {ligand_conc:.1f} µM "
        f"(from {protein_conc:.1f} µM protein × ratio {lp_ratio:.1f}×)"
    )
    st.markdown(
        f"_Effective saturation efficiency ≈ {eff_sat:.2f}_ "
        f"(power = {sat_power_dB:.1f} dB, duration = {sat_duration_s:.2f} s, "
        f"intrinsic eff = {sat:.2f})"
    )


if __name__ == "__main__":
    main()

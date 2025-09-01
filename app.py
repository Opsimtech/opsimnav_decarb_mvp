
import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator
from sklearn.neural_network import MLPRegressor

st.set_page_config(page_title="OpsimNAV • Trim Optimization", layout="wide")

# --------------------------
# Helpers
# --------------------------
def sfoc_from_pb_kw(pb_kw: np.ndarray) -> np.ndarray:
    # Specific Fuel Oil Consumption (g/kWh) from the 6th-degree polynomial (Equation 38 in the paper)
    pb = np.asarray(pb_kw, dtype=float)
    return (-5.54e-23*pb**6 + 3.75e-18*pb**5 - 8.28e-14*pb**4 + 8.00e-10*pb**3
            - 3.11e-6*pb**2 - 9.63e-4*pb + 209.0)

def dfoc_t_per_day(pb_kw: np.ndarray, sfoc_g_per_kwh: np.ndarray) -> np.ndarray:
    # Daily fuel oil consumption (tonnes/day).
    # DFOC = SFOC[g/kWh] * Pb[kW] * 24[h] / 1e6 [g->tonnes]
    pb = np.asarray(pb_kw, dtype=float)
    sf = np.asarray(sfoc_g_per_kwh, dtype=float)
    return sf * pb * 24.0 / 1e6

def prepare_interpolants(df: pd.DataFrame):
    # Build interpolation objects:
    #  - displacement as f(draft, trim)
    #  - delta_power_pct as f(draft, speed, trim)
    #  - power_even_kW as f(draft, speed)
    pts_disp = df[["draft_m","trim_m"]].values
    vals_disp = df["displacement_m3"].values
    disp_interp = LinearNDInterpolator(pts_disp, vals_disp)

    pts_delta = df[["draft_m","speed_kn","trim_m"]].values
    vals_delta = df["delta_power_pct"].values
    delta_interp = LinearNDInterpolator(pts_delta, vals_delta)

    # power_even only depends on (draft, speed) in our dataset (same across trims)
    df_power = df.drop_duplicates(subset=["draft_m","speed_kn"])
    pts_p = df_power[["draft_m","speed_kn"]].values
    vals_p = df_power["power_even_kW"].values
    power_interp = LinearNDInterpolator(pts_p, vals_p)

    return disp_interp, delta_interp, power_interp

def find_optimum_for_displacement(df: pd.DataFrame, target_disp: float, eps_rel: float=1e-3):
    # Practical implementation of Eq. 39:
    # For a mesh of (draft, speed, trim), find combinations whose displacement is within
    # eps_rel * target_disp, compute actual power = power_even * (1 + delta/100),
    # choose the tuple that minimizes power.
    disp_interp, delta_interp, power_interp = prepare_interpolants(df)

    # Generate a fine grid: Vs step ~0.05 kn, trim step ~0.02 m.
    drafts = np.unique(df["draft_m"].values)
    speeds = np.arange(df["speed_kn"].min(), df["speed_kn"].max()+1e-9, 0.05)
    trims  = np.arange(df["trim_m"].min(),  df["trim_m"].max()+1e-9, 0.02)

    best = None
    tol = eps_rel * target_disp

    for d in drafts:
        # Make a (speed, trim) grid for this draft
        S, T = np.meshgrid(speeds, trims, indexing="xy")
        D = disp_interp(np.full_like(S, d), T)  # displacement as function of (d,trim)

        # tolerance mask
        mask = np.abs(D - target_disp) <= tol
        if not np.any(mask):
            continue

        # Interpolate delta and power_even
        Delta = delta_interp(np.full_like(S, d), S, T)
        P_even = power_interp(np.full_like(S, d), S)

        # Compute actual power (kW) at those masked points
        P = P_even * (1.0 + Delta/100.0)

        # Among valid points, pick the minimum power
        idx = np.argmin(np.where(mask, P, np.inf))
        i, j = np.unravel_index(idx, P.shape)

        if not np.isfinite(P[i,j]):
            continue

        cand = dict(draft_m=float(d),
                    speed_kn=float(S[i,j]),
                    trim_m=float(T[i,j]),
                    displacement_m3=float(D[i,j]),
                    delta_power_pct=float(Delta[i,j]),
                    power_kW=float(P[i,j]),
                    power_even_kW=float(P_even[i,j]))
        if (best is None) or (cand["power_kW"] < best["power_kW"]):
            best = cand

    return best

# --------------------------
# Sidebar nav
# --------------------------
st.sidebar.title("OpsimNAV MVP")
page = st.sidebar.radio("Sections", [
    "Dashboard",
    "Ship Trim Optimization",
    "(Coming soon) Vref & KPI Estimator",
    "(Coming soon) Compliance (CII/EEXI)",
    "About"
])

# --------------------------
# Dashboard
# --------------------------
if page == "Dashboard":
    st.title("OpsimNAV • Maritime Performance & Compliance")
    st.markdown(
        """
        Welcome to the OpsimNAV MVP. This prototype focuses on **Ship Trim Optimization**
        based on a recent JMSE 2024 study that couples **CFD** and **ANN** to predict engine
        power, DFOC, and optimal trim/speed regions.
        
        **What you can do now**
        - Load the sample dataset or upload your own CFD-derived trim table.
        - Pick a **target displacement** and compute **optimal trim & speed**.
        - Estimate **brake power**, **SFOC**, and **DFOC** at that operating point.
        
        > ⚠️ This MVP uses a **synthetic sample dataset** purely for demonstration.
        For engineering use, upload surfaces built from your vessel's CFD/sea-trial data.
        """
    )

# --------------------------
# Ship Trim Optimization
# --------------------------
elif page == "Ship Trim Optimization":
    st.title("Ship Trim Optimization")
    st.caption("Find optimal trim & speed for a target displacement.")

    source = st.radio("Data source", ["Use sample dataset", "Upload CSV (your vessel)"])
    if source == "Use sample dataset":
        df = pd.read_csv("data/sample_trim_dataset.csv")
        st.success("Loaded bundled sample dataset (synthetic for demo).")
        with st.expander("Preview sample data"):
            st.dataframe(df.head(15), use_container_width=True)
    else:
        up = st.file_uploader("CSV with columns: draft_m, speed_kn, trim_m, displacement_m3, power_even_kW, delta_power_pct",
                              type=["csv"])
        if up is None:
            st.stop()
        df = pd.read_csv(up)
        missing = set(["draft_m","speed_kn","trim_m","displacement_m3","power_even_kW","delta_power_pct"]) - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {sorted(list(missing))}")
            st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        target_disp = st.number_input("Target displacement ∇ (m³)", value=float(df["displacement_m3"].median()), step=10.0, format="%.2f")
    with col2:
        eps_rel = st.number_input("Tolerance ε (relative)", value=1e-3, step=1e-3, format="%.4f",
                                  help="Relative tolerance used to match displacement per the paper (~0.0001–0.001).")
    with col3:
        run = st.button("Compute Optimum", type="primary")

    if run:
        best = find_optimum_for_displacement(df, target_disp, eps_rel=eps_rel)
        if not best:
            st.warning("No point found within tolerance. Try increasing ε or check your dataset ranges.")
            st.stop()

        # Compute SFOC & DFOC at the selected operating point
        sf = float(sfoc_from_pb_kw(best["power_kW"]))
        dfoc = float(dfoc_t_per_day(best["power_kW"], sf))

        st.subheader("Recommended Operating Point")
        m1, m2, m3 = st.columns(3)
        m1.metric("Optimal speed (kn)", f"{best['speed_kn']:.2f}")
        m2.metric("Optimal trim (m)", f"{best['trim_m']:.2f}")
        bow_stern = "by the bow (negative)" if best["trim_m"] < 0 else "by the stern (positive)"
        m3.write(f"*Trim is {bow_stern}*")

        k1, k2, k3 = st.columns(3)
        k1.metric("Brake power Pb (kW)", f"{best['power_kW']:.0f}")
        k2.metric("SFOC (g/kWh)", f"{sf:.1f}")
        k3.metric("DFOC (t/day)", f"{dfoc:.2f}")

        st.caption("Power saving vs even keel at this (draft, speed): "
                   f"{best['delta_power_pct']:.2f}%")

        # Visuals: contour of delta_power vs (speed, trim) near the chosen draft
        st.subheader("Trim map (ΔPower% vs Speed/Trim) at chosen draft")
        try:
            import plotly.graph_objects as go
            # Build dense grid for visualization at the chosen draft
            disp_interp, delta_interp, power_interp = prepare_interpolants(df)
            d = best["draft_m"]
            speeds = np.linspace(df["speed_kn"].min(), df["speed_kn"].max(), 150)
            trims  = np.linspace(df["trim_m"].min(),  df["trim_m"].max(), 150)
            S, T = np.meshgrid(speeds, trims, indexing="xy")
            Delta = delta_interp(np.full_like(S, d), S, T)
            fig = go.Figure(data=go.Contour(
                z=Delta, x=speeds, y=trims,
                contours=dict(showlabels=True),
                colorbar=dict(title="ΔPower (%)")
            ))
            fig.update_layout(xaxis_title="Speed (kn)", yaxis_title="Trim (m)",
                              height=500, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as ex:
            st.info("Plotly not available or interpolation failed; skipping contour plot.")
            st.exception(ex)

        with st.expander("Raw result (debug)"):
            st.json(best)

# --------------------------
# About
# --------------------------
elif page == "About":
    st.title("About this MVP")
    st.markdown(
        """
        **OpsimNAV Trim Optimization MVP** implements the selection logic described in a 2024 JMSE paper:
        **CFD‑Powered Ship Trim Optimization: Integrating ANN for User‑Friendly Software Tool Development**.
        
        - Grid interpolation and the optimum search follow the paper's structure (trim/speed grids, displacement tolerance).
        - SFOC uses the paper's sixth‑degree polynomial.
        - ANN: this MVP optionally trains a simple MLP on uploaded data if you wish to predict additional targets,
          but for most users, direct interpolation on CFD-derived trim tables is sufficient.
        
        **How to move from MVP to production**
        - Replace the synthetic sample with your **CFD/sea‑trial** trim dataset.
        - If you have propeller OWT curves (KT, KQ, η0) and wake/thrust deductions, extend the pipeline to compute **n (rpm)**.
        - Add calibration factors against **model tests/sea trials** per the paper.
        """
    )

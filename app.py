
import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator, griddata
import plotly.graph_objects as go

st.set_page_config(page_title="OpsimNAV • Trim Optimization", page_icon="🛳️", layout="wide")

# ====== Styles ======
st.markdown("""
<style>
/* remove Streamlit default padding for a tight hero */
.block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
.hero {
  background: radial-gradient(1200px 400px at 15% 0%, rgba(40,99,163,.35), rgba(0,0,0,0)) ,
              linear-gradient(90deg, rgba(30,136,229,.20), rgba(30,136,229,.05) 60%, rgba(30,136,229,0));
  border: 1px solid rgba(255,255,255,.06);
  border-radius: 20px;
  padding: 18px 22px;
  margin-bottom: 14px;
}
.metric-card {
  border: 1px solid rgba(255,255,255,.06);
  border-radius: 16px;
  padding: 14px;
  background: rgba(255,255,255,.02);
}
.help {
  font-size: 0.9rem; color: #9FB4C7; margin-top: -8px;
}
.small { font-size: 0.85rem; color: #9FB4C7; }
</style>
""", unsafe_allow_html=True)

def logo_header():
    left, mid, right = st.columns([1,3,1])
    with mid:
        st.image("assets/logo.png", use_column_width=True)

logo_header()
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown("### Maritime Performance & Compliance — **Ship Trim Optimization**")
st.caption("CFD/ANN‑inspired optimum search over Speed × Trim with displacement tolerance, SFOC polynomial, and DFOC estimates.")
st.markdown('</div>', unsafe_allow_html=True)

# ====== Sidebar ======
with st.sidebar:
    st.header("Navigation")
    page = st.radio("", ["Ship Trim Optimization", "Data Explorer", "About"], key="page")

# ====== Utilities ======
def sfoc_from_pb_kw(pb):
    pb = np.asarray(pb, dtype=float)
    return (-5.54e-23*pb**6 + 3.75e-18*pb**5 - 8.28e-14*pb**4 + 8.00e-10*pb**3
            - 3.11e-6*pb**2 - 9.63e-4*pb + 209.0)

def dfoc_t_per_day(pb_kw, sfoc_g_per_kwh):
    return sfoc_g_per_kwh * pb_kw * 24.0 / 1e6

def make_interpolators(df):
    disp_interp = LinearNDInterpolator(df[["draft_m","trim_m"]].values, df["displacement_m3"].values)
    delta_interp = LinearNDInterpolator(df[["draft_m","speed_kn","trim_m"]].values, df["delta_power_pct"].values)
    p_even_df = df.drop_duplicates(["draft_m","speed_kn"])
    power_interp = LinearNDInterpolator(p_even_df[["draft_m","speed_kn"]].values, p_even_df["power_even_kW"].values)
    return disp_interp, delta_interp, power_interp

def search_optimum(df, target_disp, eps_rel=1e-3):
    disp_i, delta_i, power_i = make_interpolators(df)
    drafts = np.unique(df["draft_m"].values)
    speeds = np.arange(df["speed_kn"].min(), df["speed_kn"].max()+1e-9, 0.05)
    trims  = np.arange(df["trim_m"].min(),  df["trim_m"].max()+1e-9, 0.02)
    best = None
    tol = eps_rel * target_disp
    for d in drafts:
        S, T = np.meshgrid(speeds, trims, indexing="xy")
        D = disp_i(np.full_like(S, d), T)
        mask = np.isfinite(D) & (np.abs(D - target_disp) <= tol)
        if not np.any(mask):
            continue
        Delta = delta_i(np.full_like(S, d), S, T)
        P_even = power_i(np.full_like(S, d), S)
        P = P_even * (1.0 + Delta/100.0)
        P_masked = np.where(mask, P, np.inf)
        idx = np.argmin(P_masked)
        if not np.isfinite(np.ravel(P_masked)[idx]):
            continue
        i, j = np.unravel_index(idx, P.shape)
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

def plot_trim_map(df, chosen_draft, target_disp=None, eps_rel=None, optimum=None):
    disp_i, delta_i, p_i = make_interpolators(df)
    speeds = np.linspace(df["speed_kn"].min(), df["speed_kn"].max(), 180)
    trims  = np.linspace(df["trim_m"].min(),  df["trim_m"].max(), 180)
    S, T = np.meshgrid(speeds, trims, indexing="xy")
    Delta = delta_i(np.full_like(S, chosen_draft), S, T)

    fig = go.Figure()
    fig.add_contour(
        z=Delta, x=speeds, y=trims, contours=dict(showlabels=True, labelfont=dict(size=10)),
        colorbar=dict(title="ΔPower (%)")
    )

    # Iso‑displacement overlay
    if target_disp is not None and eps_rel is not None:
        D = disp_i(np.full_like(S, chosen_draft), T)
        tol = eps_rel * target_disp
        iso = np.where(np.abs(D - target_disp) <= tol, Delta, np.nan)
        fig.add_contour(z=iso, x=speeds, y=trims, showscale=False,
                        contours=dict(coloring='lines', showlines=True), line=dict(width=3))

    # Optimum marker
    if optimum is not None and np.isfinite(optimum.get("speed_kn", np.nan)):
        fig.add_scatter(x=[optimum["speed_kn"]], y=[optimum["trim_m"]],
                        mode="markers+text",
                        text=["Optimum"],
                        textposition="top center",
                        marker=dict(size=12, symbol="x", line=dict(width=2)))

    fig.update_layout(
        height=520, margin=dict(l=10,r=10,t=20,b=10),
        xaxis_title="Speed (kn)", yaxis_title="Trim (m)"
    )
    return fig

# ====== Data source selection ======
@st.cache_data
def load_sample():
    return pd.read_csv("data/sample_trim_dataset.csv")

if "source" not in st.session_state:
    st.session_state.source = "Sample dataset"

if page == "Ship Trim Optimization":
    st.subheader("Data")
    source = st.radio("Choose data source", ["Sample dataset", "Upload CSV"], horizontal=True, key="source")
    if source == "Sample dataset":
        df = load_sample()
        st.info("Loaded bundled sample dataset (synthetic for demo).")
    else:
        up = st.file_uploader("Upload CSV with required columns", type=["csv"], accept_multiple_files=False)
        if up is None:
            st.stop()
        df = pd.read_csv(up)

    req = {"draft_m","speed_kn","trim_m","displacement_m3","power_even_kW","delta_power_pct"}
    missing = sorted(list(req - set(df.columns)))
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    with st.expander("Preview (first 20 rows)"):
        st.dataframe(df.head(20), use_container_width=True)

    # ====== Inputs ======
    st.subheader("Inputs")
    c1, c2, c3, c4 = st.columns([1.2,1,1,1.2])
    with c1:
        target_disp = st.number_input("Target displacement ∇ (m³)", value=float(df["displacement_m3"].median()), step=10.0, format="%.2f")
        st.markdown('<div class="help">Optimum will lie on the iso‑∇ curve within tolerance.</div>', unsafe_allow_html=True)
    with c2:
        eps_rel = st.number_input("Tolerance ε (relative)", value=1e-3, step=1e-3, format="%.4f")
    with c3:
        chosen_draft = st.selectbox("Visualization draft (m)", sorted(df["draft_m"].unique().tolist()))
    with c4:
        run = st.button("Compute Optimum", type="primary", use_container_width=True)

    # ====== Results ======
    if run:
        best = search_optimum(df, target_disp, eps_rel=eps_rel)
        if not best:
            st.warning("No point found within tolerance. Increase ε or expand your data ranges.")
            st.stop()

        sf = float(sfoc_from_pb_kw(best["power_kW"]))
        dfoc = float(dfoc_t_per_day(best["power_kW"], sf))

        st.subheader("Results")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Optimal speed (kn)", f"{best['speed_kn']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Optimal trim (m)", f"{best['trim_m']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Brake power Pb (kW)", f"{best['power_kW']:.0f}")
            st.markdown('</div>', unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("SFOC (g/kWh)", f"{sf:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with k2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("DFOC (t/day)", f"{dfoc:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with k3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Δ vs even‑keel (%)", f"{best['delta_power_pct']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="small">Trim is {}.</div>'.format("by the bow (negative)" if best["trim_m"]<0 else "by the stern (positive)"), unsafe_allow_html=True)

        # Tabs: Map & Charts, JSON, Table row
        tabs = st.tabs(["Trim map & iso‑∇", "Debug JSON", "Candidate summary"])
        with tabs[0]:
            fig = plot_trim_map(df, chosen_draft=float(best["draft_m"]), target_disp=target_disp, eps_rel=eps_rel, optimum=best)
            st.plotly_chart(fig, use_container_width=True)
        with tabs[1]:
            st.json(best, expanded=False)
        with tabs[2]:
            st.dataframe(pd.DataFrame([best]), use_container_width=True)

    else:
        st.info("Set inputs and click **Compute Optimum** to see results and graphics.")

elif page == "Data Explorer":
    df = pd.read_csv("data/sample_trim_dataset.csv")
    st.subheader("Sample Data Explorer")
    st.caption("Use this page as a quick sanity check for the bundled dataset. Upload your own on the main page.")
    st.dataframe(df.head(100), use_container_width=True)

elif page == "About":
    st.subheader("About")
    st.write("""
This MVP implements a displacement‑tolerant optimum search over speed/trim grids, estimates SFOC via a 6th‑degree polynomial,
and reports DFOC (t/day). Replace the bundled dataset with your CFD/sea‑trial trim tables for engineering use.
""")

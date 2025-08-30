
# opsimnav_decarb_mvp.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Opsimnav — Decarbonization MVP (CFD→AI→CII)", layout="wide")

@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def ensure_monotonic_speed(df: pd.DataFrame, speed_col: str) -> pd.DataFrame:
    df = df.sort_values(speed_col).drop_duplicates(speed_col)
    df = df[df[speed_col] > 0].copy()
    return df.reset_index(drop=True)

def compute_power_from_resistance(dfR: pd.DataFrame, speed_col: str, R_col: str, eta: float) -> pd.DataFrame:
    df = dfR.copy()
    df["power_kW"] = df[R_col] * df[speed_col] / max(eta, 1e-6) / 1000.0
    return df[[speed_col, "power_kW"]]

def fit_power_surrogate(speed: np.ndarray, power: np.ndarray, degree: int = 3):
    coeffs = np.polyfit(speed, power, deg=degree)
    def model(v):
        return np.polyval(coeffs, v)
    y_hat = model(speed)
    ss_res = np.sum((power - y_hat) ** 2)
    ss_tot = np.sum((power - np.mean(power)) ** 2) + 1e-8
    r2 = 1.0 - ss_res / ss_tot
    return model, coeffs, r2

def daily_metrics(speed_mps: np.ndarray, power_kW: np.ndarray, sfoc_g_per_kWh: float, ef_tco2_per_tfuel: float, dwt_t: float):
    fuel_kg_per_h = power_kW * sfoc_g_per_kWh / 1000.0
    fuel_t_per_day = fuel_kg_per_h * 24.0 / 1000.0
    co2_t_per_day = fuel_t_per_day * ef_tco2_per_tfuel
    nm_per_day = speed_mps * 86400.0 / 1852.0
    co2_g_per_nm = (co2_t_per_day * 1e6) / np.maximum(nm_per_day, 1e-9)
    aer = (co2_t_per_day * 1e6) / np.maximum(dwt_t * nm_per_day, 1e-9)
    return fuel_t_per_day, co2_t_per_day, nm_per_day, co2_g_per_nm, aer

def regulation_check(aer_value: float, a_thr: float, b_thr: float, c_thr: float, d_thr: float):
    if aer_value <= a_thr: return "A"
    if aer_value <= b_thr: return "B"
    if aer_value <= c_thr: return "C"
    if aer_value <= d_thr: return "D"
    return "E"

with st.sidebar:
    st.title("⚙️ Inputs")
    res_file = st.file_uploader("Resistance CSV (speed_mps, resistance_N)", type=["csv"], key="res")
    pow_file = st.file_uploader("Power CSV (speed_mps, shaft_power_kW)", type=["csv"], key="pow")
    st.subheader("Ship & Engine")
    dwt_t = st.number_input("DWT [t]", 1000.0, 400000.0, 35000.0, 1000.0)
    eta = st.slider("Propulsive efficiency η", 0.3, 0.8, 0.65, 0.01)
    sfoc = st.number_input("SFOC [g/kWh]", 150.0, 220.0, 175.0, 1.0)
    ef = st.number_input("Emission factor (tCO₂/t fuel)", 2.5, 3.5, 3.114, 0.001, format="%.3f")
    v_min = st.number_input("Min speed [m/s]", 0.5, 5.0, 2.0, 0.1)
    v_max = st.number_input("Max speed [m/s]", 3.0, 12.0, 8.0, 0.1)
    st.subheader("CII Targets (AER thresholds)")
    thr_A = st.number_input("A ≤", 0.5, 25.0, 3.0, 0.1)
    thr_B = st.number_input("B ≤", 0.5, 25.0, 4.0, 0.1)
    thr_C = st.number_input("C ≤", 0.5, 25.0, 5.0, 0.1)
    thr_D = st.number_input("D ≤", 0.5, 25.0, 6.0, 0.1)
    degree = st.select_slider("Polynomial degree for P(V)", [2,3,4,5], value=3)

sample_notice = False
if pow_file is not None:
    df_pow = ensure_monotonic_speed(load_csv(pow_file), "speed_mps")
    df_pow = df_pow.rename(columns={"shaft_power_kW":"power_kW"})
elif res_file is not None:
    df_res = ensure_monotonic_speed(load_csv(res_file), "speed_mps")
    df_pow = compute_power_from_resistance(df_res, "speed_mps", "resistance_N", eta)
else:
    sample_notice = True
    v_demo = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    p_demo = np.array([150, 420, 900, 1700, 2800, 4300, 6200], float)
    df_pow = pd.DataFrame({"speed_mps": v_demo, "power_kW": p_demo})

v_grid = np.linspace(v_min, v_max, 200)
model, coeffs, r2 = fit_power_surrogate(df_pow["speed_mps"].values, df_pow["power_kW"].values, degree=degree)
p_sur = model(v_grid)

fuel_t_day, co2_t_day, nm_day, co2_g_nm, aer = daily_metrics(v_grid, p_sur, sfoc, ef, dwt_t)
idx_min_aer = int(np.argmin(aer))

st.title("🌿 Opsimnav — Decarbonization MVP")
if sample_notice:
    st.info("No CSV uploaded — using demo dataset.")

st.metric("Best AER speed [m/s]", f"{v_grid[idx_min_aer]:.2f}")
st.metric("AER @ best [gCO₂/(dwt·nm)]", f"{aer[idx_min_aer]:.2f}")
grade = regulation_check(aer[idx_min_aer], thr_A, thr_B, thr_C, thr_D)
st.metric("CII Grade (proxy)", grade)
st.metric("Fuel @ best [t/day]", f"{fuel_t_day[idx_min_aer]:.2f}")

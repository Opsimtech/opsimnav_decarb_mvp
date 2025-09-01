# OpsimNAV • Ship Trim Optimization (MVP)

This Streamlit app reproduces the selection logic from the paper:

> **CFD‑Powered Ship Trim Optimization: Integrating ANN for User‑Friendly Software Tool Development** (JMSE 2024, 12, 1265)

It lets you:

- Load a **trim table** (speed × trim × draft) built from your CFD/sea‑trial campaign.
- Enter a **target displacement** and automatically compute **optimal trim & speed** (per Eq. 39 idea).
- Estimate **Pb (kW)**, **SFOC (g/kWh)**, and **DFOC (t/day)** using the paper's SFOC 6th‑degree polynomial (Eq. 38).

> The bundled dataset is **synthetic** for demo only. Replace it with your vessel's data before any decision-making.

---

## How to run locally

```bash
# 1) Create a fresh venv (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Launch
streamlit run app.py
```

Then open the local URL that Streamlit prints (usually http://localhost:8501).

---

## Expected CSV format (if you upload your own data)

Your CSV must have these columns (names must match exactly):

```
draft_m, speed_kn, trim_m, displacement_m3, power_even_kW, delta_power_pct
```

- **power_even_kW**: main engine **brake power at even keel** for the (draft, speed).
- **delta_power_pct**: relative power change at that (draft, speed, trim) vs even keel (negative = saving).

The app performs linear interpolation over the grid and searches for the minimum power point that matches your
**target displacement** within a relative tolerance ε.

---

## Deploy on Streamlit Community Cloud

1. Create a new **public GitHub repo**, add these files:
   - `app.py`
   - `requirements.txt`
   - `data/sample_trim_dataset.csv` (optional demo)
   - `README.md`
2. Go to https://share.streamlit.io, connect your GitHub, select the repo/branch, and set the **file** to `app.py`.

---

## Notes and extensions

- The paper recommends verifying CFD against **OWT** and **sea-trial/model-test** data and applying calibration factors.
- If you also have OWT curves (KT, KQ, η0), wake fraction `w`, and thrust deduction `t`, extend the pipeline to compute **propeller rpm (n)**.
- For ANN-based generalization to **non-simulated** conditions, you can train a simple MLP on your dataset inside the app,
  or export features and train offline, then load the model to predict `Pb`, `DFOC`, `n` for new points.

---

## License

The app code is MIT licensed. The included paper is open-access (CC BY 4.0); please cite the authors if you use their methodology.

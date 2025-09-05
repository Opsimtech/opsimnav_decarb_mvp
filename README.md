# OpsimNAV • Ship Trim Optimization (Pro UI)

A polished Streamlit MVP for **Ship Trim Optimization**, based on a JMSE 2024 study (CFD+ANN).  
This version adds a modern UI, branded header, metric cards, and high‑quality graphics.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate ; macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Upload CSV schema
```
draft_m, speed_kn, trim_m, displacement_m3, power_even_kW, delta_power_pct
```

- `power_even_kW`: brake power at even‑keel for (draft, speed)
- `delta_power_pct`: relative change vs even‑keel at (draft, speed, trim) [negative = saving]

> The bundled dataset is synthetic for demo.

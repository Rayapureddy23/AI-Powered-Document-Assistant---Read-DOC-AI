# Sharing ReadDoc AI with your supervisor — demo guide

## Option 1 — Live demo on your laptop (best experience)
Everything works: chat, EDA, experiments, results.

1. `venv\Scripts\activate` then `streamlit run app.py`
2. Share your screen in Teams/Zoom, or present in person.

## Option 2 — Same-network access (professor uses their own browser)
1. Run: `streamlit run app.py --server.address 0.0.0.0`
2. Find your IP: `ipconfig` → IPv4 Address (e.g. 192.168.1.42)
3. Professor opens: `http://192.168.1.42:8501` (must be on the same Wi-Fi).

## Option 3 — Streamlit Cloud (share a link, explore anywhere)
Ollama cannot run on Streamlit Cloud, so live answer generation is disabled
there — but the app degrades gracefully:
- Methodology, EDA (after uploading a PDF) and **Results** are fully interactive.
- Results auto-loads `data_seed/seed_results.csv` so the professor sees your
  real experiment scores, charts, heatmap and findings without running anything.

To refresh the seed data after your local runs:
1. Run all experiments locally, download `readdocai_results.csv` from Results.
2. Copy it to `data_seed/seed_results.csv` in the repo.
3. `git add data_seed/seed_results.csv && git commit -m "update demo seed" && git push`

What to tell the professor:
> "The link shows the full study — methodology with metric formulas, document
> EDA, and all experiment results with charts. Answer generation runs on a
> local model for reproducibility, so the live chat is demonstrated on my
> machine."

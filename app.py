"""
Tambay Finder — Streamlit UI
Run with: streamlit run app.py
Requires the pipeline module (pipeline.py) in the same directory.
Install deps: pip install streamlit plotly
"""

import time
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Import everything from your existing pipeline module.
# Rename your current script to pipeline.py first, or adjust the import below.
from pipeline import (
    search_google_cafes,
    fetch_google_details,
    classify_cafe_with_llm,
    upsert_cafe,
    init_db,
    build_dataframe,
    BookyScraper,
    DATABASE_FILE,
    TAMBAY_WEIGHTS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tambay Finder",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar — pipeline config ─────────────────────────────────────────────────
with st.sidebar:
    st.title("☕ Tambay Finder")
    st.caption("Find cafes perfect for long work & study sessions.")
    st.divider()

    st.subheader("Pipeline config")
    location = st.text_input("Location",        value="Makati, Philippines")
    max_cafes  = st.slider("Max cafes",   min_value=5, max_value=50, value=20, step=5)
    max_pages  = st.slider("Max pages",   min_value=1, max_value=10, value=3)
    driver_path = st.text_input("Edge WebDriver path", value="msedgedriver.exe")

    st.divider()
    st.subheader("Tambay score weights")
    with st.expander("Edit weights (must sum to 1.0)"):
        w_wifi    = st.slider("has_wifi",         0.0, 0.5, TAMBAY_WEIGHTS["has_wifi"],         0.05)
        w_outlets = st.slider("has_outlets",      0.0, 0.5, TAMBAY_WEIGHTS["has_outlets"],      0.05)
        w_quiet   = st.slider("is_quiet",         0.0, 0.5, TAMBAY_WEIGHTS["is_quiet"],         0.05)
        w_comfy   = st.slider("is_comfy",         0.0, 0.5, TAMBAY_WEIGHTS["is_comfy"],         0.05)
        w_space   = st.slider("is_spacious",      0.0, 0.5, TAMBAY_WEIGHTS["is_spacious"],      0.05)
        w_late    = st.slider("opens_until_late", 0.0, 0.5, TAMBAY_WEIGHTS["opens_until_late"], 0.05)
        weight_sum = round(w_wifi + w_outlets + w_quiet + w_comfy + w_space + w_late, 2)
        if abs(weight_sum - 1.0) > 0.01:
            st.warning(f"Weights sum to {weight_sum:.2f} — must equal 1.0")
        else:
            st.success(f"Weights sum to {weight_sum:.2f} ✓")

    run_button = st.button("▶ Run pipeline", type="primary", use_container_width=True)

# ── Helper: inject custom weights into the pipeline module at runtime ─────────
import pipeline as _pl
def _apply_weights():
    _pl.TAMBAY_WEIGHTS.update({
        "has_wifi":         w_wifi,
        "has_outlets":      w_outlets,
        "is_quiet":         w_quiet,
        "is_comfy":         w_comfy,
        "is_spacious":      w_space,
        "opens_until_late": w_late,
    })
    _pl.SEARCH_LOCATION = location
    _pl.MAX_PAGES       = max_pages

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Tambay Finder")
st.caption("Pipeline: Google Places API → Booky.ph scraper → GPT-4o-mini classifier → SQLite")

if run_button:
    if abs(weight_sum - 1.0) > 0.01:
        st.error("Fix weights before running — they must sum to exactly 1.0.")
        st.stop()

    _apply_weights()

    # ── Progress display ───────────────────────────────────────────────────────
    progress_bar = st.progress(0, text="Starting pipeline…")
    status_box   = st.empty()
    log_area     = st.expander("Pipeline log", expanded=False)
    logs: list[str] = []

    def log(msg: str):
        logs.append(msg)
        with log_area:
            st.text("\n".join(logs[-40:]))   # last 40 lines

    def step(pct: int, label: str):
        progress_bar.progress(pct, text=label)
        status_box.info(label)
        log(f"[{pct:>3}%] {label}")

    # ── Step 1: Google Places search ───────────────────────────────────────────
    step(5, "Searching Google Places…")
    google_results = search_google_cafes(max_cafes)
    log(f"       Found {len(google_results)} cafes.")

    # ── Step 2: Place details + Booky prices + LLM classify ───────────────────
    conn = sqlite3.connect(DATABASE_FILE)
    init_db(conn)
    cafes = []
    n = len(google_results)

    with BookyScraper() as scraper:
        for i, entry in enumerate(google_results, 1):
            name = entry.get("name", "Unknown")
            pct  = 10 + int(80 * i / n)

            step(pct, f"[{i}/{n}] Fetching details — {name}")
            cafe = fetch_google_details(entry["place_id"])
            if cafe is None:
                log(f"       Skipped (no details): {name}")
                continue

            step(pct, f"[{i}/{n}] Scraping Booky — {name}")
            cafe.min_price, cafe.max_price = scraper.scrape_price_range(name)
            log(f"       Price: ₱{cafe.min_price}–₱{cafe.max_price}")

            step(pct, f"[{i}/{n}] Classifying — {name}")
            classify_cafe_with_llm(cafe)
            log(f"       tambayable={cafe.tambayable}  score={cafe.tambay_score}")

            upsert_cafe(conn, cafe)
            cafes.append(cafe)

    conn.close()

    # ── Step 3: Build results ──────────────────────────────────────────────────
    step(95, "Building results…")
    df = build_dataframe(cafes)
    df.to_csv("cafes_results.csv", index=False)
    log("       Exported → cafes_results.csv")

    step(100, "Done!")
    status_box.success(f"Pipeline complete — processed {len(cafes)} cafes.")
    st.session_state["df"] = df

# ── Results (shown after pipeline OR on reload if session has data) ────────────
df: pd.DataFrame | None = st.session_state.get("df")

# Fallback: load from DB if the app is reloaded without re-running
if df is None:
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        raw = pd.read_sql("SELECT * FROM cafes", conn)
        conn.close()
        if not raw.empty:
            # Re-hydrate the dataframe into the expected shape
            raw["mid_price"] = raw.apply(
                lambda r: (r.min_price + r.max_price) / 2
                if pd.notna(r.min_price) and pd.notna(r.max_price) else None, axis=1
            )
            raw["tambay_pct"] = (raw["tambay_score"] / 10 * 100).round(1)
            df = raw
            st.info("Showing results from the last saved database run.")
    except Exception:
        pass

if df is not None and not df.empty:
    st.divider()

    # ── Metric cards ──────────────────────────────────────────────────────────
    tambayable_df = df[df["tambayable"].astype(bool)]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total cafes",       len(df))
    m2.metric("Tambayable",        len(tambayable_df),
              f"{len(tambayable_df)/len(df)*100:.0f}% of total")
    m3.metric("Avg tambay score",  f"{tambayable_df['tambay_score'].mean():.1f} / 10"
              if not tambayable_df.empty else "—")
    m4.metric("Avg Google rating", f"{df['rating'].mean():.2f}")
    m5.metric("Avg mid price",
              f"₱{df['mid_price'].dropna().mean():.0f}"
              if df["mid_price"].notna().any() else "—")

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([2, 2, 3])
    with col_f1:
        filter_mode = st.radio(
            "Show", ["All cafes", "Tambayable only", "Not tambayable"],
            horizontal=True,
        )
    with col_f2:
        sort_by = st.selectbox(
            "Sort by",
            ["tambay_score", "rating", "mid_price", "review_count"],
        )
    with col_f3:
        min_score = st.slider("Min tambay score", 0.0, 10.0, 0.0, 0.5)

    view = df.copy()
    if filter_mode == "Tambayable only":
        view = view[view["tambayable"].astype(bool)]
    elif filter_mode == "Not tambayable":
        view = view[~view["tambayable"].astype(bool)]
    view = view[view["tambay_score"] >= min_score]
    view = view.sort_values(sort_by, ascending=(sort_by == "mid_price"), na_position="last")

    # ── Results table ─────────────────────────────────────────────────────────
    st.subheader(f"Results — {len(view)} cafes")

    FEATURE_COLS = list(TAMBAY_WEIGHTS.keys())
    display_cols = ["name", "tambayable", "tambay_score", "rating",
                    "review_count", "min_price", "max_price"] + FEATURE_COLS

    available = [c for c in display_cols if c in view.columns]

    def _style(val):
        if isinstance(val, bool) or val in (0, 1, True, False):
            return "color: #0F6E56; font-weight:500" if val else "color: #A32D2D"
        return ""

    styled = (
        view[available]
        .rename(columns={
            "name":             "Cafe",
            "tambayable":       "Tambayable",
            "tambay_score":     "Score /10",
            "rating":           "Rating",
            "review_count":     "Reviews",
            "min_price":        "Min ₱",
            "max_price":        "Max ₱",
            "has_wifi":         "WiFi",
            "has_outlets":      "Outlets",
            "is_quiet":         "Quiet",
            "is_comfy":         "Comfy",
            "is_spacious":      "Spacious",
            "opens_until_late": "Late hrs",
        })
        .style.background_gradient(subset=["Score /10"], cmap="YlOrBr")
        .format({"Score /10": "{:.1f}", "Rating": "{:.1f}",
                 "Min ₱": lambda v: f"₱{int(v)}" if pd.notna(v) else "—",
                 "Max ₱": lambda v: f"₱{int(v)}" if pd.notna(v) else "—"})
        .map(_style, subset=["Tambayable", "WiFi", "Outlets", "Quiet",
                               "Comfy", "Spacious", "Late hrs"])
    )
    st.dataframe(styled, use_container_width=True, height=420)

    csv_bytes = view.drop(columns=["reason"], errors="ignore").to_csv(index=False).encode()
    st.download_button("Download CSV", csv_bytes, "cafes_results.csv", "text/csv")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.divider()
    ch1, ch2 = st.columns(2)

    with ch1:
        st.subheader("Tambay score distribution")
        fig_hist = px.histogram(
            view, x="tambay_score", nbins=10,
            color="tambayable",
            color_discrete_map={True: "#EF9F27", False: "#B4B2A9"},
            labels={"tambay_score": "Tambay score", "count": "Cafes"},
            template="simple_white",
        )
        fig_hist.update_layout(showlegend=True, margin=dict(t=10, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    with ch2:
        st.subheader("Score vs. rating")
        fig_scat = px.scatter(
            view, x="rating", y="tambay_score",
            color="tambayable", size="review_count",
            hover_name="name",
            color_discrete_map={True: "#EF9F27", False: "#B4B2A9"},
            labels={"rating": "Google rating", "tambay_score": "Tambay score"},
            template="simple_white",
        )
        fig_scat.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig_scat, use_container_width=True)

    # ── Feature breakdown (tambayable cafes only) ─────────────────────────────
    st.subheader("Feature breakdown — tambayable cafes")
    if not tambayable_df.empty:
        feat_counts = {
            col: int(tambayable_df[col].astype(int).sum())
            for col in FEATURE_COLS if col in tambayable_df.columns
        }
        feat_labels = {
            "has_wifi": "WiFi", "has_outlets": "Outlets", "is_quiet": "Quiet",
            "is_comfy": "Comfy", "is_spacious": "Spacious", "opens_until_late": "Late hours",
        }
        fig_feat = go.Figure(go.Bar(
            x=[feat_labels.get(k, k) for k in feat_counts],
            y=list(feat_counts.values()),
            marker_color="#EF9F27",
        ))
        fig_feat.update_layout(
            template="simple_white",
            yaxis_title="# tambayable cafes with feature",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    # ── Feature correlation with tambay score ─────────────────────────────────
    st.subheader("Feature correlation with tambay score")
    feat_present = [c for c in FEATURE_COLS if c in df.columns]
    if feat_present:
        corr = df[feat_present + ["tambay_score"]].corr()["tambay_score"].drop("tambay_score")
        fig_corr = go.Figure(go.Bar(
            x=corr.sort_values().index.tolist(),
            y=corr.sort_values().values.tolist(),
            marker_color=["#EF9F27" if v >= 0 else "#E24B4A" for v in corr.sort_values()],
        ))
        fig_corr.update_layout(
            template="simple_white",
            yaxis_title="Pearson correlation",
            xaxis_title="Feature",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Per-cafe detail expanders ──────────────────────────────────────────────
    st.divider()
    st.subheader("Cafe details")
    for _, row in view.iterrows():
        icon = "✅" if row.get("tambayable") else "❌"
        with st.expander(f"{icon}  {row['name']}  —  {row['tambay_score']:.1f}/10"):
            dc1, dc2 = st.columns(2)
            dc1.metric("Google rating",  f"{row['rating']:.1f}")
            dc1.metric("Reviews",        int(row["review_count"]))
            price_str = (
                f"₱{int(row['min_price'])}–₱{int(row['max_price'])}"
                if pd.notna(row.get("min_price")) else "—"
            )
            dc2.metric("Price range",  price_str)
            dc2.metric("Tambay score", f"{row['tambay_score']:.1f} / 10")

            feat_row = {feat_labels.get(k, k): bool(row.get(k, False)) for k in FEATURE_COLS if k in row}
            st.write(feat_row)

            if row.get("reason"):
                st.caption(f"AI reasoning: {row['reason']}")

else:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.info(
        "Configure the pipeline in the sidebar and click **▶ Run pipeline** to get started.  \n"
        "Results from previous runs will be loaded automatically from `cafes.db`."
    )
    with st.expander("How it works"):
        st.markdown("""
        1. **Google Places** — searches for cafes in your chosen location (up to N pages × 20 results).
        2. **Place details** — fetches reviews, ratings, and addresses per cafe.
        3. **Booky.ph scraper** — uses Selenium + Edge to extract menu price ranges.
        4. **GPT-4o-mini** — reads reviews and classifies each cafe on 6 tambay features.
        5. **SQLite** — persists everything to `cafes.db` for fast reloads.

        The **tambay score** is a weighted sum of boolean features:

        | Feature | Weight |
        |---|---|
        | has_wifi | 0.25 |
        | has_outlets | 0.25 |
        | is_quiet | 0.20 |
        | opens_until_late | 0.15 |
        | is_comfy | 0.10 |
        | is_spacious | 0.05 |
        """)
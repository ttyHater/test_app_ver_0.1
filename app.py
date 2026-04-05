import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io
import threading
import time
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tambay Finder",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0f0e0c;
    color: #e8e0d4;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1a1814 !important;
    border-right: 1px solid #2e2b26;
}
section[data-testid="stSidebar"] * {
    color: #c8bfb0 !important;
}
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input {
    background: #252219 !important;
    border: 1px solid #3a3630 !important;
    color: #e8e0d4 !important;
    border-radius: 6px;
}

/* Header */
.tambay-header {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #d4a853 0%, #f0c878 50%, #c8832a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.tambay-sub {
    font-size: 1rem;
    color: #7a7060;
    font-weight: 300;
    letter-spacing: 0.05em;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: #1a1814;
    border: 1px solid #2e2b26;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-val {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #d4a853;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: #6a6050;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}

/* Score badge */
.score-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}
.score-high  { background: #1a3320; color: #5dba7e; }
.score-mid   { background: #2e2410; color: #d4a853; }
.score-low   { background: #2e1414; color: #c05050; }

/* Run button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #c8832a, #d4a853) !important;
    color: #0f0e0c !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    width: 100%;
    transition: all 0.2s ease !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: #1a1814;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6a6050 !important;
    border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}
.stTabs [aria-selected="true"] {
    background: #2e2b26 !important;
    color: #d4a853 !important;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #2e2b26 !important;
    border-radius: 10px !important;
}

/* Progress / Status */
.status-box {
    background: #1a1814;
    border: 1px solid #2e2b26;
    border-left: 3px solid #d4a853;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    color: #c8bfb0;
    margin: 1rem 0;
}

/* Divider */
hr { border-color: #2e2b26 !important; }

/* Select/checkbox */
.stCheckbox label { color: #c8bfb0 !important; }
.stSelectbox div[data-baseweb="select"] > div {
    background: #252219 !important;
    border-color: #3a3630 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "has_wifi":         "📶 WiFi",
    "has_outlets":      "🔌 Outlets",
    "is_quiet":         "🤫 Quiet",
    "is_comfy":         "🛋️ Comfy",
    "is_spacious":      "📐 Spacious",
    "opens_until_late": "🌙 Late Hours",
}

TAMBAY_WEIGHTS = {
    "has_wifi": 0.25, "has_outlets": 0.25, "is_quiet": 0.20,
    "is_comfy": 0.10, "is_spacious": 0.05, "opens_until_late": 0.15,
}

def score_class(score):
    if score >= 7:  return "score-high"
    if score >= 4:  return "score-mid"
    return "score-low"


def render_metrics(df: pd.DataFrame):
    tambayable_df = df[df["tambayable"]]
    avg_score = tambayable_df["tambay_score"].mean() if not tambayable_df.empty else 0
    avg_rating = df["rating"].mean()
    price_data = df["mid_price"].dropna()
    avg_price = price_data.mean() if not price_data.empty else None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{len(df)}</div>
            <div class="metric-label">Cafes Scanned</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{len(tambayable_df)}</div>
            <div class="metric-label">Tambayable</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{avg_score:.1f}</div>
            <div class="metric-label">Avg Tambay Score</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        price_str = f"₱{avg_price:,.0f}" if avg_price else "N/A"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{price_str}</div>
            <div class="metric-label">Avg Mid Price</div>
        </div>""", unsafe_allow_html=True)


def chart_theme():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8bfb0", family="DM Sans"),
        xaxis=dict(gridcolor="#2e2b26", linecolor="#2e2b26"),
        yaxis=dict(gridcolor="#2e2b26", linecolor="#2e2b26"),
        colorway=["#d4a853", "#5dba7e", "#c05050", "#6a9fd4", "#a87ed4"],
    )


def render_charts(df: pd.DataFrame):
    col_l, col_r = st.columns(2)

    # ── Tambay score distribution ─────────────────────────────────────────────
    with col_l:
        st.markdown("##### Tambay Score Distribution")
        fig = px.histogram(
            df, x="tambay_score", nbins=10,
            color_discrete_sequence=["#d4a853"],
            labels={"tambay_score": "Tambay Score"},
        )
        fig.update_layout(**chart_theme(), showlegend=False,
                          margin=dict(t=10, b=10, l=0, r=0), height=260)
        fig.update_traces(marker_line_color="#0f0e0c", marker_line_width=1.5)
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature prevalence ────────────────────────────────────────────────────
    with col_r:
        st.markdown("##### Feature Prevalence")
        feat_cols = list(TAMBAY_WEIGHTS.keys())
        feat_pct = {FEATURE_LABELS[f]: df[f].mean() * 100 for f in feat_cols if f in df.columns}
        fig2 = go.Figure(go.Bar(
            x=list(feat_pct.values()),
            y=list(feat_pct.keys()),
            orientation="h",
            marker_color="#d4a853",
        ))
        fig2.update_layout(**chart_theme(), showlegend=False,
                           margin=dict(t=10, b=10, l=0, r=0), height=260)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Rating vs Tambay Score scatter ────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### Rating vs Tambay Score")
        fig3 = px.scatter(
            df, x="rating", y="tambay_score",
            color="tambayable",
            color_discrete_map={True: "#5dba7e", False: "#c05050"},
            hover_name="name",
            labels={"rating": "Google Rating", "tambay_score": "Tambay Score"},
            size_max=14,
        )
        fig3.update_layout(**chart_theme(), legend_title_text="Tambayable",
                           margin=dict(t=10, b=10, l=0, r=0), height=280)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Price range box ───────────────────────────────────────────────────────
    with col_b:
        st.markdown("##### Price Range by Tambayability")
        price_df = df.dropna(subset=["mid_price"])
        if not price_df.empty:
            fig4 = px.box(
                price_df, x="tambayable", y="mid_price",
                color="tambayable",
                color_discrete_map={True: "#5dba7e", False: "#c05050"},
                labels={"tambayable": "Tambayable", "mid_price": "Mid Price (₱)"},
            )
            fig4.update_layout(**chart_theme(), showlegend=False,
                               margin=dict(t=10, b=10, l=0, r=0), height=280)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No price data available for this chart.")


def render_top_cafes(df: pd.DataFrame):
    top = (
        df[df["tambayable"]]
        .sort_values("tambay_score", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    if top.empty:
        st.info("No tambayable cafes found. Try adjusting your search parameters.")
        return

    for _, row in top.iterrows():
        cls = score_class(row["tambay_score"])
        features_html = " ".join(
            f'<span style="background:#252219;border:1px solid #3a3630;'
            f'border-radius:4px;padding:2px 8px;font-size:0.75rem;color:#c8bfb0;">'
            f'{FEATURE_LABELS[f]}</span>'
            for f in TAMBAY_WEIGHTS if row.get(f, False)
        )
        price_str = (
            f"₱{int(row['min_price']):,} – ₱{int(row['max_price']):,}"
            if pd.notna(row.get("min_price")) and pd.notna(row.get("max_price"))
            else "Price N/A"
        )
        st.markdown(f"""
        <div style="background:#1a1814;border:1px solid #2e2b26;border-radius:12px;
                    padding:1.1rem 1.4rem;margin-bottom:0.8rem;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                    <span style="font-family:'Playfair Display',serif;font-size:1.15rem;
                                 color:#e8e0d4;font-weight:600;">{row['name']}</span>
                    <span style="margin-left:10px;font-size:0.8rem;color:#6a6050;">
                        ⭐ {row['rating']} &nbsp;·&nbsp; {price_str}
                    </span>
                </div>
                <span class="score-pill {cls}">{row['tambay_score']}/10</span>
            </div>
            <div style="margin:0.5rem 0;">{features_html}</div>
            <div style="font-size:0.82rem;color:#7a7060;font-style:italic;">{row.get('summary','')}</div>
        </div>
        """, unsafe_allow_html=True)


def render_full_table(df: pd.DataFrame):
    display_cols = ["name", "rating", "review_count", "tambay_score",
                    "tambayable", "min_price", "max_price"] + list(TAMBAY_WEIGHTS.keys())
    display_cols = [c for c in display_cols if c in df.columns]
    show_df = df[display_cols].copy()

    # Rename for readability
    show_df.rename(columns={
        "review_count": "Reviews",
        "tambay_score": "Score",
        "tambayable":   "Tambayable",
        "min_price":    "Min ₱",
        "max_price":    "Max ₱",
        **{k: v for k, v in FEATURE_LABELS.items()},
    }, inplace=True)

    bool_cols = [v for v in FEATURE_LABELS.values() if v in show_df.columns] + ["Tambayable"]
    for c in bool_cols:
        if c in show_df.columns:
            show_df[c] = show_df[c].map({True: "✅", False: "❌"})

    st.dataframe(show_df, use_container_width=True, hide_index=True)


def get_csv(df: pd.DataFrame) -> bytes:
    return df.drop(columns=["summary"], errors="ignore").to_csv(index=False).encode("utf-8")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ☕ Tambay Finder")
    st.markdown("---")

    st.markdown("**🔍 Search Settings**")
    search_location = st.text_input("Location", value="Makati Philippines")
    max_cafes       = st.number_input("Max Cafes", min_value=1, max_value=60, value=20, step=5)
    max_pages       = st.number_input("Max Pages", min_value=1, max_value=5, value=3, step=1)

    st.markdown("---")
    st.markdown("**🔑 API Keys**")
    google_key = st.text_input("Google Places API Key", type="password",
                               value=os.getenv("GOOGLE_API_KEY", ""))
    openai_key = st.text_input("OpenAI API Key", type="password",
                               value=os.getenv("OPENAI_API_KEY", ""))

    st.markdown("---")
    run_btn = st.button("▶  Run Pipeline", use_container_width=True)

    st.markdown("---")
    st.markdown("**📂 Load CSV**")
    uploaded = st.file_uploader("Upload previous results", type=["csv"])


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="tambay-header">Tambay Finder</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tambay-sub">Discover the best work/study cafes — powered by Google Places, Booky & GPT</div>',
    unsafe_allow_html=True,
)

# Session state
if "df" not in st.session_state:
    st.session_state["df"] = None
if "running" not in st.session_state:
    st.session_state["running"] = False
if "log_lines" not in st.session_state:
    st.session_state["log_lines"] = []

# Load from CSV upload
if uploaded is not None and st.session_state["df"] is None:
    try:
        df_loaded = pd.read_csv(uploaded)
        st.session_state["df"] = df_loaded
        st.success(f"Loaded {len(df_loaded)} cafes from CSV.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# ── Pipeline run ──────────────────────────────────────────────────────────────
if run_btn:
    if not google_key:
        st.error("Please enter your Google Places API Key in the sidebar.")
    elif not openai_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    else:
        st.session_state["running"] = True
        st.session_state["log_lines"] = []

        progress_bar  = st.progress(0)
        status_text   = st.empty()
        log_placeholder = st.empty()

        def update_progress(step, total, message):
            pct = int((step / total) * 100) if total > 0 else 0
            pct = min(pct, 100)
            progress_bar.progress(pct)
            status_text.markdown(
                f'<div class="status-box">⏳ {message}</div>',
                unsafe_allow_html=True,
            )
            st.session_state["log_lines"].append(message)
            # Update log display with last 15 lines
            with log_placeholder.container():
                with st.expander("📋 Live Log", expanded=True):
                    for line in st.session_state["log_lines"][-15:]:
                        st.markdown(f"› {line}")

        try:
            from pipeline import run_pipeline, PipelineConfig
            
            # Build config object
            cfg = PipelineConfig(
                search_query=search_query,
                search_location=search_location,
                max_cafes=int(max_cafes),
                max_pages=int(max_pages),
                google_api_key=google_key,
                openai_api_key=openai_key,
                driver_path=driver_path or "",
            )
            
            df_result = None
            step_count = 0
            
            # Consume the generator and update progress
            for event_type, payload in run_pipeline(cfg):
                if event_type == "log":
                    update_progress(step_count, 100, payload)
                    step_count += 1
                
                elif event_type == "progress":
                    current, total = payload
                    update_progress(current, total, f"Processing cafe {current}/{total}…")
                
                elif event_type == "cafe_done":
                    pass  # Optional: could update a cafe counter
                
                elif event_type == "result":
                    df_result = payload
                
                elif event_type == "error":
                    raise Exception(payload)

            if df_result is not None:
                progress_bar.progress(100)
                status_text.markdown(
                    '<div class="status-box" style="border-left-color:#5dba7e;">✅ Pipeline complete!</div>',
                    unsafe_allow_html=True,
                )
                st.session_state["df"] = df_result
            else:
                st.error("Pipeline completed but no results were returned.")
            
            st.session_state["running"] = False

        except ImportError as ie:
            st.error(f"Could not import pipeline: {ie}. Make sure pipeline.py is in the same directory as this app.")
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.session_state["running"] = False

# ── Results ───────────────────────────────────────────────────────────────────
df: pd.DataFrame = st.session_state.get("df")

if df is not None and not df.empty:
    st.markdown("---")

    # Boolean column fix (CSV loads bools as strings sometimes)
    bool_cols = list(TAMBAY_WEIGHTS.keys()) + ["tambayable"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: x if isinstance(x, bool)
                else str(x).strip().lower() in ("true", "1", "yes")
            )

    render_metrics(df)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🏆 Top Tambayable", "📊 Analytics", "📋 Full Table"])

    with tab1:
        render_top_cafes(df)

    with tab2:
        render_charts(df)

    with tab3:
        render_full_table(df)
        st.download_button(
            "⬇ Download CSV",
            data=get_csv(df),
            file_name="tambay_cafes.csv",
            mime="text/csv",
        )

elif not run_btn:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#4a4438;">
        <div style="font-size:4rem;margin-bottom:1rem;">☕</div>
        <div style="font-family:'Playfair Display',serif;font-size:1.4rem;color:#6a6050;">
            Ready to find your next tambay spot
        </div>
        <div style="font-size:0.9rem;margin-top:0.5rem;">
            Configure your search in the sidebar, then hit <strong style="color:#d4a853;">Run Pipeline</strong>.
        </div>
        <div style="font-size:0.85rem;margin-top:1.5rem;color:#3a3630;">
            Or upload a previous <code>cafes_results.csv</code> to explore existing results.
        </div>
    </div>
    """, unsafe_allow_html=True)
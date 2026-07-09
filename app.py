import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Retail Investor Terminal — Multi-Asset Optimizer", page_icon="🤖📈", layout="wide")


# --- Premium light-slate corporate theme CSS ---
st.markdown("""
<style>
  :root { color-scheme: light; }
  body, .stApp { background-color: #f8fafc; color: #0f172a; }
  .block-container { padding-top: 1.6rem; padding-bottom: 1.6rem; }
  h1, h2, h3, h4 { font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial; font-weight:600; letter-spacing:0.2px; color:#0f172a; }
  .stSidebar { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
  .card { background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:14px; box-shadow:0 6px 18px rgba(15,23,42,0.06); color:#0f172a; }
  .metric-row > div { padding: 6px; }
  .ai-box { background: linear-gradient(180deg,#ffffff,#fbfdff); border:1px solid #e2e8f0; padding:16px; border-radius:12px; box-shadow:0 10px 30px rgba(15,23,42,0.06); }
  .small-muted { color:#475569; font-size:13px; }
  table.dataframe { background: transparent; }
</style>
""", unsafe_allow_html=True)


# --- Asset universe and modelling inputs (unchanged) ---
assets = [
    "Domestic Equities (Nifty 50)",
    "International Equities (S&P 500)",
    "Fixed Income (Corporate Debt)",
    "Gold",
    "Liquid Cash",
]

expected_returns = np.array([0.12, 0.10, 0.08, 0.07, 0.05])
volatilities = np.array([0.16, 0.18, 0.05, 0.12, 0.01])

correlation_matrix = pd.DataFrame(
    [
        [1.00, 0.62, 0.08, -0.12, 0.03],
        [0.62, 1.00, -0.04, -0.08, 0.02],
        [0.08, -0.04, 1.00, 0.05, 0.14],
        [-0.12, -0.08, 0.05, 1.00, 0.00],
        [0.03, 0.02, 0.14, 0.00, 1.00],
    ],
    index=assets,
    columns=assets,
)

covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix.to_numpy()


def optimize_weights(risk_profile: str) -> np.ndarray:
    base_weights = {
        "Conservative": np.array([0.16, 0.10, 0.28, 0.16, 0.30]),
        "Moderate": np.array([0.30, 0.20, 0.22, 0.14, 0.14]),
        "Aggressive": np.array([0.36, 0.26, 0.13, 0.16, 0.09]),
    }
    aversion = {"Conservative": 2.8, "Moderate": 1.6, "Aggressive": 0.8}[risk_profile]
    w = base_weights[risk_profile].astype(float)
    for _ in range(250):
        gradient = expected_returns - 2 * aversion * (covariance_matrix @ w)
        w = w + 0.02 * gradient
        w = np.clip(w, 0.05, 0.40)
        w = w / w.sum()
    return w


st.title("Retail Investor Terminal — Multi-Asset Optimizer")
st.markdown("<div class='small-muted'>A friendly terminal that recommends diversified allocations and provides clear, actionable next steps.</div>", unsafe_allow_html=True)


# --- Sidebar inputs ---
with st.sidebar:
    st.header("Investor Inputs")
    investment = st.number_input("Investment Amount (INR)", min_value=10000, value=5000000, step=10000, format="%d")
    risk_profile = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"]) 
    st.markdown("---")
    st.subheader("Portfolio Logic")
    st.write("A risk-adjusted allocation engine balances return and volatility while preserving liquidity and diversification.")


# --- Core calculations (kept intact) ---
weights = optimize_weights(risk_profile)
allocation_inr = weights * investment
portfolio_return = float(weights @ expected_returns)
portfolio_volatility = float(np.sqrt(weights @ covariance_matrix @ weights))
portfolio_sharpe = (portfolio_return - 0.04) / portfolio_volatility if portfolio_volatility else 0.0


# --- Overview metrics section ---
st.markdown("## Overview")
col_t, col_r, col_v = st.columns([1,1,1])
with col_t:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Total Capital")
    st.markdown(f"<h2>₹{investment:,.0f}</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col_r:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Expected Annual Return")
    st.markdown(f"<h2>{portfolio_return*100:.2f}%</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col_v:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Risk Level")
    st.markdown(f"<h2>{risk_profile}</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)



# --- Allocation visualization and table ---
st.markdown("## Recommended Allocation")
left, right = st.columns([1.4, 0.8])
alloc_df = pd.DataFrame({
    "Asset Class": assets,
    "Recommended %": (weights * 100).round(2),
    "Amount (INR)": allocation_inr.round(0).astype(int),
})

with left:
    fig = px.bar(
        alloc_df,
        x="Recommended %",
        y="Asset Class",
        orientation="h",
        text="Recommended %",
        color="Recommended %",
        color_continuous_scale="Greys",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#f8fafc",
        font={"color": "#0f172a"},
        margin=dict(l=20, r=20, t=10, b=10),
        coloraxis_showscale=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e6e8eb", zeroline=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown('<div class="card"><h4 style="margin-top:0">Allocation Table</h4></div>', unsafe_allow_html=True)
    display_df = alloc_df.copy()
    display_df["Amount (INR)"] = display_df["Amount (INR)"].map(lambda v: f"₹{v:,}")
    st.table(display_df.set_index("Asset Class"))



# --- Correlation matrix ---
st.markdown("## Correlation Matrix")
st.write("How the asset classes move relative to each other (diversification helps reduce portfolio volatility).")
fig_corr = px.imshow(correlation_matrix.round(2), text_auto=True, color_continuous_scale="Blues", aspect="auto")
fig_corr.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#f8fafc", font={"color":"#0f172a"}, margin=dict(l=20,r=20,t=10,b=10))
fig_corr.update_xaxes(showgrid=True, gridcolor="#e6e8eb", zeroline=False)
fig_corr.update_yaxes(showgrid=True, gridcolor="#e6e8eb", zeroline=False)
st.plotly_chart(fig_corr, use_container_width=True)



# --- AI Suggestion Engine (deterministic template-based) ---
st.markdown("## 🤖 AI Investment Strategist Copilot Advice")
with st.container():
    st.markdown('<div class="ai-box">', unsafe_allow_html=True)
    st.subheader(f"Personalized Advice — {risk_profile} Profile")

    # Build rationale based on weights
    top_idx = int(np.argmax(weights))
    top_asset = assets[top_idx]
    top_weight = weights[top_idx]

    rationale_lines = []
    rationale_lines.append(f"The optimizer assigns a higher weight to {top_asset} ({top_weight*100:.1f}%) to align with the {risk_profile.lower()} objective.")
    fi_weight = weights[2]
    gold_weight = weights[3]
    cash_weight = weights[4]
    rationale_lines.append(f"Fixed Income ({fi_weight*100:.1f}%) provides income and downside protection during equity drawdowns.")
    rationale_lines.append(f"Gold ({gold_weight*100:.1f}%) is included as an inflation hedge and a risk-off diversifier.")
    rationale_lines.append(f"Liquid Cash ({cash_weight*100:.1f}%) preserves optionality for rebalancing or opportunistic deployment.")

    st.markdown("**Strategic Rationale**")
    for line in rationale_lines:
        st.write(line)

    # Macro benefits
    st.markdown("**Core Macroeconomic Benefits of This Mix**")
    macro_bullets = [
        "Exposure to global growth via international equities reduces single-market concentration risk.",
        "Domestic equities capture local market upside and currency-aligned returns.",
        "Corporate debt lowers portfolio volatility and contributes steady income when yields are favorable.",
        "Gold tends to perform in inflationary or risk-off periods, offering downside protection.",
        "Cash ensures liquidity for tactical rebalancing and emergency needs without forced selling.",
    ]
    for b in macro_bullets:
        st.write(f"- {b}")

    # Next steps action plan (3 steps)
    st.markdown("**3-Step Next Steps Action Plan**")
    steps = [
        "1) Establish core holdings: invest the recommended amounts as the base allocation and set rebalance rules.",
        "2) Build a 3–6 month cash buffer (as suggested) to avoid selling into market stress.",
        "3) Review quarterly: rebalance back to target weights and adjust if your personal risk budget changes.",
    ]
    for s in steps:
        st.write(s)

    # Customized quick notes
    st.markdown("**Quick Notes**")
    st.write(f"- Largest allocation: {top_asset} ({top_weight*100:.1f}%).")
    st.write(f"- Portfolio expected return: {portfolio_return*100:.2f}% · Expected volatility: {portfolio_volatility*100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

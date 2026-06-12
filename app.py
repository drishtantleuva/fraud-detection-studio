"""Fraud Detection Studio — interactive explainable fraud detection demo.

Live synthetic transaction feed scored by XGBoost, with SHAP waterfalls and
plain-English reasons for every flag. Inject classic fraud patterns and watch
the model catch them. All data is synthetic.
"""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from data_gen import SCENARIOS, FeedSimulator
from model import explain_row, plain_english, score, train_model

st.set_page_config(
    page_title="Fraud Detection Studio",
    page_icon="🕵️",
    layout="wide",
)


@st.cache_resource(show_spinner="Training fraud model on simulated history…")
def load_artifacts():
    return train_model()


ART = load_artifacts()


def init_state():
    if "sim" not in st.session_state:
        st.session_state.sim = FeedSimulator(seed=11)
        st.session_state.feed = pd.DataFrame()
        append_feed(st.session_state.sim.stream(40))


def append_feed(batch: pd.DataFrame):
    batch = batch.copy()
    batch["risk"] = score(ART["model"], batch)
    st.session_state.feed = pd.concat(
        [st.session_state.feed, batch], ignore_index=True
    )


init_state()
feed: pd.DataFrame = st.session_state.feed

# ---------------- sidebar ----------------

with st.sidebar:
    st.title("🕵️ Controls")

    st.subheader("Live feed")
    if st.button("▶ Stream 25 transactions", use_container_width=True):
        append_feed(st.session_state.sim.stream(25))
        st.rerun()

    st.subheader("Inject a fraud scenario")
    st.caption("Simulate an attack and watch the model catch it.")
    for key, label in SCENARIOS.items():
        if st.button(f"⚠️ {label}", use_container_width=True):
            append_feed(st.session_state.sim.inject(key))
            st.rerun()

    threshold = st.slider("Alert threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Transactions scoring above this are flagged.")

    if st.button("↺ Reset feed", use_container_width=True):
        for k in ("sim", "feed"):
            st.session_state.pop(k, None)
        st.rerun()

    m = ART["metrics"]
    st.divider()
    st.subheader("Model card")
    st.markdown(
        f"""
- **Model:** XGBoost classifier
- **Trained on:** {m['n_train']:,} simulated transactions
- **ROC-AUC:** {m['roc_auc']:.3f} · **PR-AUC:** {m['avg_precision']:.3f}
- **Explainability:** SHAP (TreeExplainer)
- **Data:** 100% synthetic — no real customers
"""
    )
    st.divider()
    st.markdown(
        "Built by **[Drishtant Leuva](https://www.linkedin.com/in/drishtant-leuva/)** · "
        "[source code](https://github.com/drishtantleuva/fraud-detection-studio)"
    )

# ---------------- header & metrics ----------------

st.title("Fraud Detection Studio")
st.caption(
    "Real-time card-fraud detection with explainable AI — every flag comes with "
    "a SHAP breakdown and a plain-English reason. Synthetic data, real techniques."
)

flagged = feed[feed["risk"] >= threshold]
injected = feed[feed["is_fraud"] == 1]
caught = injected[injected["risk"] >= threshold]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions processed", f"{len(feed):,}")
c2.metric("Flagged as suspicious", f"{len(flagged):,}")
c3.metric("Fraud injected", f"{len(injected):,}")
c4.metric(
    "Fraud caught",
    f"{len(caught):,}",
    f"{len(caught) / len(injected) * 100:.0f}% recall" if len(injected) else "no fraud yet",
)

# ---------------- risk scatter ----------------

plot_df = feed.assign(
    status=lambda d: d["risk"].ge(threshold).map({True: "Flagged", False: "Cleared"})
)
fig = px.scatter(
    plot_df,
    x="timestamp",
    y="amount",
    color="risk",
    symbol="status",
    symbol_map={"Flagged": "x", "Cleared": "circle"},
    color_continuous_scale=["#2fc8f5", "#7b5cff", "#ff5c87"],
    hover_data=["txn_id", "customer_id", "merchant", "category", "city"],
    log_y=True,
    height=380,
    labels={"amount": "Amount (AUD, log scale)", "timestamp": "Time"},
)
fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

# ---------------- feed table & investigation ----------------

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Transaction feed")
    show = feed.sort_values("timestamp", ascending=False).head(60)
    st.dataframe(
        show[["txn_id", "timestamp", "customer_id", "merchant", "city",
              "amount", "risk"]],
        column_config={
            "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "risk": st.column_config.ProgressColumn(
                "Risk", min_value=0, max_value=1, format="%.2f"
            ),
            "timestamp": st.column_config.DatetimeColumn(
                "Time", format="DD MMM HH:mm"
            ),
        },
        hide_index=True,
        use_container_width=True,
        height=420,
    )

with right:
    st.subheader("🔍 Investigate a flag")
    if flagged.empty:
        st.info("No transactions above the threshold yet. "
                "Inject a fraud scenario from the sidebar.")
    else:
        options = flagged.sort_values("risk", ascending=False)
        txn_id = st.selectbox(
            "Flagged transaction",
            options["txn_id"],
            format_func=lambda t: (
                f"{t} — ${options.loc[options.txn_id == t, 'amount'].iat[0]:,.2f} "
                f"(risk {options.loc[options.txn_id == t, 'risk'].iat[0]:.2f})"
            ),
        )
        row = feed.loc[feed["txn_id"] == txn_id].iloc[0]

        truth = "⚠️ injected fraud" if row["is_fraud"] else "legitimate (false positive)"
        scenario = SCENARIOS.get(row["scenario"], "")
        st.markdown(
            f"**{row['merchant']}** · {row['category']} · {row['city']} · "
            f"A${row['amount']:,.2f}\n\n"
            f"Customer `{row['customer_id']}` · risk **{row['risk']:.2f}** · "
            f"ground truth: **{truth}**"
            + (f" — *{scenario}*" if scenario else "")
        )

        explanation = explain_row(ART["explainer"], row)

        st.markdown("**Why the model flagged it:**")
        for reason in plain_english(explanation, row):
            st.markdown(f"- {reason}")

        fig_w, ax = plt.subplots()
        shap.plots.waterfall(explanation, max_display=9, show=False)
        st.pyplot(plt.gcf(), use_container_width=True)
        plt.close("all")

st.divider()
st.caption(
    "All customers, merchants and transactions are synthetically generated by "
    "[data_gen.py](https://github.com/drishtantleuva/fraud-detection-studio/blob/main/data_gen.py) — "
    "no real financial data is used. Fraud patterns are modelled on the public "
    "[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) and "
    "[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) "
    "research datasets."
)

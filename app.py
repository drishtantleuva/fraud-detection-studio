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

import branding
from data_gen import FEATURES, SCENARIOS, FeedSimulator
from model import explain_row, plain_english, score, train_model

st.set_page_config(
    page_title="Fraud Detection Studio",
    page_icon=":material/policy:",
    layout="wide",
)
branding.inject()


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
    st.title("Controls")

    st.subheader("Live feed")
    if st.button("Stream 25 transactions", use_container_width=True):
        append_feed(st.session_state.sim.stream(25))
        st.rerun()

    st.subheader("Inject a fraud scenario")
    st.caption("Simulate an attack and watch the model catch it.")
    for key, label in SCENARIOS.items():
        if st.button(label, use_container_width=True):
            append_feed(st.session_state.sim.inject(key))
            st.rerun()

    threshold = st.slider("Alert threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Transactions scoring above this are flagged.")

    if st.button("Reset feed", use_container_width=True):
        for k in ("sim", "feed"):
            st.session_state.pop(k, None)
        st.rerun()

# ---------------- header & metrics ----------------

branding.eyebrow("Explainable ML · Fraud & Anomaly Detection")
st.title("Fraud Detection Studio")
st.caption(
    "Real-time card-fraud detection with explainable AI — every flag comes with "
    "a SHAP breakdown and a plain-English reason. Synthetic data, real techniques."
)

flagged = feed[feed["risk"] >= threshold]
injected = feed[feed["is_fraud"] == 1]
caught = injected[injected["risk"] >= threshold]

tab_live, tab_how, tab_model = st.tabs(
    ["Live feed", "How it works", "Data & model"]
)

# ================= TAB 1: live feed =================

with tab_live:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions processed", f"{len(feed):,}")
    c2.metric("Flagged as suspicious", f"{len(flagged):,}")
    c3.metric("Fraud injected", f"{len(injected):,}")
    c4.metric(
        "Fraud caught",
        f"{len(caught):,}",
        f"{len(caught) / len(injected) * 100:.0f}% recall" if len(injected) else "no fraud yet",
    )

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
        color_continuous_scale=["#38bdf8", "#f5a623", "#ff3b46"],
        hover_data=["txn_id", "customer_id", "merchant", "category", "city"],
        log_y=True,
        height=380,
        labels={"amount": "Amount (AUD, log scale)", "timestamp": "Time"},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(56,189,248,0.03)",
                      font={"color": "#dfe3e8", "family": "IBM Plex Mono"})
    st.plotly_chart(fig, use_container_width=True)

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
        st.subheader("Investigate a flag")
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

            truth = "injected fraud" if row["is_fraud"] else "legitimate — a false positive"
            scenario = SCENARIOS.get(row["scenario"], "")
            st.markdown(
                f"**{row['merchant']}** · {row['category']} · {row['city']} · "
                f"A${row['amount']:,.2f}\n\n"
                f"Customer `{row['customer_id']}` · risk **{row['risk']:.2f}** · "
                f"ground truth: **{truth}**"
                + (f" — *{scenario}*" if scenario else "")
            )

            explanation = explain_row(ART["explainer"], row)

            st.markdown("**Why the model flagged it**")
            for r in plain_english(explanation, row):
                branding.reason(r, "neg")

            fig_w, ax = plt.subplots()
            shap.plots.waterfall(explanation, max_display=9, show=False)
            st.pyplot(branding.darken(plt.gcf()), use_container_width=True)
            plt.close("all")

# ================= TAB 2: how it works =================

with tab_how:
    st.subheader("From raw transaction to explained alert")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        branding.step(1, "Simulate a customer book",
                      "150 synthetic Australian customers with individual spending "
                      "baselines, home cities, merchant habits and night-owl "
                      "tendencies generate a realistic transaction stream.")
    with c2:
        branding.step(2, "Engineer behavioural features",
                      "Each transaction becomes signals a fraud analyst would ask "
                      "about: velocity (txns in the last hour), distance from home, "
                      "amount vs. this customer's baseline, merchant risk, "
                      "first-time merchant.")
    with c3:
        branding.step(3, "Score with XGBoost",
                      "300 trees score every transaction in real time. Training "
                      "data deliberately includes hard negatives — legitimate "
                      "shopping sprees and big-ticket buys — so the model learns "
                      "behaviour, not 'big amount = fraud'.")
    with c4:
        branding.step(4, "Explain every flag",
                      "SHAP TreeExplainer attributes each alert to its drivers, "
                      "which are translated into the plain-English reasons an "
                      "investigator needs to act.")

    st.write("")
    st.subheader("What the model actually learned")
    st.caption(
        "SHAP beeswarm across 300 held-out transactions — transaction velocity, "
        "deviation from personal baseline and geo-distance dominate, exactly the "
        "signals real fraud teams monitor."
    )
    fig, ax = plt.subplots()
    shap.plots.beeswarm(ART["global_explanation"], max_display=11, show=False)
    st.pyplot(branding.darken(plt.gcf()), use_container_width=True)
    plt.close("all")

    st.write("")
    st.subheader("The three attack patterns, and how they're caught")
    with st.expander("Stolen card spree"):
        st.markdown(
            "A pickpocketed card gets maxed out fast: 6–10 card-present purchases "
            "within ~90 minutes, far from the victim's home city, at high-risk "
            "merchants (electronics, jewellery, gift cards), with escalating "
            "amounts as the fraudster tests limits. **Caught by:** velocity + "
            "geo-distance + merchant risk + amount-vs-baseline all firing at once."
        )
    with st.expander("Account takeover"):
        st.markdown(
            "Stolen credentials are used from the attacker's location: late-night "
            "card-not-present purchases at merchants the customer has never used, "
            "in amounts far above their baseline. **Caught by:** night-time flag + "
            "online flag + first-time merchant + amount ratio."
        )
    with st.expander("Card testing"):
        st.markdown(
            "Before selling card numbers, fraudsters validate them with rapid "
            "micro-charges ($0.50–$3) at online merchants — then cash out with "
            "large purchases. The tiny amounts fool amount-based rules. "
            "**Caught by:** extreme velocity (seconds between transactions), "
            "even though every individual amount looks harmless."
        )
    with st.expander("Why the near-perfect AUC deserves scepticism"):
        st.markdown(
            "The model card reports ROC-AUC ≈ 0.999 — because synthetic fraud is "
            "cleaner than real fraud, even with hard negatives injected. Real-world "
            "systems fight concept drift, fraud rings that adapt to the model, and "
            "label delays. The value of this demo is the **workflow** — behavioural "
            "features, calibrated alerting, explainable triage — not the headline "
            "number. Knowing why your metric is too good is part of the job."
        )

    st.write("")
    st.subheader("Read the code")
    st.markdown(
        "Small, deliberately separated codebase — simulation, modelling and "
        "presentation never mix. Each module is a short read.\n\n"
        "| Module | Responsibility |\n"
        "|---|---|\n"
        "| [`data_gen.py`](https://github.com/drishtantleuva/fraud-detection-studio/blob/main/data_gen.py) | Synthetic customer book, transaction simulator, the three attack patterns, and the behavioural feature engineering shared by training and live scoring |\n"
        "| [`model.py`](https://github.com/drishtantleuva/fraud-detection-studio/blob/main/model.py) | Trains the classifier, computes SHAP attributions, and renders them as the plain-English reasons investigators read |\n"
        "| [`app.py`](https://github.com/drishtantleuva/fraud-detection-studio/blob/main/app.py) | Everything you are looking at |\n"
    )

# ================= TAB 3: data & model =================

with tab_model:
    m = ART["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training transactions", f"{m['n_train']:,}")
    c2.metric("ROC-AUC (held out)", f"{m['roc_auc']:.3f}")
    c3.metric("PR-AUC", f"{m['avg_precision']:.3f}")
    c4.metric("Fraud rate in training", f"{m['fraud_rate']:.1%}")

    st.write("")
    st.subheader("The feature vector")
    st.dataframe(pd.DataFrame({
        "feature": FEATURES,
        "what it captures": [
            "Raw transaction amount (AUD)",
            "Log-scaled amount — tames the long tail",
            "Hour of day",
            "Night-time transaction (10pm–6am)",
            "Risk weight of the merchant category",
            "Card-not-present (online) transaction",
            "Distance from the customer's home city (km)",
            "Minutes since this customer's previous transaction",
            "This customer's transactions in the past hour",
            "Amount relative to this customer's personal baseline",
            "First time this customer uses this merchant",
        ],
    }), hide_index=True, use_container_width=True)

    st.subheader("Data provenance")
    st.markdown(
        "All customers, merchants and transactions are generated by "
        "[`data_gen.py`](https://github.com/drishtantleuva/fraud-detection-studio/blob/main/data_gen.py) "
        "— no real financial data anywhere, no licensing constraints, fully "
        "reproducible. The attack patterns and feature design are modelled on two "
        "public fraud-research datasets:\n"
        "- [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — synthetic "
        "mobile-money transactions\n"
        "- [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) "
        "— real anonymised card transactions (Vesta Corp.)"
    )

branding.footer("fraud-detection-studio")

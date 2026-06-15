"""Visual identity for the Fraud Detection Studio.

Design language: a security-operations console. Near-black, dense, technical.
IBM Plex Mono for labels and metrics, signal-red for alerts and a cool cyan for
cleared activity — the way a real-time monitoring tool for a fraud team looks,
not a marketing dashboard.
"""

import streamlit as st

BG = "#0a0c10"
PANEL = "#12161d"
INK = "#dfe3e8"
MUTED = "#8b94a3"
CYAN = "#38bdf8"
ALERT = "#ff3b46"
AMBER = "#f5a623"
LINE = "#232a35"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="st-"], [data-testid="stMarkdownContainer"] {
  font-family: 'IBM Plex Sans', -apple-system, sans-serif;
}
[data-testid="stAppViewContainer"] { background: #0a0c10; }

h1 {
  font-family: 'IBM Plex Mono', monospace !important;
  font-weight: 600 !important;
  font-size: 2.2rem !important;
  letter-spacing: -0.01em;
  color: #f1f4f8 !important;
}
h2, h3 {
  font-family: 'IBM Plex Mono', monospace !important;
  font-weight: 600 !important;
  color: #eef2f6 !important;
}

/* metric cards — instrument readouts: sharp corners, mono values, a top tick */
[data-testid="stMetric"] {
  background: #12161d;
  border: 1px solid #232a35;
  border-top: 2px solid #38bdf8;
  border-radius: 3px;
  padding: 14px 16px;
}
[data-testid="stMetricLabel"] {
  color: #8b94a3 !important; text-transform: uppercase;
  letter-spacing: 0.08em; font-size: 0.72rem !important;
}
[data-testid="stMetricValue"] {
  font-family: 'IBM Plex Mono', monospace !important;
  color: #f1f4f8 !important; font-weight: 600;
}
[data-testid="stMetricDelta"] { color: #38bdf8 !important; }

[data-testid="stSidebar"] {
  background: #0c1015; border-right: 1px solid #1c232e;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  font-size: 1rem !important; text-transform: uppercase; letter-spacing: 0.1em;
}

.stButton button {
  border-radius: 3px; border: 1px solid #2b3442;
  background: #12161d; color: #cdd5df;
  font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
}
.stButton button:hover { border-color: #38bdf8; color: #fff; }

[data-testid="stTabs"] button[role="tab"] {
  font-family: 'IBM Plex Mono', monospace; font-weight: 500; color: #8b94a3;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: #38bdf8; }
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }

div[data-testid="stExpander"] {
  border: 1px solid #232a35; border-radius: 3px; background: #0f141b;
}

/* console kicker: mono, with a pulsing live dot */
.eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  text-transform: uppercase; letter-spacing: 0.16em;
  font-size: 0.72rem; color: #38bdf8; margin-bottom: 2px;
}
.live-dot {
  display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: #ff3b46; margin-right: 7px; vertical-align: middle;
  box-shadow: 0 0 0 0 rgba(255,59,70,0.6); animation: pulse 1.8s infinite;
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(255,59,70,0.5); }
  70% { box-shadow: 0 0 0 7px rgba(255,59,70,0); }
  100% { box-shadow: 0 0 0 0 rgba(255,59,70,0); }
}

.dl-step {
  background: #12161d; border: 1px solid #232a35; border-radius: 3px;
  padding: 18px; height: 100%;
}
.dl-step b { color: #38bdf8; font-family: 'IBM Plex Mono', monospace; }
.dl-step .n {
  display: inline-block; font-family: 'IBM Plex Mono', monospace;
  color: #8b94a3; font-weight: 600; margin-bottom: 8px; font-size: 0.85rem;
  border: 1px solid #2b3442; border-radius: 3px; padding: 1px 7px;
}

/* alerts — flagged reasons read like log lines */
.reason {
  padding: 9px 14px; margin: 6px 0;
  border-left: 3px solid #ff3b46; border-radius: 0 3px 3px 0;
  background: rgba(255,59,70,0.06);
  font-family: 'IBM Plex Mono', monospace; font-size: 0.86rem; color: #e6cdd0;
}
.reason.pos { border-left-color: #38bdf8; background: rgba(56,189,248,0.06); color: #cfe3ef; }
.reason.tip { border-left-color: #f5a623; background: rgba(245,166,35,0.06); color: #ecdcc3; }

table { font-size: 0.9rem; }
a { color: #38bdf8; }
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def eyebrow(text: str):
    st.markdown(
        f'<p class="eyebrow"><span class="live-dot"></span>{text}</p>',
        unsafe_allow_html=True,
    )


def reason(text: str, kind: str = "neg"):
    st.markdown(f'<div class="reason {kind}">{text}</div>', unsafe_allow_html=True)


def darken(fig):
    """Restyle a matplotlib/SHAP figure for the console's near-black panels."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    fig.patch.set_facecolor(PANEL)
    for ax in fig.axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=INK, labelcolor=INK)
        for spine in ax.spines.values():
            spine.set_color(LINE)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
        ax.title.set_color(INK)
    for text in fig.findobj(plt.Text):
        try:
            r, g, b = mcolors.to_rgb(text.get_color())
            if 0.299 * r + 0.587 * g + 0.114 * b < 0.82:
                text.set_color(INK)
        except (ValueError, TypeError):
            pass
    return fig


def step(n, title, body):
    st.markdown(
        f'<div class="dl-step"><span class="n">{n:02d}</span><br/><br/>'
        f'<b>{title}</b><br/><span style="color:#aab2bf;font-size:0.9rem">{body}</span></div>',
        unsafe_allow_html=True,
    )


def footer(repo: str):
    st.divider()
    st.markdown(
        f'<p style="color:#6c7585;font-size:0.82rem;font-family:\'IBM Plex Mono\',monospace">'
        f'built by <a href="https://drishtantleuva.github.io" target="_blank">'
        f'<b>Drishtant Leuva</b></a> — data scientist · risk &amp; anomaly detection &nbsp;·&nbsp; '
        f'<a href="https://github.com/drishtantleuva/{repo}" target="_blank">source</a> '
        f'&nbsp;·&nbsp; <a href="https://www.linkedin.com/in/drishtant-leuva/" '
        f'target="_blank">linkedin</a></p>',
        unsafe_allow_html=True,
    )

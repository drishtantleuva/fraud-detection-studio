"""Shared look-and-feel: typography, card styling, dark chart helpers."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import streamlit as st

ACCENT1 = "#7b5cff"
ACCENT2 = "#2fc8f5"
PANEL = "#15151a"
INK = "#e8e8ec"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="st-"], [data-testid="stMarkdownContainer"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Streamlit renders icons as Material Symbols ligatures; the font override
   above must never reach them or the icon name shows as literal text. */
[data-testid="stIconMaterial"], [data-testid="stExpanderToggleIcon"] {
  font-family: 'Material Symbols Rounded' !important;
}

h1 {
  font-weight: 800 !important;
  letter-spacing: -0.02em;
  background: linear-gradient(92deg, #f2f2f5 30%, #7b5cff 75%, #2fc8f5);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}
h2, h3 { font-weight: 700 !important; letter-spacing: -0.01em; }

[data-testid="stMetric"] {
  background: #15151a;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 14px;
  padding: 14px 18px;
}

[data-testid="stSidebar"] {
  background: #101014;
  border-right: 1px solid rgba(255, 255, 255, 0.06);
}

.stButton button {
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.18);
}
.stButton button:hover { border-color: #7b5cff; color: #fff; }

[data-testid="stTabs"] button[role="tab"] { font-weight: 600; font-size: 0.95rem; }

div[data-testid="stExpander"] {
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 12px;
  background: #121217;
}

.dl-footer {
  margin-top: 8px;
  color: #8a8a92;
  font-size: 0.85rem;
}
.dl-footer a { color: #2fc8f5; text-decoration: none; }

.dl-step {
  background: #15151a;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 14px;
  padding: 18px;
  height: 100%;
}
.dl-step b { color: #2fc8f5; }
.dl-step .n {
  display: inline-block; width: 26px; height: 26px; line-height: 26px;
  text-align: center; border-radius: 50%;
  background: linear-gradient(92deg, #7b5cff, #2fc8f5);
  color: white; font-weight: 700; margin-bottom: 8px; font-size: 0.85rem;
}

.reason {
  padding: 9px 14px; margin: 7px 0;
  border-left: 3px solid; border-radius: 0 10px 10px 0;
  background: rgba(255, 255, 255, 0.03);
  font-size: 0.95rem; color: #dcdce2;
}
.reason.neg { border-color: #ff5c5c; }
.reason.pos { border-color: #21c98d; }
.reason.tip { border-color: #2fc8f5; }

.pill {
  display: inline-block; padding: 8px 24px; border-radius: 999px;
  font-weight: 800; letter-spacing: 0.05em; font-size: 1.2rem;
}
.pill.ok {
  background: rgba(33, 201, 141, 0.14); color: #21c98d;
  border: 1px solid rgba(33, 201, 141, 0.4);
}
.pill.no {
  background: rgba(255, 92, 92, 0.12); color: #ff5c5c;
  border: 1px solid rgba(255, 92, 92, 0.4);
}

.eyebrow {
  text-transform: uppercase; letter-spacing: 0.22em;
  font-size: 0.72rem; color: #8a8a92; margin-bottom: -4px;
}

table { font-size: 0.92rem; }
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def eyebrow(text: str):
    st.markdown(f'<p class="eyebrow">{text}</p>', unsafe_allow_html=True)


def verdict_pill(approved: bool, label_ok: str = "Approved",
                 label_no: str = "Declined"):
    cls, label = ("ok", label_ok) if approved else ("no", label_no)
    st.markdown(f'<span class="pill {cls}">{label}</span>', unsafe_allow_html=True)


def reason(text: str, kind: str):
    """kind: 'neg' (against), 'pos' (in favour), 'tip' (guidance)."""
    st.markdown(f'<div class="reason {kind}">{text}</div>', unsafe_allow_html=True)


def darken(fig):
    """Restyle a matplotlib figure (e.g. SHAP plots) for the dark theme.

    Any text darker than mid-grey is lifted to the ink colour so axis labels,
    tick labels and SHAP annotations stay readable on the dark panel.
    """
    fig.patch.set_facecolor(PANEL)
    for ax in fig.axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=INK, labelcolor=INK)
        for spine in ax.spines.values():
            spine.set_color("#3a3a42")
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
        ax.title.set_color(INK)
    for text in fig.findobj(plt.Text):
        try:
            r, g, b = mcolors.to_rgb(text.get_color())
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            if luminance < 0.82:
                text.set_color(INK)
        except (ValueError, TypeError):
            pass
    return fig


def step(n, title, body):
    st.markdown(
        f'<div class="dl-step"><span class="n">{n}</span><br/>'
        f'<b>{title}</b><br/><span style="color:#b9b9c2;font-size:0.92rem">{body}</span></div>',
        unsafe_allow_html=True,
    )


def footer(repo: str):
    st.divider()
    st.markdown(
        f'<p class="dl-footer">Built by <a href="https://drishtantleuva.github.io" '
        f'target="_blank"><b>Drishtant Leuva</b></a> — Data Scientist · Risk &amp; '
        f'Explainable AI &nbsp;·&nbsp; '
        f'<a href="https://github.com/drishtantleuva/{repo}" target="_blank">Source on GitHub</a> '
        f'&nbsp;·&nbsp; <a href="https://www.linkedin.com/in/drishtant-leuva/" '
        f'target="_blank">LinkedIn</a></p>',
        unsafe_allow_html=True,
    )

"""
Content & Audience Predictor - Streamlit dashboard

Run from the project root:
    streamlit run app/streamlit_app.py

Visual language follows the reference reel: paper background, boxed white panels with thin dark
rules, numbered section labels, a grey action block, cobalt / orange / grey dots and a big-headline
verdict strip. All ML happens in src/ (inference.py, segmentation.py, report.py); this file only draws.

Educational project. It does not reproduce or predict Instagram's real algorithm.
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so `import src` works from anywhere

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.inference import FEATURE_LABELS, analyze, load_artifacts
from src.report import build_report_pdf
from src.segmentation import SEGMENT_INFO

st.set_page_config(page_title="Content & Audience Predictor", layout="wide",
                   initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------
# Palette (taken from the reference reel)
# ----------------------------------------------------------------------
INK, INK2, INK3 = "#111111", "#55554F", "#8A8A84"
PAPER, PANEL, HAIR, RULE = "#EFEEE9", "#FCFCFA", "#DCDBD5", "#2A2A28"
BLUE, ORANGE, DOT, RED = "#1E3FD8", "#E0561B", "#BDBDB8", "#B5474F"
SEG_COLOR = {"Casual Scrollers": DOT, "Conversation Starters": BLUE,
             "Viral Sharers": ORANGE, "Loyal Fans": "#1A1A1A"}
PERF_COLOR = {"Low": ORANGE, "Medium": DOT, "High": BLUE}
FONT = "Inter, Helvetica Neue, Arial, sans-serif"
TRANSPARENT = "rgba(0,0,0,0)"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap');
:root{--paper:#EFEEE9;--panel:#FCFCFA;--ink:#111;--ink2:#55554F;--ink3:#8A8A84;--rule:#2A2A28;--hair:#DCDBD5;
--blue:#1E3FD8;--orange:#E0561B;--dot:#BDBDB8;--red:#B5474F;
--sans:'Inter','Helvetica Neue',Arial,sans-serif;--mono:'IBM Plex Mono',ui-monospace,Menlo,Consolas,monospace;}
.stApp{background:var(--paper);color:var(--ink);font-family:var(--sans);}
header[data-testid="stHeader"],footer,#MainMenu,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none !important;}
.block-container{max-width:1160px;padding:1.5rem 1.5rem 3rem;}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] div,[data-testid="stWidgetLabel"] p,
.stApp input,.stApp button,.stApp label,div[data-baseweb="select"] *{font-family:var(--sans);}

/* header + section labels */
.hdr{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--rule);padding-bottom:.65rem;margin-bottom:.4rem;}
.hdr b{font-size:15px;font-weight:700;letter-spacing:-.01em}.hdr span{color:var(--ink3);margin-left:.7rem;font-size:14px}
.live{font-size:11px;color:var(--ink2)}.live i{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--blue);margin-right:6px}
.sec{display:flex;justify-content:space-between;align-items:baseline;margin:2.4rem 0 .75rem;gap:1rem}
.sec .l{font-size:10.5px;font-weight:800;letter-spacing:.16em;text-transform:uppercase;white-space:nowrap}
.sec .n{margin-right:1rem;font-weight:700;letter-spacing:.04em}
.sec .r{color:var(--ink3);font-size:11px;text-align:right}

/* widgets: flat, square, quiet */
[data-testid="stWidgetLabel"] p{font-size:10px !important;letter-spacing:.09em;text-transform:uppercase;color:var(--ink3) !important;font-weight:500 !important;}
div[data-baseweb="select"]>div,div[data-baseweb="input"],div[data-baseweb="base-input"]{border-radius:0 !important;background:#fff !important;border-color:var(--hair) !important;}
div[data-baseweb="select"]>div{min-height:38px;font-size:13px;}
.stApp input{font-size:13px !important;}
div[data-baseweb="slider"] div[role="slider"]{background:var(--ink) !important;box-shadow:none !important;}
[data-testid="stSliderThumbValue"],[data-testid="stTickBarMin"],[data-testid="stTickBarMax"]{font-size:11px !important;color:var(--ink2) !important;}
[data-testid="stForm"]{background:var(--panel);border:1px solid var(--rule) !important;border-radius:0 !important;padding:1.2rem 1.3rem 1.3rem;}
[data-testid="stFormSubmitButton"] button,[data-testid="stDownloadButton"] button{border-radius:0 !important;background:#8D8D8A !important;color:#fff !important;border:0 !important;
width:100%;font-weight:700;text-transform:uppercase;}
[data-testid="stFormSubmitButton"] button{min-height:236px;}
[data-testid="stDownloadButton"] button{min-height:132px;}
[data-testid="stFormSubmitButton"] button p,[data-testid="stDownloadButton"] button p{font-size:11px !important;letter-spacing:.2em;font-weight:700 !important;color:#fff !important;}
[data-testid="stFormSubmitButton"] button:hover,[data-testid="stDownloadButton"] button:hover{background:var(--ink) !important;}
.stButton button{border-radius:0 !important;background:transparent !important;border:1px solid var(--rule) !important;color:var(--ink) !important;
padding:.4rem .8rem;min-height:0;width:100%;}
.stButton button p{font-size:12px !important;font-weight:500 !important;white-space:nowrap;}
.stButton button:hover{background:var(--ink) !important;color:#fff !important;}
.stButton button:hover p{color:#fff !important;}
[data-testid="stRadio"] label p{font-size:12px !important;text-transform:none !important;letter-spacing:0 !important;color:var(--ink) !important;}
[data-testid="stButtonGroup"] button{border-radius:0 !important;border:1px solid var(--rule) !important;background:transparent !important;min-height:32px;}
[data-testid="stButtonGroup"] button p{font-size:12px !important;color:var(--ink) !important;text-transform:none !important;letter-spacing:0 !important;}
[data-testid="stBaseButton-segmented_controlActive"]{background:var(--ink) !important;}
[data-testid="stBaseButton-segmented_controlActive"] p{color:#fff !important;}
[class*="st-key-panel_"]{background:var(--panel);border:1px solid var(--rule);border-radius:0;padding:1rem 1.15rem .6rem;}
[data-testid="stExpander"]{border:1px solid var(--rule) !important;border-radius:0 !important;background:var(--panel);}
[data-testid="stExpander"] summary p{font-size:11px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;}
.exlab{font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink3);padding-top:.5rem}
.rowsep{border-top:1px solid var(--hair);margin:.5rem 0 1rem}
.fnote{font-size:12px;color:var(--ink2);line-height:1.55;margin:0 0 .3rem}
.fnote b{color:var(--ink);font-weight:700}

/* panels */
.panel{background:var(--panel);border:1px solid var(--rule);}
.empty{padding:1.6rem 1.5rem}.empty h4{margin:0 0 .35rem;font-size:14.5px;font-weight:700}
.empty p{margin:0;color:var(--ink2);font-size:12.5px;line-height:1.5}
.vtop{display:flex;gap:2.2rem;align-items:baseline;padding:1.5rem 1.5rem 1.35rem;border-bottom:1px solid var(--hair);}
.vtop h2{margin:0;padding:0;font-size:44px;font-weight:800;letter-spacing:-.035em;line-height:1;white-space:nowrap;color:var(--ink)}
.vtop p{margin:0;color:var(--ink2);font-size:13px;line-height:1.55;max-width:640px}
.strip{display:grid;grid-template-columns:repeat(5,1fr)}
.strip.s3{grid-template-columns:repeat(3,1fr)}
.cell{padding:1rem 1.2rem 1.15rem;border-right:1px solid var(--hair);min-width:0}.cell:last-child{border-right:0}
.cell .k{font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink3);margin-bottom:.75rem}
.cell .v{font-size:38px;font-weight:800;letter-spacing:-.03em;line-height:1;color:var(--ink)}
.cell .v small{font-size:13px;font-weight:500;color:var(--ink3);margin-left:3px;letter-spacing:0}
.cell .v.t{font-size:21px;line-height:1.2;letter-spacing:-.02em;padding-top:.35rem}
.cell .c{font-size:10.5px;color:var(--ink3);margin-top:.65rem;line-height:1.45}
.cell .c.m{font-family:var(--mono)}
.bar{display:flex;height:5px;margin-top:.75rem;background:var(--hair)}.bar i{display:block;height:100%}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:1px}
.note{font-size:12px;color:var(--red);margin:.5rem 0 0}
.segcard{padding:1.15rem 1.25rem 1.2rem}
.segcard .k{font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink3)}
.segname{font-size:22px;font-weight:800;letter-spacing:-.025em;margin:.5rem 0 .5rem;line-height:1.15}
.segcard p{font-size:12.5px;color:var(--ink2);line-height:1.55;margin:.1rem 0 .3rem}
.mini{display:grid;grid-template-columns:1fr 1fr;gap:.7rem 1rem;margin:.9rem 0 .2rem;padding-top:.8rem;border-top:1px solid var(--hair)}
.mini div{font-size:10.5px;color:var(--ink3)}.mini span{display:block;font-size:20px;font-weight:800;color:var(--ink);letter-spacing:-.02em}
.hint{font-size:12px;color:var(--ink2);margin-top:1rem;padding-top:.8rem;border-top:1px solid var(--hair);line-height:1.55}
.ptitle{font-size:13px;font-weight:700;margin:.1rem 0 .15rem}
.psub{font-size:11.5px;color:var(--ink3);margin-bottom:.5rem;line-height:1.5}
.ins{padding:1.2rem 1.3rem 1.3rem;min-height:414px}
.rec{display:flex;justify-content:space-between;gap:1rem;align-items:baseline;padding:.85rem 0;border-top:1px solid var(--hair)}
.rec .t{font-size:13.5px;font-weight:600;letter-spacing:-.005em}
.rec .g{font-size:26px;font-weight:800;letter-spacing:-.03em;white-space:nowrap}.rec .g small{font-size:11px;font-weight:500;color:var(--ink3);margin-left:2px}
.rep{padding:1.25rem 1.4rem;min-height:132px}
.rep h4{margin:0 0 .35rem;font-size:15px;font-weight:700}
.rep p{margin:0 0 .5rem;font-size:12.5px;color:var(--ink2);line-height:1.55}
.rep ul{margin:.2rem 0 0;padding-left:1.1rem;font-size:12px;color:var(--ink2);line-height:1.7}
.foot{margin-top:2.6rem;padding-top:.8rem;border-top:1px solid var(--hair);font-size:10.5px;color:var(--ink3);display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap}
.relnotes{margin:.2rem 0 0;padding-left:1.1rem;font-size:12.5px;color:var(--ink2);line-height:1.75}
@media(max-width:860px){.strip{grid-template-columns:1fr 1fr}.strip.s3{grid-template-columns:1fr}.vtop{flex-direction:column;gap:.9rem}.vtop h2{font-size:34px}}
</style>
"""


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def html(s: str):
    """Streamlit treats indented lines as code blocks, so flatten the HTML to one line."""
    st.markdown(" ".join(line.strip() for line in s.strip().splitlines()), unsafe_allow_html=True)


def section(n: int, title: str, note: str = ""):
    html(f'<div class="sec"><div class="l"><span class="n">{n:02d}</span>{title}</div><div class="r">{note}</div></div>')


def panel(key: str):
    """A square, bordered panel (styled by the st-key-panel_ CSS rule above)."""
    try:
        return st.container(key=key)
    except TypeError:                          # older Streamlit without container keys
        return st.container(border=True)


def toggle_group(options, key, label):
    """Black-and-white toggle like the reel's 2D | 3D switch. Falls back to radio buttons on older Streamlit."""
    if hasattr(st, "segmented_control"):
        st.segmented_control(label, options, key=key, label_visibility="collapsed")
    else:
        st.radio(label, options, key=key, horizontal=True, label_visibility="collapsed")
    return st.session_state.get(key) or options[0]


def show(fig, key):
    try:
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False}, key=key)
    except TypeError:                          # very old / very new Streamlit signature
        st.plotly_chart(fig, key=key)


def axis_style(**kw):
    """Readable axes: visible line, outside ticks, 11-12 px text, and room (automargin) for long labels."""
    base = dict(showline=True, linecolor=INK, linewidth=1, ticks="outside", ticklen=5, tickcolor=INK,
                tickfont=dict(size=12, color=INK2), automargin=True, zeroline=False)
    base.update(kw)
    return base


def axis_title(text):
    return dict(text=text, font=dict(size=12, color=INK2), standoff=10)


def base_layout(height, **kw):
    layout = dict(
        height=height, margin=dict(l=10, r=20, t=10, b=10),
        paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
        font=dict(family=FONT, size=12, color=INK),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=RULE, font=dict(family=FONT, size=12, color=INK)),
    )
    layout.update(kw)
    return layout


# ----------------------------------------------------------------------
# Load artefacts (cached) - friendly message if training has not been run yet
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_artifacts():
    art = load_artifacts()
    art["map_sample"] = art["audience_map"].sample(900, random_state=C.RANDOM_SEED)
    return art


st.markdown(CSS, unsafe_allow_html=True)
try:
    ART = get_artifacts()
except FileNotFoundError as err:
    st.error("The models have not been trained yet.")
    st.code(str(err))
    st.stop()

# ----------------------------------------------------------------------
# State: widget defaults, example posts
# ----------------------------------------------------------------------
DEFAULTS = dict(category="Comedy", duration_sec=25, caption_length=80, hashtag_count=6, post_hour=19,
                post_day="Sat", trending_audio=False, followers=20000, prev_avg_engagement=6.0,
                views=0, likes=0, comments=0, shares=0, watch=50)

PRESETS = {
    "Comedy reel, Saturday 8 pm": dict(category="Comedy", duration_sec=20, caption_length=80, hashtag_count=6,
                                       post_hour=20, post_day="Sat", trending_audio=True, followers=15000,
                                       prev_avg_engagement=7.0, views=42000, likes=3200, comments=900,
                                       shares=700, watch=68),
    "Long tutorial, 3 am": dict(category="Education", duration_sec=80, caption_length=400, hashtag_count=15,
                                post_hour=3, post_day="Tue", trending_audio=False, followers=8000,
                                prev_avg_engagement=4.5, views=6000, likes=240, comments=15, shares=12, watch=30),
    "Food reel people forward": dict(category="Food", duration_sec=28, caption_length=60, hashtag_count=5,
                                     post_hour=13, post_day="Sun", trending_audio=True, followers=30000,
                                     prev_avg_engagement=6.5, views=80000, likes=4000, comments=400,
                                     shares=2400, watch=60),
}

for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("result", None)
st.session_state.setdefault("colour_by", "Segment")
st.session_state.setdefault("seg_inspect", list(SEGMENT_INFO)[0])
st.session_state.setdefault("imp_model", "Engagement rate")


def collect_inputs() -> dict:
    return {k: st.session_state[k] for k in DEFAULTS}


def store_result(result: dict):
    """Keeps the analysis and builds its PDF report (about a second)."""
    try:
        result["report_pdf"], result["report_error"] = build_report_pdf(result, ART), None
    except Exception as err:                   # the dashboard must keep working even if the report fails
        result["report_pdf"], result["report_error"] = None, str(err)
    st.session_state["result"] = result
    if result["segment"]:
        st.session_state["seg_inspect"] = result["segment"]["segment"]


def load_preset(name: str):
    st.session_state.update(PRESETS[name])
    store_result(analyze(collect_inputs(), ART))


# ======================================================================
# Header
# ======================================================================
html('<div class="hdr"><div><b>Content &amp; Audience Predictor</b><span>Run report</span></div>'
     '<div class="live"><i></i>Runs locally</div></div>')

# ======================================================================
# 01  INPUT
# ======================================================================
section(1, "Input", "Describe a post, then analyze. Nothing is uploaded or sent anywhere.")

ex = st.columns([0.9, 1.6, 1.6, 1.6, 1.3], gap="small")
ex[0].markdown('<div class="exlab">Try an example</div>', unsafe_allow_html=True)
for col, name in zip(ex[1:4], PRESETS):
    col.button(name, key=f"ex_{name}", on_click=load_preset, args=(name,))

with st.form("post_form"):
    c1, c2, c3, c4, c5 = st.columns([1.25, 1.25, 1.15, 1.2, 0.75], gap="medium")
    with c1:
        st.selectbox("Content category", C.CATEGORIES, key="category")
        st.slider("Video duration (s)", 5, 90, key="duration_sec")
    with c2:
        st.slider("Caption length (characters)", 0, 500, key="caption_length")
        st.slider("Hashtag count", 0, 30, key="hashtag_count")
    with c3:
        st.slider("Posting hour (0-23)", 0, 23, key="post_hour")
        st.selectbox("Posting day", C.DAYS, key="post_day")
        st.toggle("Trending audio", key="trending_audio")
    with c4:
        st.number_input("Followers", min_value=100, max_value=50_000_000, step=500, key="followers")
        st.slider("Previous avg. engagement (%)", 0.5, 20.0, step=0.1, key="prev_avg_engagement")
    with c5:
        submitted = st.form_submit_button("Analyze")

    html('<div class="rowsep"></div>')
    html('<div class="fnote"><b>After publishing</b> (optional). Already posted? Add its results to see which audience '
         'it reached and how actual engagement compares with the estimate. These numbers never change the estimate.</div>')
    n1, n2, n3, n4, n5 = st.columns(5, gap="medium")
    n1.number_input("Views", min_value=0, step=100, key="views")
    n2.number_input("Likes", min_value=0, step=10, key="likes")
    n3.number_input("Comments", min_value=0, step=5, key="comments")
    n4.number_input("Shares", min_value=0, step=5, key="shares")
    n5.slider("Avg. watch (%)", 5, 100, key="watch")

if submitted:
    store_result(analyze(collect_inputs(), ART))

R = st.session_state["result"]

# ======================================================================
# 02  PREDICTION
# ======================================================================
section(2, "Prediction", "An estimate to guide decisions, not a guarantee")

if R is None:
    html('<div class="panel empty"><h4>Verdict lands here.</h4><p>Fill in the post above and press Analyze. '
         'Predicted performance, engagement rate, model confidence and audience segment appear after a run.</p></div>')
else:
    th = ART["thresholds"]
    p = R["probs"]
    bar = "".join(f'<i style="width:{p[l] * 100:.1f}%;background:{PERF_COLOR[l]}"></i>' for l in C.LABELS)
    seg = R["segment"]
    if seg:
        seg_cell = (f'<div class="v t"><span class="dot" style="background:{SEG_COLOR[seg["segment"]]}"></span>'
                    f'{seg["segment"]}</div><div class="c">from your views, interactions and watch time</div>')
    else:
        seg_cell = ('<div class="v t" style="color:#8A8A84">Not placed</div>'
                    '<div class="c">add views, likes, comments, shares and watch % to place this post</div>')
    actual = (f'<br>actual from your numbers: {R["er_actual"]:.1f}%' if R["er_actual"] is not None else "")

    html(f'''
    <div class="panel">
      <div class="vtop"><h2>{R["headline"]}</h2><p>{R["sub"]}</p></div>
      <div class="strip">
        <div class="cell"><div class="k">Predicted performance</div>
          <div class="v"><span class="dot" style="background:{PERF_COLOR[R["label"]]};width:12px;height:12px"></span>{R["label"]}</div>
          <div class="c">Low &lt; {th["low_medium_cutoff"]:.1f}% &middot; High &gt; {th["medium_high_cutoff"]:.1f}% engagement</div></div>
        <div class="cell"><div class="k">Estimated engagement</div>
          <div class="v">{R["er_pred"]:.1f}<small>%</small></div>
          <div class="c">typical error &plusmn;{R["mae"]:.1f} points{actual}</div></div>
        <div class="cell"><div class="k">Model confidence</div>
          <div class="v">{R["confidence"] * 100:.0f}<small>%</small></div>
          <div class="bar">{bar}</div>
          <div class="c m">low {p["Low"] * 100:.0f} &middot; mid {p["Medium"] * 100:.0f} &middot; high {p["High"] * 100:.0f}</div></div>
        <div class="cell"><div class="k">Audience segment</div>{seg_cell}</div>
        <div class="cell"><div class="k">Standing</div>
          <div class="v">{R["percentile"]:.0f}<small>percentile</small></div>
          <div class="c">ahead of {R["percentile"]:.0f}% of {ART["n_posts"]:,} reference posts</div></div>
      </div>
    </div>''')
    for w in R["warnings"]:
        html(f'<div class="note">{w}</div>')

# ======================================================================
# 03  AUDIENCE MAP
# ======================================================================
section(3, "Audience map", "Each dot is one post. Nearby dots reached similar audiences.")

sample = ART["map_sample"]
prof = ART["profiles"]
full = ART["audience_map"]


def build_map(colour_by: str, result):
    fig = go.Figure()
    groups = ([(s, SEG_COLOR[s], "segment") for s in SEG_COLOR] if colour_by == "Segment"
              else [(l, PERF_COLOR[l], "performance_label") for l in C.LABELS])
    for name, colour, col in groups:
        d = sample[sample[col] == name]
        fig.add_trace(go.Scatter(
            x=d.pc1, y=d.pc2, mode="markers", name=name,
            marker=dict(size=9, color=colour, opacity=0.92, line=dict(width=1, color="#FFFFFF")),
            customdata=np.stack([d.segment, d.category, d.engagement_rate.round(1), d.performance_label], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}% engagement"
                          "<br>Performance: %{customdata[3]}<extra></extra>"))

    cent = prof["centroid_pca_xy"]
    seg = result["segment"] if result else None
    if seg:                                    # lines from the user's post to every segment centre
        px, py = seg["pc1"], seg["pc2"]
        for name, (cx, cy) in cent.items():
            mine = name == seg["segment"]
            fig.add_trace(go.Scatter(
                x=[px, cx], y=[py, cy], mode="lines", showlegend=False, hoverinfo="skip",
                line=dict(color=RED if mine else "#8A8A86", width=1.8 if mine else 1, dash="solid" if mine else "dot")))
    fig.add_trace(go.Scatter(
        x=[v[0] for v in cent.values()], y=[v[1] for v in cent.values()], mode="markers", showlegend=False,
        marker=dict(size=15, color="#FFFFFF", line=dict(width=2.5, color=INK)), hoverinfo="skip"))
    for name, (cx, cy) in cent.items():        # boxed labels stay readable on top of the dots
        fig.add_annotation(x=cx, y=cy, text=f"<b>{name}</b>", showarrow=False,
                           yshift=-24 if name == "Conversation Starters" else 24,
                           bgcolor=PANEL, bordercolor=RULE, borderwidth=1, borderpad=3,
                           font=dict(size=11, color=INK, family=FONT))
    if seg:
        fig.add_trace(go.Scatter(
            x=[seg["pc1"]], y=[seg["pc2"]], mode="markers", showlegend=False,
            marker=dict(size=20, color="#FFFFFF", line=dict(width=3.5, color=INK)),
            hovertemplate=f"<b>Your post</b><br>{seg['segment']}<extra></extra>"))
        fig.add_annotation(x=seg["pc1"], y=seg["pc2"], text="<b>Your post</b>", showarrow=False, yshift=-28,
                           bgcolor=INK, bordercolor=INK, borderpad=3, font=dict(size=11, color="#FFFFFF", family=FONT))

    xs = list(full.pc1) + ([seg["pc1"]] if seg else [])
    ys = list(full.pc2) + ([seg["pc2"]] if seg else [])
    quiet = dict(showgrid=False, showticklabels=False, ticks="")
    fig.update_layout(**base_layout(
        580, margin=dict(l=10, r=20, t=10, b=10),
        xaxis=axis_style(range=[min(xs) - .3, max(xs) + .3], title=axis_title("Watch time and comments  →"), **quiet),
        yaxis=axis_style(range=[min(ys) - .5, max(ys) + .4], title=axis_title("Sharing  →"), **quiet),
        legend=dict(orientation="h", y=-0.14, x=0, font=dict(size=12), itemsizing="constant")))
    return fig


def seg_card(seg_name: str, yours: bool) -> str:
    info, p = SEGMENT_INFO[seg_name], prof["profiles"][seg_name]
    tag = " &middot; your post" if yours else ""
    hint = ("" if R and R["segment"] else
            '<div class="hint">Your post is not on the map yet. Add views, likes, comments, shares and watch % '
            'in the input panel, then press Analyze.</div>')
    return f'''<div class="panel segcard">
      <div class="k">Segment{tag}</div>
      <div class="segname"><span class="dot" style="background:{SEG_COLOR[seg_name]}"></span>{seg_name}</div>
      <p>{info["description"]}</p>
      <div class="mini">
        <div><span>{p["share_of_posts_%"]:.0f}%</span>of all posts</div>
        <div><span>{p["avg_watch_pct"]:.0f}%</span>avg. watched</div>
        <div><span>{p["comment_share"]:.0f}%</span>of actions are comments</div>
        <div><span>{p["share_share"]:.0f}%</span>of actions are shares</div>
      </div>
      <div class="hint" style="margin-top:.9rem">{info["tip"]}</div>{hint}</div>'''


left, right = st.columns([2.35, 1], gap="medium")
with left:
    colour_by = toggle_group(["Segment", "Performance"], "colour_by", "Colour by")
    with panel("panel_map"):
        show(build_map(colour_by, R), "map")
with right:
    html('<div class="exlab" style="padding-top:.2rem;margin-bottom:.3rem">Inspect a segment</div>')
    st.radio("Inspect segment", list(SEGMENT_INFO), key="seg_inspect", label_visibility="collapsed")
    chosen = st.session_state["seg_inspect"]
    yours = bool(R and R["segment"] and R["segment"]["segment"] == chosen)
    html(seg_card(chosen, yours))

# ======================================================================
# 04  FEATURE IMPORTANCE
# ======================================================================
section(4, "Feature importance", "What matters most, and what moved this post")

fa, fb = st.columns(2, gap="medium")
with fa:
    with panel("panel_imp_all"):
        html('<div class="ptitle">What matters most overall</div>'
             '<div class="psub">How much each input shapes the estimate, across all posts. Longer bar = bigger influence.</div>')
        which = toggle_group(["Engagement rate", "Performance level"], "imp_model", "Estimate")
        col = "reg_permutation" if which == "Engagement rate" else "clf_permutation"
        imp = ART["importance"][col].clip(lower=0)
        imp = (imp / imp.sum()).sort_values()
        fig = go.Figure(go.Bar(
            x=imp.values, y=[FEATURE_LABELS[i] for i in imp.index], orientation="h", marker=dict(color=INK),
            text=[f"{v:.0%}" for v in imp.values], textposition="outside", cliponaxis=False,
            textfont=dict(size=12, color=INK), hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
        fig.update_layout(**base_layout(
            380, margin=dict(l=10, r=48, t=8, b=10),
            xaxis=axis_style(tickformat=".0%", showgrid=True, gridcolor=HAIR, rangemode="tozero",
                             title=axis_title("share of total importance")),
            yaxis=axis_style(tickfont=dict(size=12, color=INK), showgrid=False)))
        show(fig, "imp_global")
with fb:
    with panel("panel_imp_post"):
        html('<div class="ptitle">What moved this post\'s estimate</div>'
             '<div class="psub">How far each input pushed this post up (blue) or down (orange), compared with a typical post.</div>')
        if R is None:
            html('<div class="empty" style="padding:5.5rem 0 6.5rem"><h4>Nothing to explain yet.</h4>'
                 '<p>Run an analysis to see what pushed this post up or down.</p></div>')
        else:
            html('<div style="height:2.2rem"></div>')
            eff = R["effects"].sort_values("effect")
            fig = go.Figure(go.Bar(
                x=eff.effect, y=eff.label, orientation="h",
                marker=dict(color=[BLUE if v >= 0 else ORANGE for v in eff.effect]),
                text=[f"{v:+.2f}" for v in eff.effect], textposition="outside", cliponaxis=False,
                textfont=dict(size=12, color=INK), hovertemplate="%{y}: %{x:+.2f} points<extra></extra>"))
            lim = max(0.5, float(eff.effect.abs().max()) * 1.4)
            fig.update_layout(**base_layout(
                380, margin=dict(l=10, r=48, t=8, b=10),
                xaxis=axis_style(range=[-lim, lim], showgrid=True, gridcolor=HAIR, zeroline=True, zerolinecolor=INK,
                                 title=axis_title("points of engagement vs a typical post")),
                yaxis=axis_style(tickfont=dict(size=12, color=INK), showgrid=False)))
            show(fig, "imp_local")

# ======================================================================
# 05  INSIGHTS
# ======================================================================
section(5, "Insights", "Each suggestion changes one thing at a time")

if R is None:
    html('<div class="panel empty"><h4>Insights land here.</h4><p>After a run: the changes the model expects to help most, '
         'a best-time-to-post map, and a strategy tip for your audience segment.</p></div>')
else:
    ia, ib = st.columns([1, 1.3], gap="medium")
    with ia:
        rows = "".join(f'<div class="rec"><div class="t">{r["text"]}</div>'
                       f'<div class="g">+{r["gain"]:.1f}<small>pt</small></div></div>' for r in R["recs"])
        if not rows:
            rows = ('<div class="rec"><div class="t" style="font-weight:500;color:#55554F">No single change is predicted to '
                    'lift engagement noticeably. This post already sits near what the model rewards.</div></div>')
        tip = ""
        if R["segment"]:
            s = R["segment"]["segment"]
            tip = (f'<div class="k" style="margin-top:1.2rem;font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:#8A8A84">'
                   f'Audience tip &middot; {s}</div><p style="font-size:12.5px;color:#55554F;line-height:1.55;margin:.4rem 0 0">'
                   f'{SEGMENT_INFO[s]["tip"]}</p>')
        html(f'''<div class="panel ins">
            <div class="ptitle">What would help most</div>
            <div class="psub">Gains are estimates (typical error &plusmn;{R["mae"]:.1f} pt), so treat small ones as hints, not promises.</div>
            {rows}{tip}</div>''')
    with ib:
        with panel("panel_heat"):
            html('<div class="ptitle">Best time to post</div>'
                 '<div class="psub">Estimated engagement for every day and hour, with everything else about the post unchanged. '
                 'The square marks your current choice.</div>')
            heat = R["heat"]
            hours = list(range(24))
            fig = go.Figure(go.Heatmap(
                z=heat.values, x=hours, y=list(heat.index), xgap=2, ygap=2,
                colorscale=[[0, "#F4F3EE"], [1, BLUE]],
                colorbar=dict(thickness=10, len=0.9, tickfont=dict(size=11, color=INK2), outlinewidth=0, ticksuffix="%"),
                hovertemplate="%{y} %{x}:00 → %{z:.1f}%<extra></extra>"))
            fig.add_trace(go.Scatter(
                x=[R["inputs"]["post_hour"]], y=[R["inputs"]["post_day"]], mode="markers", showlegend=False,
                marker=dict(symbol="square-open", size=18, line=dict(width=2.5, color=INK)), hoverinfo="skip"))
            fig.update_layout(**base_layout(
                320, margin=dict(l=10, r=10, t=8, b=10),
                xaxis=axis_style(tickmode="array", tickvals=list(range(0, 24, 3)),
                                 ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)], showgrid=False, showline=False,
                                 title=axis_title("hour of day")),
                yaxis=axis_style(autorange="reversed", showgrid=False, showline=False, tickfont=dict(size=12, color=INK))))
            show(fig, "heat")

# ======================================================================
# 06  REPORT (download)
# ======================================================================
section(6, "Report", "A written summary with next steps, ready to keep or share")

if R is None:
    html('<div class="panel empty"><h4>Report lands here.</h4><p>After a run you can download a PDF that explains the '
         'result and tells you how to improve the post and tweak it for your audience.</p></div>')
else:
    ra, rb = st.columns([3.4, 1], gap="medium")
    with ra:
        html('''<div class="panel rep"><h4>Your report is ready.</h4>
            <p>A PDF that explains this result in plain language and tells you what to change.</p>
            <ul><li>The verdict, confidence and how the post ranks</li>
            <li>What pushed the estimate up or down</li>
            <li>The changes worth trying, and the best time to post</li>
            <li>Your audience, and how to tweak the post for it or reach a different one</li></ul></div>''')
    with rb:
        if R.get("report_pdf"):
            st.download_button("Download report (PDF)", data=R["report_pdf"],
                               file_name=f"audience-report-{datetime.now():%Y%m%d-%H%M}.pdf",
                               mime="application/pdf", key="dl_report")
        else:
            html(f'<div class="note">The report could not be created: {R.get("report_error")}</div>')

# ----------------------------------------------------------------------
# How reliable is this? (plain language)
# ----------------------------------------------------------------------
st.write("")
with st.expander("How reliable is this?"):
    m = ART["metrics"]
    acc = m["classification"]["Random Forest"]["accuracy"]
    mae = m["regression"]["Random Forest"]["mae"]
    r2 = m["regression"]["Random Forest"]["r2"]
    html(f'''<div class="panel"><div class="strip s3">
        <div class="cell"><div class="k">Performance level called correctly</div>
          <div class="v">{acc * 100:.0f}<small>%</small></div>
          <div class="c">of {m["n_test"]:,} posts the model had never seen. Guessing would get 33%.</div></div>
        <div class="cell"><div class="k">Typical engagement error</div>
          <div class="v">&plusmn;{mae:.1f}<small>points</small></div>
          <div class="c">average gap between the estimate and the real rate</div></div>
        <div class="cell"><div class="k">Differences in engagement explained</div>
          <div class="v">{r2 * 100:.0f}<small>%</small></div>
          <div class="c">the rest comes down to things no input captures</div></div>
      </div></div>''')
    st.write("")
    k1, k2 = st.columns([1, 1.25], gap="medium")
    with k1:
        cm = m["confusion_matrix"]
        z = np.array(cm["matrix"])
        cfig = go.Figure(go.Heatmap(
            z=z, x=cm["labels"], y=cm["labels"], colorscale=[[0, "#F4F3EE"], [1, BLUE]], showscale=False,
            xgap=2, ygap=2, text=z, texttemplate="%{text}", textfont=dict(size=15, color=INK),
            hovertemplate="actual %{y}, estimated %{x}: %{z} posts<extra></extra>"))
        cfig.update_layout(**base_layout(
            300, margin=dict(l=10, r=10, t=8, b=10),
            xaxis=axis_style(showline=False, ticks="", title=axis_title("estimated level")),
            yaxis=axis_style(showline=False, ticks="", autorange="reversed", title=axis_title("actual level"),
                             tickfont=dict(size=12, color=INK))))
        with panel("panel_conf"):
            html('<div class="ptitle">Estimated vs actual level</div>'
                 '<div class="psub">Each cell counts test posts. The diagonal is where the estimate was right.</div>')
            show(cfig, "cm")
    with k2:
        html('''<div class="ptitle">Read the numbers with care</div>
        <ul class="relnotes">
        <li>Medium is the hardest level to call because it sits between Low and High.</li>
        <li>Confidence shows how strongly the model favours its answer. It is not a guarantee.</li>
        <li>Estimates use only what is known before posting. Views, likes, comments and shares you add afterwards
        are used only to place your post on the audience map.</li>
        <li>The model learned from 5,000 simulated example posts built for learning purposes, so treat every number
        as illustrative guidance rather than advice about a real platform.</li>
        </ul>''')

html(f'<div class="foot"><div>Educational project &middot; not Instagram&rsquo;s algorithm</div>'
     f'<div>runs locally &middot; nothing is uploaded</div></div>')
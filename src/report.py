"""
report.py
---------
Builds the downloadable PDF report using ONLY matplotlib (no extra PDF libraries).

    pdf_bytes = build_report_pdf(result, artifacts)

The words come from inference.report_content(); this file only lays them out on A4 pages:
a small layout engine that measures text, wraps it, adds page breaks and draws charts.
"""
import io
import math
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from src.inference import report_content

# ---- palette (same as the dashboard) ----
INK, INK2, INK3 = "#111111", "#55554F", "#8A8A84"
HAIR, PANEL = "#DCDBD5", "#FCFCFA"
BLUE, ORANGE, GREY, RED = "#1E3FD8", "#E0561B", "#BDBDB8", "#B5474F"
SEG_COLOR = {"Casual Scrollers": GREY, "Conversation Starters": BLUE, "Viral Sharers": ORANGE, "Loyal Fans": "#1A1A1A"}

# ---- page geometry (inches, A4) ----
W, H = 8.27, 11.69
MX = 0.75
CW = W - 2 * MX
BOTTOM = 0.9


class Doc:
    """Minimal top-to-bottom page layout: keeps a cursor `y`, starts a new page when a block does not fit."""

    def __init__(self, pdf, generated: str):
        self.pdf, self.generated, self.page = pdf, generated, 0
        self.fig = None
        self.new_page()

    # ------------------------------------------------------------ pages
    def new_page(self):
        if self.fig is not None:
            self._finish_page()
        self.page += 1
        self.fig = plt.figure(figsize=(W, H), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)                      # y grows downwards, like a page
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()
        self.text(MX, 0.48, "Content & Audience Predictor", 8.5, "bold")
        self.text(MX + self.width("Content & Audience Predictor", 8.5, "bold") + 0.12, 0.48, "Run report", 8.5, color=INK3)
        self.text(W - MX, 0.48, self.generated, 7.5, color=INK3, ha="right")
        self.rule(0.72, INK, 0.9)
        self.y = 1.0

    def _finish_page(self):
        self.rule(H - 0.72, HAIR, 0.6)
        self.text(MX, H - 0.62, "An educational estimate: not a guarantee, and not Instagram's algorithm.", 7, color=INK3)
        self.text(W - MX, H - 0.62, f"Page {self.page}", 7, color=INK3, ha="right")
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None

    def finish(self):
        self._finish_page()

    def ensure(self, h):
        if self.y + h > H - BOTTOM:
            self.new_page()

    # ------------------------------------------------------------ primitives
    def text(self, x, y, s, size=9.5, weight="normal", color=INK, ha="left", va="top", **kw):
        return self.ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, **kw)

    def rule(self, y, color=HAIR, lw=0.6, x0=MX, x1=W - MX):
        self.ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt")

    def width(self, s, size, weight="normal"):
        t = self.ax.text(0, 0, s, fontsize=size, fontweight=weight)
        w = t.get_window_extent(self.renderer).width / self.fig.dpi
        t.remove()
        return w

    def wrap(self, s, size, width, weight="normal"):
        lines = []
        for para in s.split("\n"):
            cur = ""
            for word in para.split():
                trial = f"{cur} {word}".strip()
                if not cur or self.width(trial, size, weight) <= width:
                    cur = trial
                else:
                    lines.append(cur)
                    cur = word
            lines.append(cur)
        return lines

    @staticmethod
    def lh(size, mult=1.5):
        return size * mult / 72

    # ------------------------------------------------------------ blocks
    def headline(self, s, size=30):
        h = self.lh(size, 1.15)
        self.ensure(h + 0.2)
        self.text(MX, self.y, s, size, "bold")
        self.y += h + 0.1

    def section(self, num, title, need=1.2):
        self.ensure(need)
        self.y += 0.3
        self.text(MX, self.y, f"{num:02d}", 8, "bold")
        self.text(MX + 0.38, self.y, title.upper(), 8, "bold")
        self.y += 0.2
        self.rule(self.y, INK, 0.9)
        self.y += 0.2

    def subhead(self, s, gap=0.08):
        self.ensure(0.4)
        self.text(MX, self.y, s, 9.5, "bold")
        self.y += self.lh(9.5) + gap

    def para(self, s, size=9.5, color=INK2, gap=0.12, weight="normal", x=MX, width=CW):
        lines = self.wrap(s, size, width, weight)
        h = len(lines) * self.lh(size)
        self.ensure(h)
        for i, line in enumerate(lines):
            self.text(x, self.y + i * self.lh(size), line, size, weight, color)
        self.y += h + gap

    def bullets(self, items, size=9.5, color=INK2, marker=BLUE, gap=0.07):
        for it in items:
            lines = self.wrap(it, size, CW - 0.3)
            h = len(lines) * self.lh(size)
            self.ensure(h)
            self.ax.add_patch(Rectangle((MX + 0.02, self.y + 0.045), 0.065, 0.065, fc=marker, ec="none"))
            for i, line in enumerate(lines):
                self.text(MX + 0.25, self.y + i * self.lh(size), line, size, color=color)
            self.y += h + gap
        self.y += 0.05

    def defs(self, items, marker=None):
        """items: list of (bold title, grey body)."""
        for title, body in items:
            lines = self.wrap(body, 9, CW - 0.3)
            h = self.lh(9.5) + len(lines) * self.lh(9) + 0.1
            self.ensure(h)
            if marker:
                self.ax.add_patch(Rectangle((MX + 0.02, self.y + 0.045), 0.065, 0.065, fc=marker, ec="none"))
            self.text(MX + 0.25, self.y, title, 9.5, "bold")
            for i, line in enumerate(lines):
                self.text(MX + 0.25, self.y + self.lh(9.5) + i * self.lh(9), line, 9, color=INK2)
            self.y += h
        self.y += 0.05

    def callout(self, s, size=9.5):
        lines = self.wrap(s, size, CW - 0.4)
        h = len(lines) * self.lh(size) + 0.3
        self.ensure(h + 0.1)
        self.ax.add_patch(Rectangle((MX, self.y), CW, h, fc="#F3F2EC", ec=HAIR, lw=0.7))
        for i, line in enumerate(lines):
            self.text(MX + 0.2, self.y + 0.15 + i * self.lh(size), line, size, color=INK2)
        self.y += h + 0.15

    def facts(self, pairs, cols=4):
        rows = math.ceil(len(pairs) / cols)
        rh = 0.52
        self.ensure(rows * rh)
        for k, (label, value) in enumerate(pairs):
            r, c = divmod(k, cols)
            x, y = MX + c * (CW / cols), self.y + r * rh
            self.text(x, y, label.upper(), 6, color=INK3)
            self.text(x, y + 0.17, value, 10, "bold")
        self.y += rows * rh + 0.05

    def strip(self, cells, height=1.28):
        """Boxed row of stat cells, like the dashboard's verdict strip."""
        self.ensure(height + 0.1)
        y0, n = self.y, len(cells)
        cw = CW / n
        self.ax.add_patch(Rectangle((MX, y0), CW, height, fc=PANEL, ec=INK, lw=0.9))
        for i, c in enumerate(cells):
            x = MX + i * cw
            if i:
                self.ax.plot([x, x], [y0, y0 + height], color=HAIR, lw=0.7)
            self.text(x + 0.12, y0 + 0.13, c["label"].upper(), 5.8, color=INK3)
            size = 22
            while size > 11 and self.width(c["value"], size, "bold") > cw - 0.24:
                size -= 2
            if self.width(c["value"], size, "bold") > cw - 0.24:           # still too wide: two lines
                words = c["value"].split(" ", 1)
                self.text(x + 0.12, y0 + 0.32, words[0], 10.5, "bold")
                self.text(x + 0.12, y0 + 0.51, words[1] if len(words) > 1 else "", 10.5, "bold")
                cap_y = y0 + 0.84
            else:
                self.text(x + 0.12, y0 + 0.30, c["value"], size, "bold")
                cap_y = y0 + 0.84
            if c.get("bar"):
                bx = x + 0.12
                for frac, col in c["bar"]:
                    bw = frac * (cw - 0.24)
                    self.ax.add_patch(Rectangle((bx, y0 + 0.72), bw, 0.055, fc=col, ec="none"))
                    bx += bw
                cap_y = y0 + 0.86
            for j, line in enumerate(self.wrap(c.get("caption", ""), 6.3, cw - 0.24)[:3]):
                self.text(x + 0.12, cap_y + j * self.lh(6.3, 1.35), line, 6.3, color=INK3)
        self.y += height + 0.15

    def action_rows(self, actions):
        for a in actions:
            t_lines = self.wrap(a["title"], 10.5, CW - 1.2, "bold")
            w_lines = self.wrap(a["why"], 8.8, CW - 1.2)
            h = max(0.62, len(t_lines) * self.lh(10.5) + len(w_lines) * self.lh(8.8) + 0.24)
            self.ensure(h)
            self.rule(self.y, HAIR, 0.7)
            y = self.y + 0.11
            for i, line in enumerate(t_lines):
                self.text(MX, y + i * self.lh(10.5), line, 10.5, "bold")
            y += len(t_lines) * self.lh(10.5) + 0.02
            for i, line in enumerate(w_lines):
                self.text(MX, y + i * self.lh(8.8), line, 8.8, color=INK2)
            self.text(W - MX, self.y + 0.08, f"+{a['gain']:.1f}", 20, "bold", ha="right")
            self.text(W - MX, self.y + 0.40, "points", 6.5, color=INK3, ha="right")
            self.y += h
        self.y += 0.1

    def table(self, headers, rows, col_w):
        rh = 0.3
        self.ensure(rh * (len(rows) + 1) + 0.1)
        x_edges = [MX]
        for w in col_w:
            x_edges.append(x_edges[-1] + w)
        for j, hd in enumerate(headers):
            self.text(x_edges[j] + (0 if j == 0 else col_w[j] - 0.08), self.y, hd.upper(), 6, color=INK3,
                      ha="left" if j == 0 else "right")
        self.y += 0.2
        self.rule(self.y, INK, 0.8)
        for row in rows:
            for j, cell in enumerate(row):
                self.text(x_edges[j] + (0 if j == 0 else col_w[j] - 0.08), self.y + 0.08, cell, 9,
                          "normal" if j == 0 else "bold", ha="left" if j == 0 else "right")
            self.y += rh
            self.rule(self.y, HAIR, 0.6)
        self.y += 0.12

    def chart(self, height, draw, left=0.0, right=0.0, top=0.05, bottom=0.5, gap=0.12):
        """Draws a matplotlib chart in a box `height` inches tall. `left`/`bottom` reserve room for axis labels."""
        self.ensure(height + gap)
        x0, w = MX + left, CW - left - right
        h = height - top - bottom
        ax = self.fig.add_axes([x0 / W, 1 - (self.y + top + h) / H, w / W, h / H])
        draw(ax)
        self.y += height + gap


def _lc(t: str) -> str:
    return t[:1].lower() + t[1:]


# ======================================================================
# Charts
# ======================================================================
def _clean(ax, keep=("bottom",)):
    for side, sp in ax.spines.items():
        sp.set_visible(side in keep)
        sp.set_color(INK)
        sp.set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8)


def draw_effects(ax, eff):
    d = eff.sort_values("effect")
    ax.barh(d["label"], d["effect"], color=[BLUE if v >= 0 else ORANGE for v in d["effect"]], height=0.62)
    lim = max(0.6, float(d["effect"].abs().max()) * 1.35)
    for i, v in enumerate(d["effect"]):
        ax.text(v + (0.04 if v >= 0 else -0.04), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=7.5, color=INK)
    ax.set_xlim(-lim, lim)
    ax.axvline(0, color=INK, lw=0.9)
    ax.grid(axis="x", color=HAIR, lw=0.6)
    ax.set_axisbelow(True)
    _clean(ax)
    ax.tick_params(axis="y", length=0, labelsize=8.5, colors=INK)
    ax.set_xlabel("points of engagement, compared with a typical post", fontsize=8, color=INK2)


def draw_heat(ax, heat, cur_day, cur_hour, best_day, best_hour):
    cmap = LinearSegmentedColormap.from_list("paper_blue", ["#F4F3EE", BLUE])
    im = ax.imshow(heat.values, aspect="auto", cmap=cmap)
    days = list(heat.index)
    ax.set_yticks(range(len(days)))
    ax.set_yticklabels(days, fontsize=8.5, color=INK)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], fontsize=8)
    ax.add_patch(Rectangle((best_hour - .5, days.index(best_day) - .5), 1, 1, fill=False, ec=ORANGE, lw=2.2))
    ax.add_patch(Rectangle((cur_hour - .5, days.index(cur_day) - .5), 1, 1, fill=False, ec=INK, lw=2.2))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}%")
    cb.outline.set_visible(False)


def draw_map(ax, sample, profiles, seg):
    for name, col in SEG_COLOR.items():
        d = sample[sample["segment"] == name]
        ax.scatter(d["pc1"], d["pc2"], s=9, color=col, alpha=0.85, edgecolors="white", linewidths=0.4, label=name)
    cent = profiles["centroid_pca_xy"]
    for name, (cx, cy) in cent.items():
        if seg:
            mine = name == seg["segment"]
            ax.plot([seg["pc1"], cx], [seg["pc2"], cy], color=RED if mine else "#8A8A86",
                    lw=1.5 if mine else 0.8, ls="-" if mine else ":", zorder=3)
        ax.scatter(cx, cy, s=95, facecolors="white", edgecolors=INK, linewidths=1.8, zorder=5)
        dy = -17 if name == "Conversation Starters" else 12          # keep the two close labels apart
        ax.annotate(name, (cx, cy), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=7.5, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.25", fc=PANEL, ec=INK, lw=0.6))
    if seg:
        ax.scatter(seg["pc1"], seg["pc2"], s=150, facecolors="white", edgecolors=INK, linewidths=2.6, zorder=7)
        ax.annotate("Your post", (seg["pc1"], seg["pc2"]), xytext=(0, -17), textcoords="offset points", ha="center",
                    fontsize=7.5, color="white", zorder=8, bbox=dict(boxstyle="square,pad=0.25", fc=INK, ec=INK))
    ys = list(sample["pc2"]) + ([seg["pc2"]] if seg else [])
    ax.set_ylim(min(ys) - 1.0, max(ys) + 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Watch time and comments  \u2192", fontsize=8.5, color=INK2)
    ax.set_ylabel("Sharing  \u2192", fontsize=8.5, color=INK2)
    _clean(ax, keep=("bottom", "left"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False, fontsize=7.5, markerscale=1.6)


# ======================================================================
# The report
# ======================================================================
def build_report_pdf(r: dict, art: dict) -> bytes:
    c = report_content(r, art)
    inp, th = r["inputs"], art["thresholds"]
    seg = r["segment"]
    aud = c["audience"]
    buf = io.BytesIO()
    now = datetime.now().strftime("%d %b %Y, %H:%M")

    rc = {"font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.unicode_minus": False}
    with matplotlib.rc_context(rc), PdfPages(buf, metadata={"Title": "Content & Audience Predictor: run report",
                                                             "Author": "Content & Audience Predictor"}) as pdf:
        d = Doc(pdf, now)

        # ---------------- 01 verdict ----------------
        d.section(1, "The verdict")
        d.headline(r["headline"])
        d.para(r["sub"], size=10.5, gap=0.2)
        p = r["probs"]
        actual = f"actual from your numbers: {r['er_actual']:.1f}%" if r["er_actual"] is not None else ""
        d.strip([
            {"label": "Predicted performance", "value": r["label"],
             "caption": f"Low < {th['low_medium_cutoff']:.1f}%, High > {th['medium_high_cutoff']:.1f}% engagement"},
            {"label": "Estimated engagement", "value": f"{r['er_pred']:.1f}%",
             "caption": f"typical error \u00b1{r['mae']:.1f} points. {actual}".strip()},
            {"label": "Model confidence", "value": f"{r['confidence'] * 100:.0f}%",
             "bar": [(p["Low"], ORANGE), (p["Medium"], GREY), (p["High"], BLUE)],
             "caption": f"low {p['Low'] * 100:.0f} \u00b7 mid {p['Medium'] * 100:.0f} \u00b7 high {p['High'] * 100:.0f}"},
            {"label": "Audience segment", "value": seg["segment"] if seg else "Not placed",
             "caption": "from your post results" if seg else "add post results to place it"},
            {"label": "Standing", "value": f"{r['percentile']:.0f}",
             "caption": f"ahead of {r['percentile']:.0f}% of {art['n_posts']:,} reference posts"},
        ])
        for w in r["warnings"]:
            d.para(w, size=8.5, color=RED)

        # ---------------- 02 your post ----------------
        d.section(2, "Your post")
        d.facts([
            ("Category", inp["category"]), ("Video length", f"{inp['duration_sec']} s"),
            ("Caption length", f"{inp['caption_length']} characters"), ("Hashtags", str(inp["hashtag_count"])),
            ("Posting time", f"{inp['post_day']} {int(inp['post_hour']):02d}:00"),
            ("Trending audio", "Used" if inp["trending_audio"] else "Not used"),
            ("Followers", f"{inp['followers']:,}"), ("Previous engagement", f"{inp['prev_avg_engagement']:.1f}%"),
        ])
        if inp["views"] > 0:
            d.facts([("Views", f"{inp['views']:,}"), ("Likes", f"{inp['likes']:,}"), ("Comments", f"{inp['comments']:,}"),
                     ("Shares", f"{inp['shares']:,}"), ("Avg. watch", f"{inp['watch']}%")], cols=5)

        # ---------------- 03 drivers ----------------
        d.section(3, "What is driving this result", need=4.0)
        d.para("Each bar shows how far one input pushed this post's estimated engagement up (blue) or down (orange), "
               "compared with a typical post.", size=9)
        d.chart(2.7, lambda ax: draw_effects(ax, r["effects"]), left=1.6, bottom=0.55)
        if c["lifts"]:
            d.subhead("Lifting your result")
            d.bullets(c["lifts"], marker=BLUE)
        if c["drags"]:
            d.subhead("Holding it back")
            d.bullets(c["drags"], marker=ORANGE)
        if not c["lifts"] and not c["drags"]:
            d.para("No single input moves this estimate by more than 0.2 points; the result is a balance of many small effects.")

        # ---------------- 04 improve ----------------
        d.section(4, "How to make it better")
        if c["actions"]:
            d.para(f"Each change below was tested on its own, keeping everything else about the post the same. "
                   f"With a typical error of \u00b1{r['mae']:.1f} points, treat the smaller gains as hints, not promises.", size=9)
            d.action_rows(c["actions"])
        else:
            d.callout("No single change is predicted to lift engagement noticeably. This post already sits close to what the "
                      "model rewards, so the best next step is to test it and watch how real viewers react.")
        if c["keep"] and c["actions"]:
            d.subhead("Keep doing")
            d.bullets(c["keep"], marker=BLUE)

        bt = c["best_time"]
        d.ensure(4.0)
        d.subhead("Best time to post")
        d.para(f"For this post, the strongest slot is {bt['day']} at {bt['hour']:02d}:00 (about {bt['value']:.1f}% engagement). "
               f"Your current choice, {bt['current_day']} at {bt['current_hour']:02d}:00, is estimated at {bt['current_value']:.1f}%.",
               size=9.5, gap=0.06)
        d.chart(2.6, lambda ax: draw_heat(ax, r["heat"], bt["current_day"], bt["current_hour"], bt["day"], bt["hour"]), left=0.5, bottom=0.4)
        d.para("Black square: your current choice. Orange square: the strongest slot.", size=8, color=INK3)

        # ---------------- 05 audience ----------------
        d.section(5, "Your audience and how to tweak it", need=3.8 if aud["placed"] else 2.6)
        if aud["placed"]:
            d.para(f"Based on how people reacted to your post, it reached {aud['segment']}. {aud['description']}", size=10, gap=0.15)
            d.subhead("How your post compares with this audience")
            d.table(["Measure", "Your post", "Audience typical"],
                    [(m, f"{y:.0f}%", f"{t:.0f}%") for m, y, t in aud["table"]], [3.4, 1.7, CW - 5.1])
            d.subhead(f"What works for {aud['segment']}")
            d.bullets(aud["tactics"], marker=BLUE)
            d.subhead("If you want to reach a different audience")
            d.para("Listed from the audience closest to yours to the furthest.", size=8.5, color=INK3, gap=0.06)
            d.defs([(f"To lean toward {s['segment']}", f"{s['description']} A good first step: {_lc(s['tactic'])}") for s in aud["shift"]],
                   marker=ORANGE)
            d.ensure(4.9)
            d.subhead("Where your post sits among other audiences")
            d.chart(4.0, lambda ax: draw_map(ax, art["audience_map"].sample(700, random_state=42), art["profiles"], seg),
                    left=0.45, bottom=0.95, gap=0.1)
            d.para("Each dot is one reference post. Posts that sit close together reached similar audiences.", size=8, color=INK3)
        else:
            d.para("Your post has not been placed on the audience map yet. After publishing, add its views, likes, comments, "
                   "shares and average watch time in the dashboard and run the analysis again. You will get a version of this "
                   "section showing which audience you reached and how to tweak the post for it.", size=9.5)
            d.subhead("The four audience types")
            d.defs([(o["segment"], f"{o['description']} Best lever: {o['lever']}") for o in aud["overview"]], marker=BLUE)

        # ---------------- 06 reading ----------------
        d.section(6, "Reading this report")
        d.para("Where the numbers come from. The estimates come from machine-learning models trained on 5,000 simulated example "
               "posts built for this educational project. They show how the method works and which patterns it can pick up, "
               "but they are not measurements of any real platform and not Instagram's algorithm. Use them as directional "
               "guidance and check every suggestion against your own results.", size=9)
        d.para("Confidence shows how strongly the model favours its answer; it is not a guarantee. Suggestions change one thing "
               "at a time, so combining several changes may not add up. Results you enter after publishing are used only to "
               "place your post among audiences and never change the estimate.", size=9)
        d.finish()
    return buf.getvalue()
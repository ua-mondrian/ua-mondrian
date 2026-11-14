"""Рисунки 1–5 статьи (файл -> номер рисунка в статье).

fig1_pipeline.png       — рис. 1: схема конвейера UA-Mondrian (D_public / D_ML, P_aux, пост-валидация);
fig3_delta_f1.png       — рис. 2: ΔF1 = UA-Local − B4 по 10 сплитам, средние и 95 % ДИ (Nadeau–Bengio);
fig2_f1_vs_k.png        — рис. 3: Macro-F1 в зависимости от k для UA-Local и B4, уровень Raw (RF);
fig4_ncp_dm_w_vs_k.png  — рис. 4: NCP и DM_w в зависимости от k;
fig5_attacks_vs_k.png   — рис. 5: LinkageRisk и SAI-gain в основном сценарии (Prosecutor).

Данные — только сводки results/*.csv и, для рис. 5, results_posthoc/attacks_halfopen_summary.csv.
Сохранение: figures/*.png, 300 dpi, русские подписи.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C

RES = C.RESULT_DIR
FIG = C.FIG_DIR

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 300, "savefig.dpi": 300,
})
COL_LOCAL = "#1f77b4"
COL_B4 = "#d62728"
METHOD_COLORS = {
    "B2": "#7f7f7f", "B4": "#d62728", "B4m": "#ff7f0e", "A1": "#2ca02c",
    "A2": "#9467bd", "UA-Global": "#17becf", "UA-Local": "#1f77b4",
}


def _load(name: str) -> pd.DataFrame:
    path = RES / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _mean_sd(summary: pd.DataFrame, ds, clf, k, method, metric) -> tuple[float, float]:
    g = summary[(summary["dataset"] == ds) & (summary["classifier"] == clf)
                & (summary["k"] == float(k))  # NaN (Raw/B1) сам отсечётся
                & (summary["method"] == method) & (summary["metric"] == metric)]
    if g.empty:
        return np.nan, np.nan
    row = g.iloc[0]
    return float(row["mean"]), float(row["sd"])


class _CommaFormatter(matplotlib.ticker.Formatter):
    """Обёртка над штатным форматтером оси: десятичная запятая вместо точки (как в тексте)."""

    def __init__(self, base):
        self.base = base

    def set_axis(self, axis):
        super().set_axis(axis)
        self.base.set_axis(axis)

    def set_locs(self, locs):
        self.base.set_locs(locs)

    def __call__(self, x, pos=None):
        return self.base(x, pos).replace(".", ",")

    def get_offset(self):
        return self.base.get_offset().replace(".", ",")


def _comma_axes(fig) -> None:
    """Десятичная запятая на всех числовых осях рисунка (как в тексте статьи)."""
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            fmt = axis.get_major_formatter()
            if isinstance(fmt, (_CommaFormatter, matplotlib.ticker.FixedFormatter)):
                continue
            axis.set_major_formatter(_CommaFormatter(fmt))


FIG_W = 6.0  # дюймов: при вставке в DOCX шириной 14 см масштаб ≈ 0,92


def _fits(fig, inner_bbox, artist, margin_px=2.0) -> bool:
    """True, если габариты текста лежат внутри inner_bbox (в пикселях)."""
    bb = artist.get_window_extent(renderer=fig.canvas.get_renderer())
    return (bb.x0 >= inner_bbox.x0 + margin_px and bb.x1 <= inner_bbox.x1 - margin_px
            and bb.y0 >= inner_bbox.y0 + margin_px and bb.y1 <= inner_bbox.y1 - margin_px)


def fig1_pipeline() -> None:
    """Схема конвейера UA-Mondrian (без заголовка: подпись — в DOCX).

    Переносы строк заданы вручную по ширине блока; после отрисовки каждая
    подпись проверяется на попадание в рамку, при выходе за рамку кегль
    уменьшается (не ниже 6 пт), иначе — ошибка. Свободные подписи
    проверяются на непересечение с блоками.
    """
    from matplotlib.transforms import Bbox

    fig, ax = plt.subplots(figsize=(FIG_W, 4.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995)
    boxes = []   # (inner Bbox в данных, text artist)

    def box(x, y, w, h, text, fc="#eef2f7", fs=8.0, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=fc, ec="#345", lw=1.0))
        t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    fontsize=fs, fontweight="bold" if bold else "normal",
                    linespacing=1.15)
        boxes.append(((x, y, w, h), t))

    def arrow(x1, y1, x2, y2, color="#345"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=10, color=color, lw=1.0))

    labels = []

    def label(x, y, text, color="#345", fs=6.8, ha="center", style="normal"):
        labels.append(ax.text(x, y, text, ha=ha, va="center", fontsize=fs,
                              color=color, style=style, linespacing=1.1))

    # --- блоки: три колонки (x = 1 / 29 / 73), три ряда ---
    box(1, 43, 20, 15, "D_train\n(80 %, сплит s)\nQI, SA, Y", fc="#e7f0e7", bold=True)
    box(29, 42, 36, 17, "UA-Mondrian\nScore = α·Ĩ(A_j; Y) − (1 − α)·w_sens\n"
                        "проверка k-анонимности\nи ℓ-разнообразия по B_SA",
        fc="#e7e7f7", fs=7.8)
    box(73, 43, 26, 15, "D_public = (QI*, B_SA)\nпубликуемая\nтаблица",
        fc="#e7f0e7", bold=True)
    box(29, 25, 36, 11, "PostValidate\n(независимая проверка,\nбез кеша алгоритма)",
        fc="#fbeeee")
    box(73, 25, 26, 12.5, "D_ML = QI* + P_aux:\nсередины интервалов,\nиерархическое и\none-hot-кодирование",
        fc="#eef", fs=7.4)
    box(1, 8, 20, 14, "D_test\n(20 %, сплит s)\nобобщается\nобученными\nправилами",
        fc="#f7f7e7", fs=7.6)
    box(29, 8, 36, 11, "P_aux (только внутреннее\nпредставление): не входит\nв публикуемую таблицу",
        fc="#fff7e0", fs=7.6)
    box(73, 7, 26, 13, "Модель ML\n(RF / XGBoost):\nобучение на D_ML,\nоценка на D_test",
        fc="#eef")

    # --- стрелки ---
    arrow(21.8, 50.5, 28.2, 50.5)            # D_train -> UA
    arrow(65.8, 50.5, 72.2, 50.5)            # UA -> D_public
    arrow(74.5, 42.0, 66.2, 35.6)            # D_public -> PostValidate
    arrow(86, 42.2, 86, 38.3)                # D_public -> D_ML
    arrow(65.8, 30.5, 72.2, 30.5)            # PostValidate -> D_ML
    arrow(86, 24.2, 86, 20.8)                # D_ML -> модель
    arrow(65.8, 16.0, 74.5, 24.3, color="#986")   # P_aux -> D_ML
    # D_test -> модель: полилиния под блоком P_aux
    ax.plot([11, 11, 80], [7.2, 4.0, 4.0], color="#986", lw=1.0)
    arrow(80, 4.0, 80, 6.2, color="#986")

    # --- свободные подписи (вне блоков) ---
    label(11, 38.5, "обучение — только\nна D_train: Dist, MI,\nиерархии, B_SA = q₄")
    label(87.5, 40.3, "те же листья", ha="left", fs=6.3)
    label(69, 32.6, "PASS", fs=6.0)
    label(45, 2.2, "D_test: те же правила обобщения и кодирования, без переобучения; Macro-F1 — один раз",
          color="#986", fs=6.4)
    label(50, 62.6, "Атаки (Prosecutor и др.) — только на D_public; "
                    "истинность сверяет доверенный оценщик",
          fs=6.6, style="italic")

    # --- автоматическая проверка: весь текст внутри рамок ---
    fig.canvas.draw()
    to_px = ax.transData
    for (x, y, w, h), t in boxes:
        inner = Bbox(to_px.transform([(x, y), (x + w, y + h)]))
        while not _fits(fig, inner, t):
            fs = t.get_fontsize() - 0.2
            if fs < 6.0:
                raise RuntimeError(f"текст не помещается в блок: {t.get_text()!r}")
            t.set_fontsize(fs)
            fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    for lab in labels:
        lb = lab.get_window_extent(renderer=rnd)
        for (x, y, w, h), t in boxes:
            pad = 0.6
            outer = Bbox(to_px.transform([(x - pad, y - pad), (x + w + pad, y + h + pad)]))
            if lb.overlaps(outer):
                raise RuntimeError(f"подпись пересекает блок: {lab.get_text()!r}")
        fb = ax.bbox
        if lb.x0 < fb.x0 or lb.x1 > fb.x1 or lb.y0 < fb.y0 or lb.y1 > fb.y1:
            raise RuntimeError(f"подпись вне поля рисунка: {lab.get_text()!r}")
    print("[fig1] кегли блоков:", [round(t.get_fontsize(), 1) for _, t in boxes])
    fig.savefig(FIG / "fig1_pipeline.png")
    plt.close(fig)


def _bottom_legend(fig, handles, labels, ncol, rect_bottom):
    """Общая легенда под графиками, вне областей данных."""
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, rect_bottom, 1, 1))


def fig2_f1_vs_k() -> None:
    """Macro-F1 vs k: UA-Local vs B4, 4 панели; одна легенда под рисунком."""
    s = _load("ml_summary.csv")
    if s.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 4.9), sharex=True)
    for i, ds in enumerate(["Adult", "Bank"]):
        for j, clf in enumerate(["RF", "XGBoost"]):
            ax = axes[i][j]
            for method, color, label in (("UA-Local", COL_LOCAL, "UA-Local"),
                                         ("B4", COL_B4, "B4 (ранжирование по IG)")):
                ks = list(C.K_GRID)
                means = [_mean_sd(s, ds, clf, k, method, "macro_f1")[0] for k in ks]
                sds = [_mean_sd(s, ds, clf, k, method, "macro_f1")[1] for k in ks]
                ax.errorbar(ks, means, yerr=sds, marker="o", ms=3.2, lw=1.3,
                            color=color, label=label, capsize=2.2)
            if clf == "RF":  # Raw по протоколу оценивался только с RF
                raw = s[(s["dataset"] == ds) & (s["classifier"] == "RF")
                        & (s["method"] == "Raw") & (s["metric"] == "macro_f1")]
                if not raw.empty:
                    ax.axhline(float(raw["mean"].iloc[0]), color="#555555", lw=1.0,
                               ls="--", label="Raw (только RF)")
            ax.set_xscale("log")
            ax.set_xticks(list(C.K_GRID))
            ax.set_xticklabels([str(k) for k in C.K_GRID])
            ax.minorticks_off()
            ax.set_title(f"{ds}, {clf}")
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel("k")
            if j == 0:
                ax.set_ylabel("Macro-F1 (mean ± SD)")
    h, l = axes[0][0].get_legend_handles_labels()
    _bottom_legend(fig, h, l, ncol=3, rect_bottom=0.06)
    _comma_axes(fig)
    fig.savefig(FIG / "fig2_f1_vs_k.png", bbox_inches="tight")
    plt.close(fig)


def fig3_delta_f1() -> None:
    """ΔF1 по 10 сплитам с 95% ДИ (Nadeau–Bengio); порог 0,01 — в легенде."""
    from matplotlib.lines import Line2D
    pairs = _load("h1_pairs.csv")
    verd = _load("h1_verdicts.csv")
    if pairs.empty or verd.empty:
        return
    configs = [(ds, clf, k) for ds in ["Adult", "Bank"]
               for clf in ["RF", "XGBoost"] for k in C.K_GRID]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.3))
    for ax, ds in zip(axes, ["Adult", "Bank"]):
        sub_cfgs = [(d, c, k) for (d, c, k) in configs if d == ds]
        lo_all = []
        for idx, (d, clf, k) in enumerate(sub_cfgs):
            row = verd[(verd["dataset"] == ds) & (verd["classifier"] == clf)
                       & (verd["k"] == k)]
            pts = pairs[(pairs["config"] == f"{ds}|{clf}|k={k}")]["delta_f1"]
            x = idx + 1
            ax.scatter([x] * len(pts), pts, s=5, color="#345", alpha=0.5,
                       zorder=3)
            lo_all.extend(pts.tolist())
            if not row.empty:
                r = row.iloc[0]
                lo, hi = r["ci_lo"], r["ci_hi"]
                if np.isfinite(lo):
                    ax.errorbar([x], [r["mean_delta_f1"]],
                                yerr=[[r["mean_delta_f1"] - lo],
                                      [hi - r["mean_delta_f1"]]],
                                fmt="D", ms=3.5, color=COL_LOCAL, capsize=2.5,
                                zorder=4)
                    lo_all.extend([lo, hi])
        lim = float(np.nanmin(lo_all)) - 0.004, \
            max(float(np.nanmax(lo_all)) + 0.006, 0.02)
        ax.set_ylim(*lim)
        ax.axhline(0.01, color="#d62728", lw=0.9, ls="--")
        ax.axhline(0.0, color="#999999", lw=0.7)
        labels = [f"{'XGB' if c == 'XGBoost' else c} k={k}" for (_, c, k) in sub_cfgs]
        ax.set_xticks(range(1, len(sub_cfgs) + 1))
        ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
        ax.set_title(ds)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("ΔF1 = UA-Local − B4")
    handles = [Line2D([], [], ls="none", marker="o", ms=3, color="#345", alpha=0.6),
               Line2D([], [], ls="none", marker="D", ms=3.5, color=COL_LOCAL),
               Line2D([], [], ls="--", lw=0.9, color="#d62728")]
    _bottom_legend(fig, handles, ["ΔF1 по сплитам", "среднее и 95 % ДИ",
                                  "порог практической значимости 0,01"],
                   ncol=3, rect_bottom=0.07)
    _comma_axes(fig)
    fig.savefig(FIG / "fig3_delta_f1.png", bbox_inches="tight")
    plt.close(fig)


def _line_panel(ax, s, ds, metric, ylabel=""):
    for method in ["B2", "B4", "B4m", "A1", "A2", "UA-Global", "UA-Local"]:
        g = s[(s["dataset"] == ds) & (s["method"] == method)
              & (s["metric"] == metric)]
        if g.empty:
            continue
        g = g.copy()
        g["k"] = g["k"].astype(float)
        g = g.sort_values("k")
        # B4, B4m и A1 численно совпадают: B4 — толстая линия снизу,
        # B4m — пунктир, A1 — точечная линия поверх, чтобы были видны все три
        st = {"B4": dict(lw=2.6, ms=4.0, zorder=2),
              "B4m": dict(lw=1.2, ms=0, ls="--", zorder=4),
              "A1": dict(lw=1.2, ms=2.2, ls=":", marker="s", zorder=5)}.get(
                  method, dict(lw=1.2, ms=3.0, zorder=3))
        st.setdefault("marker", "o")
        ax.plot(g["k"], g["mean"], color=METHOD_COLORS.get(method, None),
                label=method, **st)
    ax.set_xscale("log")
    ax.set_xticks(list(C.K_GRID))
    ax.set_xticklabels([str(k) for k in C.K_GRID])
    ax.minorticks_off()
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def _two_metric_fig(s, rows, fname) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 5.3), sharex=True)
    for row, (metric, ylab) in enumerate(rows):
        for col, ds in enumerate(["Adult", "Bank"]):
            ax = axes[row][col]
            _line_panel(ax, s, ds, metric, ylabel=ylab if col == 0 else "")
            if metric == "dm_w":
                ax.set_yscale("log")
            ax.set_title(f"{ds}")
            if row == 1:
                ax.set_xlabel("k")
    handles, labels = axes[0][0].get_legend_handles_labels()
    _bottom_legend(fig, handles, labels, ncol=7, rect_bottom=0.05)
    _comma_axes(fig)
    fig.savefig(FIG / fname, bbox_inches="tight")
    plt.close(fig)


def fig4_ncp_dm() -> None:
    """Рис. 4: NCP (верх) и DM_w (низ) vs k; легенда под рисунком."""
    s = _load("privacy_summary.csv")
    if s.empty:
        return
    _two_metric_fig(s, [("ncp", "NCP (mean)"), ("dm_w", "DM_w (mean, лог. шкала)")],
                    "fig4_ncp_dm_w_vs_k.png")


def fig5_attacks() -> None:
    """Рис. 5: LinkageRisk (верх) и SAI-gain (низ) vs k (Prosecutor)."""
    # Атаки с точной (полуоткрытой) семантикой разреза (attacks_halfopen.py);
    # сводка analysis/halfopen_summary.py -> results_posthoc/attacks_halfopen_summary.csv.
    path = C.CODE_DIR / "results_posthoc" / "attacks_halfopen_summary.csv"
    s = pd.read_csv(path) if path.exists() else _load("attacks_summary.csv")
    if s.empty:
        return
    s = s[s["scenario"] == "prosecutor"]
    _two_metric_fig(s, [("linkage_risk", "LinkageRisk (mean)"),
                        ("sai_gain", "SAI-gain (mean)")],
                    "fig5_attacks_vs_k.png")


def main() -> None:
    FIG.mkdir(exist_ok=True, parents=True)
    fig1_pipeline()
    fig2_f1_vs_k()
    fig3_delta_f1()
    fig4_ncp_dm()
    fig5_attacks()
    print(f"[fig] готово -> {FIG}")


if __name__ == "__main__":
    main()

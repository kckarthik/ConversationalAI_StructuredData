"""
Conversational Analytics Agent — Production Server
===================================================
Flask + SQLite + Ollama (Qwen 0.5B) + 7-Tool Agent
"""

import os
import re
import json
import glob
import uuid
import hashlib
import logging
import sqlite3
import traceback
from io import BytesIO
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from flask import (
    Flask, request, jsonify, send_from_directory,
    render_template, send_file
)
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests

# ─── Config ───────────────────────────────────────────────
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/app/uploads")
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
CHARTS_DIR = os.environ.get("CHARTS_DIR", "/app/static/charts")
DB_PATH = os.environ.get("DB_PATH", "/app/data/analytics.db")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
MODEL = os.environ.get("MODEL", "qwen2.5:0.5b")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# ─── Flask App ────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("analytics")

# ─── State ────────────────────────────────────────────────
table_info = {}  # {table_name: {columns:[], rows:int, dtypes:{}}}
schema_text = ""

CHART_PALETTE = [
    "#6366f1", "#f43f5e", "#10b981", "#f59e0b",
    "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16",
]
CHART_BG = "#0f0f14"
CHART_FG = "#e2e2e8"

# ═══════════════════════════════════════════════════════════
# CSV → SQLITE INGESTION
# ═══════════════════════════════════════════════════════════

def sanitize_table_name(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
    return re.sub(r"_+", "_", name).strip("_")


def ingest_csv(filepath: str) -> dict:
    """Ingest a single CSV into SQLite. Returns table info."""
    global table_info, schema_text
    table_name = sanitize_table_name(filepath)
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        # Try with different encodings
        for enc in ["latin1", "cp1252", "utf-8-sig"]:
            try:
                df = pd.read_csv(filepath, encoding=enc, low_memory=False)
                break
            except:
                continue
        else:
            return {"error": str(e)}

    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

    info = {
        "table": table_name,
        "columns": list(df.columns),
        "rows": len(df),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "file": os.path.basename(filepath),
        "size_mb": round(os.path.getsize(filepath) / 1024 / 1024, 2),
    }
    table_info[table_name] = info
    schema_text = _build_schema()
    return info


def _build_schema() -> str:
    """Build schema text for LLM context."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    parts = []
    for t in tables:
        cursor.execute(f"PRAGMA table_info('{t}');")
        cols = cursor.fetchall()
        col_str = ", ".join(f"{c[1]} ({c[2]})" for c in cols)
        cursor.execute(f"SELECT COUNT(*) FROM '{t}';")
        cnt = cursor.fetchone()[0]
        col_names = [c[1] for c in cols]
        cursor.execute(f"SELECT * FROM '{t}' LIMIT 2;")
        samples = cursor.fetchall()
        sample_str = "\n    ".join(str(dict(zip(col_names, r))) for r in samples)
        parts.append(f"Table: {t} ({cnt} rows)\n  Columns: {col_str}\n  Samples:\n    {sample_str}")
    conn.close()
    return "\n\n".join(parts)


def load_existing_tables():
    """Load table_info from existing DB on startup."""
    global table_info, schema_text
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for (t,) in cursor.fetchall():
        cursor.execute(f"PRAGMA table_info('{t}');")
        cols = cursor.fetchall()
        cursor.execute(f"SELECT COUNT(*) FROM '{t}';")
        cnt = cursor.fetchone()[0]
        table_info[t] = {
            "table": t,
            "columns": [c[1] for c in cols],
            "rows": cnt,
            "dtypes": {c[1]: c[2] for c in cols},
        }
    conn.close()
    if table_info:
        schema_text = _build_schema()
        log.info(f"Loaded {len(table_info)} existing tables from DB")


# ═══════════════════════════════════════════════════════════
# OLLAMA INTEGRATION
# ═══════════════════════════════════════════════════════════

def llm_generate(prompt: str, system: str = "") -> str:
    """Call Ollama API directly (no langchain dependency)."""
    try:
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096},
        }
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        log.error(f"Cannot connect to Ollama at {OLLAMA_URL}")
        return "[Error: Cannot connect to Ollama. Make sure it's running.]"
    except Exception as e:
        log.error(f"LLM error: {e}")
        return f"[LLM Error: {e}]"


# ═══════════════════════════════════════════════════════════
# 7 ANALYSIS TOOLS
# ═══════════════════════════════════════════════════════════

def _styled_fig(figsize=(11, 5.5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor("#16161d")
    ax.tick_params(colors=CHART_FG, labelsize=9)
    ax.xaxis.label.set_color(CHART_FG)
    ax.yaxis.label.set_color(CHART_FG)
    ax.title.set_color(CHART_FG)
    for s in ax.spines.values():
        s.set_color("#2a2a35")
    return fig, ax


def _save_chart(fig, name_hint: str) -> str:
    fname = f"{name_hint}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(CHARTS_DIR, fname)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return f"/static/charts/{fname}"


# ── TOOL 1: run_sql ──────────────────────────────────────
def tool_run_sql(query: str) -> dict:
    try:
        q = query.strip()
        if not q.upper().startswith(("SELECT", "WITH")):
            return {"error": "Only SELECT queries allowed."}
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(q, conn)
        conn.close()
        if len(df) > 100:
            df = df.head(100)
        return {"rows": len(df), "columns": list(df.columns), "data": df.to_dict("records")}
    except Exception as e:
        return {"error": str(e)}


# ── TOOL 2: create_chart ─────────────────────────────────
def tool_create_chart(sql: str, chart_type: str = "bar", x: str = "",
                      y: str = "", title: str = "Chart", color: str = "") -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        if df.empty:
            return {"error": "Query returned no data."}

        cols = list(df.columns)
        if not x: x = cols[0]
        if not y and len(cols) > 1: y = cols[1]

        fig, ax = _styled_fig()

        if chart_type == "bar":
            if color and color in df.columns:
                pivot = df.pivot_table(index=x, columns=color, values=y, aggfunc="sum").fillna(0)
                pivot.plot.bar(ax=ax, color=CHART_PALETTE[:len(pivot.columns)], edgecolor="none", width=0.75)
                ax.legend(facecolor=CHART_BG, edgecolor="#2a2a35", labelcolor=CHART_FG)
            else:
                bars = ax.bar(df[x].astype(str), df[y], color=CHART_PALETTE[:len(df)], edgecolor="none", width=0.65)
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., h, f"{h:,.0f}",
                            ha="center", va="bottom", fontsize=8, color=CHART_FG, fontweight="bold")
        elif chart_type == "line":
            if color and color in df.columns:
                for i, (name, group) in enumerate(df.groupby(color)):
                    ax.plot(group[x], group[y], label=name, marker="o",
                            color=CHART_PALETTE[i % len(CHART_PALETTE)], linewidth=2, markersize=4)
                ax.legend(facecolor=CHART_BG, edgecolor="#2a2a35", labelcolor=CHART_FG)
            else:
                ax.plot(df[x], df[y], marker="o", color=CHART_PALETTE[0], linewidth=2, markersize=4)
                ax.fill_between(range(len(df)), df[y], alpha=0.15, color=CHART_PALETTE[0])
        elif chart_type == "pie":
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.patch.set_facecolor(CHART_BG)
            ax.set_facecolor(CHART_BG)
            wedges, texts, autos = ax.pie(
                df[y], labels=df[x], autopct="%1.1f%%", startangle=90,
                colors=CHART_PALETTE[:len(df)],
                wedgeprops={"edgecolor": CHART_BG, "linewidth": 2},
                textprops={"color": CHART_FG},
            )
            for t in autos: t.set_fontweight("bold"); t.set_color("#ffffff")
        elif chart_type == "scatter":
            ax.scatter(df[x], df[y], alpha=0.65, c=CHART_PALETTE[0], s=55,
                       edgecolors="white", linewidth=0.3)
            if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
                z = np.polyfit(df[x].astype(float), df[y].astype(float), 1)
                p = np.poly1d(z)
                xl = np.linspace(float(df[x].min()), float(df[x].max()), 100)
                ax.plot(xl, p(xl), "--", color=CHART_PALETTE[1], alpha=0.8, linewidth=1.5)
        elif chart_type == "hist":
            ax.hist(df[x], bins=25, color=CHART_PALETTE[0], edgecolor=CHART_BG, alpha=0.85)
            m = float(df[x].mean())
            ax.axvline(m, color=CHART_PALETTE[1], linestyle="--", label=f"Mean: {m:,.1f}")
            ax.legend(facecolor=CHART_BG, edgecolor="#2a2a35", labelcolor=CHART_FG)
        elif chart_type == "heatmap":
            vals_col = cols[2] if len(cols) > 2 else y
            pivot = df.pivot_table(index=x, columns=y, values=vals_col, aggfunc="sum").fillna(0)
            im = ax.imshow(pivot.values, cmap="magma", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            cb = plt.colorbar(im, ax=ax)
            cb.ax.yaxis.set_tick_params(color=CHART_FG)
            cb.outline.set_edgecolor("#2a2a35")
            plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=CHART_FG)

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color=CHART_FG)
        if chart_type not in ("pie", "heatmap"):
            ax.set_xlabel(x, fontsize=10)
            ax.set_ylabel(y or "", fontsize=10)
            ax.grid(axis="y", alpha=0.1, color="#444")
            plt.xticks(rotation=40, ha="right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        chart_url = _save_chart(fig, f"chart_{chart_type}")

        summary = {}
        if y and y in df.columns and pd.api.types.is_numeric_dtype(df[y]):
            summary = {
                "max": f"{df[y].max():,.2f}", "min": f"{df[y].min():,.2f}",
                "mean": f"{df[y].mean():,.2f}",
                "max_label": str(df.loc[df[y].idxmax(), x]),
                "min_label": str(df.loc[df[y].idxmin(), x]),
            }
        return {"chart": chart_url, "summary": summary, "rows": len(df)}
    except Exception as e:
        return {"error": str(e)}


# ── TOOL 3: profile_table ────────────────────────────────
def tool_profile_table(table: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()

        profile = {
            "table": table, "rows": len(df), "columns": len(df.columns),
            "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
            "column_profiles": [],
        }

        for col in df.columns:
            cp = {"name": col, "dtype": str(df[col].dtype),
                  "nulls": int(df[col].isna().sum()), "unique": int(df[col].nunique())}
            if pd.api.types.is_numeric_dtype(df[col]):
                d = df[col].describe()
                cp.update({k: round(float(d[k]), 2) for k in ["mean", "std", "min", "25%", "50%", "75%", "max"]})
                cp["skew"] = round(float(df[col].skew()), 3)
            else:
                top = df[col].value_counts().head(5)
                cp["top_values"] = {str(k): int(v) for k, v in top.items()}
            profile["column_profiles"].append(cp)

        # Distribution chart
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
        if num_cols:
            n = len(num_cols)
            ncols = min(n, 3)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows))
            fig.patch.set_facecolor(CHART_BG)
            if n == 1: axes = np.array([axes])
            axes = axes.flatten()
            for i, col in enumerate(num_cols):
                axes[i].hist(df[col].dropna(), bins=22, color=CHART_PALETTE[i % len(CHART_PALETTE)],
                             edgecolor=CHART_BG, alpha=0.85)
                axes[i].set_title(col, fontsize=10, fontweight="bold", color=CHART_FG)
                axes[i].set_facecolor("#16161d")
                axes[i].tick_params(colors=CHART_FG, labelsize=7)
                axes[i].spines["top"].set_visible(False)
                axes[i].spines["right"].set_visible(False)
                for s in axes[i].spines.values(): s.set_color("#2a2a35")
            for j in range(n, len(axes)):
                axes[j].set_visible(False)
            plt.suptitle(f"Distributions: {table}", fontweight="bold", color=CHART_FG, fontsize=12)
            plt.tight_layout()
            profile["chart"] = _save_chart(fig, f"profile_{table}")

        return profile
    except Exception as e:
        return {"error": str(e)}


# ── TOOL 4: detect_anomalies ─────────────────────────────
def tool_detect_anomalies(table: str, column: str, method: str = "iqr") -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()

        if column not in df.columns:
            return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}

        series = df[column].dropna()
        if method == "zscore":
            z = np.abs(scipy_stats.zscore(series))
            mask = z > 2.5
        else:
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            mask = (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)

        anom_count = int(mask.sum())
        mean_val = float(series.mean())
        std_val = float(series.std())

        fig, ax = _styled_fig((12, 5))
        ax.plot(range(len(series)), series.values, color=CHART_PALETTE[0], alpha=0.5, linewidth=0.8)
        anom_idx = np.where(mask.values[:len(series)])[0]
        ax.scatter(anom_idx, series.values[anom_idx], color=CHART_PALETTE[1], s=70, zorder=5,
                   edgecolors="white", linewidth=0.5, label=f"Anomalies ({len(anom_idx)})")
        ax.axhline(mean_val, color=CHART_PALETTE[2], linestyle="--", alpha=0.6, label=f"Mean: {mean_val:,.1f}")
        ax.fill_between(range(len(series)), mean_val - 2 * std_val, mean_val + 2 * std_val,
                        alpha=0.08, color=CHART_PALETTE[2])
        ax.set_title(f"Anomaly Detection: {table}.{column} ({method.upper()})", fontweight="bold")
        ax.legend(facecolor=CHART_BG, edgecolor="#2a2a35", labelcolor=CHART_FG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        chart_url = _save_chart(fig, f"anomaly_{table}_{column}")

        return {
            "chart": chart_url, "method": method,
            "total_rows": len(df), "anomaly_count": anom_count,
            "anomaly_pct": round(anom_count / len(df) * 100, 2),
            "stats": {"mean": round(mean_val, 2), "std": round(std_val, 2),
                      "min": round(float(series.min()), 2), "max": round(float(series.max()), 2)},
        }
    except Exception as e:
        return {"error": str(e)}


# ── TOOL 5: compare_groups ───────────────────────────────
def tool_compare_groups(table: str, group_col: str, metric_col: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()

        grouped = df.groupby(group_col)[metric_col]
        agg = grouped.agg(["mean", "median", "std", "min", "max", "count"]).round(2).reset_index()
        means = grouped.mean()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.patch.set_facecolor(CHART_BG)
        for a in (ax1, ax2):
            a.set_facecolor("#16161d")
            a.tick_params(colors=CHART_FG, labelsize=9)
            a.title.set_color(CHART_FG)
            a.xaxis.label.set_color(CHART_FG)
            a.yaxis.label.set_color(CHART_FG)
            for s in a.spines.values(): s.set_color("#2a2a35")

        groups = df[group_col].unique()
        data_groups = [df[df[group_col] == g][metric_col].dropna().values for g in groups]
        bp = ax1.boxplot(data_groups, labels=groups, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(CHART_PALETTE[i % len(CHART_PALETTE)])
            patch.set_alpha(0.7)
        for w in bp["whiskers"]: w.set_color(CHART_FG); w.set_alpha(0.5)
        for c in bp["caps"]: c.set_color(CHART_FG); c.set_alpha(0.5)
        for m in bp["medians"]: m.set_color("#fff"); m.set_linewidth(2)
        ax1.set_title(f"{metric_col} Distribution", fontweight="bold")
        ax1.grid(axis="y", alpha=0.1, color="#444")
        plt.setp(ax1.get_xticklabels(), rotation=40, ha="right")

        bars = ax2.bar(means.index.astype(str), means.values,
                       color=CHART_PALETTE[:len(means)], edgecolor="none", width=0.6)
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h, f"{h:,.0f}",
                     ha="center", va="bottom", fontweight="bold", fontsize=9, color=CHART_FG)
        ax2.set_title(f"Mean {metric_col}", fontweight="bold")
        ax2.grid(axis="y", alpha=0.1, color="#444")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        plt.setp(ax2.get_xticklabels(), rotation=40, ha="right")

        plt.tight_layout()
        chart_url = _save_chart(fig, f"compare_{table}")

        best = str(means.idxmax())
        worst = str(means.idxmin())
        spread = float((means.max() - means.min()) / means.mean() * 100)

        return {
            "chart": chart_url,
            "stats": agg.to_dict("records"),
            "insights": {
                "best_group": best, "best_mean": round(float(means.max()), 2),
                "worst_group": worst, "worst_mean": round(float(means.min()), 2),
                "spread_pct": round(spread, 1), "num_groups": len(groups),
            },
        }
    except Exception as e:
        return {"error": str(e)}


# ── TOOL 6: trend_analysis ───────────────────────────────
def tool_trend_analysis(table: str, date_col: str, metric_col: str, freq: str = "M") -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        ts = df.groupby(pd.Grouper(key=date_col, freq=freq))[metric_col].sum().reset_index()
        ts.columns = ["period", "value"]
        ts = ts[ts["value"] != 0]

        window = min(3, len(ts))
        ts["ma"] = ts["value"].rolling(window=window, center=True).mean()

        if len(ts) >= 2:
            fh = ts["value"][:len(ts)//2].mean()
            sh = ts["value"][len(ts)//2:].mean()
            trend_pct = float((sh - fh) / fh * 100) if fh else 0
            trend_dir = "📈 Upward" if trend_pct > 5 else "📉 Downward" if trend_pct < -5 else "➡️ Flat"
        else:
            trend_pct, trend_dir = 0, "Insufficient data"

        peak_idx = int(ts["value"].idxmax())
        trough_idx = int(ts["value"].idxmin())

        fig, ax = _styled_fig((12, 5.5))
        ax.bar(range(len(ts)), ts["value"], color=CHART_PALETTE[0], alpha=0.35, width=0.8)
        ax.plot(range(len(ts)), ts["ma"], color=CHART_PALETTE[1], linewidth=2.5, marker="o", markersize=4)
        ax.scatter([peak_idx], [ts.loc[peak_idx, "value"]], color=CHART_PALETTE[2], s=120,
                   zorder=5, marker="^", edgecolors="white")
        ax.scatter([trough_idx], [ts.loc[trough_idx, "value"]], color=CHART_PALETTE[1], s=120,
                   zorder=5, marker="v", edgecolors="white")
        labels = [str(d.strftime("%Y-%m") if hasattr(d, "strftime") else d) for d in ts["period"]]
        ax.set_xticks(range(len(ts)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"Trend: {metric_col} ({trend_dir})", fontweight="bold", fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.1, color="#444")
        plt.tight_layout()
        chart_url = _save_chart(fig, f"trend_{table}")

        return {
            "chart": chart_url, "trend": trend_dir,
            "trend_change_pct": round(trend_pct, 1),
            "peak": {"period": str(ts.loc[peak_idx, "period"]), "value": round(float(ts.loc[peak_idx, "value"]), 2)},
            "trough": {"period": str(ts.loc[trough_idx, "period"]), "value": round(float(ts.loc[trough_idx, "value"]), 2)},
            "total_periods": len(ts),
        }
    except Exception as e:
        return {"error": str(e)}


# ── TOOL 7: correlation_matrix ───────────────────────────
def tool_correlation_matrix(table: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()

        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return {"error": "Need at least 2 numeric columns."}

        corr = num_df.corr().round(3)

        fig, ax = _styled_fig((8, 7))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        for i in range(len(corr)):
            for j in range(len(corr)):
                v = corr.values[i, j]
                c = "#fff" if abs(v) > 0.5 else CHART_FG
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=c, fontsize=9, fontweight="bold")
        cb = plt.colorbar(im, ax=ax)
        cb.outline.set_edgecolor("#2a2a35")
        cb.ax.yaxis.set_tick_params(color=CHART_FG)
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=CHART_FG)
        ax.set_title(f"Correlations: {table}", fontweight="bold", fontsize=13)
        plt.tight_layout()
        chart_url = _save_chart(fig, f"corr_{table}")

        pairs = []
        for i in range(len(corr)):
            for j in range(i+1, len(corr)):
                pairs.append({"col1": corr.columns[i], "col2": corr.columns[j], "corr": float(corr.values[i, j])})
        pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)

        return {"chart": chart_url, "top_correlations": pairs[:5]}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════
# 🧠 INTENT PLANNER + ORCHESTRATOR
# ═══════════════════════════════════════════════════════════

INTENT_PATTERNS = {
    "profile": ["profile", "describe", "overview", "summary of table", "data quality", "statistics", "what does"],
    "anomaly": ["anomal", "outlier", "unusual", "weird", "spike", "abnormal"],
    "compare": ["compare", "vs", "versus", "difference between", "by department", "by region", "by segment", "across", "group by"],
    "trend": ["trend", "over time", "monthly", "weekly", "daily", "growth", "time series", "seasonal"],
    "correlation": ["correlat", "relationship between", "related", "association", "affect", "impact on"],
    "chart": ["chart", "plot", "graph", "visuali", "show me", "draw", "pie", "bar chart", "line chart", "scatter", "histogram", "heatmap"],
    "query": [],
}


def classify_intent(msg: str) -> list:
    m = msg.lower()
    intents = [i for i, pats in INTENT_PATTERNS.items() if any(p in m for p in pats)]
    return intents or ["query"]


def detect_chart_type(msg: str) -> str:
    m = msg.lower()
    for ct in ["pie", "line", "scatter", "hist", "heatmap"]:
        if ct in m:
            return ct
    return "bar"


def detect_table(msg: str) -> str:
    m = msg.lower()
    for t in table_info:
        if t in m:
            return t
    kw_map = {
        "sales": ["revenue", "sales", "product", "units", "cost", "channel"],
        "customers": ["customer", "segment", "lifetime", "churn", "spend", "signup"],
        "employees": ["employee", "salary", "department", "tenure", "performance", "satisfaction", "hr"],
        "products": ["product info", "category", "margin", "launch", "unit_cost"],
    }
    for t, kws in kw_map.items():
        if t in table_info and any(k in m for k in kws):
            return t
    return next(iter(table_info), "")


def detect_metric(msg: str, table: str) -> str:
    m = msg.lower()
    info = table_info.get(table, {})
    cols = info.get("columns", [])
    dtypes = info.get("dtypes", {})
    for c in cols:
        if c.lower() in m:
            return c
    # First numeric column that's not an ID
    for c in cols:
        d = str(dtypes.get(c, ""))
        if any(t in d for t in ["int", "float", "REAL", "INTEGER"]) and "id" not in c.lower():
            return c
    return cols[-1] if cols else ""


def detect_group_col(msg: str, table: str) -> str:
    m = msg.lower()
    info = table_info.get(table, {})
    cols = info.get("columns", [])
    dtypes = info.get("dtypes", {})
    for c in cols:
        if c.lower() in m:
            d = str(dtypes.get(c, ""))
            if "object" in d or "TEXT" in d or c in ["region", "segment", "department", "product", "channel", "category"]:
                return c
    for c in cols:
        d = str(dtypes.get(c, ""))
        if "object" in d or "TEXT" in d:
            if "name" not in c.lower() and "date" not in c.lower() and "id" not in c.lower():
                return c
    return cols[0] if cols else ""


def generate_sql(user_msg: str) -> str:
    prompt = f"""Given this schema:
{schema_text}

Write ONLY a SQL SELECT query for: {user_msg}
Return ONLY the SQL. No explanation."""

    resp = llm_generate(prompt, system="You are a SQL expert. Return ONLY SQL, nothing else.")
    sql = re.sub(r"```sql\s*", "", resp)
    sql = re.sub(r"```", "", sql).strip().rstrip(";")
    match = re.search(r"(SELECT.+)", sql, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else None


def synthesize(user_msg: str, tool_results: list) -> str:
    results_text = ""
    for name, result in tool_results:
        r = dict(result) if isinstance(result, dict) else result
        if isinstance(r, dict):
            for k in ["data", "matrix", "anomaly_samples", "column_profiles"]:
                r.pop(k, None)
            if "stats" in r and isinstance(r["stats"], list):
                r["stats"] = r["stats"][:3]
        results_text += f"\n[{name}]: {json.dumps(r, default=str)[:600]}\n"

    prompt = f"""User asked: {user_msg}

Analysis results:
{results_text}

Give a clear, insightful answer. State specific numbers. Say which is highest/lowest. Note patterns. Be concise (3-5 sentences)."""

    return llm_generate(prompt, system="You are a data analyst. Be specific with numbers. Be concise.")


def orchestrate(user_msg: str) -> dict:
    """Main agent: classify → select tools → execute → synthesize."""
    if not table_info:
        return {"answer": "No data loaded yet. Please upload CSV files first.", "charts": [], "tools_used": []}

    intents = classify_intent(user_msg)
    table = detect_table(user_msg)
    metric = detect_metric(user_msg, table)
    group_col = detect_group_col(user_msg, table)
    chart_type = detect_chart_type(user_msg)

    log.info(f"Plan: intents={intents} table={table} metric={metric} group={group_col} chart={chart_type}")

    tool_results = []
    charts = []

    for intent in intents:
        result = None
        if intent == "profile":
            result = ("profile_table", tool_profile_table(table))
        elif intent == "anomaly":
            result = ("detect_anomalies", tool_detect_anomalies(table, metric))
        elif intent == "compare":
            result = ("compare_groups", tool_compare_groups(table, group_col, metric))
        elif intent == "trend":
            date_cols = [c for c in table_info.get(table, {}).get("columns", []) if "date" in c.lower()]
            dc = date_cols[0] if date_cols else "date"
            result = ("trend_analysis", tool_trend_analysis(table, dc, metric))
        elif intent == "correlation":
            result = ("correlation_matrix", tool_correlation_matrix(table))
        elif intent == "chart":
            sql = generate_sql(user_msg)
            if sql:
                result = ("create_chart", tool_create_chart(sql=sql, chart_type=chart_type, title=user_msg[:60]))
            else:
                fallback = f"SELECT {group_col}, SUM({metric}) as total FROM {table} GROUP BY {group_col}"
                result = ("create_chart", tool_create_chart(sql=fallback, chart_type=chart_type, title=user_msg[:60]))
        elif intent == "query":
            sql = generate_sql(user_msg)
            if sql:
                result = ("run_sql", tool_run_sql(sql))
            else:
                result = ("run_sql", {"error": "Could not generate SQL"})

        if result:
            tool_results.append(result)
            r = result[1]
            if isinstance(r, dict) and "chart" in r:
                charts.append(r["chart"])

    answer = synthesize(user_msg, tool_results)
    tools_used = [r[0] for r in tool_results]

    # Collect data for table display
    data_table = None
    for name, r in tool_results:
        if isinstance(r, dict):
            if "data" in r and isinstance(r["data"], list) and r["data"]:
                data_table = {"columns": list(r["data"][0].keys()), "rows": r["data"][:20]}
                break
            if "stats" in r and isinstance(r["stats"], list):
                data_table = {"columns": list(r["stats"][0].keys()), "rows": r["stats"][:20]}
                break

    return {
        "answer": answer,
        "charts": charts,
        "tools_used": tools_used,
        "intents": intents,
        "data_table": data_table,
    }


# ═══════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_files():
    """Upload CSV files and ingest into SQLite."""
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    results = []
    files = request.files.getlist("files")
    for f in files:
        if not f.filename:
            continue
        fname = secure_filename(f.filename)
        if not fname.lower().endswith(".csv"):
            results.append({"file": fname, "error": "Not a CSV file"})
            continue
        path = os.path.join(UPLOAD_DIR, fname)
        f.save(path)
        info = ingest_csv(path)
        results.append(info)

    return jsonify({"results": results, "tables": list(table_info.keys())})


@app.route("/api/tables", methods=["GET"])
def get_tables():
    """Return current table info."""
    return jsonify({"tables": table_info, "schema": schema_text})


@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chat endpoint."""
    data = request.json or {}
    msg = data.get("message", "").strip()
    if not msg:
        return jsonify({"error": "Empty message"}), 400
    try:
        result = orchestrate(msg)
        return jsonify(result)
    except Exception as e:
        log.error(f"Chat error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear", methods=["POST"])
def clear_data():
    """Clear all data."""
    global table_info, schema_text
    table_info = {}
    schema_text = ""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    for f in glob.glob(os.path.join(CHARTS_DIR, "*.png")):
        os.remove(f)
    return jsonify({"status": "cleared"})


@app.route("/api/health", methods=["GET"])
def health():
    # Check Ollama connectivity
    ollama_ok = False
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        ollama_ok = r.status_code == 200
    except:
        pass
    return jsonify({
        "status": "ok", "tables": len(table_info),
        "model": MODEL, "ollama": ollama_ok,
    })


# ═══════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════

load_existing_tables()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

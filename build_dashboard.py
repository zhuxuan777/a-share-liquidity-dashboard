#!/usr/bin/env python3
import argparse
import concurrent.futures
import datetime as dt
import gzip
import html
import json
import math
import os
import time
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_FILE = DATA_DIR / "all_a_daily.csv.gz"
METRICS_FILE = DATA_DIR / "market_metrics.csv"
LATEST_FILE = DATA_DIR / "latest_stock_snapshot.csv"
FAILURES_FILE = DATA_DIR / "fetch_failures.csv"
OUTPUT_FILE = BASE_DIR / "A股流动性与市场宽度看板.html"
START_DATE = "20251101"
END_DATE = dt.datetime.now(ZoneInfo("Asia/Shanghai")).date().strftime("%Y%m%d")
SOURCE_URL = "https://quote.eastmoney.com/center/gridlist.html#hs_a_board"
MARGIN_URL = "https://data.eastmoney.com/rzrq/zhtjday.html"


def fetch_history(code, start_date, end_date, retries=3):
    for attempt in range(retries):
        try:
            frame = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
                timeout=20,
            )
            if frame is None or frame.empty:
                return code, frame, None
            frame = frame.copy()
            frame["代码"] = code
            return code, frame, None
        except Exception as exc:
            if attempt == retries - 1:
                return code, None, f"{type(exc).__name__}: {exc}"
            time.sleep(0.6 * (attempt + 1))


def fetch_spot_with_retry(retries=5):
    for attempt in range(retries):
        try:
            return ak.stock_zh_a_spot_em()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"全A快照第 {attempt + 1} 次获取失败：{type(exc).__name__}: {exc}", flush=True)
            time.sleep(2 ** attempt)


def load_or_fetch(refresh, workers, limit):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists() and not refresh:
        print(f"读取缓存: {CACHE_FILE}", flush=True)
        daily = pd.read_csv(CACHE_FILE, dtype={"代码": str})
        daily["日期"] = pd.to_datetime(daily["日期"])
        return daily, pd.DataFrame(columns=["代码", "错误"]), pd.DataFrame()

    print("获取全A股票列表与最新快照...", flush=True)
    try:
        spot = fetch_spot_with_retry()
    except Exception:
        if not CACHE_FILE.exists():
            raise
        print("全A快照不可用，使用现有缓存；网页将保留缓存对应的行情日期。", flush=True)
        daily = pd.read_csv(CACHE_FILE, dtype={"代码": str})
        daily["日期"] = pd.to_datetime(daily["日期"])
        active_spot = pd.read_csv(LATEST_FILE, dtype={"代码": str}) if LATEST_FILE.exists() else pd.DataFrame()
        return daily, pd.DataFrame(columns=["代码", "错误"]), active_spot
    spot["代码"] = spot["代码"].astype(str).str.zfill(6)
    spot = spot[spot["代码"].str.fullmatch(r"\d{6}")].copy()
    spot = spot.drop_duplicates("代码")
    active_spot = spot[spot["最新价"].notna()].copy()
    active_spot.to_csv(LATEST_FILE, index=False, encoding="utf-8-sig")
    codes = spot["代码"].tolist()
    if limit:
        codes = codes[:limit]
    total = len(codes)
    print(f"开始抓取 {total} 只股票近一年日线，线程数 {workers}", flush=True)

    frames = []
    failures = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_history, code, START_DATE, END_DATE): code for code in codes}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            code, frame, error = future.result()
            if error:
                failures.append({"代码": code, "错误": error})
            elif frame is not None and not frame.empty:
                frames.append(frame)
            if completed % 100 == 0 or completed == total:
                elapsed = time.time() - started
                print(
                    f"进度 {completed}/{total}，成功 {len(frames)}，失败 {len(failures)}，耗时 {elapsed:.0f} 秒",
                    flush=True,
                )

    if not frames:
        raise RuntimeError("未获取到任何个股历史数据")
    daily = pd.concat(frames, ignore_index=True)
    daily["日期"] = pd.to_datetime(daily["日期"])
    daily = daily.sort_values(["代码", "日期"]).reset_index(drop=True)
    daily.to_csv(CACHE_FILE, index=False, compression="gzip", encoding="utf-8-sig")
    failure_frame = pd.DataFrame(failures, columns=["代码", "错误"])
    failure_frame.to_csv(FAILURES_FILE, index=False, encoding="utf-8-sig")
    return daily, failure_frame, active_spot


def calculate_metrics(daily):
    columns = ["日期", "代码", "收盘", "成交量", "成交额", "涨跌幅"]
    frame = daily[columns].copy()
    for column in ["收盘", "成交量", "成交额", "涨跌幅"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["日期", "代码"]).sort_values(["代码", "日期"])
    frame["成交额"] = frame["成交额"].clip(lower=0)
    frame["成交量"] = frame["成交量"].clip(lower=0)
    frame["有效收益"] = frame["涨跌幅"].where(frame["成交额"] > 0)
    first_observation = frame.groupby("代码").cumcount() == 0
    frame.loc[first_observation, "有效收益"] = np.nan

    grouped = frame.groupby("代码", group_keys=False)
    frame["MA20"] = grouped["收盘"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    frame["MA50"] = grouped["收盘"].transform(lambda x: x.rolling(50, min_periods=50).mean())
    frame["高于MA20"] = np.where(frame["MA20"].notna(), (frame["收盘"] > frame["MA20"]).astype(float), np.nan)
    frame["高于MA50"] = np.where(frame["MA50"].notna(), (frame["收盘"] > frame["MA50"]).astype(float), np.nan)
    frame["上涨"] = np.where(frame["有效收益"].notna(), (frame["有效收益"] > 0).astype(float), np.nan)
    frame["下跌"] = np.where(frame["有效收益"].notna(), (frame["有效收益"] < 0).astype(float), np.nan)
    frame["平盘"] = np.where(frame["有效收益"].notna(), (frame["有效收益"] == 0).astype(float), np.nan)
    frame["上涨成交量"] = frame["成交量"].where(frame["有效收益"] > 0, 0)
    frame["下跌成交量"] = frame["成交量"].where(frame["有效收益"] < 0, 0)
    frame["上涨成交额"] = frame["成交额"].where(frame["有效收益"] > 0, 0)
    frame["下跌成交额"] = frame["成交额"].where(frame["有效收益"] < 0, 0)
    frame["Amihud"] = (frame["有效收益"].abs() / (frame["成交额"] / 1e8)).where(
        (frame["成交额"] > 0) & frame["有效收益"].notna()
    )
    frame["大跌5"] = np.where(frame["有效收益"].notna(), (frame["有效收益"] <= -5).astype(float), np.nan)
    frame["大涨5"] = np.where(frame["有效收益"].notna(), (frame["有效收益"] >= 5).astype(float), np.nan)

    metrics = frame.groupby("日期").agg(
        股票数=("有效收益", "count"),
        上涨家数=("上涨", "sum"),
        下跌家数=("下跌", "sum"),
        平盘家数=("平盘", "sum"),
        成交额=("成交额", "sum"),
        上涨成交量=("上涨成交量", "sum"),
        下跌成交量=("下跌成交量", "sum"),
        上涨成交额=("上涨成交额", "sum"),
        下跌成交额=("下跌成交额", "sum"),
        个股收益中位数=("有效收益", "median"),
        Amihud中位数=("Amihud", "median"),
        大跌5家数=("大跌5", "sum"),
        大涨5家数=("大涨5", "sum"),
        MA20覆盖率=("高于MA20", "mean"),
        MA50覆盖率=("高于MA50", "mean"),
        MA20样本数=("高于MA20", "count"),
        MA50样本数=("高于MA50", "count"),
    ).reset_index()
    metrics["成交额"] = metrics["成交额"] / 1e8
    metrics["上涨家数占比"] = metrics["上涨家数"] / metrics["股票数"] * 100
    metrics["下跌家数占比"] = metrics["下跌家数"] / metrics["股票数"] * 100
    metrics["MA20覆盖率"] *= 100
    metrics["MA50覆盖率"] *= 100
    metrics["成交额MA20"] = metrics["成交额"].rolling(20, min_periods=5).mean()
    metrics["成交额相对20日"] = metrics["成交额"] / metrics["成交额MA20"]
    metrics["净上涨家数"] = metrics["上涨家数"] - metrics["下跌家数"]
    metrics["AD_Line"] = metrics["净上涨家数"].cumsum()
    ema19 = metrics["净上涨家数"].ewm(span=19, adjust=False).mean()
    ema39 = metrics["净上涨家数"].ewm(span=39, adjust=False).mean()
    metrics["McClellan"] = ema19 - ema39

    def safe_arms(count_ratio, volume_ratio):
        if not np.isfinite(count_ratio) or not np.isfinite(volume_ratio) or count_ratio <= 0 or volume_ratio <= 0:
            return np.nan
        return count_ratio / volume_ratio

    metrics["TRIN"] = metrics.apply(
        lambda row: safe_arms(
            row["上涨家数"] / row["下跌家数"] if row["下跌家数"] else np.nan,
            row["上涨成交量"] / row["下跌成交量"] if row["下跌成交量"] else np.nan,
        ),
        axis=1,
    )
    metrics["ARMS_A"] = metrics.apply(
        lambda row: safe_arms(
            row["上涨家数"] / row["下跌家数"] if row["下跌家数"] else np.nan,
            row["上涨成交额"] / row["下跌成交额"] if row["下跌成交额"] else np.nan,
        ),
        axis=1,
    )
    metrics["ARMS_A_MA5"] = metrics["ARMS_A"].rolling(5, min_periods=3).mean()
    metrics["LogARMS_A"] = np.log(metrics["ARMS_A"])

    index_frame = ak.index_zh_a_hist(
        symbol="000300", period="daily", start_date=START_DATE, end_date=END_DATE
    )
    index_frame["日期"] = pd.to_datetime(index_frame["日期"])
    index_frame["沪深300涨跌幅"] = pd.to_numeric(index_frame["涨跌幅"], errors="coerce")
    metrics = metrics.merge(index_frame[["日期", "沪深300涨跌幅"]], on="日期", how="left")
    metrics["权重个股偏离"] = metrics["沪深300涨跌幅"] - metrics["个股收益中位数"]

    margin = ak.stock_margin_account_info()
    margin["日期"] = pd.to_datetime(margin["日期"])
    for column in ["融资余额", "融资买入额", "平均维持担保比例"]:
        margin[column] = pd.to_numeric(margin[column], errors="coerce")
    margin = margin.sort_values("日期")
    margin["融资余额5日变化"] = margin["融资余额"].pct_change(5) * 100
    metrics = metrics.merge(
        margin[["日期", "融资余额", "融资买入额", "融资余额5日变化", "平均维持担保比例"]],
        on="日期",
        how="left",
    )
    metrics["融资买入占比"] = metrics["融资买入额"] / metrics["成交额"] * 100
    metrics["融资余额"] = metrics["融资余额"].ffill()
    metrics["融资余额5日变化"] = metrics["融资余额5日变化"].ffill()
    metrics["平均维持担保比例"] = metrics["平均维持担保比例"].ffill()
    metrics = metrics.sort_values("日期").reset_index(drop=True)
    return metrics, frame


def percentile(series, value):
    clean = pd.Series(series).dropna()
    if clean.empty or pd.isna(value):
        return np.nan
    return float((clean <= value).mean() * 100)


def fmt(value, digits=2, suffix=""):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}{suffix}"


def sign_text(value, digits=2, suffix=""):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+,.{digits}f}{suffix}"


def base_layout(title, yaxis_title="", height=370):
    return go.Layout(
        title={"text": title, "font": {"size": 16, "color": "#dbe7f3"}, "x": 0.02},
        height=height,
        margin={"l": 54, "r": 24, "t": 52, "b": 42},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "-apple-system, BlinkMacSystemFont, 'PingFang SC', sans-serif", "color": "#91a4b7"},
        hovermode="x unified",
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "rangeslider": {"visible": False},
        },
        yaxis={"title": yaxis_title, "gridcolor": "rgba(148,163,184,.12)", "zerolinecolor": "rgba(148,163,184,.3)"},
        legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"},
    )


def chart_html(metrics, latest_distribution):
    import plotly.io as pio

    charts = []
    config = {"responsive": True, "displaylogo": False, "scrollZoom": False}

    def render(fig, div_id):
        charts.append(
            pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs=False,
                div_id=div_id,
                config=config,
            )
        )

    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics["日期"], y=metrics["成交额"] / 10000, name="全A成交额", marker_color="#3b82f6"))
    fig.add_trace(go.Scatter(x=metrics["日期"], y=metrics["成交额MA20"] / 10000, name="20日均值", line={"color": "#f59e0b", "width": 2}))
    fig.update_layout(base_layout("全A成交额与20日均值", "万亿元"))
    render(fig, "chart-amount")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics["日期"], y=metrics["上涨家数占比"], name="上涨家数占比", line={"color": "#ef5b5b", "width": 2}))
    fig.add_trace(go.Scatter(x=metrics["日期"], y=metrics["MA20覆盖率"], name="站上MA20", line={"color": "#38bdf8", "width": 2}))
    fig.add_trace(go.Scatter(x=metrics["日期"], y=metrics["MA50覆盖率"], name="站上MA50", line={"color": "#a78bfa", "width": 1.8}))
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(148,163,184,.45)")
    fig.update_layout(base_layout("参与宽度与趋势宽度", "%"))
    fig.update_yaxes(range=[0, 100])
    render(fig, "chart-breadth")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics["日期"], y=metrics["ARMS_A"], name="ARMS_A", line={"color": "#7dd3fc", "width": 1.6}))
    fig.add_trace(go.Scatter(x=metrics["日期"], y=metrics["ARMS_A_MA5"], name="5日均值", line={"color": "#f59e0b", "width": 2.6}))
    fig.add_hline(y=1, line_dash="dash", line_color="#ef5b5b", annotation_text="1：多空均衡")
    fig.update_layout(base_layout("成交额阿姆斯指数 ARMS_A", "倍；>1卖方成交占优", height=430))
    render(fig, "chart-arms")

    colors = np.where(metrics["权重个股偏离"] >= 0, "#f59e0b", "#38bdf8")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics["日期"], y=metrics["权重个股偏离"], name="沪深300−个股中位数", marker_color=colors))
    fig.add_hline(y=0, line_color="rgba(148,163,184,.5)")
    fig.update_layout(base_layout("权重承接偏离", "百分点；正值=权重相对强"))
    render(fig, "chart-support")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics["日期"], y=metrics["Amihud中位数"], name="Amihud中位数", line={"color": "#c084fc", "width": 2}, fill="tozeroy", fillcolor="rgba(192,132,252,.08)"))
    fig.update_layout(base_layout("日频价格冲击代理", "|涨跌幅%| / 成交额亿元；越高越差"))
    render(fig, "chart-amihud")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics["日期"], y=metrics["个股收益中位数"], name="个股收益中位数", line={"color": "#38bdf8", "width": 1.8}))
    fig.add_trace(go.Bar(x=metrics["日期"], y=metrics["大跌5家数"], name="跌幅≥5%家数", marker_color="#22c55e", yaxis="y2"))
    fig.update_layout(base_layout("个股中位数与尾部下跌", "%"))
    fig.update_layout(yaxis2={"title": "家数", "overlaying": "y", "side": "right", "showgrid": False})
    render(fig, "chart-tail")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics["日期"], y=metrics["融资余额"], name="融资余额", line={"color": "#f59e0b", "width": 2}))
    fig.add_trace(go.Scatter(x=metrics["日期"], y=metrics["融资买入占比"], name="融资买入占比", line={"color": "#38bdf8", "width": 1.6}, yaxis="y2"))
    fig.update_layout(base_layout("杠杆资金", "融资余额（亿元）"))
    fig.update_layout(yaxis2={"title": "融资买入/全A成交（%）", "overlaying": "y", "side": "right", "showgrid": False})
    render(fig, "chart-margin")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics["日期"], y=metrics["McClellan"], name="McClellan Oscillator", line={"color": "#38bdf8", "width": 1.8}))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(148,163,184,.5)")
    fig.update_layout(base_layout("麦克莱伦振荡器", "EMA19(净上涨家数)−EMA39"))
    render(fig, "chart-mcclellan")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=latest_distribution, nbinsx=45, marker_color="#3b82f6", name="股票数"))
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(148,163,184,.6)")
    fig.update_layout(base_layout("最新交易日全A涨跌幅分布", "股票数量"))
    fig.update_xaxes(title="日涨跌幅（%）")
    render(fig, "chart-distribution")
    return charts


def us_chart_html(us):
    import plotly.io as pio

    charts = {}
    config = {"responsive": True, "displaylogo": False, "scrollZoom": False}

    def render(name, fig):
        charts[name] = pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id=f"chart-us-{name}", config=config)

    def unavailable(name, title):
        fig = go.Figure()
        fig.add_annotation(text="数据不可用 · 不计入信号分母", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"color": "#91a4b7", "size": 15})
        fig.update_layout(base_layout(title))
        render(name, fig)

    concentration = us.get("concentration")
    if concentration:
        series = concentration["series"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series["date"], y=series["top10"], name="Top 10", line={"color": "#f59e0b", "width": 2}))
        fig.add_trace(go.Scatter(x=series["date"], y=series["top5"], name="Top 5", line={"color": "#38bdf8", "width": 2}))
        fig.update_layout(base_layout("标普500集中度", "权重之和（%）"))
        render("concentration", fig)
    else:
        unavailable("concentration", "标普500集中度")

    ai = us["prices"].get("ai_breadth")
    if ai is not None and not ai.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ai.index, y=ai["above50"], name="高于MA50", line={"color": "#38bdf8", "width": 2}))
        fig.add_trace(go.Scatter(x=ai.index, y=ai["above200"], name="高于MA200", line={"color": "#a78bfa", "width": 2}))
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(148,163,184,.5)")
        fig.update_layout(base_layout("AI固定篮子宽度", "19只股票；占比（%）"))
        fig.update_yaxes(range=[0, 100])
        render("ai", fig)
    else:
        unavailable("ai", "AI固定篮子宽度")

    ratios = us["prices"].get("ratios", {})
    if ratios:
        labels = list(ratios)
        changes = [ratios[label]["chg20"] for label in labels]
        fig = go.Figure(go.Bar(x=labels, y=changes, marker_color=["#ef5b5b" if value >= 0 else "#22c55e" for value in changes]))
        fig.add_hline(y=0, line_color="rgba(148,163,184,.5)")
        fig.update_layout(base_layout("美股领导力：20日相对强弱变化", "比率变化（%）"))
        render("leadership", fig)
    else:
        unavailable("leadership", "美股领导力")

    cot = us.get("cot")
    if cot and cot.get("summary"):
        fig = go.Figure()
        for symbol, color in [("SPX", "#38bdf8"), ("NDX", "#f59e0b"), ("RTY", "#a78bfa")]:
            item = cot["summary"].get(symbol)
            if item:
                series = item["series"]
                series = series[series["date"] >= series["date"].max() - pd.Timedelta(days=365 * 3)]
                fig.add_trace(go.Scatter(x=series["date"], y=series["z156"], name=symbol, line={"color": color, "width": 1.8}))
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(148,163,184,.5)")
        fig.update_layout(base_layout("股指CFTC杠杆资金净持仓", "156周z-score"))
        render("cot", fig)
        ndx = cot["summary"].get("NDX")
        if ndx:
            series = ndx["series"]
            series = series[series["date"] >= series["date"].max() - pd.Timedelta(days=365 * 5)]
            fig = go.Figure(go.Scatter(x=series["date"], y=series["lev_net"], name="NDX净持仓", line={"color": "#38bdf8", "width": 2}))
            fig.add_hline(y=0, line_dash="dot", line_color="rgba(148,163,184,.5)")
            fig.update_layout(base_layout("CTA代理：NDX杠杆资金净持仓", "合约数；周度"))
            render("cta", fig)
        else:
            unavailable("cta", "CTA代理")
    else:
        unavailable("cot", "股指CFTC持仓")
        unavailable("cta", "CTA代理")

    vix = us.get("vix")
    if vix:
        series = vix["series"].tail(500)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series["VIX"], name="VIX", line={"color": "#f59e0b", "width": 1.8}))
        fig.add_trace(go.Scatter(x=series.index, y=series["VIX3M"], name="VIX3M", line={"color": "#38bdf8", "width": 1.8}))
        fig.update_layout(base_layout("VIX期限结构：现货与3个月", "指数点"))
        render("vix", fig)
        labels = ["VIX9D", "VIX", "VIX3M"]
        values = [vix["vix9d"], vix["vix"], vix["vix3m"]]
        fig = go.Figure(go.Scatter(x=labels, y=values, mode="lines+markers", line={"color": "#a78bfa", "width": 3}, marker={"size": 10}))
        fig.update_layout(base_layout("最新VIX期限曲线", f"观察日 {vix['date'].date()} · 指数点"))
        render("vixterm", fig)
    else:
        unavailable("vix", "VIX期限结构")
        unavailable("vixterm", "VIX期限曲线")

    corr = us["prices"].get("correlation")
    if corr is not None and not corr.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=corr.index, y=corr["corr2w"], name="2周", line={"color": "#f59e0b", "width": 2}))
        fig.add_trace(go.Scatter(x=corr.index, y=corr["corr1m"], name="1个月", line={"color": "#38bdf8", "width": 2}))
        fig.add_hline(y=0.60, line_dash="dash", line_color="#ef5b5b")
        fig.update_layout(base_layout("跨板块平均相关性", "11只行业ETF日收益的平均两两相关"))
        render("corr", fig)
    else:
        unavailable("corr", "跨板块相关性")

    spy = us["prices"].get("spy")
    if spy is not None and not spy.empty:
        series = spy.dropna(subset=["volume_ratio"])
        colors = np.where(series["ret1"] < 0, "#22c55e", "#ef5b5b")
        fig = go.Figure(go.Bar(x=series.index, y=series["volume_ratio"], marker_color=colors))
        fig.add_hline(y=1.5, line_dash="dash", line_color="#f59e0b", annotation_text="1.5x")
        fig.update_layout(base_layout("SPY成交量 / 此前20日均值", "绿色=下跌日；红色=上涨日"))
        render("spyvol", fig)
    else:
        unavailable("spyvol", "SPY成交量倍数")
    return charts


def build_html(metrics, frame, failures, active_spot, limited=False, us=None, output_file=OUTPUT_FILE):
    latest = metrics.iloc[-1]
    previous = metrics.iloc[-2]
    latest_date = latest["日期"].strftime("%Y-%m-%d")
    margin_rows = metrics.dropna(subset=["融资买入额"])
    margin_date = margin_rows.iloc[-1]["日期"].strftime("%Y-%m-%d") if not margin_rows.empty else "—"
    latest_frame = frame[(frame["日期"] == latest["日期"]) & frame["有效收益"].notna()]
    latest_distribution = latest_frame["有效收益"].clip(-30, 30)
    arms_state = "卖方成交占优" if latest["ARMS_A"] > 1.1 else "买方成交占优" if latest["ARMS_A"] < 0.9 else "多空成交均衡"
    breadth_state = "宽度偏强" if latest["上涨家数占比"] >= 60 else "宽度偏弱" if latest["上涨家数占比"] <= 40 else "宽度分化"
    amihud_pct = percentile(metrics["Amihud中位数"], latest["Amihud中位数"])
    arms_pct = percentile(metrics["ARMS_A"], latest["ARMS_A"])
    support_pct = percentile(metrics["权重个股偏离"], latest["权重个股偏离"])
    generated = dt.datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    coverage = int(latest["股票数"])
    failed_count = len(failures)
    universe_note = "测试样本" if limited else "东方财富当前可取全A列表"
    charts = chart_html(metrics, latest_distribution)
    us = us or {}
    us_charts = us_chart_html(us) if us else {}

    def card(label, value, note, tone=""):
        return f'''<div class="card {tone}"><div class="card-label">{label}</div><div class="card-value">{value}</div><div class="card-note">{note}</div></div>'''

    cards = [
        card("全A成交额", fmt(latest["成交额"] / 10000, 2, " 万亿"), f"20日均值的 {fmt(latest['成交额相对20日'] * 100, 0, '%')}"),
        card("上涨家数占比", fmt(latest["上涨家数占比"], 1, "%"), f"{int(latest['上涨家数'])}涨 / {int(latest['下跌家数'])}跌 · {breadth_state}", "positive" if latest["上涨家数占比"] >= 60 else ""),
        card("个股收益中位数", sign_text(latest["个股收益中位数"], 2, "%"), f"沪深300 {sign_text(latest['沪深300涨跌幅'], 2, '%')}"),
        card("成交额 ARMS_A", fmt(latest["ARMS_A"], 2), f"5日均值 {fmt(latest['ARMS_A_MA5'], 2)} · {arms_state}", "warning" if latest["ARMS_A"] > 1.1 else ""),
        card("经典 TRIN", fmt(latest["TRIN"], 2), "成交量口径 · 与ARMS_A交叉验证"),
        card("权重个股偏离", sign_text(latest["权重个股偏离"], 2, " pct"), f"近一年分位 {fmt(support_pct, 0, '%')} · 正值=权重较强"),
        card("Amihud中位数", fmt(latest["Amihud中位数"], 3), f"近一年分位 {fmt(amihud_pct, 0, '%')} · 越高代表价格冲击越大"),
        card("融资余额", fmt(latest["融资余额"], 0, " 亿"), f"5日 {sign_text(latest['融资余额5日变化'], 2, '%')} · 实际日期{margin_date}"),
        card("站上MA20", fmt(latest["MA20覆盖率"], 1, "%"), f"MA50 {fmt(latest['MA50覆盖率'], 1, '%')} · 趋势宽度"),
        card("跌幅≥5%股票", fmt(latest["大跌5家数"], 0, " 家"), f"涨幅≥5% {fmt(latest['大涨5家数'], 0, ' 家')} · 尾部宽度"),
    ]

    us_tabs_html = ""
    us_panels_html = ""
    us_method_note = ""
    if us:
        concentration = us.get("concentration")
        cot = us.get("cot")
        vix = us.get("vix")
        crypto = us.get("crypto", {})
        gex = us.get("gex", {})
        ai = us["prices"].get("ai_breadth")
        ai_latest = ai.dropna().iloc[-1] if ai is not None and not ai.dropna().empty else None
        qqq_ratio = us["prices"].get("ratios", {}).get("QQQ/SPY")
        basis_score = cot["basis_score"] if cot else np.nan
        crowding_cards = [
            card("标普500 Top10 / Top5", f"{fmt(concentration['top10'], 1, '%')} / {fmt(concentration['top5'], 1, '%')}" if concentration else "不可用", f"{concentration['snapshots']}个日度快照 · {concentration['date'].date()}" if concentration else "数据源不可用"),
            card("AI篮子高于MA50 / MA200", f"{fmt(ai_latest['above50'], 0, '%')} / {fmt(ai_latest['above200'], 0, '%')}" if ai_latest is not None else "不可用", f"固定19只供应链股票 · 实际{us['prices'].get('ai_count', 0)}只"),
            card("基差交易拥挤代理", fmt(basis_score, 0, " / 100"), "2Y/5Y/10Y：资管做多与杠杆基金做空分位的均值", "warning" if pd.notna(basis_score) and basis_score >= 80 else ""),
            card("QQQ / SPY 20日变化", sign_text(qqq_ratio["chg20"], 1, "%") if qqq_ratio else "不可用", f"3年分位 {fmt(qqq_ratio['chg20_pctile'], 0, '%')}" if qqq_ratio else ""),
        ]
        basis_rows = "".join(
            f"<tr><td>{symbol}</td><td>{fmt(item['score'], 1)}</td><td>{fmt(item['am_long_pct'], 1)}</td><td>{fmt(item['lf_short_pct'], 1)}</td></tr>"
            for symbol, item in (cot.get("basis", {}) if cot else {}).items()
        )
        amp_status_cn = {"confirmed": "已确认放大", "partial": "部分触发", "quiet": "未确认", "na": "数据不足"}.get(us.get("amp_status"), "数据不足")
        amp_cards = [
            card("放大器状态", f"{us.get('amp_on', 0)} / {us.get('amp_evaluable', 0)}", f"{amp_status_cn} · 至少2项触发才确认", "warning" if us.get("amp_on", 0) >= 2 else ""),
            card("SPY净Gamma", f"{fmt(gex['SPY']['net_gex'], 2)} bn" if "SPY" in gex else "不可用", f"Cboe期权链自算 · {gex['SPY']['asof']}" if "SPY" in gex else "不计入分母"),
            card("QQQ净Gamma", f"{fmt(gex['QQQ']['net_gex'], 2)} bn" if "QQQ" in gex else "不可用", f"每1%标的变化 · {gex['QQQ']['contracts']}张合约" if "QQQ" in gex else "不计入分母"),
            card("VIX3M − VIX", sign_text(vix["spread"], 2, " pts") if vix else "不可用", f"{'Contango' if vix['contango'] else 'Backwardation'} · {vix['date'].date()}" if vix else "不计入分母"),
        ]
        recovery_cards = [
            card("恢复条件", f"{us.get('recovery_met', 0)} / {us.get('recovery_evaluable', 0)}", "已满足 / 可评估 · 不可用项剔除分母", "positive" if us.get("recovery_evaluable") and us.get("recovery_met") == us.get("recovery_evaluable") else ""),
            card("BTC资金费率", sign_text(crypto["BTC"]["rate_pct"], 4, "%") if "BTC" in crypto else "不可用", "Gate.io永续 · 正数计为满足"),
            card("ETH资金费率", sign_text(crypto["ETH"]["rate_pct"], 4, "%") if "ETH" in crypto else "不可用", "Gate.io永续 · 正数计为满足"),
            card("NDX CTA代理", sign_text(cot["summary"]["NDX"]["wow"], 0, " 张") if cot and "NDX" in cot.get("summary", {}) else "不可用", f"周环比 · {cot['summary']['NDX']['date'].date()}" if cot and "NDX" in cot.get("summary", {}) else ""),
        ]

        def signal_table(items, on_label):
            rows = []
            for item in items:
                state = "na" if item["on"] is None else "on" if item["on"] else "off"
                state_text = "不可用" if state == "na" else on_label if state == "on" else "未满足"
                rows.append(f"<tr><td>{html.escape(item['name'])}</td><td>{html.escape(str(item['value']))}</td><td><span class='signal-pill {state}'>{state_text}</span></td><td>{html.escape(item['note'])}</td></tr>")
            return f"<section class='table-panel table-wrap'><table><thead><tr><th>信号</th><th>当前值</th><th>状态</th><th>规则与限制</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>"

        us_tabs_html = '''<button onclick="showTab('us-crowding',this)">美股拥挤</button><button onclick="showTab('us-amplifiers',this)">美股放大器</button><button onclick="showTab('us-recovery',this)">美股恢复</button>'''
        us_panels_html = f'''
<div id="panel-us-crowding" class="tab-panel"><div class="section-heading"><h2>美股拥挤：集中度、领导力与持仓</h2><span>周频COT与日频行情分开标注，不合成方向预测</span></div><section class="grid">{''.join(crowding_cards)}</section><div class="section-heading"><h2>集中度与领导力</h2><span>集中度使用原看板公开归档的SSGA快照</span></div><section class="chart-grid"><div class="chart-panel">{us_charts.get('concentration', '')}</div><div class="chart-panel">{us_charts.get('ai', '')}</div><div class="chart-panel">{us_charts.get('leadership', '')}</div><div class="chart-panel">{us_charts.get('cot', '')}</div></section><div class="section-heading"><h2>基差交易代理构成</h2><span>2010年以来CFTC历史；代理并非真实基差仓位</span></div><section class="table-panel table-wrap"><table><thead><tr><th>品种</th><th>拥挤分</th><th>资管多头分位</th><th>杠杆基金空头分位</th></tr></thead><tbody>{basis_rows}</tbody></table></section></div>
<div id="panel-us-amplifiers" class="tab-panel"><div class="section-heading"><h2>美股放大器与触发器</h2><span>负Gamma、VIX倒挂、相关性升高、下跌放量</span></div><section class="grid">{''.join(amp_cards)}</section><div class="section-heading"><h2>触发清单</h2><span>{amp_status_cn} · 不可用项不进入分母</span></div>{signal_table(us.get('amplifiers', []), '触发')}<div class="section-heading"><h2>市场结构与成交量</h2><span>GEX为模型估计，不进入恢复确认数</span></div><section class="chart-grid"><div class="chart-panel">{us_charts.get('vix', '')}</div><div class="chart-panel">{us_charts.get('corr', '')}</div><div class="chart-panel hero-chart">{us_charts.get('spyvol', '')}</div></section></div>
<div id="panel-us-recovery" class="tab-panel"><div class="section-heading"><h2>美股恢复：去杠杆是否缓和</h2><span>恢复信号描述抛售是否结束，与脆弱性分开</span></div><section class="grid">{''.join(recovery_cards)}</section><div class="section-heading"><h2>恢复确认清单</h2><span>满足 {us.get('recovery_met', 0)} / 可评估 {us.get('recovery_evaluable', 0)}</span></div>{signal_table(us.get('recovery', []), '满足')}<div class="section-heading"><h2>期限结构与CTA代理</h2><span>正资金费率不是A股看多信号，仅保留原美股框架</span></div><section class="chart-grid"><div class="chart-panel">{us_charts.get('vixterm', '')}</div><div class="chart-panel">{us_charts.get('cta', '')}</div></section></div>'''
        error_count = len(us.get("errors", {}))
        us_method_note = f'''<section class="method-panel"><b>美股模块方法与数据健康</b><ul><li>参考<a href="https://github.com/Vincizh/market-fragility-dashboard" target="_blank" rel="noopener">Market Fragility Dashboard</a>：拥挤、放大、恢复分开；恢复计数为已满足/可评估。</li><li>日线：Sina/AKShare前复权；VIX和期权链：Cboe；COT：CFTC 2010年以来官方年度包；资金费率：Gate.io；集中度：原看板归档的SSGA快照。</li><li>当前有 {error_count} 项缓存或抓取提示。GEX采用Call正、Put负的Black-Scholes Gamma聚合，前45天、最多10个到期日，至少200张有效合约和3个到期日。</li></ul></section>'''

    recent = metrics.tail(12).iloc[::-1]
    table_rows = []
    for _, row in recent.iterrows():
        table_rows.append(
            "<tr>"
            f"<td>{row['日期'].strftime('%Y-%m-%d')}</td>"
            f"<td>{fmt(row['成交额'] / 10000)}</td>"
            f"<td>{fmt(row['上涨家数占比'], 1)}%</td>"
            f"<td>{sign_text(row['个股收益中位数'])}%</td>"
            f"<td>{fmt(row['ARMS_A'])}</td>"
            f"<td>{fmt(row['TRIN'])}</td>"
            f"<td>{sign_text(row['权重个股偏离'])}</td>"
            f"<td>{fmt(row['Amihud中位数'], 3)}</td>"
            f"<td>{fmt(row['融资余额'], 0)}</td>"
            "</tr>"
        )

    methodology = {
        "ARMS_A": "(上涨家数/下跌家数) ÷ (上涨股票成交额/下跌股票成交额)。>1为卖方成交占优，<1为买方成交占优；5日均值用于降低噪声。",
        "TRIN": "经典阿姆斯指数，结构与ARMS_A相同，但成交分布使用成交量。两种口径并列，不混用。",
        "Amihud": "个股 |日涨跌幅(%)| ÷ 成交额(亿元)，再取全市场当日中位数。属于日频价格冲击代理，不是Level-2盘口深度。",
        "权重个股偏离": "沪深300日涨跌幅 − 全A个股日涨跌幅中位数。正值较大表示权重相对强，但不能据此识别买方身份。",
        "McClellan": "每日净上涨家数的EMA19 − EMA39。AD Line从样本起点累计，仅观察形态，不比较绝对零值。",
        "融资指标": "融资余额、融资买入额来自东方财富融资融券账户统计。融资余额图按上一实际披露日对齐，不使用未来数据。",
    }
    method_html = "".join(f"<li><b>{html.escape(k)}</b>：{html.escape(v)}</li>" for k, v in methodology.items())
    failure_note = f"抓取失败 {failed_count} 只，详见失败清单。" if failed_count else "本次个股历史抓取无失败代码。"
    limited_warning = '<div class="banner warning-banner">当前为测试样本，不能代表全市场。</div>' if limited else ""
    css = """
:root{--bg:#08111d;--panel:#0e1a29;--line:rgba(148,163,184,.16);--text:#e2edf7;--muted:#9bafc2;--blue:#38bdf8;--red:#ef5b5b;--amber:#f59e0b;--green:#22c55e}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 20% 0,#10253a 0,transparent 32%),var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.55;font-size:14px}a{color:var(--blue);text-decoration:none}.wrap{max-width:1500px;margin:auto;padding:22px 28px 44px}.topbar{position:sticky;top:0;z-index:60;background:rgba(8,17,29,.94);backdrop-filter:blur(16px);border-bottom:1px solid var(--line)}.topbar-inner{max-width:1500px;margin:auto;padding:10px 28px;display:flex;align-items:center;gap:22px}.mini-title{font-weight:650;white-space:nowrap}.mini-title span{display:block;font-size:10px;font-weight:400;color:var(--muted)}.tabs{display:flex;gap:5px;overflow:auto}.tabs button,.toolbar button{border:1px solid transparent;background:transparent;color:var(--muted);padding:7px 12px;border-radius:7px;cursor:pointer;white-space:nowrap;font-size:13px}.tabs button:hover,.toolbar button:hover{color:var(--text);background:rgba(56,189,248,.07)}.tabs button.active,.toolbar button.active{color:#fff;border-color:rgba(56,189,248,.5);background:rgba(56,189,248,.12)}.toplinks{margin-left:auto;white-space:nowrap;font-size:12px}.hero{display:flex;justify-content:space-between;gap:20px;align-items:flex-end;margin-bottom:14px}.eyebrow{font-size:11px;letter-spacing:.14em;color:var(--blue)}h1{font-size:26px;margin:4px 0;letter-spacing:-.02em}.subtitle,.meta{color:var(--muted);font-size:12px}.meta{text-align:right}.banner{padding:11px 15px;border:1px solid rgba(56,189,248,.22);border-radius:10px;margin:0 0 14px;background:rgba(56,189,248,.06);font-size:13px;color:#b7c8d8}.banner b{color:var(--text)}.warning-banner{background:rgba(245,158,11,.1);color:var(--amber)}.toolbar{position:sticky;top:57px;z-index:45;display:flex;gap:7px;align-items:center;margin:0 0 14px;padding:8px 0;background:rgba(8,17,29,.94);backdrop-filter:blur(12px)}.toolbar button{background:var(--panel);border-color:var(--line)}.toolbar-hint{margin-left:auto;color:var(--muted);font-size:11px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.card{background:linear-gradient(145deg,rgba(20,36,54,.98),rgba(12,25,40,.98));border:1px solid var(--line);border-radius:12px;padding:16px;min-height:112px}.card-label{font-size:12px;color:var(--muted);margin-bottom:7px}.card-value{font-size:27px;font-weight:650;letter-spacing:-.02em;font-variant-numeric:tabular-nums}.card-note{font-size:11px;color:var(--muted);margin-top:7px}.card.positive{border-top:2px solid var(--red)}.card.warning{border-top:2px solid var(--amber)}.secondary-metrics{margin:12px 0 0}.secondary-metrics summary{cursor:pointer;color:var(--muted);font-size:12px;padding:8px 2px}.secondary-grid{grid-template-columns:repeat(3,minmax(0,1fr));margin-top:8px}.tab-panel{display:none}.tab-panel.active{display:block}.section-heading{display:flex;align-items:baseline;justify-content:space-between;gap:16px;margin:24px 0 12px}.section-heading h2,.section-title{font-size:18px;margin:0}.section-heading span{color:var(--muted);font-size:12px}.chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.chart-panel,.table-panel,.method-panel{background:rgba(14,26,41,.9);border:1px solid var(--line);border-radius:12px;overflow:hidden}.hero-chart{grid-column:1/-1}.chart-help{display:flex;gap:18px;flex-wrap:wrap;padding:11px 18px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}.chart-help b{color:var(--text)}.chart-help .sell{color:var(--amber)}.chart-help .buy{color:var(--red)}.js-plotly-plot{width:100%}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}th,td{padding:10px 13px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{color:var(--muted);font-weight:500;background:rgba(8,17,29,.5)}th:first-child,td:first-child{text-align:left}.method-panel{padding:18px 22px;color:var(--muted);font-size:13px;margin-top:14px}.method-panel li{margin:8px 0}.method-panel b{color:var(--text)}.notes{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:14px}.note-box{border:1px solid var(--line);border-radius:10px;padding:15px;background:rgba(14,26,41,.65);font-size:12px;color:var(--muted)}.note-box b{display:block;color:var(--text);margin-bottom:5px}.footer{margin-top:22px;color:var(--muted);font-size:11px}.quality{display:inline-block;padding:2px 7px;border:1px solid var(--line);border-radius:5px;margin-left:5px}@media(max-width:1100px){.topbar-inner{gap:10px}.toplinks{display:none}.grid{grid-template-columns:repeat(2,1fr)}.secondary-grid{grid-template-columns:repeat(2,1fr)}.chart-grid{grid-template-columns:1fr}.notes{grid-template-columns:1fr}.hero{align-items:flex-start;flex-direction:column}.meta{text-align:left}.hero-chart{grid-column:auto}}@media(max-width:640px){.wrap{padding:16px 12px 34px}.topbar-inner{padding:8px 12px}.mini-title{display:none}.toolbar{top:53px}.grid,.secondary-grid{grid-template-columns:1fr}h1{font-size:23px}.toolbar-hint{display:none}.card{min-height:100px}.section-heading{display:block}.section-heading span{display:block;margin-top:4px}}
"""
    css += """
.signal-pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600}.signal-pill.on{color:#fbbf24;background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.35)}.signal-pill.off{color:#94a3b8;background:rgba(148,163,184,.08)}.signal-pill.na{color:#64748b;border:1px dashed rgba(100,116,139,.4)}#panel-us-recovery .signal-pill.on{color:#34d399;background:rgba(16,185,129,.1);border-color:rgba(16,185,129,.35)}
"""
    range_start = metrics["日期"].max().strftime("%Y-%m-%d")
    script = f"""
function showTab(name,btn){{document.querySelectorAll('.tab-panel').forEach(p=>p.classList.toggle('active',p.id==='panel-'+name));document.querySelectorAll('.tabs button').forEach(b=>b.classList.remove('active'));btn.classList.add('active');const panel=document.getElementById('panel-'+name);panel.querySelectorAll('.js-plotly-plot').forEach(function(el){{if(el.data)Plotly.Plots.resize(el);}});window.scrollTo({{top:0,behavior:'smooth'}});}}
function setRange(days,btn){{const end=new Date('{range_start}T00:00:00');let start=new Date(end);if(days>0)start.setDate(start.getDate()-days);else start=new Date('2000-01-01');const startText=start.toLocaleDateString('en-CA');document.querySelectorAll('.js-plotly-plot').forEach(function(el){{if(el.data&&el.data.length&&el.data[0].x&&typeof el.data[0].x[0]==='string'&&/^\d{{4}}-\d{{2}}-\d{{2}}/.test(el.data[0].x[0]))Plotly.relayout(el,{{'xaxis.range':[startText,'{range_start}']}});}});document.querySelectorAll('.toolbar button').forEach(b=>b.classList.remove('active'));btn.classList.add('active');}}
setRange(90,document.getElementById('range60'));
"""
    plotly_js = get_plotlyjs()
    html_doc = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A股流动性与市场宽度看板</title><style>{css}</style><script>{plotly_js}</script></head><body>
<div class="topbar"><div class="topbar-inner"><div class="mini-title">A股＋美股<span>{latest_date}</span></div><nav class="tabs"><button class="active" onclick="showTab('core',this)">A股核心</button><button onclick="showTab('risk',this)">A股风险</button><button onclick="showTab('leverage',this)">A股杠杆</button>{us_tabs_html}<button onclick="showTab('data',this)">数据方法</button></nav><div class="toplinks"><a href="data/market_metrics.csv" download>指标CSV</a> · <a href="data/latest_stock_snapshot.csv" download>最新快照</a></div></div></div>
<div class="wrap"><header class="hero"><div><div class="eyebrow">A-SHARE LIQUIDITY & BREADTH</div><h1>A股流动性与市场宽度看板</h1><div class="subtitle">先看当日状态，再按主题查看细节</div></div><div class="meta">行情日期 <b>{latest_date}</b> · 生成 {generated} 北京时间<br><span class="quality">{universe_note}</span><span class="quality">有效 {coverage} 只</span></div></header>
{limited_warning}<div class="banner"><b>当日速读：{breadth_state}，{arms_state}。</b> MA20覆盖率 {fmt(latest['MA20覆盖率'], 1, '%')}，融资余额5日 {sign_text(latest['融资余额5日变化'], 2, '%')}。权重偏离不用于确认国家队身份。</div>
<div class="toolbar"><button id="range20" onclick="setRange(30,this)">近20交易日</button><button id="range60" class="active" onclick="setRange(90,this)">近60交易日</button><button id="rangeall" onclick="setRange(0,this)">近一年</button><span class="toolbar-hint">默认显示近60个交易日；鼠标移到图上可查看每日数值</span></div>
<section class="grid">{''.join(cards[:4])}</section><details class="secondary-metrics"><summary>展开更多指标：TRIN、权重偏离、Amihud、融资与趋势宽度</summary><section class="grid secondary-grid">{''.join(cards[4:])}</section></details>
<div id="panel-core" class="tab-panel active"><div class="section-heading"><h2>核心：成交压力与市场宽度</h2><span>ARMS_A 为成交额口径，TRIN 为成交量口径</span></div><section class="chart-grid"><div class="chart-panel hero-chart">{charts[2]}<div class="chart-help"><span><b>当前 {fmt(latest['ARMS_A'], 2)}</b> · {arms_state}</span><span class="buy">低于1：买方成交相对占优</span><span class="sell">高于1：卖方成交相对占优</span><span>结合上涨占比与权重偏离一起判断</span></div></div><div class="chart-panel">{charts[1]}</div><div class="chart-panel">{charts[3]}</div></section></div>
<div id="panel-risk" class="tab-panel"><div class="section-heading"><h2>成交、价格冲击与尾部风险</h2><span>放量不等于增量资金，必须结合价格方向</span></div><section class="chart-grid"><div class="chart-panel">{charts[0]}</div><div class="chart-panel">{charts[4]}</div><div class="chart-panel">{charts[5]}</div><div class="chart-panel">{charts[8]}</div></section></div>
<div id="panel-leverage" class="tab-panel"><div class="section-heading"><h2>杠杆资金与宽度动量</h2><span>两融实际披露日期 {margin_date}，不与当日行情混同</span></div><section class="chart-grid"><div class="chart-panel">{charts[6]}</div><div class="chart-panel">{charts[7]}</div></section></div>
{us_panels_html}
<div id="panel-data" class="tab-panel"><div class="section-heading"><h2>最近12个交易日</h2><span>完整序列可通过顶部“指标CSV”下载</span></div><section class="table-panel table-wrap"><table><thead><tr><th>日期</th><th>成交额(万亿)</th><th>上涨占比</th><th>收益中位数</th><th>ARMS_A</th><th>TRIN</th><th>权重偏离</th><th>Amihud</th><th>融资余额(亿)</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></section><section class="method-panel"><b>指标口径</b><ul>{method_html}</ul></section><section class="notes"><div class="note-box"><b>数据质量</b>最新交易日有效样本 {coverage} 只；{html.escape(failure_note)} 历史以当前可取列表回溯，可能存在少量退市股票缺失造成的幸存者偏差。</div><div class="note-box"><b>尚未纳入</b>ETF连续份额、Level-2盘口、全市场公募日度申赎和公告级回购执行数据库。因此本版不输出“国家队实时净买入”。</div><div class="note-box"><b>数据来源</b><a href="{SOURCE_URL}">东方财富全A行情</a>、<a href="{MARGIN_URL}">融资融券账户统计</a>；沪深300行情亦取自东方财富。前复权价格用于收益与均线，成交额和成交量保持原始口径。</div></section></div>
{us_method_note}<div class="footer">仅供量化研究与数据验证，不构成任何投资建议。固定阈值需结合更长历史与市场制度变化回测。</div></div><script>{script}</script></body></html>'''
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html_doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="生成A股流动性与市场宽度量化看板")
    parser.add_argument("--refresh", action="store_true", help="重新抓取全A历史行情")
    parser.add_argument("--workers", type=int, default=12, help="历史行情抓取线程数")
    parser.add_argument("--limit", type=int, default=0, help="仅用于测试的前N只股票")
    parser.add_argument("--skip-us", action="store_true", help="仅重新生成A股模块")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="HTML输出路径")
    args = parser.parse_args()
    daily, failures, active_spot = load_or_fetch(args.refresh, args.workers, args.limit)
    print("计算市场宽度、ARMS、Amihud与两融指标...", flush=True)
    metrics, frame = calculate_metrics(daily)
    metrics.to_csv(METRICS_FILE, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"指标区间: {metrics['日期'].min().date()} 至 {metrics['日期'].max().date()}，共 {len(metrics)} 个交易日", flush=True)
    us = {}
    if not args.skip_us:
        from us_modules import build_us_bundle
        print("获取美股拥挤、放大器与恢复数据...", flush=True)
        try:
            us = build_us_bundle()
        except Exception as exc:
            print(f"美股模块整体不可用: {type(exc).__name__}: {exc}", flush=True)
    print("生成交互式HTML看板...", flush=True)
    build_html(metrics, frame, failures, active_spot, limited=bool(args.limit), us=us, output_file=args.output)
    print(f"完成: {args.output}", flush=True)


if __name__ == "__main__":
    main()

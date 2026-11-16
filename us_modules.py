import concurrent.futures
import datetime as dt
import io
import json
import math
import re
import time
import zipfile
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PRICE_DIR = DATA_DIR / "us_price_cache"
CFTC_DIR = DATA_DIR / "cftc_cache"
AI_BASKET = [
    "NVDA", "MU", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "ORCL", "VRT", "ETN",
    "PWR", "CEG", "TSM", "ANET", "DELL", "SMCI", "KLAC", "AMAT", "LRCX",
]
SECTORS = ["XLK", "XLV", "XLF", "XLI", "XLE", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC"]
PRICE_SYMBOLS = sorted(set(AI_BASKET + SECTORS + ["SPY", "RSP", "IWM", "QQQ", "SMH", "SOXX", "GLD", "USO"]))
RATIO_PAIRS = {"RSP/SPY": ("RSP", "SPY"), "IWM/SPY": ("IWM", "SPY"), "QQQ/SPY": ("QQQ", "SPY"), "SMH/SPY": ("SMH", "SPY"), "SOXX/SPY": ("SOXX", "SPY")}
COT_MARKETS = {
    "SPX": "E-MINI S&P 500",
    "NDX": "NASDAQ-100 Consolidated",
    "RTY": "RUSSELL E-MINI",
    "UST2Y": "UST 2Y NOTE",
    "UST5Y": "UST 5Y NOTE",
    "UST10Y": "UST 10Y NOTE",
}
COT_CODES = {
    "SPX": "13874A", "NDX": "20974+", "RTY": "239742",
    "UST2Y": "042601", "UST5Y": "044601", "UST10Y": "043602",
}
COT_COLUMNS = [
    "Market_and_Exchange_Names", "CFTC_Contract_Market_Code", "Report_Date_as_YYYY-MM-DD", "Open_Interest_All",
    "Asset_Mgr_Positions_Long_All", "Lev_Money_Positions_Long_All", "Lev_Money_Positions_Short_All",
]


def latest_pctile(values):
    series = pd.Series(values).dropna()
    if series.empty:
        return np.nan
    return float((series <= series.iloc[-1]).mean() * 100)


def fresh_enough(date_value, max_days):
    if date_value is None or pd.isna(date_value):
        return False
    return (dt.date.today() - pd.Timestamp(date_value).date()).days <= max_days


def fetch_price(symbol):
    cache = PRICE_DIR / f"{symbol}.csv"
    try:
        frame = ak.stock_us_daily(symbol=symbol, adjust="qfq")
        if frame is None or frame.empty:
            raise ValueError("empty price history")
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[["date", "close", "volume"]].dropna(subset=["date", "close"]).sort_values("date")
        frame.to_csv(cache, index=False)
        return symbol, frame, "Sina/AKShare", None
    except Exception as exc:
        if cache.exists():
            frame = pd.read_csv(cache)
            frame["date"] = pd.to_datetime(frame["date"])
            return symbol, frame, "local cache", f"{type(exc).__name__}: {exc}"
        return symbol, None, None, f"{type(exc).__name__}: {exc}"


def fetch_prices():
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    prices, sources, errors = {}, {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(fetch_price, symbol) for symbol in PRICE_SYMBOLS]
        for future in concurrent.futures.as_completed(futures):
            symbol, frame, source, error = future.result()
            if frame is not None and not frame.empty:
                prices[symbol] = frame.tail(1000).copy()
                sources[symbol] = source
            if error:
                errors[f"price_{symbol}"] = error
    return prices, sources, errors


def build_price_indicators(prices):
    latest_date = max(frame["date"].max() for frame in prices.values()) if prices else pd.NaT
    output = {"latest_date": latest_date, "ratios": {}}
    for label, (left, right) in RATIO_PAIRS.items():
        if left not in prices or right not in prices:
            continue
        left_frame = prices[left].set_index("date")["close"].rename("left")
        right_frame = prices[right].set_index("date")["close"].rename("right")
        joined = pd.concat([left_frame, right_frame], axis=1).dropna().sort_index()
        ratio = (joined["left"] / joined["right"]).rename("ratio").to_frame()
        ratio["ma50"] = ratio["ratio"].rolling(50).mean()
        ratio["ma200"] = ratio["ratio"].rolling(200).mean()
        ratio["chg20"] = ratio["ratio"].pct_change(20) * 100
        ratio["chg60"] = ratio["ratio"].pct_change(60) * 100
        latest = ratio.iloc[-1]
        output["ratios"][label] = {
            "series": ratio,
            "value": float(latest["ratio"]),
            "chg20": float(latest["chg20"]),
            "chg60": float(latest["chg60"]) if pd.notna(latest["chg60"]) else np.nan,
            "ma50": float(latest["ma50"]),
            "ma200": float(latest["ma200"]) if pd.notna(latest["ma200"]) else np.nan,
            "chg20_pctile": latest_pctile(ratio["chg20"].tail(756).dropna()),
            "date": ratio.index[-1],
        }

    ai_rows = []
    for symbol in AI_BASKET:
        if symbol not in prices:
            continue
        close = prices[symbol].set_index("date")["close"].sort_index()
        ai_rows.append(pd.DataFrame({
            "above50": close > close.rolling(50).mean(),
            "above200": close > close.rolling(200).mean(),
            "valid50": close.rolling(50).mean().notna(),
            "valid200": close.rolling(200).mean().notna(),
        }).assign(symbol=symbol))
    if ai_rows:
        ai = pd.concat(ai_rows)
        ai = ai.reset_index().rename(columns={"index": "date"})
        ai50 = ai[ai["valid50"]].groupby("date")["above50"].mean() * 100
        ai200 = ai[ai["valid200"]].groupby("date")["above200"].mean() * 100
        output["ai_breadth"] = pd.concat([ai50.rename("above50"), ai200.rename("above200")], axis=1).dropna(how="all")
        output["ai_count"] = int(ai["symbol"].nunique())
    else:
        output["ai_breadth"] = pd.DataFrame(columns=["above50", "above200"])
        output["ai_count"] = 0

    sector_closes = []
    for symbol in SECTORS:
        if symbol in prices:
            sector_closes.append(prices[symbol].set_index("date")["close"].rename(symbol))
    if len(sector_closes) >= 4:
        returns = pd.concat(sector_closes, axis=1).sort_index().pct_change(fill_method=None)
        corr10, corr21 = [], []
        for end in range(len(returns)):
            date = returns.index[end]
            for window, target in [(10, corr10), (21, corr21)]:
                sample = returns.iloc[max(0, end - window + 1):end + 1].dropna(how="all")
                if len(sample) < window - 2:
                    target.append((date, np.nan))
                    continue
                matrix = sample.corr().to_numpy()
                mask = np.triu(np.ones(matrix.shape, dtype=bool), 1)
                values = matrix[mask]
                values = values[np.isfinite(values)]
                target.append((date, float(values.mean()) if len(values) else np.nan))
        corr = pd.DataFrame(corr10, columns=["date", "corr2w"]).set_index("date").join(
            pd.DataFrame(corr21, columns=["date", "corr1m"]).set_index("date")
        )
        output["correlation"] = corr.dropna(how="all")
    else:
        output["correlation"] = pd.DataFrame(columns=["corr2w", "corr1m"])

    if "SPY" in prices:
        spy = prices["SPY"].set_index("date").sort_index().copy()
        spy["ret1"] = spy["close"].pct_change() * 100
        spy["volume_ratio"] = spy["volume"] / spy["volume"].shift(1).rolling(20).mean()
        output["spy"] = spy
    return output


def read_csv_url(url):
    response = requests.get(url, timeout=40)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content))


def fetch_vix():
    frames = {}
    for name in ["VIX", "VIX3M", "VIX9D"]:
        frame = read_csv_url(f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv")
        frame.columns = [column.upper() for column in frame.columns]
        frame["date"] = pd.to_datetime(frame["DATE"], format="%m/%d/%Y")
        frame[name] = pd.to_numeric(frame["CLOSE"], errors="coerce")
        frames[name] = frame.set_index("date")[[name]]
    combined = pd.concat(frames.values(), axis=1).sort_index()
    combined["VIX3M_minus_VIX"] = combined["VIX3M"] - combined["VIX"]
    combined.to_csv(DATA_DIR / "us_vix.csv", encoding="utf-8-sig")
    common = combined.dropna(subset=["VIX", "VIX3M"])
    if common.empty:
        raise ValueError("no common VIX/VIX3M date")
    latest_date = common.index[-1]
    latest = common.loc[latest_date]
    return {
        "series": combined,
        "date": latest_date,
        "vix": float(latest["VIX"]),
        "vix3m": float(latest["VIX3M"]),
        "vix9d": float(combined.loc[latest_date, "VIX9D"]) if latest_date in combined.index and pd.notna(combined.loc[latest_date, "VIX9D"]) else np.nan,
        "spread": float(latest["VIX3M_minus_VIX"]),
        "contango": bool(latest["VIX3M_minus_VIX"] > 0),
        "fresh": fresh_enough(latest_date, 4),
    }


def fetch_concentration():
    url = "https://vincizh.github.io/market-fragility-dashboard/data/spy-concentration-history.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    frame = pd.DataFrame(response.json())
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")
    frame.to_csv(DATA_DIR / "spy_concentration.csv", index=False, encoding="utf-8-sig")
    latest = frame.iloc[-1]
    return {"series": frame, "date": latest["date"], "top5": float(latest["top5"]), "top10": float(latest["top10"]), "snapshots": len(frame), "source": "SSGA snapshots archived by the reference dashboard"}


def fetch_cftc_year(year):
    cache = CFTC_DIR / f"fut_fin_txt_{year}.zip"
    if not cache.exists():
        response = requests.get(f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip", timeout=35)
        response.raise_for_status()
        cache.write_bytes(response.content)
    with zipfile.ZipFile(cache) as archive:
        name = archive.namelist()[0]
        header = pd.read_csv(archive.open(name), nrows=0)
        date_column = next(
            column for column in ["Report_Date_as_YYYY-MM-DD", "Report_Date_as_YYYY_MM_DD", "Report_Date_as_MM_DD_YYYY"]
            if column in header.columns
        )
        frame = pd.read_csv(
            archive.open(name),
            usecols=[date_column] + [column for column in COT_COLUMNS if column != "Report_Date_as_YYYY-MM-DD"],
            low_memory=False,
        )
    return frame.rename(columns={date_column: "Report_Date_as_YYYY-MM-DD"})


def fetch_cot():
    CFTC_DIR.mkdir(parents=True, exist_ok=True)
    frames, errors = [], {}
    years = list(range(2010, dt.date.today().year + 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_cftc_year, year): year for year in years}
        for future in concurrent.futures.as_completed(futures):
            year = futures[future]
            try:
                frames.append(future.result())
            except Exception as exc:
                errors[str(year)] = f"{type(exc).__name__}: {exc}"
    if not frames:
        raise RuntimeError("no CFTC history available")
    cot = pd.concat(frames, ignore_index=True)
    cot["date"] = pd.to_datetime(cot["Report_Date_as_YYYY-MM-DD"], format="mixed")
    cot["code"] = cot["CFTC_Contract_Market_Code"].astype(str).str.strip()
    cot = cot.drop_duplicates(subset=["code", "date"], keep="last")
    selected = []
    summary, basis = {}, {}
    for symbol in COT_MARKETS:
        market = cot[cot["code"] == COT_CODES[symbol]].copy()
        if market.empty:
            errors[symbol] = "market not found"
            continue
        market = market.sort_values("date")
        market["symbol"] = symbol
        market["lev_net"] = market["Lev_Money_Positions_Long_All"] - market["Lev_Money_Positions_Short_All"]
        market["net_wow"] = market["lev_net"].diff()
        market["net_chg4w"] = market["lev_net"].diff(4)
        market["z156"] = (market["lev_net"] - market["lev_net"].rolling(156).mean()) / market["lev_net"].rolling(156).std()
        selected.append(market)
        latest = market.iloc[-1]
        summary[symbol] = {
            "date": latest["date"],
            "net": float(latest["lev_net"]),
            "pctile": latest_pctile(market["lev_net"]),
            "z156": float(latest["z156"]),
            "chg4w": float(latest["net_chg4w"]),
            "wow": float(latest["net_wow"]),
            "oi": float(latest["Open_Interest_All"]),
            "fresh": fresh_enough(latest["date"], 12),
            "series": market[["date", "lev_net", "z156"]],
        }
        if symbol.startswith("UST"):
            am_pct = latest_pctile(market["Asset_Mgr_Positions_Long_All"])
            lf_short_pct = latest_pctile(market["Lev_Money_Positions_Short_All"])
            basis[symbol] = {"score": float(np.mean([am_pct, lf_short_pct])), "am_long_pct": am_pct, "lf_short_pct": lf_short_pct}
    selected_frame = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    if not selected_frame.empty:
        selected_frame[["date", "symbol", "lev_net", "net_wow", "net_chg4w", "z156"]].to_csv(DATA_DIR / "us_cot_positioning.csv", index=False, encoding="utf-8-sig")
    return {"summary": summary, "basis": basis, "basis_score": float(np.mean([item["score"] for item in basis.values()])) if basis else np.nan, "errors": errors}


def bs_gamma(spot, strike, years, sigma, rate=0.04):
    if min(spot, strike, years, sigma) <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * years) / (sigma * math.sqrt(years))
    return math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (spot * sigma * math.sqrt(years))


def fetch_gex(symbol):
    response = requests.get(f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json", timeout=50)
    response.raise_for_status()
    payload = response.json()
    asof = dt.datetime.strptime(payload["timestamp"], "%Y-%m-%d %H:%M:%S").date()
    spot = float(payload["data"]["close"])
    contracts = []
    for option in payload["data"]["options"]:
        match = re.search(r"(\d{6})([CP])(\d{8})$", option.get("option", ""))
        if not match:
            continue
        expiry = dt.datetime.strptime(match.group(1), "%y%m%d").date()
        days = (expiry - asof).days
        strike = int(match.group(3)) / 1000
        oi = float(option.get("open_interest") or 0)
        iv = float(option.get("iv") or 0)
        if 0 < days <= 45 and oi > 0 and 0.001 < iv <= 5 and strike > 0:
            contracts.append((expiry, match.group(2) == "C", strike, oi, iv))
    expiries = sorted({row[0] for row in contracts})[:10]
    contracts = [row for row in contracts if row[0] in expiries]
    if len(contracts) < 200 or len(expiries) < 3:
        raise ValueError(f"thin option chain: {len(contracts)} contracts, {len(expiries)} expiries")
    net = 0.0
    for expiry, is_call, strike, oi, iv in contracts:
        dollar = bs_gamma(spot, strike, (expiry - asof).days / 365, iv) * oi * 100 * spot * spot * 0.01
        net += dollar if is_call else -dollar
    return {"symbol": symbol, "asof": asof, "spot": spot, "net_gex": net / 1e9, "contracts": len(contracts), "expiries": len(expiries), "source": "Cboe delayed option chain; homebrew calls + / puts −"}


def fetch_crypto():
    output, errors = {}, {}
    for symbol, contract in [("BTC", "BTC_USDT"), ("ETH", "ETH_USDT")]:
        try:
            response = requests.get(f"https://api.gateio.ws/api/v4/futures/usdt/contracts/{contract}", timeout=25)
            response.raise_for_status()
            payload = response.json()
            value = payload.get("funding_rate")
            field = "funding_rate"
            if value is None:
                value = payload.get("funding_rate_indicative")
                field = "funding_rate_indicative"
            if value is None:
                raise ValueError("funding rate missing")
            output[symbol] = {"rate_pct": float(value) * 100, "field": field, "source": "Gate.io USDT perpetual", "fetched": dt.datetime.now(dt.timezone.utc).isoformat()}
        except Exception as exc:
            errors[symbol] = f"{type(exc).__name__}: {exc}"
    return output, errors


def signal(name, value, on, note=""):
    return {"name": name, "value": value, "on": on, "note": note}


def build_us_bundle():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    errors = {}
    prices, price_sources, price_errors = fetch_prices()
    errors.update(price_errors)
    price_data = build_price_indicators(prices)

    try:
        vix = fetch_vix()
    except Exception as exc:
        errors["vix"] = f"{type(exc).__name__}: {exc}"
        vix = None
    try:
        concentration = fetch_concentration()
    except Exception as exc:
        errors["concentration"] = f"{type(exc).__name__}: {exc}"
        concentration = None
    try:
        cot = fetch_cot()
        errors.update({f"cot_{key}": value for key, value in cot["errors"].items()})
    except Exception as exc:
        errors["cot"] = f"{type(exc).__name__}: {exc}"
        cot = None
    crypto, crypto_errors = fetch_crypto()
    errors.update({f"crypto_{key}": value for key, value in crypto_errors.items()})

    gex = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(fetch_gex, symbol): symbol for symbol in ["SPY", "QQQ"]}
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                gex[symbol] = future.result()
            except Exception as exc:
                errors[f"gex_{symbol}"] = f"{type(exc).__name__}: {exc}"

    corr_latest = None
    if not price_data["correlation"].empty:
        corr_valid = price_data["correlation"].dropna()
        if not corr_valid.empty:
            corr_latest = corr_valid.iloc[-1]
            corr_latest_date = corr_valid.index[-1]
        else:
            corr_latest_date = pd.NaT
    else:
        corr_latest_date = pd.NaT
    corr_available = corr_latest is not None and fresh_enough(corr_latest_date, 4)
    corr_amplifying = bool(corr_latest["corr2w"] >= 0.60 and corr_latest["corr2w"] >= corr_latest["corr1m"]) if corr_available else None
    corr_declining = bool(corr_latest["corr2w"] < corr_latest["corr1m"]) if corr_available else None

    spy_latest = None
    spy_available = False
    if "spy" in price_data:
        spy_valid = price_data["spy"].dropna(subset=["ret1", "volume_ratio"])
        if not spy_valid.empty:
            spy_latest = spy_valid.iloc[-1]
            spy_latest_date = spy_valid.index[-1]
            spy_available = fresh_enough(spy_latest_date, 4)
    spy_down_volume = bool(spy_latest["volume_ratio"] >= 1.5 and spy_latest["ret1"] < 0) if spy_available else None
    vix_backwardation = (not vix["contango"]) if vix and vix["fresh"] else None
    gex_negative = bool(gex["SPY"]["net_gex"] < 0) if "SPY" in gex and fresh_enough(gex["SPY"]["asof"], 4) else None
    amplifiers = [
        signal("SPY Dealer Gamma", f"{gex['SPY']['net_gex']:+.2f} $bn/1%" if "SPY" in gex else "不可用", gex_negative, "负Gamma=可能放大波动；模型估计"),
        signal("VIX期限结构", f"{vix['spread']:+.2f} pts" if vix else "不可用", vix_backwardation, "倒挂=短期波动需求急升"),
        signal("跨板块相关性", f"2W {corr_latest['corr2w']:.3f} / 1M {corr_latest['corr1m']:.3f}" if corr_latest is not None else "不可用", corr_amplifying, "2W≥0.60且不低于1M"),
        signal("SPY下跌放量", f"{spy_latest['volume_ratio']:.2f}x / {spy_latest['ret1']:+.2f}%" if spy_latest is not None else "不可用", spy_down_volume, "下跌日且成交量≥此前20日均值1.5倍"),
    ]
    amp_evaluable = [item for item in amplifiers if item["on"] is not None]
    amp_on = sum(item["on"] for item in amp_evaluable)
    amp_status = "confirmed" if amp_on >= 2 else "partial" if amp_on == 1 else "quiet" if amp_evaluable else "na"

    ndx = cot["summary"].get("NDX") if cot else None
    cta_covering = bool(ndx["wow"] > 0) if ndx and ndx["fresh"] else None
    recovery = [
        signal("VIX Contango", f"{vix['spread']:+.2f} pts · {vix['date'].date()}" if vix else "不可用", bool(vix["contango"]) if vix and vix["fresh"] else None, "VIX3M>VIX"),
        signal("BTC资金费率", f"{crypto['BTC']['rate_pct']:+.4f}%" if "BTC" in crypto else "不可用", bool(crypto["BTC"]["rate_pct"] > 0) if "BTC" in crypto else None, "Gate.io；与Binance口径可能不同"),
        signal("ETH资金费率", f"{crypto['ETH']['rate_pct']:+.4f}%" if "ETH" in crypto else "不可用", bool(crypto["ETH"]["rate_pct"] > 0) if "ETH" in crypto else None, "Gate.io；与Binance口径可能不同"),
        signal("CTA代理：NDX空头回补", f"WoW {ndx['wow']:+,.0f} · {ndx['date'].date()}" if ndx else "不可用", cta_covering, "CFTC杠杆资金净持仓周环比上升"),
        signal("相关性下降", f"2W {corr_latest['corr2w']:.3f} < 1M {corr_latest['corr1m']:.3f}" if corr_latest is not None else "不可用", corr_declining, "2W低于1M"),
    ]
    recovery_evaluable = [item for item in recovery if item["on"] is not None]
    recovery_met = sum(item["on"] for item in recovery_evaluable)

    bundle = {
        "prices": price_data,
        "vix": vix,
        "concentration": concentration,
        "cot": cot,
        "crypto": crypto,
        "gex": gex,
        "amplifiers": amplifiers,
        "amp_on": amp_on,
        "amp_evaluable": len(amp_evaluable),
        "amp_status": amp_status,
        "recovery": recovery,
        "recovery_met": recovery_met,
        "recovery_evaluable": len(recovery_evaluable),
        "errors": errors,
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
    }
    summary = {
        "generated": bundle["generated"],
        "amp_on": amp_on,
        "amp_evaluable": len(amp_evaluable),
        "amp_status": amp_status,
        "recovery_met": recovery_met,
        "recovery_evaluable": len(recovery_evaluable),
        "basis_score": cot["basis_score"] if cot else None,
        "errors": errors,
        "gex": {key: {k: (v.isoformat() if isinstance(v, dt.date) else v) for k, v in value.items()} for key, value in gex.items()},
        "crypto": crypto,
    }
    (DATA_DIR / "us_current_signals.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle

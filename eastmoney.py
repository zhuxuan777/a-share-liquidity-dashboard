import time

import pandas as pd
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
}
UT = "7eea3edcaed734bea9cbfc24409ed989"


def get_json(url, params, timeout=20, retries=4):
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(0.6 * (2 ** attempt))


def stock_spot():
    data = get_json(
        "https://push2.eastmoney.com/api/qt/clist/get",
        {
            "pn": "1",
            "pz": "10000",
            "po": "1",
            "np": "1",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": "2",
            "invt": "2",
            "fid": "f12",
            "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
            "fields": "f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f14,f15,f16,f17,f18,f20,f21,f22,f23,f24,f25",
        },
    )["data"]
    rows = data["diff"]
    if not rows or len(rows) < data["total"]:
        raise RuntimeError(f"全A快照分页不完整：{len(rows)}/{data['total']}")
    frame = pd.DataFrame(rows).rename(columns={
        "f12": "代码", "f14": "名称", "f2": "最新价", "f3": "涨跌幅", "f4": "涨跌额",
        "f5": "成交量", "f6": "成交额", "f7": "振幅", "f15": "最高", "f16": "最低",
        "f17": "今开", "f18": "昨收", "f10": "量比", "f8": "换手率", "f9": "市盈率-动态",
        "f23": "市净率", "f20": "总市值", "f21": "流通市值", "f22": "涨速",
        "f24": "60日涨跌幅", "f25": "年初至今涨跌幅", "f11": "5分钟涨跌",
    })
    columns = [
        "代码", "名称", "最新价", "涨跌幅", "涨跌额", "成交量", "成交额", "振幅",
        "最高", "最低", "今开", "昨收", "量比", "换手率", "市盈率-动态", "市净率",
        "总市值", "流通市值", "涨速", "60日涨跌幅", "年初至今涨跌幅", "5分钟涨跌",
    ]
    frame = frame[columns]
    for column in columns[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.insert(0, "序号", range(1, len(frame) + 1))
    return frame


def kline_frame(data, code):
    if not data or not data.get("klines"):
        return pd.DataFrame()
    frame = pd.DataFrame([item.split(",") for item in data["klines"]])
    frame = frame.iloc[:, :11]
    frame.columns = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
    frame.insert(1, "股票代码", code)
    frame["日期"] = pd.to_datetime(frame["日期"], errors="coerce").dt.date
    for column in frame.columns[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def tencent_stock_history(code, start_date, end_date):
    market = "sh" if code.startswith("6") else ("bj" if code.startswith(("4", "8", "9")) else "sz")
    symbol = market + code
    start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
    url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
    params = {"param": f"{symbol},day,{start},{end},400,qfq"}
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
            response.raise_for_status()
            payload = response.json()
            break
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(0.6 * (2 ** attempt))
    data = (payload.get("data") or {}).get(symbol) or {}
    rows = data.get("qfqday") or data.get("day") or []
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    frame = pd.DataFrame({
        "日期": pd.to_datetime(raw[0], errors="coerce"),
        "股票代码": code,
        "开盘": pd.to_numeric(raw[1], errors="coerce"),
        "收盘": pd.to_numeric(raw[2], errors="coerce"),
        "最高": pd.to_numeric(raw[3], errors="coerce"),
        "最低": pd.to_numeric(raw[4], errors="coerce"),
        "成交量": pd.to_numeric(raw[5], errors="coerce") / (100 if code.startswith(("688", "689")) else 1),
        "成交额": pd.to_numeric(raw[8], errors="coerce") * 10000,
        "换手率": pd.to_numeric(raw[7], errors="coerce"),
    })
    previous = frame["收盘"].shift(1)
    frame["涨跌幅"] = ((frame["收盘"] / previous - 1) * 100).round(2)
    frame["涨跌额"] = (frame["收盘"] - previous).round(2)
    frame["振幅"] = ((frame["最高"] - frame["最低"]) / previous * 100).round(2)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frame = frame[(frame["日期"] >= start_ts) & (frame["日期"] <= end_ts)].copy()
    frame["日期"] = frame["日期"].dt.date
    return frame[["日期", "股票代码", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]]


def stock_history(code, start_date, end_date):
    market = 1 if code.startswith("6") else 0
    try:
        data = get_json(
            "https://push2his.eastmoney.com/api/qt/stock/kline/get",
            {
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "ut": UT,
                "klt": "101",
                "fqt": "1",
                "secid": f"{market}.{code}",
                "beg": start_date,
                "end": end_date,
            },
            retries=1,
        ).get("data")
        frame = kline_frame(data, code)
        if not frame.empty:
            return frame
    except (requests.RequestException, ValueError, KeyError):
        pass
    return tencent_stock_history(code, start_date, end_date)


def index_history(code, start_date, end_date):
    market = 1 if code.startswith(("0", "9")) else 0
    data = get_json(
        "https://push2his.eastmoney.com/api/qt/stock/kline/get",
        {
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "ut": UT,
            "klt": "101",
            "fqt": "0",
            "secid": f"{market}.{code}",
            "beg": start_date,
            "end": end_date,
        },
    ).get("data")
    return kline_frame(data, code)


def margin_account_info():
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "reportName": "RPTA_WEB_MARGIN_DAILYTRADE",
        "columns": "ALL",
        "pageNumber": "1",
        "pageSize": "500",
        "sortColumns": "STATISTICS_DATE",
        "sortTypes": "-1",
    }
    first = get_json(url, params)["result"]
    rows = list(first["data"])
    for page in range(2, first["pages"] + 1):
        page_params = dict(params, pageNumber=str(page))
        rows.extend(get_json(url, page_params)["result"]["data"])
    frame = pd.DataFrame(rows).rename(columns={
        "STATISTICS_DATE": "日期", "FIN_BALANCE": "融资余额", "LOAN_BALANCE": "融券余额",
        "FIN_BUY_AMT": "融资买入额", "LOAN_SELL_AMT": "融券卖出额",
        "SECURITY_ORG_NUM": "证券公司数量", "OPERATEDEPT_NUM": "营业部数量",
        "PERSONAL_INVESTOR_NUM": "个人投资者数量", "ORG_INVESTOR_NUM": "机构投资者数量",
        "INVESTOR_NUM": "参与交易的投资者数量", "MARGINLIAB_INVESTOR_NUM": "有融资融券负债的投资者数量",
        "TOTAL_GUARANTEE": "担保物总价值", "AVG_GUARANTEE_RATIO": "平均维持担保比例",
    })
    columns = [
        "日期", "融资余额", "融券余额", "融资买入额", "融券卖出额", "证券公司数量",
        "营业部数量", "个人投资者数量", "机构投资者数量", "参与交易的投资者数量",
        "有融资融券负债的投资者数量", "担保物总价值", "平均维持担保比例",
    ]
    frame = frame[columns]
    frame["日期"] = pd.to_datetime(frame["日期"], errors="coerce").dt.date
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("日期", ignore_index=True)

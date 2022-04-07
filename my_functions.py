#   定义各种函数
import pandas as pd
import numpy as np
import talib as ta
from scipy.stats import rankdata


def _rolling_rank(data):
    value = rankdata(data)[-1] if not np.isnan(data[-1]) else np.nan
    return value


def _rolling_prod(data):
    return np.nanprod(data)


def _ts_sum(data, window):
    data = data.copy()
    data[np.isinf(data)] = np.nan
    value = np.array(pd.Series(data.flatten()).rolling(window, min_periods=1).sum().tolist())
    return value


def _sma(data, window):
    data = data.copy()
    data[np.isinf(data)] = np.nan
    value = np.array(pd.Series(data.flatten()).rolling(window, min_periods=1).mean().tolist())
    return value


def _stddev(data, window):
    data = data.copy()
    data[np.isinf(data)] = np.nan
    value = np.array(pd.Series(data.flatten()).rolling(window, min_periods=1).std().tolist())
    return value


def _ts_rank(data, window):
    data = data.copy()
    data[np.isinf(data)] = np.nan
    value = np.array(pd.Series(data.flatten()).rolling(window, min_periods=1).apply(_rolling_rank, raw=True).tolist())
    return value


def _product(data, window):
    data = data.copy()
    data[np.isinf(data)] = np.nan
    value = np.array(pd.Series(data.flatten()).rolling(window, min_periods=1).apply(_rolling_prod, raw=True).tolist())
    return value


def _ts_min(data, window):
    data = data.copy()
    data[np.isinf(data)] = np.nan
    value = np.array(pd.Series(data.flatten()).rolling(window, min_periods=1).min().tolist())
    return value


def _ts_max(data, window):
    data = data.copy()
    data[np.isinf(data)] = np.nan
    value = np.array(pd.Series(data.flatten()).rolling(window, min_periods=1).max().tolist())
    return value


def _delta(data, window):
    data = data.copy()
    data = data.flatten()
    data[np.isinf(data)] = np.nan
    value = data[window:] - data[:(len(data) - window)]
    value = np.append(np.array([0]*window), value)
    return value


def _delay(data, window):
    data = data.copy()
    data[np.isinf(data)] = np.nan
    value = pd.Series(data.flatten()).shift(window)
    value = value.values
    return value


def _ts_argmax(data, window):
    data = data.copy()
    data[np.isinf(data)] = np.nan
    value = pd.Series(data.flatten()).rolling(window, min_periods=1).apply(np.nanargmax, raw=True) + 1
    value = value.values
    return value


def _ts_argmin(data, window):
    data = data.copy()
    data[np.isinf(data)] = np.nan
    value = pd.Series(data.flatten()).rolling(window, min_periods=1).apply(np.nanargmin, raw=True) + 1
    value = value.values
    return value


def _ts_corr(data1, data2, window):
    data1 = data1.copy()
    data2 = data2.copy()
    data1[np.isinf(data1)] = np.nan
    data2[np.isinf(data2)] = np.nan
    value = pd.Series(data1).rolling(window, min_periods=1).corr(pd.Series(data2).rolling(window, min_periods=1))
    value = value.values
    return value


def _ts_dema(data, window):
    data = data.copy()
    data[np.isinf(data)] = np.nan
    try:
        value = ta.DEMA(data, window)
    except:
        value = np.zeros(len(data)) * np.nan
    return value


def _ts_kama(data, window):
    data = data.copy()
    data[np.isinf(data)] = np.nan
    try:
        value = ta.KAMA(data, window)
    except:
        value = np.zeros(len(data)) * np.nan
    return value


def _ts_ma(data, window):
    data = data.copy()
    data[np.isinf(data)] = np.nan
    try:
        value = ta.MA(data, window)
    except:
        value = np.zeros(len(data)) * np.nan
    return value


def _ts_midpoint(data, window):
    data = data.copy()
    data[np.isinf(data)] = np.nan
    try:
        value = ta.MIDPOINT(data, window)
    except:
        value = np.zeros(len(data)) * np.nan
    return value


def _ts_midprice(high, low, window):
    high = high.copy()
    low = low.copy()
    high[np.isinf(high)] = np.nan
    low[np.isinf(low)] = np.nan
    try:
        value = ta.MIDPRICE(high, low, window)
    except:
        value = np.zeros(len(high)) * np.nan
    return value


def _ts_aroonosc(high, low, window):
    high = high.copy()
    low = low.copy()
    high[np.isinf(high)] = np.nan
    low[np.isinf(low)] = np.nan
    try:
        value = ta.AROONOSC(high, low, window)
    except:
        value = np.zeros(len(high)) * np.nan
    return value


def _ts_willr(high, low, close, window):
    high = high.copy()
    low = low.copy()
    close = close.copy()
    high[np.isinf(high)] = np.nan
    low[np.isinf(low)] = np.nan
    close[np.isinf(close)] = np.nan
    try:
        value = ta.WILLR(high, low, close, window)
    except:
        value = np.zeros(len(close)) * np.nan
    return value


def _ts_cci(high, low, close, window):
    high = high.copy()
    low = low.copy()
    close = close.copy()
    high[np.isinf(high)] = np.nan
    low[np.isinf(low)] = np.nan
    close[np.isinf(close)] = np.nan
    try:
        value = ta.CCI(high, low, close, window)
    except:
        value = np.zeros(len(close)) * np.nan
    return value


def _ts_adx(high, low, close, window):
    high = high.copy()
    low = low.copy()
    close = close.copy()
    high[np.isinf(high)] = np.nan
    low[np.isinf(low)] = np.nan
    close[np.isinf(close)] = np.nan
    try:
        value = ta.ADX(high, low, close, window)
    except:
        value = np.zeros(len(close)) * np.nan
    return value


def _ts_mfi(high, low, close, volume, window):
    high = high.copy()
    low = low.copy()
    close = close.copy()
    volume = volume.copy()
    high[np.isinf(high)] = np.nan
    low[np.isinf(low)] = np.nan
    close[np.isinf(close)] = np.nan
    volume[np.isinf(volume)] = np.nan
    try:
        value = ta.MFI(high, low, close, volume, window)
    except:
        value = np.zeros(len(close)) * np.nan
    return value


def _ts_natr(high, low, close, window):
    high = high.copy()
    low = low.copy()
    close = close.copy()
    high[np.isinf(high)] = np.nan
    low[np.isinf(low)] = np.nan
    close[np.isinf(close)] = np.nan
    try:
        value = ta.NATR(high, low, close, window)
    except:
        value = np.zeros(len(close)) * np.nan
    return value


def _ts_linearreg_slope(data, window):
    data = data.copy()
    data[np.isinf(data)] = np.nan
    try:
        value = ta.LINEARREG_SLOPE(data, window)
    except:
        value = np.zeros(len(data)) * np.nan
    return value



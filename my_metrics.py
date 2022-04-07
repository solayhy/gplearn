#   定义CTA交易的适应度: 赚取的价差点数，用样本内交易收益
import numpy as np
import pandas as pd
import statsmodels.api as sm
def _cta_spread_trading_metric(y, y_pred, w, *args):
    #   对于期货价差CTA交易的适应度，不用*args中的参数
    #   y: 样本内价差序列
    #   y_pred: 遗传算法生成的信号序列
    #   交易规则： 当信号序列往上突破最近10000个信号序列的80%分位数时，做多； 做多后如果信号序列跌破70%分位数，平多
    #            当信号序列往下跌破最近10000个信号序列的20%分位数时，做空； 做空后如果信号序列往上突破30%分位数时，平空

    #   首先检查y和y_pred长度是否相等
    if len(y) != len(y_pred):
        raise Exception('y and y_pred must have equal length !')
    position = 0
    money = 0
    commission_fee = 5.0 / 10000    #  交易费率
    for i in range(1000, len(y)):
        if position == 0:
            #   如果当前没有持仓
            if y_pred[i] > np.nanquantile(y_pred[i-1000: i+1], 0.8):
                #   做多
                position = 1
                money -= money * commission_fee
            elif y_pred[i] < np.nanquantile(y_pred[i-1000: i+1], 0.2):
                #   做空
                position = -1
                money -= money * commission_fee
                short_open_price = y[i]
        elif position == 1:
            #   如果当前持有多单
            if y_pred[i] >= np.nanquantile(y_pred[i - 1000: i + 1], 0.7):
                #   继续持有多单
                money += (y[i] - y[i-1])
            else:
                #   平掉多单
                money += (y[i] - y[i - 1])
                money -= money * commission_fee
                position = 0
        elif position == -1:
            #   如果当前持有空单
            if y_pred[i] <= np.nanquantile(y_pred[i - 1000: i + 1], 0.3):
                #   继续持有空单
                money += -(y[i] - y[i-1])
            else:
                #   平掉空单
                money += -(y[i] - y[i - 1])
                money -= money * commission_fee
                position = 0

    return money
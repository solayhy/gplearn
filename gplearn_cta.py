import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
from gplearn.functions import make_function, _Function
from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import make_fitness
from my_functions import _rolling_rank, _rolling_prod, _ts_sum, _sma, _stddev, _ts_rank, _product, \
    _ts_min, _ts_max, _delta, _delay, _ts_argmax, _ts_argmin, _ts_corr, _ts_dema, _ts_kama, _ts_ma, _ts_midpoint, _ts_linearreg_slope
#from gplearn.functions import _protected_sqrt, _protected_log
#from load_data import get_stock_data, get_stock_data_from_local, get_logsize_sectormark_status
from my_metrics import _cta_spread_trading_metric
import sys
sys.path

# 自定义函数群
ts_sum = _Function(function=_ts_sum, name='ts_sum', arity=1, is_ts=True)
sma = _Function(function=_sma, name='sma', arity=1, is_ts=True)
stddev = _Function(function=_stddev, name='stddev', arity=1, is_ts=True)
ts_rank = _Function(function=_ts_rank, name='ts_rank', arity=1, is_ts=True)
product = _Function(function=_product, name='product', arity=1, is_ts=True)
ts_min = _Function(function=_ts_min, name='ts_min', arity=1, is_ts=True)
ts_max = _Function(function=_ts_max, name='ts_max', arity=1, is_ts=True)
delta = _Function(function=_delta, name='delta', arity=1, is_ts=True)
delay = _Function(function=_delay, name='delay', arity=1, is_ts=True)
ts_argmax = _Function(function=_ts_argmax, name='ts_argmax', arity=1, is_ts=True)
ts_argmin = _Function(function=_ts_argmin, name='ts_argmin', arity=1, is_ts=True)
ts_corr = _Function(function=_ts_corr, name='ts_corr', arity=2, is_ts=True)
ts_dema = _Function(function=_ts_dema, name='ts_dema', arity=0, is_ts=True, params_need=['spread'])
ts_kama = _Function(function=_ts_kama, name='ts_kama', arity=0, is_ts=True, params_need=['spread'])
ts_ma = _Function(function=_ts_ma, name='ts_ma', arity=0, is_ts=True, params_need=['spread'])
ts_midpoint = _Function(function=_ts_midpoint, name='ts_midpoint', arity=0, is_ts=True, params_need=['spread'])
ts_linearreg_slope = _Function(function=_ts_linearreg_slope, name='ts_linearreg_slope', arity=0, is_ts=True, params_need=['spread'])


function_set = ['add', 'sub', 'mul', 'div']
# ts_function_set = [ts_sum, sma, stddev, ts_rank, product, ts_min, ts_max, delta, delay, ts_argmax, ts_argmin, ts_corr,
#                    ts_dema, ts_kama, ts_ma, ts_midpoint, ts_linearreg_slope]
ts_function_set = [ts_sum, sma, stddev, ts_rank, product, ts_min, ts_max, delta, delay, ts_argmax, ts_argmin, ts_corr]
fixed_function_set = [ts_dema, ts_kama, ts_midpoint, ts_linearreg_slope]

#   读取股票数据，并将特征在时间序列上归一化到区间[0.01, 1]
def nanminmaxscaler(x, feature_range):
    denominator = np.nanmax(x, axis=0) - np.nanmin(x, axis=0)
    denominator[np.abs(denominator) < 1e-6] = np.nan
    x_std = (x - np.nanmin(x, axis=0)) / denominator
    x_scaled = x_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    return x_scaled
feature_range = (0.01, 1)

df_price = pd.read_csv('spread_整理后的.csv', index_col=0, header=0)
series_spread = df_price['spread']
raw_array_spread = series_spread.values
raw_array_spread = raw_array_spread.astype(float)
rescaled_array_spread = nanminmaxscaler(raw_array_spread.reshape(-1, 1), feature_range=(0.01, 1)).flatten()
fields = ['spread']
#   特征数组shape: [n_samples, n_features, n_stocks]
n_samples = len(series_spread)
n_features = len(fields)
X = np.zeros((n_samples, n_features))
for i in range(len(fields)):
    X[:, i] = rescaled_array_spread[-n_samples:]

y = raw_array_spread


#   定义适应度
#   CTA交易的适应度: 赚取的价差点数，用样本内交易收益
metric_name = 'cta_spread_trading'
my_metric = make_fitness(function=_cta_spread_trading_metric, greater_is_better=True, wrap=False, is_custom=True)

#   生成表达式
generations = 5
metric = my_metric
population_size = 500
random_state = 0
est_gp = SymbolicTransformer(feature_names=fields, function_set=function_set, ts_function_set=ts_function_set,
                             fixed_function_set=fixed_function_set,  d_ls=list(range(3, 11)), generations=generations,
                             metric=metric, population_size=population_size, tournament_size=20,
                             stopping_criteria=1e10,  const_range=None,
                             random_state=random_state, max_samples=1., verbose=1, n_jobs=20)
est_gp.fit(X, y, None, True, False, np.array([]), np.array([]), np.array([]))
#   保存模型
output_file = open('./est_gp_%s.pkl' % metric_name, 'wb')
pickle.dump(est_gp, output_file)
output_file.close()
# 获取较优的表达式
best_programs = est_gp._best_programs
best_programs_dict = {}
for p in best_programs:
    factor_name = 'alpha_' + str(best_programs.index(p) + 1)
    best_programs_dict[factor_name] = {'fitness': p.fitness_, 'expression': str(p), 'depth': p.depth_, 'length': p.length_}
best_programs_dict = pd.DataFrame(best_programs_dict).T
# best_programs_dict = best_programs_dict.sort_values(by='fitness')
tmp_xnew_list = []
array_best_programs_y_pred = est_gp.transform(X, is_custom=True)

#   保存生成的信号因子
import pickle
output_file = open('./array_best_programs_y_pred_%s.pkl' % metric_name, 'wb')
pickle.dump(array_best_programs_y_pred, output_file)
output_file.close()











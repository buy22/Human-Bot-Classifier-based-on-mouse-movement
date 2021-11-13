import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def inf_to_nan(x):
	for i in range(len(x)):
		if x[i] == float("inf") or x[i] == float("-inf"):
			x[i] = float('nan') # or x or return whatever makes sense
	return x


def read_data(path='data/dsjtzs_txfz_training.txt'):
	train = pd.read_csv(path, sep=' ', header=None, encoding='utf-8', names=['id', 'data', 'target', 'label'])
	train['data'] = train['data'].apply(lambda x: [list(map(float, point.split(','))) for point in x.split(';')[:-1]])
	train['target'] = train['target'].apply(lambda x: list(map(float, x.split(","))))
	train['data_x'] = train['data'].apply(lambda x: np.array([i[0] for i in x]))
	train['data_y'] = train['data'].apply(lambda x: np.array([i[1] for i in x]))
	train['data_t'] = train['data'].apply(lambda x: np.array([i[2] for i in x]))
	del train['data']

	train['target_x'] = train['target'].apply(lambda x: x[0])
	train['target_y'] = train['target'].apply(lambda x: x[1])
	# delt
	train['delt_x'] = train['data_x'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
	train['delt_y'] = train['data_y'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
	train['delt_t'] = train['data_t'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
	train['delt_xy'] = train.delt_x ** 2 + train.delt_y ** 2
	train['delt_xy'] = train['delt_xy'].apply(lambda x: np.sqrt(x))
	# speed
	train['speed_x'] = (train.delt_x / train.delt_t).apply(lambda x: inf_to_nan(x)).apply(
		lambda x: np.nan_to_num(np.array(x)))
	train['speed_y'] = (train.delt_y / train.delt_t).apply(lambda x: inf_to_nan(x)).apply(
		lambda x: np.nan_to_num(np.array(x)))
	train['speed_xy'] = (train.delt_xy / train.delt_t).apply(lambda x: inf_to_nan(x)).apply(
		lambda x: np.nan_to_num(np.array(x)))

	# delt_speed
	train['delt_speed_x'] = train['speed_x'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
	train['delt_speed_y'] = train['speed_y'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
	train['delt_speed_xy'] = train['speed_xy'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
	train['delt_speed_t'] = train['delt_t'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])

	# acc
	train['acc_speed_x'] = (train.delt_speed_x / train.delt_speed_t).apply(lambda x: inf_to_nan(x)).apply(
		lambda x: np.nan_to_num(np.array(x)))
	train['acc_speed_y'] = (train.delt_speed_y / train.delt_speed_t).apply(lambda x: inf_to_nan(x)).apply(
		lambda x: np.nan_to_num(np.array(x)))
	train['acc_speed_xy'] = (train.delt_speed_xy / train.delt_speed_t).apply(lambda x: inf_to_nan(x)).apply(
		lambda x: np.nan_to_num(np.array(x)))

	# 轨迹长度
	train['len_x'] = train['data_x'].apply(lambda x: len(x))

	# first
	train['first_data_x'] = train['data_x'].apply(lambda x: x[0])
	train['first_speed_x'] = train['speed_x'].apply(lambda x: x[0] if x.shape[0] > 0 else 0)
	train['first_data_y'] = train['data_y'].apply(lambda x: x[0])
	train['first_delt_t'] = train['delt_t'].apply(lambda x: x[0] if len(x) > 0 else 0)

	#

	train['time_delta_min'] = train['delt_t'].map(lambda x: x.min() if x.shape[0] > 0 else 0)
	train['distance_deltas_max'] = train['delt_xy'].map(lambda x: x.max() if x.shape[0] > 0 else 0)
	train['median_speed'] = train['speed_x'].map(lambda x: np.median(x) if x.shape[0] > 0 else 0)
	train['xs_delta_var'] = train['delt_x'].map(lambda x: x.var() if x.shape[0] > 0 else 0)
	train['xs_delta_max'] = train['delt_x'].map(lambda x: x.max() if x.shape[0] > 0 else 0)
	train['x_min'] = train['data_x'].map(lambda x: x.min())
	train['y_min'] = train['data_y'].map(lambda x: x.min())

	#####################
	# X_max与X_target关系
	train['X_max'] = train['data_x'].map(lambda x: x.max() if x.shape[0] > 0 else 0)
	train['get_target'] = train['X_max'] - train['target_x']

	# 特征75 76
	train['data_x_return'] = train['delt_x'].apply(
		lambda x: np.where(x < 0)[0].shape[0] > 0 if x.shape[0] > 0 else False)
	# #y坐标唯一值个数的标准差比平均值
	train['data_y_unique_value'] = train['data_y'].map(
		lambda x: np.unique(x, return_counts=True) if x.shape[0] > 0 else 0)
	train['data_y_unique_value_stdmean'] = train['data_y_unique_value'].map(
		lambda x: x[1].std() / x[1].mean() if x != 0 else 0)
	del train['data_y_unique_value']

	train['dxspeed_mean'] = train["delt_speed_x"].apply(lambda x: np.mean(x) if x.shape[0] > 0 else 0)  # x方向速度差分的平均值
	train['dxspeed_std'] = train["delt_speed_x"].apply(lambda x: np.std(x) if x.shape[0] > 0 else 0)  # x方向速度差分的标准差
	train['dxspeed_median'] = train["delt_speed_x"].apply(lambda x: np.median(x) if x.shape[0] > 0 else 0)
	# x方向速度全局标准差与最后N个标准差比值
	train['speed_xstd'] = train["speed_x"].apply(lambda x: np.std(x) if x.shape[0] > 0 else 0)  # x方向速度差分的标准差
	train['speed_x_laststd'] = train["speed_x"].apply(lambda x: np.std(x[-9:]) if x.shape[0] > 0 else 0)  # x方向速度差分的标准差
	train['speed_xstd_laststd'] = train['speed_xstd'] / train['speed_x_laststd']
	train['speed_xstd_laststd'] = train['speed_xstd_laststd'].apply(
		lambda x: float('nan') if (x == float('-inf') or x == float('inf')) else x)

	# delt_speed
	train['delt_acc_x'] = train['delt_speed_x'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
	# train['delt_acc_y'] = train['delt_speed_y'].apply(lambda x: np.array(x)[1:]-np.array(x)[:-1])
	train['delt_acc_t'] = train['delt_speed_t'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
	train['delt_acc_speed_x'] = (train.delt_acc_x / train.delt_acc_t).apply(lambda x: inf_to_nan(x)).apply(
		lambda x: np.nan_to_num(np.array(x)))
	train['dxacc_mean'] = train["delt_acc_speed_x"].apply(lambda x: np.mean(x) if x.shape[0] > 0 else 0)  # x方向速度差分的平均值
	train['dxacc_std'] = train["delt_acc_speed_x"].apply(lambda x: np.std(x) if x.shape[0] > 0 else 0)  # x方向速度差分的标准差
	train['dxacc_median'] = train["delt_acc_speed_x"].apply(lambda x: np.median(x) if x.shape[0] > 0 else 0)

	# 0718 19
	from scipy import stats
	train['log_delt_x'] = train['data_x'].apply(lambda x: np.log1p(np.array(x)[1:] - np.array(x)[:-1]))
	train['log_delt_y'] = train['data_y'].apply(lambda x: np.log1p(np.array(x)[1:] - np.array(x)[:-1]))
	train['log_delt_t'] = train['data_t'].apply(lambda x: np.log1p(np.array(x)[1:] - np.array(x)[:-1]))

	train['xy_angles'] = train['log_delt_x'] - train['log_delt_y']
	train['xt_angles'] = train['log_delt_x'] - train['log_delt_t']
	train['xy_angles_kurtosis'] = train['xy_angles'].apply(lambda x: stats.kurtosis(x))
	train['xt_angles_max'] = train['xt_angles'].apply(lambda x: np.max(x) if x.shape[0] > 0 else 0)  # x方向速度差分的平均值
	feats = [
			'id', 'label',
			'first_data_x',
			'first_speed_x',
			'first_data_y',
			'first_delt_t',
			'time_delta_min',
			'distance_deltas_max',
			'median_speed',
			'xs_delta_var',
			'xs_delta_max',
			'x_min',
			'y_min',
			'X_max',
			'get_target',
			'data_x_return',
			'data_y_unique_value_stdmean',
			'dxspeed_mean',
			'dxspeed_std',
			'dxspeed_median',
			'speed_xstd',
			'speed_x_laststd',
			'speed_xstd_laststd',
			'dxacc_mean',
			'dxacc_std',
			'xt_angles_max',
			'xy_angles_kurtosis'
			]
	return train[feats].fillna(0)


import torch
import numpy as np
import torch.nn as nn
import plotly.graph_objects as go
import statsmodels.tsa.api as stats_api
from statsmodels.tsa.stattools import adfuller
from back_tester import RollingWalkForward



class VAR():
	def __init__(self):
		self.name = 'VAR'
		self.n_input = 1
		self.n_window = 10
		self.diff_order = 1
		self.lag = 10
		
	def select_features(self, data):
		features = []
		print(data.shape)
		for i in range(data.shape[1]):
			print(i, end=' ')
			if self.test_stationarity(data[:, i]) == 'Stationary':
				features.append(i)
				print("is Stationary", end=' ')
		if len(features) == 1:
			features = [features[0],features[0]]
		return features
	
	def differencing(self, data):
		res = []
		for i in range(data.shape[1]):
			res.append(np.diff(data[:, i], self.diff_order))
		return np.array(res).T
	
	def fit(self, data):
		out = self.differencing(data)
		self.featurs = self.select_features(out)
		out = np.take(out, self.featurs, axis=1)
		self.var = stats_api.VAR(out)
		self.fitted_var = self.var.fit(self.lag)

	def predict(self, data):
		out = self.differencing(data)
		X = np.take(out, self.featurs, axis=1)
		y_pred = []
		for i in range(self.lag, X.shape[0]):
			y_pred.append(self.fitted_var.forecast(X[i-self.lag:i, :], 1)[0])
		y_pred = np.array(y_pred)
		y_pred = np.concatenate((X[:self.lag, :], y_pred))
		l = nn.MSELoss(reduction = 'none')
		y_pred = torch.tensor(y_pred).float()
		X = torch.tensor(X).float()
		MSE = l(y_pred, X)
		loss = MSE.detach().numpy()
		return loss
	
	def test_stationarity(data_col, signif=0.05):
		if not data_col.any():
			return "Non-Stationary"
		adf_test = adfuller(data_col, autolag='AIC')
		p_value = adf_test[1]
		if p_value <= signif:
			return "Stationary"
		else:
			return "Non-Stationary"

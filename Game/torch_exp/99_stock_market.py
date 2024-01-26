import numpy as np
import pandas
import torch.nn
import yfinance as yf
from loguru import logger
import pandas as pd
import pandas_ta as ta
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


def draw_data(data, pred_data=None):
	plt.figure(figsize=(15, 10))
	plt.plot(data['Adj Close'])

	if pred_data is not None:
		plt.plot(pred_data['Adj Close'])

	# plt.plot(data['RSI'])
	plt.title('Top 500 stocks')
	plt.show(block=False)


# todo - fundamental bug. Applying scalar to make data from 0 to 1 makes model applicable only for this dataset.
#  If price goes up, it goes outside training range. So model needs to be retrained with last 1y data for every prediction???
#  possible solution: add a row with lowest and higest values + some margin so no matter actual data, there's always
#  a threshold for min and max values. Entire MinMaxScaler becomes stable and static


class LSTMModel(nn.Module):

	def __init__(self, input_size, hidden_nodes_per_layer, hidden_layers: int = 1):
		super().__init__()

		self.input_size = input_size
		self.hidden_nodes_per_layer = hidden_nodes_per_layer
		self.hidden_layers = hidden_layers

		self.layer_1 = nn.Flatten()
		self.layer_2 = nn.LSTM(input_size, hidden_nodes_per_layer, hidden_layers, batch_first=True)
		self.layer_3 = nn.Linear(hidden_nodes_per_layer, 1)

		self.__optimisation_fn = torch.optim.Adam(self.parameters(), lr=0.001)
		# self.__loss_fn = nn.CrossEntropyLoss()
		self.__loss_fn = nn.MSELoss()

	def forward(self, x):
		# h0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_nodes_per_layer).to('cpu')
		# c0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_nodes_per_layer).to('cpu')

		h0 = torch.randn(self.hidden_layers, x.size(0), self.hidden_nodes_per_layer)
		c0 = torch.randn(self.hidden_layers, x.size(0), self.hidden_nodes_per_layer)

		# x = self.layer_1(x)

		x, _ = self.layer_2(x, (h0, c0))
		x = self.layer_3(x[:, -1, :])  # todo not sure why i need to flip it???
		return x

	def output_activation_fn(self, x):
		return self.__output_activation_fn(x)

	def optimisation_fn(self):
		return self.__optimisation_fn

	def loss_fn(self, pred, actual):
		return self.__loss_fn(pred, actual)


def train_one_epoch(model: LSTMModel, train_dataloader):
	batch_loss = 0

	for batch, (train_X, train_y) in enumerate(train_dataloader):
		model.train()

		logits = model(train_X).squeeze()

		loss = model.loss_fn(logits, train_y)
		batch_loss += loss.item()

		model.optimisation_fn().zero_grad()

		loss.backward()

		model.optimisation_fn().step()

	logger.warning({'average batch loss': batch_loss / len(train_dataloader)})


def evaluate_one_epoch(model, test_dataloader):
	model.eval()
	with torch.inference_mode():
		test_loss_sum = 0
		for batch_index, (test_X, test_y) in enumerate(test_dataloader):
			pred_y_logit = model(test_X).squeeze()

			test_loss = model.loss_fn(pred_y_logit, test_y)
			test_loss_sum += test_loss.item()

		logger.warning(f'Test loss average per epoch: {test_loss_sum / len(test_dataloader):0.4f}')


def train_loop(epochs: int, model: LSTMModel, train_dataloader, test_dataloader):
	logger.error(len(train_dataloader))
	logger.error(len(test_dataloader))
	for epoch in range(epochs):
		logger.info(f'Epoch: {epoch}')
		train_one_epoch(model, train_dataloader)
		evaluate_one_epoch(model, test_dataloader)


class RuiDataset(Dataset):

	def __init__(self, train: bool = None):
		self.loaded_data_x, self.loaded_data_y, = self.load_data(train)

	def load_data(self, train: bool = None):
		data = yf.download('^RUI', start='2022-01-01', end='2023-12-31', interval='1d')  # type: pd.DataFrame

		# Technical indicators
		data['RSI'] = ta.rsi(data.Close, length=15)
		data['EMAF'] = ta.ema(data.Close, length=20)  # fast moving average
		data['EMAM'] = ta.ema(data['Close'], length=100)  # medium moving average
		data['EMAS'] = ta.ema(data['Close'], length=150)  # slow moving average

		data['TargetNextClose'] = data['Adj Close'].shift(-1)

		data.reset_index(inplace=True)
		data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)

		# first 150 don't have EMAS and last one doesn't have TargetNextClose
		data = data[150:-1]

		train_amount = int(len(data) * 0.8)

		if train is True:
			data = data[:train_amount]
		elif train is False:
			data = data[train_amount:]

		# print(data)
		# logger.error(len(data))

		# # apply scaler so data is between 0 and 1 for neural network to work
		sc = MinMaxScaler()
		scaled_data = sc.fit_transform(data)
		# logger.info(scaled_data[:5])

		X, y = self.construct_input_layer_data(scaled_data)
		logger.error(len(X))
		return X, y

	def construct_input_layer_data(self, data: list) -> [np.array, np.array]:
		X = []
		y = []
		back_candles = 10

		# print(data.shape)  # (223, 9)

		# for j in range(8):  # for each column in data except the last one -- TargetNextClose
		# 	X.append([])
		#
		# 	for i in range(back_candles, data[0].shape):  # for each row from beginning + back candles
		#
		# 		X[j].append(data[i - back_candles:i, j])

		for index, row in enumerate(data):
			if index + 1 < back_candles:
				continue

			candle_history = data[index + 1 - back_candles:index + 1]
			candle_history = [r[0:-1] for r in candle_history]  # don't add last column - it's an answer
			X.append(candle_history)
			y.append(row[-1])

		return np.array(X), np.array(y)

	def __len__(self):
		return len(self.loaded_data_x)

	def __getitem__(self, index):
		return (
			torch.tensor(self.loaded_data_x[index]).type(torch.float32),
			torch.tensor(self.loaded_data_y[index]).type(torch.float32)
		)


rui_dataset = RuiDataset(train=True)
train_dataloader = DataLoader(rui_dataset, batch_size=32, shuffle=True)

test_rui_dataset = RuiDataset(train=False)
test_dataloader = DataLoader(test_rui_dataset, batch_size=32, shuffle=False)

logger.info(f'Train Data Loader length: {len(train_dataloader)} in batches of {train_dataloader.batch_size}')

# train_features_batch, train_labels_batch = next(iter(train_dataloader))

# train_size = int(len(X) * 0.8)
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]
#
# logger.warning(f'{X[0][0]=}')
# logger.warning(f'{X[0][0][0]=}')
# logger.warning(f'{y[0]=}')

# dt = Dataset()

# BATCH_SIZE = 32
# train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
# test_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

# logger.warning(X_train.shape)
# logger.warning(X_test.shape)

# input_size = X_train.shape[1] * X_train.shape[2]
# input_size = train_features_batch.size()[1] * train_features_batch.size()[2]
# logger.warning(f'{input_size=}')
model_0 = LSTMModel(8, 10, 2)

train_loop(200, model_0, train_dataloader, test_dataloader)

all_rui_dataset = RuiDataset()
all_dataloader = DataLoader(all_rui_dataset, batch_size=1, shuffle=False)

with torch.inference_mode():
	model_0.eval()

	preds = []
	actual_values = []

	for X, y in all_dataloader:
		pred_y = model_0(X)
		preds.append(pred_y.item())
		actual_values.append(y)

plt.figure(figsize=(15, 10))
plt.plot(actual_values, c='black', label='Actual values')
plt.plot(preds, c='green', label='Predicted values')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Close')
plt.show(block=True)

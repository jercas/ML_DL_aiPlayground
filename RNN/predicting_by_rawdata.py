from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np
import subprocess

def main():
	stock_data = load_stock_trading_records("60-2016-01-01.txt")
	stock_code = get_stock_code(stock_data)
    
	for key in stock_code:
		model = train(stock_data, key)
		length = stock_data[:,0].size
		
		today = predict(model, stock_data[length - 60:, 1:7])
		print(today,"\n")
		break
	#print "testing!\n"

def load_stock_trading_records(filename):
	subprocess.call(["perl", "./switch.pl"])
	linetype = np.dtype({
		'names':['StockCode', 'YestClose', 'TodayOpen', 'TodayHigh', 'TodayLow', 'TodayClose', 'TodayAvg',
			'UpDownValue', 'UpDownRate', 'TradeSharesRate', 'TradeShares', 'Turnover',
			'TradeCount', 'Amplitude', 'SuspendDays', 'PE', 'TradeDays'],
           
    		'formats':['i', 'f', 'f', 'f', 'f', 'f', 'f',
			'f', 'f', 'f', 'i', 'i',
			'i', 'i', 'i', 'S32', 'i']})
	f = file("test_data")
	f.readline()
	stock_data = np.loadtxt(f, delimiter=",")
	f.close()
	return stock_data

def get_stock_code(stock_data):
	stockcode = np.unique(stock_data[:,0])
	return stockcode

def train(stock_data, stock_code):
	origin_training_data = stock_data[stock_data[:,0] == stock_code]

	data_dim = 6
	timesteps = 60

	length = origin_training_data[:,0].size

	b = np.array([], dtype = np.int32)
	for i in range(length - timesteps):
		a = range(i,i+timesteps,1)
		b = np.concatenate([b,a])	
	test_origin = origin_training_data[b,1:7]
	test = test_origin.reshape((length - timesteps, timesteps, 6))
	labels_origin = origin_training_data[timesteps:,1:7]
	labels = labels_origin.reshape(length - timesteps, 6)

	# expected input data shape: (batch_size, timesteps, data_dim)
	model = Sequential()
	model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
	model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
	model.add(Dropout(0.8))
	model.add(LSTM(32))  # return a single vector of dimension 32
	model.add(Dropout(0.8))
	model.add(Dense(6, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	model.fit(test, labels, epochs=10, batch_size=32)

	return model

def predict(model, stock_data):
	inputdata = stock_data.reshape(1, 60, 6)
	
	result = model.predict(inputdata)
	
#	result = stock_data[-1]

	return result

if __name__ == '__main__':
	main()


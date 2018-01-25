# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 20:05:48 2017

@author: jercas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sn

import calendar
from datetime import datetime

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.learning_curve import learning_curve
from sklearn .metrics import explained_variance_score
from sklearn.grid_search import GridSearchCV

def visualization(data):
	"""
		visualization()

		data set visualization

		Parameters
		----------
			data: data set to be shown

		Returns
		-------
			Null
	"""
	# preview top 5 row of data
	print("\n--------Data preview--------\n{0}"
		  .format(data.head()))
	print("\nNull value status as follow:\n{0}".format(data.isnull().sum()))
	cols = [col for col in data.columns]
	print("\nNumber of original features: {0}".format(len(cols)))
	print("\nFeatures types:\n{0}".format(data[cols].dtypes.value_counts()))

	counts = [[], [], []]
	for col in cols:
		# the data type of each feature
		typ = data[col].dtype
		# the number of differents value in each feature
		uniq = len(np.unique(data[col]))
		# constant value feature
		if uniq == 1:
			counts[0].append(col)
		# binary value feature
		elif uniq == 2 and typ == np.int64:
			counts[1].append(col)
		# multiple value feature
		else:
			counts[2].append(col)

	print('\nConstant features: {}\nBinary features: {} \nCategorical features: {}\n'.format(*[len(c) for c in counts]))
	print('Constant features:', counts[0])
	print('Binary features:', counts[1])
	print('Categorical features:', counts[2])

	fig, axes = plt.subplots(2,2)
	fig.set_size_inches(12, 10)
	sn.boxplot(data=data,y="count",orient="v",ax=axes[0][0])
	sn.boxplot(data=data,y="count",x="season",orient="v",ax=axes[0][1])
	sn.boxplot(data=data,y="count",x="hour",orient="v",ax=axes[1][0])
	sn.boxplot(data=data,y="count",x="workingday",orient="v",ax=axes[1][1])

	axes[0][0].set(ylabel='Count',title="Box Plot On Count")
	axes[0][1].set(xlabel='Season', ylabel='Count',title="Box Plot On Count Across Season")
	axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")
	axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")
	plt.show()

	fig,(ax1,ax2,ax3,ax4)= plt.subplots(nrows=4)
	fig.set_size_inches(12,20)
	sortOrder = [1,2,3,4,5,6,7,8,9,10,11,12]
	hueOrder = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

	monthAggregated = pd.DataFrame(data.groupby("month")["count"].mean()).reset_index()
	monthSorted = monthAggregated.sort_values(by="count",ascending=False)
	sn.barplot(data=monthSorted,x="month",y="count",ax=ax1,order=sortOrder)
	ax1.set(xlabel='Month', ylabel='Avearage Count',title="Average Count By Month")

	hourAggregated = pd.DataFrame(data.groupby(["hour","season"],sort=True)["count"].mean()).reset_index()
	sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["season"],
	             data=hourAggregated, join=True,ax=ax2)
	ax2.set(xlabel='Hour Of The Day', ylabel='Users Count',
	        title="Average Users Count By Hour Of The Day Across Season",label='big')

	hourAggregated = pd.DataFrame(data.groupby(["hour","weekday"],sort=True)["count"].mean()).reset_index()
	sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["weekday"],hue_order=hueOrder,
	             data=hourAggregated, join=True,ax=ax3)
	ax3.set(xlabel='Hour Of The Day', ylabel='Users Count',
	        title="Average Users Count By Hour Of The Day Across Weekdays",label='big')

	hourTransformed = pd.melt(data[["hour","casual","registered"]], id_vars=['hour'], value_vars=['casual', 'registered'])
	hourAggregated = pd.DataFrame(hourTransformed.groupby(["hour","variable"],sort=True)["value"].mean()).reset_index()
	sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["value"],hue=hourAggregated["variable"],
	             hue_order=["casual","registered"], data=hourAggregated, join=True,ax=ax4)
	ax4.set(xlabel='Hour Of The Day', ylabel='Users Count',
	        title="Average Users Count By Hour Of The Day Across User Type",label='big')
	plt.show()


def preprocess(data):
	"""
		preprocess(data)

		data preprocess for extract features of training data

		Parameters
		----------
			data: data set to be processed

		Returns
		-------
			data: already had processed data
	"""
	print("\n--------Data preview--------\n{0}".format(data.head()))
	# transform datatime columns to four columns includes the year、month、day、hour
	data['year'] = pd.DatetimeIndex(data['datetime']).year
	data['month'] = pd.DatetimeIndex(data['datetime']).month
	data['day'] = pd.DatetimeIndex(data['datetime']).day
	data['hour'] = pd.DatetimeIndex(data['datetime']).hour

	data["date"] = data.datetime.apply(lambda x : x.split()[0])
	data["weekday"] = data.date.apply(lambda dateString :
	                                  calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
	# after transformed delete the 'datatime' column
	dataDroped = data.drop(['datetime'], axis=1)
	print("\n-------\nAfter preprocess(transform time display format to avoid object data type)\n-------")
	return dataDroped


def EDA(data):
	"""
		EDA(data)

		exploratory data analysis

		Parameters
		----------
			data: data to be explore data analysis

		Return
		------
			Null
	"""
	# mean value curve

	fig,axs = plt.subplots(5,1, sharey='all')
	fig.set_size_inches(10, 15)
	data.groupby('weather').mean().plot(y='count', marker='o', ax=axs[0])
	data.groupby('humidity').mean().plot(y='count', marker='o', ax=axs[1])
	data.groupby('temp').mean().plot(y='count', marker='o', ax=axs[2])
	data.groupby('windspeed').mean().plot(y='count', marker='o', ax=axs[3])
	print('\n')
	data.groupby('hour').mean().plot(y='count', marker='o', ax=axs[4])
	plt.title('mean count per hour')
	plt.tight_layout()
	plt.show()

	# grouping scatter
	fig,axs = plt.subplots(2,3, sharey='all')
	fig.set_size_inches(12, 8)
	data.plot(x='temp',      y='count', kind='scatter', ax=axs[0,0], color='magenta')
	data.plot(x='humidity',  y='count', kind='scatter', ax=axs[0,1], color='bisque')
	data.plot(x='windspeed', y='count', kind='scatter', ax=axs[0,2], color='coral')
	data.plot(x='month',     y='count', kind='scatter', ax=axs[1,0], color='darkblue')
	data.plot(x='day',       y='count', kind='scatter', ax=axs[1,1], color='cyan')
	data.plot(x='hour',      y='count', kind='scatter', ax=axs[1,2], color='deeppink')
	plt.tight_layout()
	plt.show()

	# correlation analysis
	corrMatt = data[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()
	mask = np.array(corrMatt)
	mask[np.tril_indices_from(mask)] = False

	fig,ax= plt.subplots()
	fig.set_size_inches(20,10)
	sn.heatmap(corrMatt, mask=mask, vmax=.8, square=True, annot=True, cmap="Greens")
	plt.show()


def buildAndTrain(trainingData):
	"""
		buildAndTrain(trainingData, testData)

		build training model then training to fit data set

		Parameters
		----------
			trainingData: training data

		Return
		------
			model: trained prediction model
			feature: feature values for plot
			target: actual values for plot
	"""
	name = trainingData.drop(['count', 'casual', 'registered'], axis=1).columns
	target = trainingData['count'].values
	feature = trainingData.drop(['count', 'casual', 'registered'], axis=1).values
	# feature scaling
	feature_scaled = preprocessing.scale(feature)
	# 0.5 cross validate
	cv = cross_validation.ShuffleSplit(len(feature_scaled), n_iter=5, test_size=0.2, random_state=0)
	# build model, then training and get accuracy of it
	print('\n---------岭回归结果--------\n')
	for train, test in cv:
		regLR = linear_model.Ridge().fit(feature_scaled[train], target[train])
		print('train score:{0:.3f}, test score:{1:.3f}\n'.format(
																regLR.score(feature_scaled[train], target[train]),
		                                                        regLR.score(feature_scaled[test],  target[test])))
	print('\n---------svm结果--------\n')
	for train, test in cv:
		regSvm = svm.SVR().fit(feature_scaled[train], target[train])
		print('train score:{0:.3f}, test score:{1:.3f}\n'.format(
														regSvm.score(feature_scaled[train], target[train]),
														regSvm.score(feature_scaled[test],  target[test])))
	print('\n---------随机森林结果--------\n')
	for train, test in cv:
		regRF = RandomForestRegressor(n_estimators=100).fit(feature_scaled[train], target[train])
		print('train score:{0:.3f}, test score:{1:.3f}\n'.format(
														regRF.score(feature_scaled[train], target[train]),
														regRF.score(feature_scaled[test],  target[test])))
	# reduce some low correction feature
	featureReduced = trainingData.drop(['count', 'casual', 'registered', 'holiday', 'workingday', 'day'], axis=1).values
	featureReduced_scaled = preprocessing.scale(featureReduced)
	print('\n---------减少特征维度以避免过拟合后的随机森林结果--------\n')
	for train, test in cv:
		regRFImpr = RandomForestRegressor(n_estimators=100).fit(featureReduced_scaled[train], target[train])
		print('train score:{0:.3f}, test score:{1:.3f}\n'.format(
														regRFImpr.score(featureReduced_scaled[train], target[train]),
														regRFImpr.score(featureReduced_scaled[test],  target[test])))
	# use grid search algorithm to improve random forest regression
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(
																feature_scaled, target, test_size=0.2, random_state=0)
	tuned_parameters = [{'n_estimators': [10,100,500], 'max_depth': [2,3,4,5,6,7,8,9,10]}]
	scores = ['r2']

	for score in scores:
		print(score)
		clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring=score)
		clf.fit(X_train, y_train)
		print(clf.best_estimator_)
		print('each parameter combination is ')
		for params, mean_score, scores in clf.grid_scores_:
			print('{0:.3f} (+/-{1:.03f}) for {2}'.format(mean_score, scores.std()/2, params))

	print('--------最优参数下的随机森林结果--------')
	for train, test in cv:
		regRFBest = RandomForestRegressor(n_estimators=100, max_depth=10).fit(feature_scaled[train], target[train])
		print('train score:{0:.3f}, test score:{1:.3f}\n'.format(
																regRFBest.score(feature_scaled[train], target[train]),
																regRFBest.score(feature_scaled[test],  target[test])))
	return regRFBest, feature_scaled, target


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
	"""
		plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5))

		plot trained model's learning curve for better analysis the accuracy of it

		Parameters
		----------
			estimator: predictor model
			title: figure's title
			X: X axis's data
			y: y axis's data
			ylim: y axis's limitation
			cv: cross_validation
			n_jobs:
			train_sizes: data range

		Returns
		-------
			plt: plot object
	"""
	fig = plt.figure()
	plt.title(title)
	if ylim is not None:
		plt.ylim(ylim)
	plt.xlabel("Training example")
	plt.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
	                                                        train_sizes=train_sizes)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
	                 color='r')
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
	                 color='g')
	plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
	plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

	plt.legend(loc='best')
	return plt


def main():
	# read training data set
	df_train = pd.read_csv('train.csv')
	# data preprocess
	df_train = preprocess(df_train)
	# visualize original data set
	visualization(df_train)
	# exploratory data analysis
	EDA(df_train)
	# drop some analysis feature for easily training
	df_train.drop(['date', 'weekday'], axis=1, inplace=True)
	# build model
	reg, feature_scaled, target = buildAndTrain(df_train)
	# plot learning curve
	title = 'Learning Curves (Ramdom Forest - n_estimator=100)'
	cv = cross_validation.ShuffleSplit(feature_scaled.shape[0], n_iter=10, test_size=0.2, random_state=0)
	plt = plot_learning_curve(reg, title, feature_scaled, target, (0.0, 1.01), cv=cv, n_jobs=1)
	plt.show()

	# read test data set
	df_test  = pd.read_csv('test.csv')
	test = preprocess(df_test)
	test.drop(['date', 'weekday'], axis=1, inplace=True)
	test_scaled = preprocessing.scale(test.values)

	pred = np.array((reg.predict(test_scaled)))
	datetime = np.array(df_test['datetime'])
	dt = {'datetime': datetime, 'pred_result':pred}
	df_pred_result = pd.DataFrame(dt)
	print(df_pred_result.head(24))

if __name__ == "__main__":
	main()

"""
	Data Fields
		datetime - hourly date + timestamp
		season - 1 = spring, 2 = summer, 3 = fall, 4 = winter
		holiday - whether the day is considered a holiday
		workingday - whether the day is neither a weekend nor holiday
		weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
				  2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
				  3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
				  4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
		temp - temperature in Celsius
		atemp - "feels like" temperature in Celsius
		humidity - relative humidity
		windspeed - wind speed
		casual - number of non-registered user rentals initiated
		registered - number of registered user rentals initiated
		count - number of total rentals
"""
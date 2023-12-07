import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import train_test_split

#for filename in glob.glob("./results/opt/*.csv"):
for filename in glob.glob("./results/optimisation_4/*test*.csv"):
	dataset = filename.split("_")[-1][:-4]

	df = pd.read_csv(filename,sep="\t",header=None)
	train_r2 = []
	val_r2 = []


	input_df = pd.read_csv("./dataset/{}.csv".format(dataset))
	X = input_df.drop(columns=['target']).to_numpy()
	y = input_df['target'].to_numpy()
	nr_inputs = input_df.columns

	for index, row in df.iterrows():
		random_seed = df.iloc[index,0]
		expression = df.iloc[index,1]
		mse = df.iloc[index,2]

		original_expression = expression

		np.random.seed(random_seed)
		X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=random_seed)

		expression = expression.replace("sin","np.sin")
		expression = expression.replace("cos","np.cos")
		expression = expression.replace("ln","np.log")
		expression = expression.replace("sqrt","np.sqrt")

		for i in reversed(range(len(nr_inputs)-1)):
			expression = expression.replace("x_{}".format(str(i)),"x[:,{}]".format(str(i)))
		x = X_train
		output = eval(expression)
		print(np.mean((output-y_train)**2))
		if np.abs(mse - np.mean((output-y_train)**2))>1:
			print(mse, np.mean((output-y_train)**2))
			print(filename,"train")
			print(expression)
			quit()

		# x = X_val
		# output = eval(expression)
		# if np.abs(mse - np.mean((output-y_val)**2))>1:
		# 	print(filename,"val")
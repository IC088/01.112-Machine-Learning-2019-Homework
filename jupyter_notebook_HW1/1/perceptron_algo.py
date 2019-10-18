import numpy as np
import csv
import matplotlib.pyplot as plt

'''
Perceptron algorithm
1. Initialise weights theta(0) = 0 
2. check if the condition sign(t)(theta . x(t)) <= 0?
3. if true then update theta(k+1) = theta(k) + sign(t) x(t)
	else: continue
4. repeat until end of file
'''

'''
How to run this:
1. have python 3.7++ installed to be safe
2. make sure to have the data in the same directory as this since I didn't bother moving anything else or take into account 
if you are a windows user:
	if you have an IDE:
		run the code in your IDE
	else:
		got to cmd and run 
			python perceptron_algo.py
if you are a unix based user:
	open terminal:
		run python3 perceptron_algo.py
		       or
		run python perceptron_algo.py
'''


'''
function perceptron_train function takes in the filename of the dataset and the number of epochs that the training is done

returns the theta value(orthogonal vector of the boundary line), a list of data points for plotting
'''

def perceptron_train(filename, epochs=100):
	theta = 0
	points = []
	for i in range(epochs):
		with open(filename) as f:

			csv_reader = csv.reader(f, delimiter=',')

			for row in csv_reader:
				x = np.array([row[0], row[1]] , dtype = 'float64')
				points.append(x)
				# check if sign(t)( theta . x(t)) <= 0
				y = np.asarray(row[2], dtype='float64')
				thetax = sum(theta * x)
				if y * thetax <= 0:
					theta = theta + np.dot(y, x)
	return theta , points


'''
function test_val takes in the filename of the dataset and theta value as input

returns the accuracy of the theta value/ model of the training
'''

def test_val(filename, theta):
	line = 0
	total = 0
	with open(filename) as f:
		csv_reader = csv.reader(f, delimiter=',')
		for row in csv_reader:
			x = np.array([row[0], row[1]] , dtype = 'float64')
			'''
			check if they are in the on the same side of the boundary line as the predicted value
			'''
			check = np.dot(theta, x)
			result = np.array([check, row[2]], dtype = 'float64')
			
			if np.multiply(result[0],result[1]) > 0:
				total+=1
			line +=1
	return total/line




'''
For 5 iterations
'''

theta_val, points = perceptron_train('train_1_5.csv',5)

print('Accuracy: ' + str(test_val('test_1_5.csv', theta_val)))


'''
For 10 iterations
'''


theta_val, points = perceptron_train('train_1_5.csv',10)

print('Accuracy: ' + str(test_val('test_1_5.csv', theta_val)))





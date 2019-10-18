Perceptron algorithm HW1
1. Initialise weights theta(0) = 0 
2. check if the condition sign(t)(theta . x(t)) <= 0?
3. if true then update theta(k+1) = theta(k) + sign(t) x(t)
	else: continue
4. repeat until end of file
'''

'''
How to run this:
1. have python 3.7++ installed to be safe and make sure to run the requirements.txt to get the lastest versions of numpy, and matplotlib 
2. make sure to have the data in the same directory as the python file since I didn't bother moving anything else or take into account 
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
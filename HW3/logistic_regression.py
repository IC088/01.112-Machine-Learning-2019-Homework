import pandas as pd
import numpy as np

df = pd.read_csv('./HW3_data/4/diabetes_train.csv', header = None)
y = df.iloc[:, :1]
x = df.iloc[:, 1:]

y = np.array(y,dtype='float64')

x = np.array(x,dtype='float64')


def n_cost_function(theta, x, y):
    '''
    X - dataframe for the training set
    y - dataframe for the actual set
    theta - features
    '''
#     ones = np.ones(shape=x.shape) #biases
#     X = np.hstack((x,ones))
    pred = np.dot(x, theta)  # 3000, 1 shape
    cost = np.log(1 + np.exp(np.multiply(-y, pred)))
    return cost.mean()

def n_likelihood_function(theta, x, y):
    '''
    X - dataframe for the training set
    y - dataframe for the actual set
    theta - features
    '''
#     ones = np.ones(shape=x.shape) #biases
#     X = np.hstack((x,ones))
    pred = np.dot(x, theta)  # 3000, 1 shape
    cost = np.log(1/(1 + np.exp(np.multiply(-y, pred))))
    return cost.mean()

def n_d_cost(theta, x, y):
    '''
    X - dataframe for the training set
    y - dataframe for the actual set
    theta - features
    update parameters: −y(t)x(t)/1 + exp(y(t)(θ·x(t)))
    '''
    top = np.multiply(-y, x)
    pred = np.dot(x, theta)
    bot = 1 + np.exp(np.multiply(y, pred))
    return (top/bot)
#     a = np.random.choice(range(n), n, replace=False)

#     a = np.random.choice(range(n), n, replace=False)
from random import randrange
def n_sgd(arrx, arry,learning_rate=0.01, epochs=50):
    theta = np.zeros(shape=(21,1))
    ones = np.ones(shape = (3000,1))
    arrx_array = np.hstack((ones, arrx))
    arry_array = np.array(arry, dtype='float64')
    n = len(arrx)
    minloss = 10**10
    bestTheta = 0
    err = []  
    like = []
    for j in range(epochs): #This is for the number of training iteration.
        i = randrange(n)
        loss = n_cost_function(theta,arrx_array, arry_array)
        likelihood = n_likelihood_function(theta,arrx_array, arry_array)
        theta -= (learning_rate * n_d_cost(theta,arrx_array[i],arry_array[i]).reshape(21,1))      
        if loss < minloss:
            minloss = loss
            bestTheta = np.copy(theta)
        err.append(loss)
        like.append(likelihood)
        if (j + 1) % 100 == 0:
            # save weights of the function
            np.save(f'./models/model_{j+1}',theta, allow_pickle=True, fix_imports=True)
            print(f'>>> Model_{j+1} saved to models')
    return bestTheta , minloss, err, like

n_theta, n_loss, n_err, n_like = n_sgd(x,y,0.1,10000)


print(f'Theta: {N_theta} \n')

print(f'Loss: {n_loss} \n')

print(f'Training Error: {n_err} \n')


import matplotlib.pyplot as plt


print("--- Cost for every 100 iteration ---")
plt.title('Cost for every 100 iteration')
plt.ylabel('Cost')
plt.xlabel('Number of iterations (100)')
cost100 = n_err[0::100]
plt.plot(np.arange(0,len(cost100)),cost100[0:],'-r', label='Cost')
plt.legend()
plt.show()


print("--- Log-likelihood for every 100 iteration ---")
plt.title('Log-likelihood for every 100 iteration')
plt.ylabel('Log-likelihood')
plt.xlabel('Number of iterations (100)')
like100 = n_like[0::100]
plt.plot(np.arange(0,len(like100)),like100[0:],'-r', label='Log-likelihood')
plt.legend()
plt.show()

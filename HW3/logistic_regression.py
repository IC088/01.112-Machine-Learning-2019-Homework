import pandas as pd
import numpy as np

df = pd.read_csv('./HW3_data/4/diabetes_train.csv', header = None)
y = df.iloc[:, :1]
x = df.iloc[:, 1:]

y = np.array(y,dtype='float64')

x = np.array(x,dtype='float64')


def cost_function(theta, x, y):
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

def d_cost(theta, x, y):
    '''
    X - dataframe for the training set
    y - dataframe for the actual set
    theta - features
    update parameters: −y(t)x(t)/1 + exp(y(t)(θ·x(t)))
    '''
    top = np.multiply(-y, x)
    pred = np.dot(x, theta)
    bot = 1 + np.exp(np.multiply(y, pred))
    return (top/bot).mean(axis=0)

#     a = np.random.choice(range(n), n, replace=False)

def sgd(arrx, arry,learning_rate=0.01, epochs=50):
    theta = np.zeros(shape=(21,1))
    ones = np.ones(shape = (3000,1))
    arrx_array = np.hstack((ones, arrx))
    arry_array = np.array(arry, dtype='float64')
    n = len(arrx)
    minloss = 10**10
    bestTheta = 0
    err = []       
    for j in range(epochs): #This is for the number of training iteration.
        loss = cost_function(theta,arrx_array, arry_array)
        theta -= (learning_rate * d_cost(theta,arrx_array,arry_array)).reshape(21,1)         
        if loss < minloss:
            minloss = loss
            bestTheta = np.copy(theta)
        err.append(loss)
        if (j + 1) % 100 == 0:
            # save weights of the function
            np.save(f'./models/model_{j+1}',theta, allow_pickle=True, fix_imports=True)
            print(f'>>> Model_{j+1} saved to models')
    return bestTheta , minloss, err

theta, loss, err = sgd(x,y,0.1,10000)


print(f'Theta: {theta} \n')

print(f'Loss: {loss} \n')

print(f'Training Error: {err} \n')


import matplotlib.pyplot as plt
plt.plot(err)
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 15:36:12 2017

@author: stuart
"""

#another attempt at a simple neural network
import csv
import math
import numpy as np


#set a seed for testing
#np.random.seed(1)

#define the sigmoid function to perform sigmoid on every element in the 
#z vector
def sig(z):
    
    sig = 1/(1 + np.exp(-z))
    
    return sig


def sig_prime(z):
    
    #z needs to be converted into an array to avoid confusion with treating z as
    #a matrix for the squaring function

    z = np.array(z)    
    
    sig_p = np.exp(-z)/((1 + np.exp(-z))**2)
    
    #convert back to a matrix for the output to avoid extra problems
    
    return np.matrix(sig_p)


#create a class for our network

class N_network:
    
    #initialise the matrix. This v.basic network can only work with simple
    #neurons applying a preset sigmoid function. No split layers/convolutions
    #/RNN etc.
    def __init__(self, layer_matrix):
        
        self.cost = []
        #layer matrix will contain a list of numbers with a number for the number
        #of nodes in each layer

        
        #work through the requested layers generating weights and biases for each layer
        weights = []
        biases = []
        for layer_no, layer in enumerate(layer_matrix[1:]):
            
            lay_weights = np.random.rand(layer_matrix[layer_no], layer) * 0.05

            lay_bias = np.random.rand(layer_matrix[layer_no + 1])
            
            weights.append(np.matrix(lay_weights))
            
            biases.append(np.matrix(lay_bias))

            
        self.ws = weights
        
        self.bs = biases

        self.shape = layer_matrix
        
        self.Z = []
        self.A = []
            
            
        
    def feed_forward(self, X):
        
        a = X
        
        self.Z = []
        self.A = []
        
        for layer, w in enumerate(self.ws):
            
            #use matrix maths to find the value of z for each node in the layer
            #then apply the bias
            #The bias has been temporeraly removed to reduce the complication of
            #back propogation for the first version of this network
            z = a * w + self.bs[layer]

            self.Z.append(z)

            #apply the sigmoid function to the result of the matrix calculation
            #to find the neuron output for each node
            a = sig(z)
            
            self.A.append(a)
            
            
        return a
        
    def find_one_epoch(self, input_data, target):
        
        #first find the results of running this network on the test data
        y_hat = self.feed_forward(input_data)
        
        delta3 = np.multiply(-(target - y_hat), sig_prime(self.Z[-1]))

        dJdW2 = self.A[-2].transpose() * delta3
        
        self.dJdWi = [dJdW2]
        
        self.dJddi = [delta3]
        
        self.delta = [delta3]
        

        
        for layer_no in range(len(self.shape))[:-2][::-1]:
            
            stage1 = self.delta[0] * self.ws[layer_no+1].transpose()
            
            new_delta = np.multiply(stage1, sig_prime(self.Z[layer_no]))
            
            
            self.delta.insert(0, new_delta)
            
            if layer_no == 0:
            
                dJdW = input_data.transpose() * new_delta
                
            else:
                
                dJdW = self.A[layer_no - 1].transpose() * new_delta
            
            self.dJdWi.insert(0, dJdW)
            
            self.dJddi.insert(0, new_delta)
            
            
        return [self.delta, self.dJdWi, self.dJddi]
        
        
        
    def run_back_prop(self, train_data, train_target, test_data, test_Y, epochs, nu):
        
        for i in range(epochs):

            [cost, dJdWs] = self.run_one_epoch(train_data, train_target, nu)
            
            self.cost.append(cost)
            
            if i%100 == 0:
                print('epoch %s, train cost:- %s'%(str(i), str(cost)))
                
        Y_hat = self.feed_forward(test_data)
    
        test_C = Y_hat - test_Y

        cost = np.sum(abs(test_C))/len(test_C)
        
        return cost

    def run_back_prop_batch(self, train_data, train_target, test_data, test_Y, epochs, nu, batch_size=50):
        
        for i in range(epochs):
        
            [X, Y] = batch_sample(train_data, train_target, batch_size)

            [cost, dJdWs] = self.run_one_epoch(X, Y, nu)
            
            self.cost.append(cost)
            
            if i%100 == 0:
                print('epoch %s, train cost:- %s'%(str(i), str(cost)))
                
        Y_hat = self.feed_forward(test_data)
    
        test_C = 0.5 * np.square(Y_hat - test_Y)

        cost = (np.sum(test_C)/len(test_C))
        
        
        return cost
        
        
    def run_one_epoch(self, input_data, target_data, nu):
        
        [deltas, dJdWs, dJddi] = self.find_one_epoch(input_data, target_data)
            
        for layer_no, dJdW in enumerate(dJdWs):

            self.ws[layer_no] = self.ws[layer_no] - (nu * dJdW)
            
        for layer_no, dJdb in enumerate(dJddi):
            
            self.bs[layer_no] = self.bs[layer_no] - (nu * sum(dJdb))


        Y_hat = self.feed_forward(input_data)
    
        test_C = 0.5 * np.square(Y_hat - target_data)

        cost = (np.sum(test_C)/len(test_C))
        
        return [cost, dJdWs]

#sample a batch from the training data
def batch_sample(input_data, target_data, batch_size):
    
    positions = np.random.randint(len(input_data), size=batch_size)

    X = input_data[positions, :]
    
    Y = target_data[positions, :]

    return [X, Y]

#read a csv in
def load_csv(filename):

    rows = []
    with open(filename, newline='\n') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        
        for row in csvreader:
            rows.append(row)

    return rows

#convert character data in csv into something the network can read
def process_rows4neural_net(rows):
    
    target = []
    X = []
    for row in rows:
        
        this_target = np.zeros(10)
        
        
        this_target[int(row[0])] = 1
        
        target.append(this_target)
        
        #normalise x
        a = [float(x)/255 for x in row[1:]]
        
        X.append(np.array(a).astype('float64'))
    
    
    test_x = np.matrix(X)
    X = np.matrix(X)
    target = np.matrix(target).astype('float64')

    return [X, target]

#show a row from the data to confirm it's labeled right and for debug
def show_an_X(X, row_no):
    
    dat = np.array(X)[row_no]
    dat_rows = []
    
    for i, d in enumerate(dat):
    
        if i % 28 == 0:
            
            dat_rows.append([])
            
        dat_rows[-1].append(d)
    
    img = np.array(dat_rows)
    
    print(np.shape(dat_rows))
    
    plt.figure()
    plt.imshow(img, cmap='gray')
    
###################################
### End of function definitions ###
###################################


test_filename = './MNIST_digit_dataset/mnist_test.csv'

train_filename = './MNIST_digit_dataset/mnist_train.csv'

print('loading csvs')

#train_rows = load_csv(train_filename)

train_rows = load_csv(train_filename)


[train_X, train_target] = process_rows4neural_net(train_rows)

print('Loaded train data')

test_rows = load_csv(test_filename)

[test_X, test_target] = process_rows4neural_net(test_rows)

print('test data')

#Test with the more complex data

eg_shape = [784, 300, 10]

#training rate
nu = 0.005

#initialise the network
net = N_network(eg_shape)

print('training')

cost = net.run_back_prop_batch(train_X, train_target, test_X, test_target, 20000, nu)

print("Final cost is:- %f"%float(cost))


#Lets display the abilities of this neural network in some other interesting ways.

#Lets find out how many it got wrong

result = net.feed_forward(test_X)

wrongs = 0
for i, y in enumerate(result):
    
    if np.argmax(result[i]) != np.argmax(test_target[i]):
        wrongs += 1
        
per = float(wrongs)/len(result) * 100
print("Got the wrong answer %d times, %d%% of the time"%(wrongs, per))


#plot the cost over the training
import matplotlib.pyplot as plt

plt.figure()
plt.plot(net.cost)
plt.title('Training results on MNIST dataset')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()


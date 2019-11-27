

import numpy as np

# X = (note 1, note 2, note 3, note 4), y = note 5
xAll = np.array(([1, 2,3,4],[4, 5,6,7],[2,3,4,5],[3, 4,5,6],[6,7,8,1],[5, 6,7,8],[5, 6,7,8],[5,6,7,8]), dtype=float) # input data
y = np.array(([1,1,1,1,100,1,1,1],[1,1,1,1,1,1,1,100], [1,1,1,1,1,100,1,1],[1,1,1,1,1,1,100,1],[1,100,1,1,1,1,1,1],[100,1,1,1,1,1,1,1],[100,1,1,1,1,100,1,1]), dtype=float) # output
notes = np.array(('C','D','E','F','G','A','B','H'))
# scale units
xAll = xAll/np.amax(xAll, axis=0) # scaling input data par rapport aux plus grand x ou y ici 5 ou 10
y = y/100 # scaling output data (max test score is 100)

# split data
X = np.split(xAll, [7])[0] # training data
xPredicted = np.split(xAll, [7])[1] # testing data

class Neural_Network(object):
  def __init__(self):
  #parameters
    self.inputSize = 4
    self.outputSize = 8
    self.hiddenSize = 8

  #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (2x3) weight matrix from input to hidden layer
    self.W2h1 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer
    self.W2h2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights HIDDEN
    self.z2h1 = self.sigmoid(self.z) # activation function

    self.z2 = np.dot(self.z2h1, self.W2h1)
    self.z2h2 = self.sigmoid(self.z2)

    self.z3 = np.dot(self.z2h2, self.W2h2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function (1,2)
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propagate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2h1.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_errorh2 = self.o_delta.dot(self.W2h2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2h1) # applying derivative of sigmoid to z2 error
    self.z2_deltah2 = self.z2_errorh2*self.sigmoidPrime(self.z2h2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2h1 += self.z2h1.T.dot(self.z2_deltah2) # adjusting second set (hidden --> output) weights
    self.W2h2 += self.z2h2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2h1.txt", self.W2h1, fmt="%s")
    np.savetxt("w2h2.txt", self.W2h2, fmt="%s")

  def predict(self):
    print ("Predicted data based on trained weights: ")
    print ("Input (scaled): \n" + str(xPredicted))
    rep = self.forward(xPredicted)
    print (rep)
    print ("nouvelle Note choisi par le reseauxN: ")
    posNote = np.argmax(rep)
    print(notes[posNote])
    print ("Mélodie compléte : ")
    print ("G")
    print ("A")
    print ("B")
    print ("H")
    print(notes[posNote])
  

   # print ("Rep algo: \n" + str(self.forward(xPredicted)))

NN = Neural_Network()
for i in range(1000): # trains the NN 1,000 times
  print ("# " + str(i) + "\n")
  print ("Input (scaled): \n" + str(X))
  print ("Actual Output: \n" + str(y))
  print ("Predicted Output: \n" + str(NN.forward(X)))
  print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
  print ("\n")
  NN.train(X, y)

NN.saveWeights()
NN.predict()



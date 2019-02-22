import numpy as np
import hw1 
from hw1 import random_normal_weight_init as weight_init
from hw1 import zeros_bias_init as bias_init
np.random.seed(1)
class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        # You shouldn't need to edit anything in init
        self.alpha = alpha #Learning rate
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))
    

    def __call__(self, x, eval):
        return self.forward(x, eval)

    def forward(self, x, eval):
        self.x = x
        if not eval:
            self.mean = np.mean(x, axis=0) # ok
            self.var = np.var(x, axis=0) # ok
            self.norm = (self.x - self.mean) / np.sqrt(self.var + self.eps) #ok
            print (self.mean.shape,self.var.shape,self.norm.shape)
            self.out = self.gamma * self.norm  + self.beta #ok
            # update running batch statistics
            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var
            return self.out
        else:
            norm = (self.x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * norm + self.beta

    def backward(self, delta):
        N,D = self.x.shape
        self.dgamma = np.sum(delta * self.norm, axis=0) #ok
        self.dbeta = np.sum(delta, axis=0) #ok  
        x_mu = self.x - self.mean 
        i_var = 1 / np.sqrt(self.var + self.eps)  #std_inv
        dxn = delta * self.gamma
        #dvar = np.sum(dxn * self.norm * -0.5 * i_var**3 ,axis=0)
        #dmu = np.sum(dxn * (-1) * i_var, axis=0) + dvar * np.mean(-2. * x_mu, axis=0)
        dvar = np.sum(-1.0/2*dxn*(self.x-self.mean)*(self.var+self.eps)**(-3.0/2), axis =0)
        # dmu = np.sum(-1/np.sqrt(self.var+self.eps)* dxn, axis = 0) - 2.0/N*dvar *np.sum((self.x-self.mean), axis = 0) 
        dmu = np.sum(dxn * -i_var, axis=0) + dvar * np.mean(-2. * x_mu, axis=0)
        return ((dxn * i_var) + (dvar * 2 * x_mu / N) + dmu / N)



mlp = hw1.MLP(784, 10, [64, 32], [hw1.Sigmoid(), hw1.Sigmoid(), hw1.Identity()],
                  weight_init, bias_init, hw1.SoftmaxCrossEntropy(), 0.008,
                  momentum=0.0, num_bn_layers=1)

inp = np.random.randn(3,784)
y = np.random.randn(3,10)
#print (inp)
out = mlp(inp)
#print (out)

mlp.backward(y)
print (mlp.dW)
print (mlp.db)
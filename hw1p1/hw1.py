"""
    Follow the instructions provided in the writeup to completely
    implement the class specifications for a basic MLP, optimizer, .
    You will be able to test each section individually by submitting
    to autolab after implementing what is required for that section
    -- do not worry if some methods required are not implemented yet.
    
    Notes:
    
    The __call__ method is a special reserved method in
    python that defines the behaviour of an object when it is
    used as a function. For example, take the Linear activation
    function whose implementation has been provided.
    
    # >>> activation = Identity()
    # >>> activation(3)
    # 3
    # >>> activation.forward(3)
    # 3
    """

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os


class Activation(object):
    
    """
        Interface for activation functions (non-linearities).
        
        In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
        """
    
    # No additional work is needed for this class, as it acts like an abstract base class for the others
    
    def __init__(self):
        self.state = None
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        raise NotImplemented
    
    def derivative(self):
        raise NotImplemented


class Identity(Activation):
    
    """
        Identity function (already implemented).
        """
    
    # This class is a gimme as it is already implemented for you as an example
    
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        self.state = x
        return x
    
    def derivative(self):
        return 1.0



class Sigmoid(Activation):
    
    """
        Sigmoid non-linearity
        """
    
    # Remember do not change the function signatures as those are needed to stay the same for AL
    
    def __init__(self):
        super(Sigmoid, self).__init__()
    
    def forward(self, x):
        # Might we need to store something before returning?
        y = 1 / (1 + np.exp(-x))
        self.state = y
        return y
    
    def derivative(self):
        # Maybe something we need later in here...
        y = self.state
        return y * (1-y)


class Tanh(Activation):
    
    """
        Tanh non-linearity
        """
    
    # This one's all you!
    
    def __init__(self):
        super(Tanh, self).__init__()
    
    def forward(self, x):
        y = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        self.state = y
        return y
    
    def derivative(self):
        y = self.state
        return 1 - y**2


class ReLU(Activation):
    
    """
        ReLU non-linearity
        """
    
    def __init__(self):
        super(ReLU, self).__init__()
    
    def forward(self, x):
        x[x<0] = 0
        y=x
        self.state = y
        return y
    
    def derivative(self):
        y = self.state
        y[y>0] = 1
        y[y<0] = 0
        return y

# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):
    
    """
        Interface for loss functions.
        """
    
    # Nothing needs done to this class, it's used by the following Criterion classes
    
    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None
    
    def __call__(self, x, y):
        return self.forward(x, y)
    
    def forward(self, x, y):
        raise NotImplemented
    
    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):
    
    """
        Softmax loss
        """
    
    def __init__(self):
        
        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None
    
    def forward(self, x, y):
        self.logits = x
        self.labels = y
        ce=[]
        self.sm=[]
        for i in range(0,len(x)):
            xi,yi = x[i],y[i]
            sm_sum = sum(np.exp(xi))
            q,p = np.exp(xi) / sm_sum , yi
            self.sm.append(q)
            ce.append(sum(- p * np.log(q)))
        self.sm = np.array(self.sm)
        return np.array(ce)
    
    def derivative(self):
        return self.sm - self.labels


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


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return np.random.randn(d0,d1)


def zeros_bias_init(d):
    return np.zeros((1,d))


class MLP(object):
    
    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):
        
        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------
        
        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        unit = [self.input_size] + hiddens + [self.output_size]
        list_W,list_b=[],[]
        for i in range(len(unit)-1):
            s = weight_init_fn(unit[i],unit[i+1])
            t = bias_init_fn(unit[i+1])
            list_s = list(s)
            list_t = list(t)
            list_W.append(np.array(list_s))
            list_b.append(np.array(list_t))
        self.dW = None
        self.db = None
        self.W = list_W
        self.b = list_b
        self.state=[]
        self.intermediate=None
        #set initial value in momentum
        self.delta = 0.0
        self.bdelta= 0.0
        # HINT: self.foo = [ bar(???) for ?? in ? ]
        
        # if batch norm, add batch norm parameters
        if self.num_bn_layers:
            self.bn_layers = [BatchNorm(unit[1], 0.90)]


# Feel free to add any other attributes useful to your implementation (input, output, ...)

def forward(self, x):
    y = x
        self.intermediate = y
        for k in range(0,self.nlayers):
            if self.num_bn_layers>0 and k == 0:
                z = np.dot(y,self.W[k])
                z = self.bn_layers[k].forward(z,not self.train_mode)
            else:
                z = np.dot(y,self.W[k])+self.b[k]
            
            y = self.activations[k].forward(z)
        self.state = y
        return y
    
    def zero_grads(self):
        #self.dW = np.zeros((np.array(self.W).shape))
        #self.db = np.zeros((np.array(self.b).shape))
        self.dW = None
        self.db = None
        return

def step(self):
    if self.momentum>0:
        self.delta = self.momentum * self.delta - self.lr * self.dW
            self.W = self.W + self.delta
            self.bdelta = self.momentum * self.bdelta - self.lr * self.db
            for idx in range(len(self.b)):
                self.b[idx] = self.b[idx] + self.bdelta[idx]
    else:
        self.W = self.W - self.lr * self.dW
            for idx in range(len(self.b)):
                self.b[idx] = self.b[idx] - self.lr * self.db[idx]


if self.num_bn_layers>0:
    for i in range(0,len(self.bn_layers)):
        self.bn_layers[i].gamma = self.bn_layers[i].gamma - self.lr * self.bn_layers[i].dgamma
        self.bn_layers[i].beta = self.bn_layers[i].beta - self.lr * self.bn_layers[i].dbeta
        return
    
    def backward(self, labels):
        np.set_printoptions(precision=8)
        loss = self.criterion.forward(self.state,labels)
        result = self.criterion.derivative()
        res_w,res_b=[],[]
        dy = result
        for k in range(self.nlayers-1,-1,-1):
            
            dz = self.activations[k].derivative() * np.array(dy)
            
            if self.num_bn_layers>0 and k == 0:
                dz = self.bn_layers[k].backward(dz)
            dy = np.dot(dz,np.transpose(self.W[k]))
            if k==0:
                dw = np.dot(np.transpose(self.intermediate),dz)
            else:
                dw = np.dot(np.transpose(self.activations[k-1].state),dz)
            db = dz
            res_b.append(sum(db))
            res_w.append(dw)
    if self.num_bn_layers:
        self.bn_layers.reverse()
        res_w.reverse()
        res_b.reverse()
        self.dW = np.array(res_w) / len(labels)
        self.db = np.array(res_b) / len(labels)
        return sum(loss)

def __call__(self, x):
    return self.forward(x)
    
    def train(self):
        self.train_mode = True
    
    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):
    train, val, test = dset
    trainx, trainy = train
    trainy_ = np.zeros((trainy.shape[0], 10))
    trainy_[np.arange(trainy.shape[0]), trainy] = 1
    trainy = trainy_
    valx, valy = val
    valy_ = np.zeros((valy.shape[0], 10))
    valy_[np.arange(valy.shape[0]), valy] = 1
    valy = valy_
    testx, testy = test
    testy_ = np.zeros((testy.shape[0], 10))
    testy_[np.arange(testy.shape[0]), testy] = 1
    testy = testy_
    idxs = np.arange(len(trainx))
    
    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []
    
    # Setup ...
    
    for e in range(nepochs):
        count_t = 0.0
        t_error_rate = 0.0
        for b in range(0, len(trainx), batch_size):
            training_loss = 0.0
            mlp.train()
            mlp.zero_grads()
            mlp.forward(trainx[b:b + batch_size])
            t_pred = mlp(trainx[b:b+batch_size])
            t_label = trainy[b:b+batch_size]
            t_pred = t_pred.argmax(1)
            t_label = t_label.argmax(1)
            training_loss = mlp.backward(trainy[b:b + batch_size])
            t_error_rate += np.sum(t_pred != t_label)
            mlp.backward(trainy[b:b+ batch_size])
            mlp.step()
            training_loss /= len(trainx)
            count_t += training_loss
        #print('Training Loss:', training_loss)
        training_error = t_error_rate / len(trainy)
        training_losses.append(count_t)
        training_errors.append(training_error)
        
        count_v = 0.0
        v_error_rate=0.0
        for b in range(0, len(valx), batch_size):
            mlp.eval()
            validation_loss = 0.0
            mlp.zero_grads()
            mlp.forward(valx[b:b + batch_size])
            v_pred = mlp(valx[b:b+batch_size])
            v_label = valy[b:b+batch_size]
            v_pred = v_pred.argmax(1)
            v_label = v_label.argmax(1)
            validation_loss = mlp.backward(valy[b:b + batch_size])
            v_error_rate += np.sum(v_pred != v_label)
            mlp.step()
            validation_loss /= len(valx)
            count_v += validation_loss
#print('Validation Loss', validation_loss)
validation_error = v_error_rate / len(valy)
    validation_losses.append(count_v)
    validation_errors.append(validation_error)
    
    
    #for b in range(0, len(testx), batch_size):
    
    return (training_losses, training_errors, validation_losses, validation_errors)







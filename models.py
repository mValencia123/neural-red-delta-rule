from utils import get_random_number
import matplotlib.pyplot as plt
import numpy as np
from enums import Calc
from enums import Schemas
import random

class NeuronalRed:
    def __init__(self,layers,neurons,x,d,epochs,step=0.2,calc=Calc.GENERALLY_DELTA,clasification_multiple=False,schema=Schemas.SGD,batch=100,delta=0):
        # Layers of neural red
        self.layers   = []
        # Output of neural red
        self.y        = []
        # Training data
        self.X        = x
        # Desire results
        self.D        = d
        # learning factor
        self.step     = step
        # epochs
        self.epochs   = epochs
        # index training data
        self.index_training_data = 0
        # error convergence
        self.J = []
        # Type of calculate
        self.calc = calc
        # if our neural red does clasification multiple
        self.clasification_multiple = clasification_multiple
        # use multiple layers
        self.shallow = not (layers > 3)
        # Schema to update w
        self.schema = schema
        self.batch = batch

        self.delta = delta
        for i in range(0,layers):
            input_layer = False
            output_layer  = False
            if i == 0:
                input_layer = True
            if i == layers - 1:
                output_layer = True
            _layer = Layer(i,neurons,input_layer,output_layer,self)
            self.layers.append(_layer)

    def __str__(self):
        return f"I have {len(self.layers)} layers"
    
    def start_training(self):
        for i in range(0,self.epochs):
            print(f'epoch {i + 1}/{self.epochs} ')
            error = 0
            D_aux = self.D
            D_use = self.D

            X_aux = self.X
            X_use = self.X
            if self.batch != 100:
                D_use = []
                X_use = []
                amount = len(self.D) * self.batch / 100
                for i in range(0,int(amount)):
                    n = random.randint(0,len(D_aux) - 1)
                    D_use.append(D_aux[n])
                    #D_aux.pop(n)
                    np.delete(D_aux,n)

                    X_use.append(X_aux[n])
                    #X_aux.pop(n)
                    np.delete(X_aux,n)
            for i in range(0,len(D_use)):
                self.index_training_data = i
                for layer in self.layers:
                    if not layer.input_layer:
                        layer.calculate_layer()
                    else:
                        layer.output = X_use[i]
                if (self.clasification_multiple == False):
                    error = error + pow((D_use[i][0] - self.layers[-1].output[0]),2)
                else:
                    sub = np.subtract(self.D[i],self.layers[-1].output)
                    sum = round(np.sum(sub),6)
                    abs = np.abs(sum)
                    error = error + abs
                for i in range(len(self.layers) - 1,0,-1):
                    if not self.layers[i].input_layer:
                        self.layers[i].backpropagation()
            error = error / len(self.D)
            if self.schema != Schemas.SGD:
                for layer in self.layers:
                    error_aux = np.sum(layer.errors) / len(D_use)
                    layer.update_w_with_batch(error_aux)
                    layer.errors = [0] * len(layer.neurons)
            self.J.append(error)
        plt.plot(range(0,self.epochs),self.J)
        plt.xlabel('epochs')
        plt.ylabel('error')
        plt.title('average square error')
        plt.show()
    
    def test(self):
        for i in range(0,len(self.D)):
            self.index_training_data = i
            for layer in self.layers:
                if not layer.input_layer:
                    layer.calculate_layer()
                else:
                    layer.output = self.X[i]
                if layer.output_layer:
                    print(f'for input {self.X[i]} -> {self.D[i][0]} {layer.output}')
    
    def test_individul(self,D):
        for i in range(0,len(D)):
            for layer in self.layers:
                if not layer.input_layer:
                    layer.calculate_layer()
                else:
                    layer.output = self.X[i]
                if layer.output_layer:
                    print(f'for input {D[i]} -> {layer.output}')
        pass

class Layer:
    def __init__(self,index,neurons,input_layer,output_layer,neural_red):
        # neural red
        self._neural_red = neural_red
        # Neurons of layer
        self.neurons = []
        # is this layer the input layer?
        self.input_layer  = input_layer
        # is this layer the output layer?
        self.output_layer  = output_layer
        # Index layer from 0 to m
        self.index = index
        # Output of layer
        self.output = [0] * neurons[self.index]
        # Deltas of layer
        self.deltas = [0] * neurons[self.index]
        # Output of layer
        self.output_b = [0] * neurons[self.index]

        self.errors = [0] * neurons[self.index]
        if output_layer:
            self.next_length_w = 0
        else:
            self.next_length_w = neurons[index + 1]
        # factor to use with softmax
        self.factor_soft_max = [0] * neurons[self.index]
        for i in range(0,neurons[index]):
            _neuron = Neuron(i,self.next_length_w,self)
            self.neurons.append(_neuron)

    def update_w_with_batch(self,error):
        for neuron in self.neurons:
            neuron.update_w_with_batch(error)
    
    def calculate_layer(self):
        for neuron in self.neurons:
            neuron.calculate_neuron()
        if(self._neural_red.clasification_multiple == True and self.output_layer == True):
            exp_v = np.exp(self.output - np.max(self.output))
            self.output = exp_v / exp_v.sum()
    
    def get_weights_by_index(self,index):
        _w = []
        for neuron in self.neurons:
            _w.append(neuron.W[index])
        return _w
    
    def backpropagation(self):
        for neuron in self.neurons:
            neuron.calculate_error()
            neuron.calculate_delta()
            if self._neural_red.schema == Schemas.SGD:
                neuron.update_w()
    

class Neuron:
    def __init__(self,index,lenW,layer):
        #layer of this neuron belongs
        self._layer = layer
        # Get amount of neurons to get weights
        self.W = []
        self.delta = 0
        self.output = 0
        self.index = index
        self.error = 0
        self.momentum = []
        for i in range(0,lenW):
            _w = get_random_number()
            print(f'adding {i} w to neuron {self.index} on layer {self._layer.index}: {_w}')
            self.W.append(_w)
            self.momentum.append([0])
    
    def update_w_with_batch(self,error):
        for i in range(0,len(self.W)):
            self.W[i] = self.W[i] + error

    def update_w(self):
        # Amount neurons in next layer
        neurons_prev_layer = self.get_prev_layer().neurons
        for neuron in neurons_prev_layer:
            neuron.update_weigths(self.index)

    def update_weigths(self,index):
        delta_w = (self._layer._neural_red.step * self.get_next_layer().deltas[index] * self._layer.output[self.index])
        m     = delta_w + (self._layer._neural_red.delta * self.momentum[index][self._layer._neural_red.index_training_data])
        # print(self.momentum)
        self.momentum[index].append(m)
        self.W[index] = self.W[index] + m
        
    def calculate_error(self):
        if self._layer.output_layer:
            red = self._layer._neural_red
            self.error = red.D[red.index_training_data][self.index] - self.output
        else:
            _w  = np.array(self.W)
            _deltas = np.transpose(np.array(self.get_next_layer().deltas))
            self.error = (_w @ _deltas)
        self._layer.errors[self.index] = self._layer.errors[self.index] + self.error

    def relu_derivate(self):
        aux = (self.output > 0)
        return aux
    

    def calculate_delta(self):
        if self._layer._neural_red.calc == Calc.GENERALLY_DELTA :
            self.delta = self.output  * (1 - self.output) * self.error
        else:
            if not self._layer.output_layer:
                self.delta = self.output  * (1 - self.output) * self.error
            else:
                self.delta = self.error
        self._layer.deltas[self.index] = self.delta

    def activation(self,v):
        return 1/(1 + np.exp(-v))
    
    def activation_relu(self,v):
        return max(0,v)

    def calculate_neuron(self):
        _prev_layer = self.get_prev_layer()
        xi = np.transpose(np.array(_prev_layer.output))
        w  = np.array(_prev_layer.get_weights_by_index(self.index))
        v = (w @ xi)
        if (self._layer._neural_red.clasification_multiple == False):
            self.output = self.activation(v)
        else:
            if (self._layer.output_layer == False):
                if (self._layer._neural_red.shallow):
                    self.output = self.activation(v)
                else:
                    self.output = self.activation_relu(v)
            else:
                self.output = v
                # self._layer.output_b[self.index] = v
        self._layer.output[self.index] = self.output
    
    def get_error(self):
        red = self._layer._neural_red
        return red.D[red.index_training_data] - self.output

    def get_prev_layer(self):
        return self._layer._neural_red.layers[self._layer.index - 1]
    
    def get_next_layer(self):
        return self._layer._neural_red.layers[self._layer.index + 1]
    
    def calculate_neuron_with_softmax(self):
        self.output = self._layer.factor_soft_max[self.index]
        self._layer.output[self.index] = self.output
    
    def soft_max(self):
        return (self.output / self._layer.factor_soft_max)
    
    def __str__(self):
        return f"Im a Neuron and I have {len(self.W)} weights"
from utils import get_random_number
import matplotlib.pyplot as plt
import numpy as np
from enums import Calc

class NeuronalRed:
    def __init__(self,layers,neurons,x,d,epochs,step=0.2,calc=Calc.GENERALLY_DELTA,clasification_multiple=False):
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
            for i in range(0,len(self.D)):
                self.index_training_data = i
                for layer in self.layers:
                    if not layer.input_layer:
                        layer.calculate_layer()
                    else:
                        layer.output = self.X[i]
                if (self.clasification_multiple == False):
                    error = error + pow((self.D[i][0] - self.layers[-1].output[0]),2)
                else:
                    error = 0
                for i in range(len(self.layers) - 1,0,-1):
                    if not self.layers[i].input_layer:
                        self.layers[i].backpropagation()
            error = error / len(self.D)
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
                    print(f'for input {self.X[i]} -> {layer.output}')

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
        if output_layer:
            self.next_length_w = 0
        else:
            self.next_length_w = neurons[index + 1]
        # factor to use with softmax
        self.factor_soft_max = 0
        for i in range(0,neurons[index]):
            _neuron = Neuron(i,self.next_length_w,self)
            self.neurons.append(_neuron)
    
    def calculate_layer(self):
        for neuron in self.neurons:
            neuron.calculate_neuron()
        if(self._neural_red.clasification_multiple == True and self.output_layer == True):
            # print("last layer ")
            # print(self.output_layer)
            # print("output b")
            # print(self.output_b)
            self.factor_soft_max = np.sum(self.output_b)
            # print(self.factor_soft_max)
            for neuron in self.neurons:
                neuron.calculate_neuron_with_softmax()
            self.factor_soft_max = 0
            # print(self.output)
    
    def get_weights_by_index(self,index):
        _w = []
        for neuron in self.neurons:
            _w.append(neuron.W[index])
        return _w
    
    def backpropagation(self):
        for neuron in self.neurons:
            neuron.calculate_error()
            neuron.calculate_delta()
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
        for i in range(0,lenW):
            _w = get_random_number()
            print(f'adding {i} w to neuron {self.index} on layer {self._layer.index}: {_w}')
            self.W.append(_w)

    def update_w(self):
        # Amount neurons in next layer
        neurons_prev_layer = self.get_prev_layer().neurons
        for neuron in neurons_prev_layer:
            neuron.update_weigths(self.index)

    def update_weigths(self,index):
        self.W[index] = round(self.W[index] + (self._layer._neural_red.step * self.get_next_layer().deltas[index] * self._layer.output[self.index]),2)

    def calculate_error(self):
        if self._layer.output_layer:
            red = self._layer._neural_red
            # print("errors")
            # print(red.D[red.index_training_data][self.index])
            # print(self.output)
            self.error = red.D[red.index_training_data][self.index] - self.output
        else:
            _w  = np.array(self.W)
            _deltas = np.transpose(np.array(self.get_next_layer().deltas))
            self.error = (_w @ _deltas)

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

    def calculate_neuron(self):
        _prev_layer = self.get_prev_layer()
        xi = np.transpose(np.array(_prev_layer.output))
        w  = np.array(_prev_layer.get_weights_by_index(self.index))
        v = (w @ xi)
        if (self._layer._neural_red.clasification_multiple == False):
            self.output = self.activation(v)
        else:
            if (self._layer.output_layer == False):
                self.output = self.activation(v)
            else:
                aux = np.exp(v)
                self.output = aux
                # print(f"Index {self.index} v {v} e^v {aux}")
                self._layer.output_b[self.index] = aux
                # aux = np.exp(v)
                # aux2 = self._layer.factor_soft_max
                # self._layer.factor_soft_max = aux2 + aux
        self._layer.output[self.index] = self.output
    
    def get_error(self):
        red = self._layer._neural_red
        return red.D[red.index_training_data] - self.output

    def get_prev_layer(self):
        return self._layer._neural_red.layers[self._layer.index - 1]
    
    def get_next_layer(self):
        return self._layer._neural_red.layers[self._layer.index + 1]
    
    def calculate_neuron_with_softmax(self):
        # print(f"output before {self.output}")
        self.output = self.soft_max()
        self._layer.output[self.index] = self.output
    
    def soft_max(self):
        # print(f"using {self._layer.factor_soft_max}")
        return (self.output / self._layer.factor_soft_max)
    
    def __str__(self):
        return f"Im a Neuron and I have {len(self.W)} weights"
# neural-red-delta-rule
This is a basic and easy implementation of a neural red using delta rule.

# usage

1. Initialize neural red creating an instance, params:
   a. layers: int of layers (including input and output layers)
   b. neurons: array with amount of neurons by layer
   c. epochs: int of epochs of training
   d. x: training dataset
   e. d: training results dataset
   f. step (optional): learning step
 red = NeuronalRed(layers=3,neurons=[2,4,1],epochs=10000,x=[[0,0],[0,1],[1,0],[1,1]],d=[[0],[1],[1],[0]],step=0.9)

2. Execute method start_training() to start training of neural red
3. Execute method test() to start testing of neural red with input dataset (training dataset) and training results dataset

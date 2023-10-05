
from models import NeuronalRed
from enums import Calc
import csv
import numpy as np

def main():

    with open('iris.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    data_array = np.array(data)
    X = np.array(data_array[1:,0:-1],dtype=float)
    d = np.array(data_array[1:,-1:])
    norms = np.linalg.norm(X, axis=1)
    for x in range(0,len(X)):
        for y in range(0,len(X[x])):
            X[x][y] = X[x][y] / norms[x]
    print(X)
    D = []
    hotEncoding = {
        'Setosa' : [1,0,0],
        'Versicolor' : [0,1,0],
        'Virginica' : [0,0,1]
    }
    for x in d :
        D.append(hotEncoding[x[0]])

    red = NeuronalRed(layers=3,neurons=[4,8,3],epochs=700,x=X,d=D,step=0.4,calc=Calc.CROSS_ENTROPY,clasification_multiple=True)
    red.start_training()
    red.test()

    # red2 = NeuronalRed(layers=3,neurons=[2,4,1],epochs=10000,x=[[0,0],[0,1],[1,0],[1,1]],d=[[0],[1],[1],[0]],step=0.7,calc=Calc.CROSS_ENTROPY)
    # red2.start_training()
    # red2.test()

if __name__ == "__main__":
    main()
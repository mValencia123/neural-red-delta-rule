
from models import NeuronalRed
from enums import Calc, Schemas
import csv
import numpy as np


def main():


    # X = [
    #     [0,1,1,0,0,
    #     0,0,1,0,0, 
    #     0,0,1,0,0, 
    #     0,0,1,0,0, 
    #     0,1,1,1,0],# 1

    #     [1,1,1,1,0, 
    #     0,0,0,0,1, 
    #     0,1,1,1,0, 
    #     1,0,0,0,0, 
    #     1,1,1,1,1],# 2

    #   [ 1,1,1,1,0, 
    #     0,0,0,0,1,
    #     0,1,1,1,0, 
    #     0,0,0,0,1, 
    #     1,1,1,1,0 ],# 3

    #   [ 0,0,0,1,0, 
    #     0,0,1,1,0, 
    #     0,1,0,1,0, 
    #     1,1,1,1,1, 
    #     0,0,0,1,0  ],# 4

    #   [ 1,1,1,1,1, 
    #     1,0,0,0,0, 
    #     1,1,1,1,0, 
    #     0,0,0,0,1, 
    #     1,1,1,1,0 ],# 5
    # ]

    # D = [
    #     [1,0,0,0,0],
    #     [0,1,0,0,0],
    #     [0,0,1,0,0],
    #     [0,0,0,1,0],
    #     [0,0,0,0,1]
    # ]

    # red = NeuronalRed(layers=5,neurons=[25,20,20,20,5],epochs=100,x=X,d=D,step=0.01,calc=Calc.CROSS_ENTROPY,clasification_multiple=True)
    # red.start_training()
    # red.test()

    # with open('approve.csv', 'r') as f:
    #     reader = csv.reader(f)
    #     data = list(reader)
    # data_array = np.array(data)
    # X = np.array(data_array[1:,0:-1],dtype=float)
    # d = np.array(data_array[1:,-1:])
    # norms = np.linalg.norm(X, axis=1)
    # for x in range(0,len(X)):
    #     for y in range(0,len(X[x])):
    #         X[x][y] = X[x][y] / norms[x]
    # print(X)
    # D = []
    # for i in range(0,len(d)):
    #     D.append([int(d[i][0])])
    # print(D)
    # red = NeuronalRed(layers=3,neurons=[2,2,1],epochs=1000,x=X,d=D,step=0.3,calc=Calc.CROSS_ENTROPY,schema=Schemas.SGD,batch=80)
    # red.start_training()
    # red.test()
    red = NeuronalRed(layers=3,neurons=[3,4,1],epochs=6000,x=[[0,0,1],[0,1,1],[1,0,1],[1,1,1]],d=[[0],[1],[1],[0]],step=0.3,calc=Calc.CROSS_ENTROPY,delta=0.3)
    red.start_training()
    red.test()


if __name__ == "__main__":
    main()
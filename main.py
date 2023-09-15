
from models import NeuronalRed
from enums import Calc

def main():
    red = NeuronalRed(layers=3,neurons=[3,4,1],epochs=4000,x=[[0,0,1],[0,1,1],[1,0,1],[1,1,1]],d=[[0],[1],[1],[0]],step=0.4,calc=Calc.CROSS_ENTROPY)
    red.start_training()
    red.test()

    # red2 = NeuronalRed(layers=3,neurons=[2,4,1],epochs=4500,x=[[0,0],[0,1],[1,0],[1,1]],d=[[0],[1],[1],[0]],step=0.9,calc=Calc.CROSS_ENTROPY)
    # red2.start_training()
    # red2.test()

if __name__ == "__main__":
    main()
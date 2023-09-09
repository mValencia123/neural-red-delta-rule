
from models import NeuronalRed


def main():
    # red = NeuronalRed(layers=3,neurons=[3,4,1],epochs=10000,x=[[0,0,1],[0,1,1],[1,0,1],[1,1,1]],d=[[0],[1],[1],[0]],step=0.9)
    red = NeuronalRed(layers=3,neurons=[2,4,1],epochs=10000,x=[[0,0],[0,1],[1,0],[1,1]],d=[[0],[1],[1],[0]],step=0.9)
    red.start_training()
    red.test()

if __name__ == "__main__":
    main()
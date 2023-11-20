from enum import Enum

class Calc(Enum):
    GENERALLY_DELTA = 0
    CROSS_ENTROPY   = 1

class Schemas(Enum):
    SGD = 0
    BATCH = 1
    MINI_BATCH = 2
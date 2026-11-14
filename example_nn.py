import random
from micrograd.nn import Neuron, Layer, MLP


def inspect(module):
    params = module.parameters()
    print(f'{module} with {len(params)} parameters:')
    print(f'{module.parameters()}\n')


random.seed(1337)
inspect(Neuron(2))
inspect(Layer(2, 4))
inspect(MLP(2, [4, 4, 1]))

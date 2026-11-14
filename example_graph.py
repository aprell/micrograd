import random
from micrograd.engine import Value
from micrograd.nn import Neuron
from micrograd.utils import draw_graph

# A very simple example
x = Value(1.0)
y = (x * 2 + 1).relu()
y.backward()
print(draw_graph(y))

# A single neuron with two inputs
random.seed(1337)
n = Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
y.backward()
draw_graph(y).render('neuron.dot', format='png')

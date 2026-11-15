import random
import time
from micrograd.engine import Value
from micrograd.nn import MLP


# See https://github.com/tekknolagi/micrograd/blob/c15b6b8fd373c48014be369c4f7bd0917932a53b/test.py#L75
def timer(func, msg):
    print(msg)
    start = time.perf_counter()
    result = func()
    stop = time.perf_counter()
    print(f" ↳ {stop - start:.2f} s")
    return result


xor_truth_table = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

inputs = [[Value(x), Value(y)] for (x, y), _ in xor_truth_table]
results = [Value(z) for _, z in xor_truth_table]

random.seed(1337)
model = timer(lambda: MLP(2, [4, 1]), "Building model...")
print()


def forward():
    outputs = [model(inp) for inp in inputs]
    errors = [(res - out)**2 for res, out in zip(results, outputs)]
    accuracies = [round(out.data) == int(res.data) for res, out in zip(results, outputs)]
    loss = sum(errors) / len(errors)
    accuracy = sum(accuracies) / len(accuracies)
    return loss, accuracy


def train(epochs, verbose=False):
    def update(k):
        # learning_rate = 1.0 - 0.9 * k / epochs
        learning_rate = 0.1
        for p in model.parameters():
            p.data -= learning_rate * p.grad
    for k in range(1, epochs + 1):
        # Forward pass
        loss, accuracy = forward()
        # Backward pass (backpropagation)
        model.zero_grad()
        loss.backward()
        # Update model parameters
        update(k)
        if verbose and (k < 10 or k % 10 == 0):
            print(f"Step {k:2d}: loss = {loss.data:.6f}, accuracy = {accuracy:.1%}")


timer(lambda: train(50, verbose=True), "Training model...")
print()

print("Testing model...")
for (x, y), z in xor_truth_table:
    r = round(model([x, y]).data)
    print(f"{x} XOR {y} = {r} {'\u2713' if r == z else '\u2717'}")

from graphviz import Digraph


def trace(root):
    nodes, edges = set(), set()

    def visit(n):
        if n not in nodes:
            nodes.add(n)
            for child in n._prev:
                edges.add((child, n))
                visit(child)

    visit(root)
    return nodes, edges


def draw_graph(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    for n in nodes:
        dot.node(name=str(id(n)), label=f'{{ data: {n.data:.4f} | grad: {n.grad:.4f} }}', shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

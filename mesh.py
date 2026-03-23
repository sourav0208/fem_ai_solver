import numpy as np

def generate_nodes(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0) -> np.ndarray:
    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)

    print(x)
    print(y)

    nodes = []

    for j in range(ny):
        for i in range(nx):
            nodes.append([x[i], y[j]])
        
    print(nodes)

    return np.array(nodes, dtype=float)

def generate_elements(nx: int, ny: int) -> np.ndarray:
    elements = []

    for j in range(ny -1):
        for i in range(nx-1):
            n0 = j*nx + i
            n1 = n0 + 1
            n2 = n0 + nx
            n3 = n2 + 1

            elements.append([n0, n1, n3])
            elements.append([n0, n3, n2])

    return np.array(elements, dtype=int)


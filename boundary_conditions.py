import numpy as np

def find_boundary_nodes(nodes, lx: float, ly: float, tol = 1e-12):
    boundary_nodes = []

    for i , (x,y) in enumerate(nodes):
        if(
            abs(x -0.0) < tol
            or abs(x-lx) < tol
            or abs(y-0.0) < tol
            or abs(y - ly) < tol
        ):
            boundary_nodes.append(i)

    return boundary_nodes
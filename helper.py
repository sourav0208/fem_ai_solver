import numpy as np
import time
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from mesh import *
from plots import *
from boundary_conditions import *

def solve_problem(nx, ny, lx=1.0, ly=1.0, source=1.0, use_sparse=False):
    nodes = generate_nodes(nx,ny,lx=lx,ly=ly)
    elements = generate_elements(nx,ny)

    if use_sparse:
        K = assemble_global_stiffness_sparse(nodes, elements)
    else:
        K = assemble_global_stiffness_dense(nodes, elements)

    f = assemble_global_load(nodes, elements, source=source)
    boundary_nodes = find_boundary_nodes(nodes, lx, ly)
    K, f = apply_dirichlet(K, f, boundary_nodes, value=0.0)

    if use_sparse:
        u = spsolve(K,f)
    else:
        u = np.linalg.solve(K,f)

    return nodes, elements, K, f, u
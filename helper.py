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

def dense_memory_bytes(K_dense):
    return K_dense.nbytes

def sparse_memory_bytes(K_sparse):
    return K_sparse.data.nbytes + K_sparse.indices.nbytes + K_sparse.indptr.nbytes

def benchmark_case(nx, ny, lx=1.0,ly=1.0, source=1.0):
    print(f"\n --- Benchmark for mesh {nx}x{ny} ---")

    nodes = generate_nodes(nx,ny,lx,ly)
    elements = generate_elements(nx,ny)
    boundary_nodes = find_boundary_nodes(nodes, lx, ly)

    t0 = time.perf_counter()
    K_dense = assemble_global_stiffness_dense(nodes, elements)
    f_dense = assemble_global_load(nodes, elements, source)
    K_dense, f_dense = apply_dirichlet(K_dense, f_dense, boundary_nodes, value=1.0)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    u_dense = np.linalg.solve(K_dense, f_dense)
    t3 = time.perf_counter()

    dense_assemble_time = t1 - t0
    dense_solve_time = t3-t2
    dense_mem_usage = dense_memory_bytes(K_dense)

    t4 = time.perf_counter()
    K_sparse = assemble_global_stiffness_sparse(nodes, elements)
    f_sparse = assemble_global_load(nodes, elements, source)
    K_sparse, f_sparse = apply_dirichlet(K_sparse, f_sparse, boundary_nodes, value=0.0)
    t5 = time.perf_counter()

    t6 = time.perf_counter()
    u_sparse = spsolve(K_sparse, f_sparse)
    t7 = time.perf_counter()

    sparse_assemble_time = t5 - t4
    sparse_solve_time = t7-t6
    sparse_mem_usage = sparse_memory_bytes(K_sparse)

    print("Dense matrix shape:", K_dense.shape)
    print("Sparse matrix shape:", K_sparse.shape)
    print("Sparse nonzeros:", K_sparse.nnz)

    print(f"Dense assembly time: {dense_assemble_time:.6f} s")
    print(f"Dense solve time: {dense_solve_time:.6f} s")
    print(f"Sparse assembly time: {sparse_assemble_time:.6f} s")
    print(f"Sparse solve time: {sparse_solve_time:.6f} s")

    print(f"Dense memory usage: {dense_mem_usage/1024:.2f} KB")
    print(f"Sparse memory usage: {sparse_mem_usage/1024:.2f} KB")

    print("Max difference between dense and sparse solution:",
          np.max(np.abs(u_dense - u_sparse)))

    



    

    






from boundary_conditions import find_boundary_nodes, apply_dirichlet, assemble_global_load
from mesh import generate_elements, generate_nodes, assemble_global_stiffness_dense
from plots import plot_nodes, plot_mesh, plot_solution
from helper import solve_problem, benchmark_case
import numpy as np
import pandas as pd



def main():
    nx, ny = 5, 5
    lx, ly = 1.0, 1.0

    nodes, elements, K, f, u = solve_problem(nx, ny, lx, ly, source = 1.0, use_sparse = True)
    print("Matrix type:", type(K))
    print("Number of nonzeros:", K.nnz)
    print("Max value of U:", np.max(u))

    u_grid = u.reshape((ny, nx))
    print("Solution as grid:")
    print(u_grid)
    plot_nodes(nodes)
    plot_mesh(nodes, elements)
    plot_solution(nodes, elements, u)


def refinement_study():
    lx=1.0
    ly=1.0
    mesh_size = [5,9,17,33,43,50]
    
    for n in mesh_size:
        nx, ny = n,n

        nodes = generate_nodes(nx,ny,lx,ly)
        elements = generate_elements(nx,ny)

        K = assemble_global_stiffness_dense(nodes, elements)
        f = assemble_global_load(nodes, elements, source=1.0)
        boundary_nodes = find_boundary_nodes(nodes,lx,ly)
        K,f = apply_dirichlet(K,f,boundary_nodes)
        u = np.linalg.solve(K,f)
        max_u = np.max(u)
        plot_nodes(nodes)
        plot_mesh(nodes, elements)
        plot_solution(nodes, elements, u)
        print(f"Mesh {n}x{n}: max(u) = {max_u:.6f}")

def benchmark_study():
    mesh_size = [5,9,17,33,43,50]
    results = []
    for n in mesh_size:
        nx, ny = n,n
        data= benchmark_case(nx,ny,lx=1.0,ly=1.0,source=1.0)
        results.append(data)
        
    df = pd.DataFrame(results)
    print("\nBenchmark Table:\n")
    print(df)
    df.to_csv("Benchmark_results.csv", index=True)
    return df
        
        

if __name__ == "__main__":
    main()
    #refinement_study()
    benchmark_study()





from mesh import generate_elements, generate_nodes, triangle_area, local_stiffness, assemble_global_stiffness
from plots import plot_nodes, plot_mesh
import numpy as np


def main():
    nx, ny = 5, 5
    lx, ly = 1.0, 1.0

    nodes = generate_nodes(nx, ny, lx, ly)
    elements = generate_elements(nx, ny)

    print("Nodes shape:", nodes.shape)
    print("Elements shape:", elements.shape)
    print("First 5 nodes:\n", nodes[:5])
    print("First 5 elements:\n", elements[:5])

    k = 0
    coords = nodes[elements[k]]

    print("Element index:", k)
    print("Node indices of elements:", elements[k])
    print("Coordinates of elements:\n", coords)

    area = triangle_area(coords)
    print("Area of elements:", area)

    ke = local_stiffness(coords)
    print("Local stiffness matrix:\n", ke)
    print(np.allclose(ke,ke.T))
    print(np.sum(ke, axis=1))


    k = 1
    coords = nodes[elements[k]]

    print("Element index:", k)
    print("Node indices of elements:", elements[k])
    print("Coordinates of elements:\n", coords)

    area = triangle_area(coords)
    print("Area of elements:", area)

    ke = local_stiffness(coords)
    print("Local stiffness matrix:\n", ke)
    print(np.allclose(ke,ke.T))
    print(np.sum(ke, axis=1))



    k = 2
    coords = nodes[elements[k]]

    print("Element index:", k)
    print("Node indices of elements:", elements[k])
    print("Coordinates of elements:\n", coords)

    area = triangle_area(coords)
    print("Area of elements:", area)

    ke = local_stiffness(coords)
    print("Local stiffness matrix:\n", ke)
    print(np.allclose(ke,ke.T))
    print(np.sum(ke, axis=1))

    K = assemble_global_stiffness(nodes, elements)

    print("Global stiffness matrix shape:", K.shape)
    print("Top-left 10x10 block of K:\n", K[:10,:10])
    print("Is global matrix symmetric?", np.allclose(K,K.T))
    print("Row sums of K (first 10):", np.sum(K, axis=1)[:10])




    plot_nodes(nodes)
    plot_mesh(nodes, elements)


if __name__ == "__main__":
    main()





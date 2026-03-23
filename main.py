from mesh import generate_elements, generate_nodes
from plots import plot_nodes, plot_mesh


def main():
    nx, ny = 5, 5

    nodes = generate_nodes(nx, ny)
    elements = generate_elements(nx, ny)

    print("Nodes shape:", nodes.shape)
    print("Elements shape:", elements.shape)
    print("First 5 nodes:\n", nodes[:5])
    print("First 5 elements:\n", elements[:5])

    plot_nodes(nodes)
    plot_mesh(nodes, elements)


if __name__ == "__main__":
    main()





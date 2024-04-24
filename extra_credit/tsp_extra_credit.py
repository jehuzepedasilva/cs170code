import os
import pickle

import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np


def minimum_spanning_tree(matrix):
    csr_matrix = csr_matrix(matrix)

    mst_matrix = minimum_spanning_tree(csr_matrix)

    mst_dense = mst_matrix.toarray()

    return mst_dense


def minimum_weight_matching(matrix, mst):
    odd_vertices = find_odd_degree_vertices(mst)
    adjacency_list = create_adjacency_list(matrix)

    matching = []
    matched = set()

    for v in odd_vertices:
        if v not in matched:
            u, w = find_closest_neighbor(v, odd_vertices, adjacency_list, matched)
            matching.append((u, v))
            matched.add(u)
            matched.add(v)

    return matching


def find_odd_degree_vertices(graph):
    odd_vertices = []
    for i, row in enumerate(graph):
        degree = sum(1 for weight in row if weight > 0)
        if degree % 2 != 0:
            odd_vertices.append(i)
    return odd_vertices


def create_adjacency_list(matrix):
    adjacency_list = {}
    n = len(matrix)
    for i in range(n):
        adjacency_list[i] = [(j, matrix[i][j]) for j in range(n) if matrix[i][j] > 0]
    return adjacency_list


def find_closest_neighbor(vertex, odd_vertices, adjacency_list, matched):
    min_weight = float('inf')
    closest_neighbor = None
    for neighbor, weight in adjacency_list[vertex]:
        if neighbor in odd_vertices and neighbor not in matched and weight < min_weight:
            min_weight = weight
            closest_neighbor = neighbor
    return closest_neighbor, min_weight


def combine_graphs(mst, matching):
    multigraph = np.copy(mst)

    for u, v in matching:
        multigraph[u][v] = multigraph[v][u] = 1  # Adding edges for matching

    return multigraph


def eulerian_tour(graph):
    nx_graph = nx.from_numpy_array(graph)

    euler_circuit = list(nx.eulerian_circuit(nx_graph))

    tour = [edge[0] for edge in euler_circuit]
    tour.append(euler_circuit[-1][1]) 

    return tour


def shortcut_tour(euler_tour):
    return euler_tour


def improve_tour(tour, matrix):
    return tour


def christofides_tsp_approximation(matrix):
    mst = minimum_spanning_tree(matrix)
    matching = minimum_weight_matching(matrix, mst)
    multigraph = combine_graphs(mst, matching)
    euler_tour = eulerian_tour(multigraph)
    tsp_tour = shortcut_tour(euler_tour)
    improved_tour = improve_tour(tsp_tour, matrix)
    return improved_tour

def improved_tsp_approximation(matrix):
    # YOUR CODE HERE
    return christofides_tsp_approximation(matrix)


def validate_tour(tour, matrix):
    """
    Provided function to verify the validity of your TSP approximation function.
    Returns the length of the tour if it is valid, -1 otherwise.
    Feel free to use or modify this function however you please,
    as the autograder will only call your tsp_approximation function.
    """
    n = len(tour)
    cost = 0
    for i in range(n):
        if matrix[tour[i - 1]][tour[i]] == float("inf"):
            return -1
        cost += matrix[tour[i - 1]][tour[i]]
    return cost


def evaluate_tsp(tsp_approximation):
    """
    Provided function to evaluate your TSP approximation function.
    Feel free to use or modify this function however you please,
    as the autograder will only call your tsp_approximation function.
    """

    test_cases = pickle.load(open(os.path.join("tsp_cases.pkl"), "rb"))

    total_cost = 0
    for i, case in enumerate(test_cases["files"]):
        tour = tsp_approximation(case)
        cost = validate_tour(tour, case)
        assert cost != -1
        total_cost += cost
        print(f"Case {i}: {cost}")

    for i, case in enumerate(test_cases["generated"], start=len(test_cases["files"])):
        tour = tsp_approximation(case)
        cost = validate_tour(tour, case)
        assert cost != -1
        total_cost += cost
        print(f"Case {i}: {cost}")

    print(f"Total cost: {total_cost}")
    return total_cost


if __name__ == "__main__":
    evaluate_tsp(improved_tsp_approximation)

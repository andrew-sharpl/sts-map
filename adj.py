import json
import numpy as np
from scipy.optimize import linprog


def from_json(data, weighted=False):
    edge_lst = data['edges']

    # Get unique nodes
    nodes = {(n['x'], n['y']) for n in data['nodes']}
    nodes.add((-1, -1))  # s
    nodes.add((np.inf, np.inf))  # t

    # Add edges from source to first level
    edge_lst = [{"src_x": -1, "src_y": -1, "dst_x": n[0], "dst_y": n[1]}
                for n in nodes
                if n[1] == 0] + edge_lst

    # Add edges from last level to destination
    edge_lst = edge_lst + [{"src_x": n[0], "src_y": n[1], "dst_x": np.inf, "dst_y": np.inf}
                           for n in nodes
                           if n[1] == 14]

    # Sort nodes
    node_lst = sorted(nodes, key=lambda n: (n[1], n[0]))
    node_index = {node: index for index, node in enumerate(node_lst)}
    index_node = {index: node for index, node in enumerate(node_lst)}

    # Initialize incidence matrix
    num_nodes = len(node_lst)
    num_edges = len(edge_lst)
    incidence_matrix = np.zeros((num_nodes, num_edges), dtype=int)

    # Update the incidence matrix
    for col, edge in enumerate(edge_lst):
        src = (edge['src_x'], edge['src_y'])
        dst = (edge['dst_x'], edge['dst_y'])

        src_index = node_index[src]
        dst_index = node_index[dst]

        if weighted:
            incidence_matrix[src_index, col] = -1 * node_value(get_node_type(src, data))
            incidence_matrix[dst_index, col] = node_value(get_node_type(dst, data))
        else:
            incidence_matrix[src_index, col] = -1
            incidence_matrix[dst_index, col] = 1

    return incidence_matrix, index_node



def get_node_type(node, data):
    if node == (-1, -1):
        return "Source"
    elif node == (np.inf, np.inf):
        return "Sink"
    return next(n['class'] for n in data['nodes'] if (n['x'], n['y']) == node)


def node_value(node_type):
    match node_type:
        case "MonsterRoom":
            return 11
        case "ShopRoom":
            return 22
        case "RestRoom":
            return 33
        case "MonsterRoomElite":
            return 44
        case "EventRoom":
            return 55
        case "TreasureRoom":
            return 66
        case "Source" | "Sink":
            return 0

def node_risk(node_type):
    match node_type:
        case "MonsterRoom":
            return 11
        case "ShopRoom":
            return 22
        case "RestRoom":
            return 33
        case "MonsterRoomElite":
            return 44
        case "EventRoom":
            return 55
        case "TreasureRoom":
            return 66
        case "Source" | "Sink":
            return 0


def traverse_incidence_matrix(incidence_matrix, vertex, cind=None, visited=None):
    if cind is None:
        cind = []

    if visited is None:
        visited = []

    visited.append(vertex)

    rows, cols = incidence_matrix.shape

    for col in range(cols):
        if incidence_matrix[vertex][col] == -1 and col not in cind:
            cind.append(col)

            # Find the row where the entry is +1 in the same column
            next_vertex = np.where(incidence_matrix[:, col] == 1)[0][0]

            # Continue recursively if the next_vertex is not visited
            if next_vertex not in visited:
                traverse_incidence_matrix(incidence_matrix, next_vertex, cind, visited)

    return cind

if __name__ == '__main__':
    f = open('maps/673465884448_Act1.json')
    json_data = json.load(f)
    incidence, node_in = from_json(json_data)

    edge_values = [] # objective function coefficients
    edge_risks = []

    rhs_eq = [0 for _ in range(incidence.shape[0])]
    rhs_eq[0] = -1
    rhs_eq[-1] = 1
    bnd = [(0, np.inf) for _ in range(incidence.shape[1])]
    for vertex in incidence.T:
        i = np.argwhere(vertex == 1)[0][0]
        t = get_node_type(node_in[i], json_data)
        edge_values.append(node_value(t))
        edge_risks.append(node_risk(t))

    opt = linprog(c=edge_values, A_eq=incidence, b_eq=rhs_eq, bounds=bnd)
    print(opt)
    chosen_edges = [i for i in range(len(opt.x)) if opt.x[i] > 0.1]

    new_start = 62
    cind = traverse_incidence_matrix(incidence, new_start)
    inc = incidence[:, cind]


    edge_values = []  # objective function coefficients
    edge_risks = []
    rhs_eq = [0 for _ in range(inc.shape[0])]
    rhs_eq[new_start] = -1
    rhs_eq[-1] = 1
    bnd = [(0, np.inf) for _ in range(inc.shape[1])]
    first = True
    for vertex in inc.T:
        i = np.argwhere(vertex == 1)[0][0]
        t = get_node_type(node_in[i], json_data)
        edge_values.append(node_value(t))
        edge_risks.append(node_risk(t))

    opt = linprog(c=edge_values, A_eq=inc, b_eq=rhs_eq, bounds=bnd)
    print(opt)


#%%

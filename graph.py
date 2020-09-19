class Node():
    def __init__(self, id, label, coordinates):
        self.id = id
        self.label = label
        self.coordinates = tuple(coordinates)

        self.x, self.y = coordinates


class Edge():
    def __init__(self, source, target):
        self.source = source
        self.target = target


class Graph():
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

        self._nodes_by_id = {
            node.id: node for node in nodes
        }

    def get_node_by_id(self, id):
        return self._nodes_by_id[id]


def create_graph(nodes, edges):
    return Graph(
        nodes=[Node(**node) for node in nodes],
        edges=[Edge(**edge) for edge in edges],
    )

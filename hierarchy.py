import networkx as nx

heirarchy = open('data/AmazonHeirarchy.json').read()

labels = json.loads(heirarchy)

root_labels = {}

for lab in labels:
    root_labels[int(lab['BrowseNodeId'])] = lab['Name']


from sets import Set

def yield_graph(dictionary):
    nodes, names, edges = [], [], []
    parent_id = int(dictionary['BrowseNodeId'])
    nodes.append(parent_id)
    names.append(dictionary['Name'])
    if dictionary.has_key('Children'):
        to_visit = dictionary['Children']
        if to_visit.__class__ is list:
            for child in to_visit:
                e = (parent_id, int(child['BrowseNodeId']))
                edges.append(e)
                sub_nodes, sub_edges, sub_names = yield_graph(child)
                nodes += sub_nodes
                edges += sub_edges
                names += sub_names
        else:
            e = (parent_id, int(to_visit['BrowseNodeId']))
            edges.append(e)
            sub_nodes, sub_edges, sub_names = yield_graph(to_visit)
            nodes += sub_nodes
            edges += sub_edges
            names += sub_names

    return nodes, edges, names


G = nx.DiGraph()

G.add_node('amazon')

V, E, names = [], [], []

graphs =

for lab in labels:
    E += [('amazon', lab['BrowseNodeId'])]
    v, e, na = yield_graph(lab)
    V += v
    E += e
    names += na

# for nodeid, productname in zip(V, names):
#   G.add_node(nodeid, name = productname)

G.add_nodes_from(V)
G.add_edges_from(E)



G = nx.Graph()




graphs = []

for lab in labels:
    V, E, names = [], [], []
    v, e, _ = yield_graph(lab)
    V += v
    E += e
    G = nx.Graph()

    G.add_nodes_from(V)
    G.add_edges_from(E)

    graphs.append(G)








def depth_first_search():
    pass
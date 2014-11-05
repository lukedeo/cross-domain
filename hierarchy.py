"""
Hierarchy.py
Functions and loaders for treating with the categories hierarchy of Amazon products
"""
import json
import networkx as nx
from sets import Set

DEFAULT_AMAZON_HIERARCHY = 'data/AmazonHeirarchy.json'

def yield_graph(subcat_hierarchy):
    """
    Recursively builds a graph from a subcategory hierarchy 
    """
    nodes, names, edges = [], [], []

    parent_id = int(subcat_hierarchy['BrowseNodeId'])
    nodes.append(parent_id)
    names.append(subcat_hierarchy['Name'])
    if subcat_hierarchy.has_key('Children'):
        to_visit = subcat_hierarchy['Children']
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

def get_categories(nodeid, graphs):
    """
    Gets categories for a given nodeid and a list of category graphs
    """
    return [category for category, G in graphs.iteritems() if nodeid in G.nodes()]

def get_amazon_graphs(filename=DEFAULT_AMAZON_HIERARCHY):
    """
    Returns the complete amazon hierarchy given in a list of graphs.
    Each graph represents the complete hierarchy of a main category
    """

    hierarchy = open(filename).read()
    labels = json.loads(hierarchy)

    graphs = {}

    for lab in labels:
        V, E, names = [], [], []
        v, e, _ = yield_graph(lab)
        V += v
        E += e
        G = nx.Graph()
        
        G.add_nodes_from(V)
        G.add_edges_from(E)
        
        graphs[lab['Name']] = G

    return graphs

# visualization.py
"""
This module handles the visualization of expression trees using the
graphviz library. It separates the plotting logic from the core algorithm.
"""
try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    class Digraph:
        def __init__(self, *args, **kwargs): pass
        def node(self, *args, **kwargs): pass
        def edge(self, *args, **kwargs): pass
        def render(self, *args, **kwargs): pass

def draw_tree(tree, filename="best_solution_tree"):
    """
    Visualizes an expression tree using graphviz and saves it as a PNG file.
    """
    if not GRAPHVIZ_AVAILABLE:
        print("Warning: graphviz library not found. Tree visualization will be skipped.")
        return

    dot = Digraph(comment='Best Evolved Expression')
    
    def add_nodes_edges(t, dot_graph, node_id=0):
        node_name = f"node{node_id}"
        dot_graph.node(node_name, t.get_node_name())
        
        child_start_id = node_id + 1
        for child in t.children:
            child_node_name = f"node{child_start_id}"
            dot_graph.edge(node_name, child_node_name)
            child_start_id = add_nodes_edges(child, dot_graph, child_start_id)
            
        return child_start_id

    add_nodes_edges(tree, dot)
    try:
        dot.render(filename, format='png', cleanup=True)
        print(f"Best solution tree saved to '{filename}.png'")
    except Exception as e:
        print(f"Error during tree rendering: {e}")
        print("Please ensure Graphviz is installed and in your system's PATH.")
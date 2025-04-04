from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import re

def print_network_structure(model):
    """
    Print out the structure of the Bayesian Network.
    
    Args:
        model: A pgmpy BayesianNetwork model
    """
    print("\nBayesian Network Structure:")
    print("Nodes:", model.nodes)
    print("\nEdges:")
    for edge in model.edges:
        print(f"{edge[0]} -> {edge[1]}")
    print("\nParents of likelihood_of_success:", model.get_parents('likelihood_of_success'))

def create_bayesian_network() -> BayesianNetwork:
    """
    Creates a Bayesian Network based on logical statements from logical_statements.txt
    
    Returns:
        BayesianNetwork: A pgmpy BayesianNetwork object representing the relationships
    """
    print("Starting Bayesian Network creation...")
    
    # Read logical statements
    print("Reading logical statements...")
    with open('logical_statements.txt', 'r') as f:
        statements = f.read()
    print(f"Found {len(statements.splitlines())} statements")

    # Initialize Bayesian Network
    print("Initializing network...")
    model = BayesianNetwork()
    
    # Parse IF-THEN statements to extract variables and relationships
    statements = statements.splitlines()
    edges = set()
    variables = set()
    
    # Group conditions by category
    condition_groups = {
        'experience': set(),
        'education': set(),
        'skills': set(),
        'network': set(),
        'background': set(),
        'other': set()
    }
    
    print("Parsing statements...")
    for i, statement in enumerate(statements, 1):
        # Skip empty lines or lines without IF
        if not statement.strip() or 'IF' not in statement:
            continue
            
        # Extract conditions and outcome
        match = re.search(r'\*\*IF\*\* (.*?) \*\*THEN\*\* likelihood_of_success', statement)
        if not match:
            print(f"Warning: Could not parse statement {i}: {statement[:50]}...")
            continue
            
        conditions = match.group(1)
        # Split on **AND** to get individual conditions
        conditions = [c.strip() for c in conditions.split('**AND**')]
        
        # Add each condition as a node and create edge to likelihood_of_success
        for condition in conditions:
            # Clean up the condition text
            condition = condition.strip()
            if condition:
                condition_node = condition.lower().replace(' ', '_')
                
                # Categorize the condition
                if any(word in condition_node for word in ['experience', 'expertise', 'track_record']):
                    condition_groups['experience'].add(condition_node)
                elif any(word in condition_node for word in ['education', 'degree', 'university']):
                    condition_groups['education'].add(condition_node)
                elif any(word in condition_node for word in ['skill', 'technical', 'ability']):
                    condition_groups['skills'].add(condition_node)
                elif any(word in condition_node for word in ['network', 'connection', 'investor']):
                    condition_groups['network'].add(condition_node)
                elif any(word in condition_node for word in ['background', 'history', 'past']):
                    condition_groups['background'].add(condition_node)
                else:
                    condition_groups['other'].add(condition_node)
                
                variables.add(condition_node)
                edges.add((condition_node, 'likelihood_of_success'))
    
    # Print condition groups
    print("\nCondition Groups:")
    for group, conditions in condition_groups.items():
        print(f"{group}: {len(conditions)} conditions")
    
    print(f"\nTotal unique conditions: {len(variables)}")
    print(f"Total edges: {len(edges)}")
            
    variables.add('likelihood_of_success')
    
    # Add nodes and edges to model
    print("\nAdding nodes and edges to model...")
    model.add_nodes_from(list(variables))
    model.add_edges_from(list(edges))
    
    # Initialize CPDs with placeholder probabilities
    print("Initializing CPDs...")
    for var in variables:
        if var != 'likelihood_of_success':
            # Binary variables for conditions
            cpd = TabularCPD(variable=var, variable_card=2,
                           values=[[0.5], [0.5]])
            model.add_cpds(cpd)
    
    # Add CPD for likelihood_of_success with a maximum of 5 parents per group
    print("\nAdding CPD for likelihood_of_success...")
    max_parents_per_group = 5
    selected_parents = []
    
    for group, conditions in condition_groups.items():
        # Randomly select up to max_parents_per_group parents from each group
        selected = list(conditions)[:max_parents_per_group]
        selected_parents.extend(selected)
    
    # Limit total parents to 20
    selected_parents = selected_parents[:20]
    print(f"Selected {len(selected_parents)} parents for likelihood_of_success")
    
    parent_card = [2] * len(selected_parents)
    if parent_card:
        success_cpd = TabularCPD(variable='likelihood_of_success',
                                variable_card=2,
                                values=[[0.5] * (2 ** len(parent_card)),
                                       [0.5] * (2 ** len(parent_card))],
                                evidence=selected_parents,
                                evidence_card=parent_card)
        model.add_cpds(success_cpd)
    
    print("Network creation complete!")
    return model

def display_bayesian_network(model):
    """
    Display the Bayesian Network using networkx and matplotlib.
    
    Args:
        model: A pgmpy BayesianNetwork model
    """
    print("Starting network visualization...")
    import matplotlib.pyplot as plt
    import networkx as nx
    
    # Convert pgmpy model to networkx graph
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes)
    G.add_edges_from(model.edges)
    
    # Create layout for nodes
    print("Creating node layout...")
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the graph
    print("Drawing network...")
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=2000,
                          alpha=0.7)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20)
    
    # Add labels
    labels = {node: node.replace('_', '\n') for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, 
                           labels=labels,
                           font_size=8,
                           font_weight='bold')
    
    plt.title("Startup Success Prediction Bayesian Network")
    plt.axis('off')
    plt.tight_layout()
    print("Displaying network...")
    plt.show()

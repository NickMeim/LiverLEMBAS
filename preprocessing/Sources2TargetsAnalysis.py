import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle,FancyArrowPatch

def extract_and_visualize_first_neighbors(network_df, desired_node):
    # Build the graph from the dataframe
    G = nx.DiGraph()
    for _, row in network_df.iterrows():
        G.add_edge(row['source'], row['target'], interaction=row['interaction'])
    # Get 1st neighbors of the desired node
    first_neighbors = set(nx.all_neighbors(G, desired_node))
    sub_nodes = first_neighbors.union({desired_node})
    sub_G = G.subgraph(sub_nodes).copy()
    # Save the subnetwork to CSV
    subnetwork_df = pd.DataFrame(
        [(u, v, d["interaction"]) for u, v, d in sub_G.edges(data=True)],
        columns=["source", "target", "interaction"]
    )
    subnetwork_df.to_csv(desired_node+"_first_neighbors_subnetwork.csv", index=False)
    # Visualize the subnetwork
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    pos = nx.spring_layout(sub_G)
    # Draw nodes and labels
    nx.draw_networkx_nodes(sub_G, pos, node_size=500, ax=ax, node_color="lightblue")
    nx.draw_networkx_labels(sub_G, pos, font_size=10, ax=ax)
    # Draw edges with different styles based on interaction sign
    for edge in sub_G.edges(data=True):
        source, target, interaction = edge
        sign = interaction["interaction"]
        # Draw edges based on the interaction sign
        if sign > 0:  # Positive interaction
            style = 'solid'
            color = "black"
            width = 2
            # Draw an arrow with solid line
            nx.draw_networkx_edges(
                sub_G,
                pos,
                edgelist=[(source, target)],
                width=width,
                edge_color=color,
                style=style,
                ax=ax,
                arrows=True
            )
        elif sign < 0:  # Negative interaction
            style = 'solid'
            color = "red"
            width = 2
            # Draw a T-bar arrow
            nx.draw_networkx_edges(
                sub_G,
                pos,
                edgelist=[(source, target)],
                width=width,
                edge_color=color,
                style=style,
                ax=ax,
                arrows=True
            )
        else:  # Uncertain interaction (interaction == 0)
            style = 'dashed'
            color = "gray"
            width = 1
            # Draw a dashed line for uncertain interactions
            nx.draw_networkx_edges(
                sub_G,
                pos,
                edgelist=[(source, target)],
                width=width,
                edge_color=color,
                style=style,
                ax=ax
            )
    # Turn off unnecessary x and y axis labels
    ax.set_axis_off()
    plt.savefig(desired_node+"_first_neighbors_subnetwork.png", bbox_inches="tight", dpi=600)
    return sub_G

# Example input network dataframe
network_df = pd.read_csv("preprocessed_data/PKN/prior_drug_signaling_net_simplified.csv")
DrugInfo = pd.read_csv('preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv', sep='\t')
# List of drugs and the desired target node
# source_list = network_df[network_df["source"].isin(DrugInfo.drug.unique())]["source"].unique().tolist()
source_list = ["TGFBR1","TGFBR2"]
# ["IFNAR1","IFNAR2","IFNGR1","IFNGR2","TGFBR1","TGFBR2"]
desired_node = "STAT2"

# Step 1: Create a directed graph with edge attributes
G = nx.DiGraph()
for _, row in network_df.iterrows():
    G.add_edge(row["source"], row["target"], interaction=row["interaction"])
print('Created directed graph with edge attributes')

## Visualize first neighbors
sub_G = extract_and_visualize_first_neighbors(network_df, desired_node)
print('Found and visualized first neighbors of '+desired_node)

# Step 2: Find all paths from the sources to the desired node
subnetwork_edges = set()  # Use a set to avoid duplicate edges
i=0
for source in source_list:
    i+=1
    if source in G:  # Ensure the source exists in the network
        try:
            paths = nx.all_shortest_paths(G, source=source, target=desired_node)
            for path in paths:
                # Extract edges from the path
                    edges = [
                        (path[i], path[i + 1], G[path[i]][path[i + 1]]["interaction"])
                        for i in range(len(path) - 1)
                    ]
                    subnetwork_edges.update(edges)
        except nx.NetworkXNoPath:
            continue
    print('Finished source '+str(i)+' out of '+str(len(source_list)))
print('Found all paths from the sources to '+desired_node)


# Step 3: Create a dataframe for the subnetwork
subnetwork_df = pd.DataFrame(list(subnetwork_edges), columns=["source", "target", "interaction"])
# Save to CSV
# subnetwork_df.to_csv("subnetwork.csv", index=False)
# print('Saved subnetwork to '+desired_node)

# Step 4: Visualize the subnetwork
sub_G = nx.DiGraph()
for source, target, interaction in subnetwork_edges:
    sub_G.add_edge(source, target, interaction=interaction)
# Visualize
# Visualize the subnetwork with custom edge styles
fig = plt.figure(num=None, figsize=(8, 8))
ax = fig.add_subplot(111)
pos = nx.spring_layout(sub_G)
# Draw nodes and labels
nx.draw_networkx_nodes(sub_G, pos, node_size=500, ax=ax, node_color="lightblue")
nx.draw_networkx_labels(sub_G, pos, font_size=10, ax=ax)
# Draw edges with different styles based on interaction sign
for edge in sub_G.edges(data=True):
    source, target, interaction = edge
    sign = interaction["interaction"]
    # Draw edges based on the interaction sign
    if sign > 0:  # Positive interaction
        style = 'solid'
        color = "black"
        width = 2
        # Draw an arrow with solid line
        nx.draw_networkx_edges(
            sub_G,
            pos,
            edgelist=[(source, target)],
            width=width,
            edge_color=color,
            style=style,
            ax=ax,
            arrows=True
        )
    elif sign < 0:  # Negative interaction
        style = 'solid'
        color = "red"
        width = 2
        # Draw a T-bar arrow
        nx.draw_networkx_edges(
            sub_G,
            pos,
            edgelist=[(source, target)],
            width=width,
            edge_color=color,
            style=style,
            ax=ax,
            arrows=True
        )
    else:  # Uncertain interaction (interaction == 0)
        style = 'dashed'
        color = "gray"
        width = 1
        # Draw a dashed line for uncertain interactions
        nx.draw_networkx_edges(
            sub_G,
            pos,
            edgelist=[(source, target)],
            width=width,
            edge_color=color,
            style=style,
            ax=ax
        )
# Turn off unnecessary x and y axis labels
ax.set_axis_off()
# Save and display the figure
fig.savefig(desired_node+'_subnetwork.png', 
            bbox_inches='tight', dpi=600)
print('Visualized subnetwork to '+desired_node)

print('Finished')
import networkx as nx
import matplotlib.pyplot as plt
import json
import tqdm

# Read metadata JSON file in order to build a dictionary
metadata = open("metadata.json")
metadata_dict = json.load(metadata)

# Add a number as unique identifier of each one of the nodes
i = 0
for idx,obj in enumerate(metadata_dict):
    metadata_dict[idx]["node_id"] = i
    i+=1

# Read citations JSON file in order to build a dictionary
citations = open('citations.json')
citations_dict = json.load(citations)

# Build the citations graph
citations_graph = nx.DiGraph()

for citation_obj in tqdm(citations_dict):
    source = citation_obj['source']
    target = citation_obj['target']
    for obj in metadata_dict:
        if obj['id'] == source:
            source_node = obj['node_id']
        elif obj['id'] == target:
            target_node = obj['node_id']
    citations_graph.add_edge(source_node,target_node)

nx.draw(citations_graph)
plt.show() 






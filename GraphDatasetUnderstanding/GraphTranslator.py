# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license
# This file is responsible for translating subgraphs to a graph description language.

import os
import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx


class GraphTranslator:
    def __init__(self, dataset, dataset_name, subgraphs_folder, metrics_list):
        """
        Initializes the GraphTranslator with the dataset, path to the folder containing saved subgraphs,
        and a list of metrics to calculate.

        Parameters:
        - dataset: The PyTorch Geometric dataset from which subgraphs were sampled.
        - dataset_name: Name of the dataset to be translated.
        - subgraphs_folder: Path to the folder containing saved subgraphs.
        - metrics_list: List of metrics to calculate.
        """
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.subgraphs_folder = subgraphs_folder
        self.metrics_list = metrics_list

        # Predefined categorization of metrics into intensive and non-intensive
        self.intensive_metrics_names = ['local_average_shortest_path_length', 'local_graph_diameter',
                                        'local_average_closeness_centrality', 'local_average_betweenness_centrality']
        self.non_intensive_metrics_names = ['node_count', 'edge_count', 'average_degree',
                                            'density', 'average_clustering_coefficient',
                                            'connected_components', 'assortativity',
                                            'average_degree_centrality', 'average_eigenvector_centrality']

        if self.dataset_name == 'ogbn-proteins':
            # Move 'average_clustering_coefficient' from non-intensive to intensive list
            self.non_intensive_metrics_names.remove('average_clustering_coefficient')
            self.intensive_metrics_names.append('average_clustering_coefficient')

        # Filter the requested metrics into intensive and non-intensive based on the provided list
        self.intensive_metrics = [m for m in self.intensive_metrics_names if m in metrics_list]
        self.non_intensive_metrics = [m for m in self.non_intensive_metrics_names if m in metrics_list]

    def translate(self):
        """
        Computes the specified metrics, distinguishing between intensive and non-intensive metrics as defined.
        """
        # Check if the dataset has edge attributes
        has_edge_attr = 'edge_attr' in self.dataset.keys()

        # Convert the entire dataset to a NetworkX graph
        if has_edge_attr:
            if self.dataset_name == 'ogbn-proteins':
                full_graph = to_networkx(self.dataset, to_undirected=True, node_attrs=['node_species'],
                                         edge_attrs=['edge_attr'])
            else:
                full_graph = to_networkx(self.dataset, to_undirected=True, node_attrs=['x'],
                                         edge_attrs=['edge_attr'])
        else:
            # If no edge attributes, do not specify edge_attrs in the conversion
            full_graph = to_networkx(self.dataset, to_undirected=True, node_attrs=['x'])

        # Calculate and print non-intensive metrics
        results = {}
        for metric in self.non_intensive_metrics:
            results[metric] = getattr(self, f"compute_{metric}")(full_graph)
            print(metric)

        # Load subgraphs and compute intensive metrics
        intensive_results = {metric: [] for metric in self.intensive_metrics}
        for path in self.list_subgraph_files():
            subgraph = self.load_subgraph(path)
            for metric in self.intensive_metrics:
                metric_value = getattr(self, f"compute_{metric}")(subgraph)
                intensive_results[metric].append(metric_value)

        # Aggregate results for intensive metrics
        for metric, values in intensive_results.items():
            results[metric] = np.nanmean(values)        # Using nanmean to safely handle any NaN values

        if self.dataset_name == 'ogbn-proteins' and 'average_clustering_coefficient' in results:
            # Change the key in the results dictionary
            results['local_average_clustering_coefficient'] = results.pop('average_clustering_coefficient')

        return results

    def list_subgraph_files(self):
        """
        Lists all .pt files in the subgraphs_folder.
        :return: A list of full paths to the subgraph files.
        """
        return [os.path.join(self.subgraphs_folder, f) for f in os.listdir(self.subgraphs_folder) if f.endswith('.pt')]

    def load_subgraph(self, file_path):
        """
        Loads a saved subgraph from a file and converts it to a NetworkX graph object.
        It also handles both node-level and graph-level labels.
        :param file_path: The path to the file where the subgraph is saved.
        :return: A NetworkX graph object representing the subgraph, with node and edge features, and possibly
        graph-level labels.
        """
        # Load the PyTorch Geometric Data object from file
        subgraph = torch.load(file_path)

        # Check if the subgraph has edge attributes
        has_edge_attr = 'edge_attr' in subgraph.keys()

        # Convert the entire dataset to a NetworkX graph
        if has_edge_attr:
            G = to_networkx(subgraph, to_undirected=True, node_attrs=['x'], edge_attrs=['edge_attr'])
        else:
            # If no edge attributes, do not specify edge_attrs in the conversion
            G = to_networkx(subgraph, to_undirected=True, node_attrs=['x'])

        # If 'y' exists and is graph-level (1D tensor), add it as a graph attribute
        if 'y' in subgraph and subgraph.y.dim() == 1 and subgraph.y.numel() == 1:
            G.graph['label'] = subgraph.y.item()        # Assuming it's a single label

        # If 'y' exists and is node-level, add it as node attributes
        #elif 'y' in subgraph and subgraph.y.dim() > 1:
        elif 'y' in subgraph:
            for i, node_data in enumerate(subgraph.y):
                # Assuming nodes are relabeled from 0 to N-1 in the subgraph
                if self.dataset_name == 'ogbn-proteins':
                    G.nodes[i]['label'] = node_data.tolist()
                else:
                    G.nodes[i]['label'] = node_data.item()

        return G

    @staticmethod
    def compute_node_count(nx_graph):
        """Computes the node count of the graph."""
        return nx_graph.number_of_nodes()

    @staticmethod
    def compute_edge_count(nx_graph):
        """Computes the edge count of the graph."""
        return nx_graph.number_of_edges()

    @staticmethod
    def compute_density(nx_graph):
        """Computes the density of the graph."""
        return nx.density(nx_graph)

    @staticmethod
    def compute_average_degree(nx_graph):
        """Computes the average degree of the graph."""
        total_degree = sum(dict(nx_graph.degree()).values())
        avg_degree = total_degree / nx_graph.number_of_nodes()
        return avg_degree

    @staticmethod
    def compute_average_clustering_coefficient(nx_graph):
        """Computes the average clustering coefficient of the graph."""
        return nx.average_clustering(nx_graph)

    @staticmethod
    def compute_connected_components(nx_graph):
        """Computes the number of connected components in the graph."""
        return nx.number_connected_components(nx_graph)

    @staticmethod
    def compute_assortativity(nx_graph):
        """Computes the degree assortativity coefficient of the graph."""
        return nx.degree_assortativity_coefficient(nx_graph)

    @staticmethod
    def compute_average_degree_centrality(nx_graph):
        """Computes the average degree centrality of the graph."""
        degree_centrality = nx.degree_centrality(nx_graph)
        return np.mean(list(degree_centrality.values()))

    @staticmethod
    def compute_average_eigenvector_centrality(nx_graph):
        """Computes the average eigenvector centrality of the graph."""
        eigenvector_centrality = nx.eigenvector_centrality(nx_graph, max_iter=1000)
        return np.mean(list(eigenvector_centrality.values()))

    @staticmethod
    def compute_local_average_shortest_path_length(nx_graph):
        """Computes the average shortest path length for the largest connected component of the graph."""
        if nx.is_connected(nx_graph):
            return nx.average_shortest_path_length(nx_graph)
        else:
            largest_cc = max(nx.connected_components(nx_graph), key=len)
            subgraph = nx_graph.subgraph(largest_cc)
            return nx.average_shortest_path_length(subgraph)

    @staticmethod
    def compute_local_graph_diameter(nx_graph):
        """Computes the diameter for the largest connected component of the graph."""
        if nx.is_connected(nx_graph):
            return nx.diameter(nx_graph)
        else:
            largest_cc = max(nx.connected_components(nx_graph), key=len)
            subgraph = nx_graph.subgraph(largest_cc)
            return nx.diameter(subgraph)

    @staticmethod
    def compute_local_average_closeness_centrality(nx_graph):
        """Computes the average closeness centrality of the graph."""
        closeness_centrality = nx.closeness_centrality(nx_graph)
        return np.mean(list(closeness_centrality.values()))

    @staticmethod
    def compute_local_average_betweenness_centrality(nx_graph):
        """Computes the average betweenness centrality of the graph."""
        betweenness_centrality = nx.betweenness_centrality(nx_graph)
        return np.mean(list(betweenness_centrality.values()))


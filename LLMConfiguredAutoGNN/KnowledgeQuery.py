# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license

from nas_bench_graph.readbench import light_read, read
from nas_bench_graph.architecture import Arch
import heapq
import itertools


class KnowledgeQuery:
    def __init__(self, k=1, mode='best'):
        """
        Initializes the KnowledgeQuery with the number of top models to fetch.
        :param k: Number of top models to fetch for each selected dataset.
        """
        self.k = k
        self.mode = mode
        self.link_list = [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 2],
            [0, 0, 1, 3],
            [0, 1, 1, 1],
            [0, 1, 1, 2],
            [0, 1, 2, 2],
            [0, 1, 2, 3]
        ]
        self.gnn_list = [
            "gat",  # GAT with 2 heads
            "gcn",  # GCN
            "gin",  # GIN
            "cheb",  # chebnet
            "sage",  # sage
            "arma",
            "graph",  # k-GNN
            "fc",  # fully-connected
            "skip"  # skip connection
        ]
        self.gnn_list_proteins = [
            "gcn",  # GCN
            "sage",  # sage
            "arma",
            "fc",  # fully-connected
            "skip"  # skip connection
        ]
        self.benchmark_mapping = {
            "Cora" : "cora",
            "CiteSeer" : "citeseer",
            "PubMed" : "pubmed",
            "CS" : "cs",
            "Physics" : "physics",
            "Photo" : "photo",
            "Computers" : "computers",
            "ogbn-arxiv" : "arxiv",
            "ogbn-proteins" : "proteins"
            }

    def get_models_for_source_datasets(self, source_to_selected_dict):
        """
        For each source dataset, records the selected top model designs from its selected datasets.
        :param source_to_selected_dict: A dictionary mapping each source dataset to its selected similar datasets.
        :return: A dictionary where each key is a source dataset, and its value is a list of top model designs from the selected datasets.
        """
        source_models_dict = {}
        for source_dataset, selected_datasets in source_to_selected_dict.items():
            models_list = []
            for selected_dataset in selected_datasets:
                best_k_models, worst_k_models = self.get_top_k_models(selected_dataset, self.k, self.mode)
                models_info = {"selected_dataset": selected_dataset, "top_models": best_k_models,
                               "bad_models": worst_k_models}
                models_list.append(models_info)
            source_models_dict[source_dataset] = models_list

        return source_models_dict

    def get_top_k_models(self, dataset, k=1, mode='best'):
        """
        Fetches the top k performing model designs for a specified dataset using the nas_bench_graph library.
        :param dataset: The name of the dataset for which to query the top model designs.
        :param k: The number of top models to fetch.
        :return: A list of tuples, where each tuple contains the link structure and operations of a top model.
        """
        if dataset == "ogbn-arxiv":
            dataset = "arxiv"
        if dataset == "ogbn-proteins":
            dataset = "proteins"
        # Read the dataset's NAS-Bench-Graph data
        nas_bench = light_read(dataset)

        # Find the hashes of the top k models
        best_k_hashes, worst_k_hashes = self.find_k_best(nas_bench, metric='perf', k=k, mode=mode)

        # Iterate through each hash of the top k models
        best_k_models = []
        worst_k_models = []
        if best_k_hashes:
            for hash_value in best_k_hashes:
                # Reverse-engineer the architecture from the hash
                lk, ops = self.reverse_hash(hash_value)
                if lk is not None and ops is not None:
                    best_k_models.append((lk, ops))
                else:
                    print(f"Could not find architecture for model with hash {hash_value}")
                    raise ValueError
        if worst_k_hashes:
            for hash_value in worst_k_hashes:
                # Reverse-engineer the architecture from the hash
                lk, ops = self.reverse_hash(hash_value)
                if lk is not None and ops is not None:
                    worst_k_models.append((lk, ops))
                else:
                    print(f"Could not find architecture for model with hash {hash_value}")
                    raise ValueError

        return best_k_models, worst_k_models

    @staticmethod
    def find_k_best(nas_bench, metric='perf', k=1, mode='best'):
        """
        Finds the top k models in the nas_bench data based on a specified performance metric.

        :param nas_bench: The NAS-Bench-Graph data for a dataset.
        :param metric: The performance metric to use for ranking models (default is 'perf').
        :param k: The number of top models to retrieve.
        :return: A list of hashes for the top k models.
        """
        # Create a list of tuples (model performance, model hash) for all models in nas_bench
        scores_hashes = [(inner_dict[metric], outer_key) for outer_key, inner_dict in nas_bench.items()]

        # Use a heap queue to efficiently find the top k models based on performance and mode
        best_k_hashes, worst_k_hashes = None, None
        if mode in ['best', 'both']:
            best_k = heapq.nlargest(k, scores_hashes)
            best_k_hashes = [hash_value for _, hash_value in best_k]
        if mode in ['worst', 'both']:
            worst_k = heapq.nsmallest(k, scores_hashes)
            worst_k_hashes = [hash_value for _, hash_value in worst_k]

        return best_k_hashes, worst_k_hashes

    def reverse_hash(self, target_hash, use_proteins=False):
        """
        Reverse-engineers the architecture (link structure and operations) of a model from its hash.

        :param target_hash: The hash of the model architecture to reverse-engineer.
        :param use_proteins: A flag indicating whether to use the GNN list for the proteins dataset.
        :return: A tuple of (link_structure, operations) if the architecture is found; otherwise, (None, None).
        """
        gnn_list_to_use = self.gnn_list_proteins if use_proteins else self.gnn_list

        # Iterate over all possible link structures and operation combinations
        for lk in self.link_list:
            for ops_combination in itertools.product(gnn_list_to_use, repeat=len(lk)):
                temp_arch = Arch(lk, list(ops_combination))

                # If the hash matches the target hash, return the architecture details
                if temp_arch.valid_hash() == target_hash:
                    return lk, list(ops_combination)
        return None, None

    def extract_performances(self, dataset, suggested_designs):
        """
        Extracts the performance of suggested model designs for each source dataset.

        :param suggested_designs: A dictionary with source dataset names as keys and their suggested model designs as values.
        :return: A dictionary with source dataset names as keys and the performance of their suggested model designs as values.
        """
        performances = []
        dataset = self.benchmark_mapping[dataset]
        nas_bench = light_read(dataset)
        for design in suggested_designs:
            arch = Arch(design['link'], design['ops'])
            h = arch.valid_hash()
            if h == "88888" or h==88888:
                perf = 0
            else:
                perf = nas_bench[h]['perf']
            performances.append(perf)

        return performances
    
    @staticmethod
    def extract_single_performance(dataset, design):
        """
        Extracts the performance of a single model design for a specified dataset.

        :param dataset: The name of the dataset for which the model design was suggested.
        :param design: The model design suggested for the dataset.
        :return: The performance of the model design on the dataset.
        """
        ops = design[dataset]['ops']
        link = design[dataset]['link']
        if dataset == "Planetoid:Cora":
            dataset = "cora"
        elif dataset == "Planetoid:CiteSeer":
            dataset = "citeseer"
        elif dataset == "Planetoid:PubMed":
            dataset = "pubmed"
        elif dataset == "Coauthor:CS":
            dataset = "cs"
        elif dataset == "Coauthor:Physics":
            dataset = "physics"
        elif dataset == "Amazon:Photo":
            dataset = "photo"
        elif dataset == "Amazon:Computers":
            dataset = "computers"
        elif dataset == "ogbn-arxiv":
            dataset = "arxiv"
        elif dataset == "ogbn-proteins":
            dataset = "proteins"
        else:
            raise ValueError(f"Dataset '{dataset}' not recognized.")
        nas_bench = light_read(dataset)
        arch = Arch(link, ops)

        h = arch.valid_hash()
        if h == "88888" or h==88888:
            perf = 0
            total_parameters = 0
            latency = 0
        else:
            perf = nas_bench[h]['perf']
            total_parameters = nas_bench[h]['para']
            latency = nas_bench[h]['latency']
        detailed_infos = {
                "ops": ops,
                "link": link,
                "perf": perf,
                #"detailed_log": detailed_log,
                "total_parameters": total_parameters,
                "latency": latency
            }
        return detailed_infos
    
    @staticmethod
    def extract_single_log(dataset, design):
        """
        Extracts the performance of a single model design for a specified dataset.

        :param dataset: The name of the dataset for which the model design was suggested.
        :param design: The model design suggested for the dataset.
        :return: The performance of the model design on the dataset.
        """
        ops = design[dataset]['ops']
        link = design[dataset]['link']
        if dataset == "Planetoid:Cora":
            dataset = "cora0.bench"
        elif dataset == "Planetoid:CiteSeer":
            dataset = "citeseer0.bench"
        elif dataset == "Planetoid:PubMed":
            dataset = "pubmed0.bench"
        elif dataset == "Coauthor:CS":
            dataset = "cs0.bench"
        elif dataset == "Coauthor:Physics":
            dataset = "physics0.bench"
        elif dataset == "Amazon:Photo":
            dataset = "photo0.bench"
        elif dataset == "Amazon:Computers":
            dataset = "computers0.bench"
        elif dataset == "ogbn-arxiv":
            dataset = "arxiv0.bench"
        elif dataset == "ogbn-proteins":
            dataset = "proteins0.bench"
        else:
            raise ValueError(f"Dataset '{dataset}' not recognized.")
        nas_bench = read('D:\\NBG\\' + dataset)
        hash = Arch(link, ops).valid_hash()
        if hash == "88888" or hash==88888:
            return []
        info = nas_bench[hash]
        detailed_log = []
        for epoch in range(len(info['dur'])):
            if epoch % 25 == 0:
                train_accuracy = info['dur'][epoch][0]
                val_accuracy = info['dur'][epoch][1]
                test_accuracy = info['dur'][epoch][2]
                train_loss = info['dur'][epoch][3]
                val_loss = info['dur'][epoch][4]
                test_loss = info['dur'][epoch][5]
                best_performance = info['dur'][epoch][6]
            
                detailed_log.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy,
                    'test_accuracy': test_accuracy,
                    'val_loss': val_loss,
                    'test_loss': test_loss,
                    'best_performance': best_performance
                })

        return detailed_log

    @staticmethod
    def print_best_candidate(candidate_pools):
        """
        Prints the best-performing model design and its performance scores for each source dataset.

        :param candidate_pools: A dictionary with structure for source_dataset, models_info in candidate_pools.items(),
                                and within each models_info, a selected_dataset, link_structure, and operations.
        """
        for source_dataset, models_info in candidate_pools.items():
            if source_dataset == "ogbn-arxiv":
                source_dataset = "arxiv"
            if source_dataset == "ogbn-proteins":
                source_dataset = "proteins"
            best_perf = float('-inf')
            best_model = None

            # Iterate through each candidate model for the source dataset
            nas_bench = light_read(source_dataset)
            for model_info in models_info:
                for model_design in model_info['top_models']:
                    link_structure, operations = model_design

                    # Fetch performance using nas_bench, light_read, and Arch
                    arch = Arch(link_structure, operations)
                    h = arch.valid_hash()
                    if h == "88888" or h==88888:
                        perf = 0
                    else:
                        perf = nas_bench[h]['perf']

                    # Update best model if current model outperforms
                    if perf > best_perf:
                        best_perf = perf
                        best_model = (link_structure, operations, perf)

            # Print the best-performing model and its performance
            if best_model:
                link_structure, operations, perf = best_model
                print(f"Best model for '{source_dataset}':")
                print(f"Architecture: {link_structure}, Operations: {operations}, Performance: {perf}")
            else:
                print(f"No model data available for '{source_dataset}'.")
        print("")


# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license
# This file will contain the logic for reading the dataset.

import os
import torch
import torch_geometric.datasets as pyg_datasets
from torch_geometric.transforms import Compose, LargestConnectedComponents, ToSparseTensor, RandomNodeSplit
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.model_selection import train_test_split

from .DatasetSplit import SemiSplit


class DatasetReader:
    def __init__(self, dataset_name, root_dir='datasets/', local_dataset=False, local_path=None):
        """
        Initialize the DatasetReader.
        :param dataset_name: Name of the dataset to be loaded from PyG.
        :param root_dir: Root directory where the dataset will be stored.
        :param local_dataset: Flag to indicate if the dataset is local.
        :param local_path: Path to the local dataset, used if local_dataset is True.
        """
        self.benchmark_dataset = dataset_name in ["Planetoid:Cora", "Planetoid:CiteSeer", "Planetoid:PubMed",
                                                  "Coauthor:CS", "Coauthor:Physics", "Amazon:Photo",
                                                  "Amazon:Computers", "ogbn-arxiv", "ogbn-proteins"]
        self.is_ogb = 'ogb' in dataset_name
        self.dataset_name, self.specific_dataset = self.parse_identifier(dataset_name)
        self.root_dir = root_dir
        self.local_dataset = local_dataset
        self.local_path = local_path

    def parse_identifier(self, dataset_name):
        """
        Splits the dataset identifier into dataset name and specific dataset using a delimiter.
        :param dataset_name: Name of the dataset chain to be loaded from PyG.
        :return: (dataset_name, specific_dataset) tuple. specific_dataset is None if not specified.
        """
        parts = dataset_name.split(':')
        dataset_name = parts[0]
        specific_dataset = parts[1] if len(parts) > 1 else None
        return dataset_name, specific_dataset

    def read_dataset(self):
        """
        Reads the dataset.
        :return: The dataset object.
        """
        if self.local_dataset:
            # Logic for reading local dataset
            # This part is minimal and would need to be expanded based on local dataset format.
            if self.local_path and os.path.exists(self.local_path):
                dataset = self._read_local_dataset()
            else:
                raise FileNotFoundError(f"Local dataset path '{self.local_path}' does not exist.")
        elif self.benchmark_dataset:
            # Reading benchmark dataset
            dataset = self._read_benchmark_dataset()
        elif self.is_ogb:
            # Reading from OGB
            dataset = self._read_obg_dataset()
        else:
            # Reading from PyTorch Geometric
            dataset = self._read_pyg_dataset()

        return dataset

    def _read_benchmark_dataset(self):
        """
        Reads a benchmark dataset using PyTorch Geometric or OGB.
        :return: The dataset object from PyG.
        """
        if self.is_ogb:
            if self.dataset_name == 'ogbn-arxiv':
                dataset = PygNodePropPredDataset(root=self.root_dir + self.dataset_name, name=self.dataset_name)
            elif self.dataset_name == 'ogbn-proteins':
                dataset = PygNodePropPredDataset(root=self.root_dir + self.dataset_name, name=self.dataset_name,
                                                 transform=ToSparseTensor(attr='edge_attr'))
            else:
                raise ValueError(f"Dataset '{self.dataset_name}' not found in OGB datasets.")
            n_class = dataset.num_classes
            dataset = self.add_mask(dataset)
        elif self.specific_dataset in ["CS", "Physics"]:
            #dataset = pyg_datasets.Coauthor(root=self.root_dir + self.specific_dataset, name=self.specific_dataset,
            #                                pre_transform=SemiSplit(num_train_per_class=20,
            #                                                        num_val_per_class=30,
            #                                                        lcc=False))
            #dataset = pyg_datasets.Coauthor(root=self.root_dir + self.specific_dataset, name=self.specific_dataset,
            #                                pre_transform=RandomNodeSplit(split="train_rest",
            #                                                              num_val=0.2,
            #                                                              num_test=0.2))
            dataset = pyg_datasets.Coauthor(root=self.root_dir + self.specific_dataset, name=self.specific_dataset)
            n_class = dataset.num_classes
            dataset = dataset[0]
            split = RandomNodeSplit(split="train_rest", num_val=0.2, num_test=0.2)
            dataset = split(dataset)
        elif self.specific_dataset in ["Photo", "Computers"]:
            #dataset = pyg_datasets.Amazon(root=self.root_dir + self.specific_dataset, name=self.specific_dataset,
            #                              pre_transform=Compose([LargestConnectedComponents(),
            #                                                     SemiSplit(num_train_per_class=20,
            #                                                               num_val_per_class=30,
            #                                                               lcc=False)]))
            dataset = pyg_datasets.Amazon(root=self.root_dir + self.specific_dataset, name=self.specific_dataset)
            #dataset = pyg_datasets.Amazon(root=self.root_dir + self.specific_dataset, name=self.specific_dataset)
            n_class = dataset.num_classes
            dataset = dataset[0]
            split = RandomNodeSplit(split="train_rest", num_val=0.2, num_test=0.2)
            dataset = split(dataset)
        elif self.specific_dataset in ["Cora", "CiteSeer", "PubMed"]:
            dataset = pyg_datasets.Planetoid(root=self.root_dir + self.specific_dataset, name=self.specific_dataset,
                                             split="full")
            n_class = dataset.num_classes
            dataset = dataset[0]
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not a benchmark dataset.")
        setattr(dataset, "num_classes", n_class)
        return dataset

    def _read_pyg_dataset(self):
        """
        Reads a dataset using PyTorch Geometric.
        :return: The dataset object from PyG.
        """
        try:
            dataset_class = getattr(pyg_datasets, self.dataset_name)
            if self.specific_dataset is not None:
                dataset = dataset_class(root=self.root_dir + self.specific_dataset, name=self.specific_dataset)
            else:
                dataset = dataset_class(root=self.root_dir + self.dataset_name)
            
            if self.dataset_name == "Flickr":
                n_class = 7
            elif self.dataset_name == "Actor":
                n_class = 5
            else:
                n_class = dataset.num_classes
            dataset = dataset[0]
        except AttributeError:
            raise ValueError(f"Dataset '{self.dataset_name}' not found in PyTorch Geometric datasets.")
        setattr(dataset, "num_classes", n_class)
        
        if self.dataset_name == "Actor":
            for mask_type in ['train_mask', 'val_mask', 'test_mask']:
                setattr(dataset, mask_type, getattr(dataset, mask_type)[:, 0])

        if self.specific_dataset == "DBLP":
            num_nodes = dataset.num_nodes
            all_indices = torch.arange(num_nodes)
            # Split indices into training and remaining nodes
            train_indices, remaining_indices = train_test_split(all_indices, test_size=0.4, random_state=42)
            # Split remaining indices into validation and testing nodes
            val_indices, test_indices = train_test_split(remaining_indices, test_size=0.5, random_state=42)
            # Create masks
            dataset.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            dataset.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            dataset.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            dataset.train_mask[train_indices] = True
            dataset.val_mask[val_indices] = True
            dataset.test_mask[test_indices] = True

        return dataset

    def _read_obg_dataset(self):
        """
        Reads a dataset using OGB.
        :return: The dataset object from PyG.
        """
        try:
            dataset = PygNodePropPredDataset(name=self.dataset_name, root=self.root_dir + self.dataset_name)
            n_class = dataset.num_classes
            dataset = self.add_mask(dataset)
        except AttributeError:
            raise ValueError(f"Dataset '{self.dataset_name}' not found in OGB datasets.")
        setattr(dataset, "num_classes", n_class)
        return dataset

    def _read_local_dataset(self):
        """
        Reads a local dataset.
        :return: The local dataset object.
        """
        # Implement local dataset reading logic here
        # This part would need specific implementation based on how your local data is structured.
        # For example, if it's a graph stored in a file, you would parse that file and create a PyG dataset.
        local_dataset = None
        return local_dataset

    def add_mask(self, dataset):
        """
        Utility function from NAS-Bench-Graph.
        :param dataset: The dataset object.
        :return: The dataset object.
        """
        split_idx = dataset.get_idx_split()
        dat = dataset[0]
        if self.dataset_name == 'ogbn-proteins':
            dat.x = dat.adj_t.mean(dim=1)
            dat.adj_t.set_value_(None)
            # dat.adj_t = dat.adj_t.set_diag()

            # Pre-compute GCN normalization.
            adj_t = dat.adj_t.set_diag()
            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

            setattr(dat, "edge_index", adj_t)
        elif self.dataset_name == 'ogbn-arxiv':
            setattr(dat, "edge_index", torch.cat((dat.edge_index, dat.edge_index[[1, 0]]), dim=1))
        setattr(dat, "y", dat.y.squeeze())
        setattr(dat, "train_mask", self.index_to_mask(split_idx["train"], size=dat.num_nodes))
        setattr(dat, "val_mask", self.index_to_mask(split_idx["valid"], size=dat.num_nodes))
        setattr(dat, "test_mask", self.index_to_mask(split_idx["test"], size=dat.num_nodes))
        return dat

    @staticmethod
    def index_to_mask(index, size):
        """
        Utility function from NAS-Bench-Graph.
        """
        mask = torch.zeros(size, dtype=torch.bool, device=index.device)
        mask[index] = 1
        return mask


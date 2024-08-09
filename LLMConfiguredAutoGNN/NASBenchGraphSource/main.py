# Source code from: Yijian Qin, Ziwei Zhang, Xin Wang, Zeyang Zhang, Wenwu Zhu,
# NAS-Bench-Graph: Benchmarking Graph Neural Architecture Search (NeurIPS 2022)

import torch
import pkg_resources
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.transforms import RandomNodeSplit, Compose, LargestConnectedComponents, ToSparseTensor
from ogb.nodeproppred import PygNodePropPredDataset
from hpo import random_search, all_archs, HP, Arch
from worker import Worker
import time
from split import SemiSplit
import sys

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def add_mask(dname, dataset):
    split_idx = dataset.get_idx_split() 
    dat = dataset[0]
    if dname == 'ogbn-proteins':
        dat.x = dat.adj_t.mean(dim=1)
        dat.adj_t.set_value_(None)
        #dat.adj_t = dat.adj_t.set_diag()

        # Pre-compute GCN normalization.
        adj_t = dat.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

        setattr(dat, "edge_index", adj_t)
    elif dname == 'ogbn-arxiv':
        setattr(dat, "edge_index", torch.cat((dat.edge_index, dat.edge_index[[1,0]]), dim = 1))
    setattr(dat, "y", dat.y.squeeze())
    setattr(dat, "train_mask", index_to_mask(split_idx["train"], size=dat.num_nodes))
    setattr(dat, "val_mask", index_to_mask(split_idx["valid"], size=dat.num_nodes))
    setattr(dat, "test_mask", index_to_mask(split_idx["test"], size=dat.num_nodes))
    return dat

def get_dataset(dname):
    if dname[:3] == 'ogb':
        if dname == 'ogbn-arxiv':
            dataset = PygNodePropPredDataset(root='~/data', name=dname)
        elif dname == 'ogbn-proteins':
            #dataset = PygNodePropPredDataset(root='~/data', name=dname, transform = ToSparseTensor(attr = 'edge_attr', remove_edge_index = False))
            dataset = PygNodePropPredDataset(root='~/data', name=dname, transform = ToSparseTensor(attr = 'edge_attr'))
        n_class = dataset.num_classes
        dataset = add_mask(dname, dataset)
    elif dname in ["CS", "Physics"]:
        dataset = Coauthor(root='~/data', name=dname, pre_transform = SemiSplit(num_train_per_class=20, num_val_per_class = 30, lcc = False))
        n_class = dataset.num_classes
        dataset = dataset[0]
    elif dname in ["Photo", "Computers"]:
        dataset = Amazon(root='~/data', name=dname, pre_transform = Compose([LargestConnectedComponents(), SemiSplit(num_train_per_class=20, num_val_per_class = 30, lcc = False)]))
        n_class = dataset.num_classes
        dataset = dataset[0]
    else: # PubMed
        dataset = Planetoid(root='~/data', name=dname)
        n_class = dataset.num_classes
        dataset = dataset[0]
    setattr(dataset, "num_classes", n_class)
    return dataset 

def main(dname):
    dataset = get_dataset(dname)

    i = int(sys.argv[1])
    best_hp = random_search(dataset, 60, 20, dname, i)
    print(best_hp)

main('ogbn-arxiv')
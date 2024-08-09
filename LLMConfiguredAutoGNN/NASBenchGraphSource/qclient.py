# Source code from: Yijian Qin, Ziwei Zhang, Xin Wang, Zeyang Zhang, Wenwu Zhu,
# NAS-Bench-Graph: Benchmarking Graph Neural Architecture Search (NeurIPS 2022)

from twisted.internet import reactor, protocol
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.transforms import RandomNodeSplit, Compose, LargestConnectedComponents, ToSparseTensor
import qprotocol
import argparse
import random
import time
import sys
import torch
from worker import Worker
import random
from hpo import HP
from split import SemiSplit

class QClient(protocol.Protocol):
    def connectionMade(self):
        print("Connected!")
        mes = qprotocol.construct_login_message(self.factory.name)
        self.transport.write(mes)

    def dataReceived(self, data):
        msg = qprotocol.parse_message(data)
        print(msg)
        if msg['type'] == "dismiss": 
            print("dismiss")
            self.transport.loseConnection()
        elif msg['type'] == 'task':
            task = msg['task']
            print(task)
            #hp['perf'] = random.random()
            acc = self.factory.get_acc(task)
            mes = qprotocol.construct_acc_message(self.factory.name, task, acc, self.factory.name)
            self.transport.write(mes)

class QCFactory(protocol.ClientFactory):
    def __init__(self, name, gpu_to_use, hp, data, dname):
        self.name = name
        self.protocol = QClient
        self.device = torch.device('cuda:{}'.format(gpu_to_use % 10))
        self.worker = Worker(hp, data, self.device, dname)

    def get_acc(self, task):
        try:
            acc = self.worker.run(task, True)
        except RuntimeError:
            return None
        return acc

    def clientConnectionFailed(self, connector, reason):
        print("connection failed", reason.getErrorMessage())
        reactor.stop()

    def clientConnectionLost(self, connector, reason):
        print("connection lost", reason.getErrorMessage())
        reactor.stop()

def start_fac(hostname, gpu, hp, dataset, dname):
    fac = QCFactory(hostname, gpu, hp, dataset, dname)
    reactor.connectTCP('localhost', 59486, fac)
    reactor.run()

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
    #print(torch.cat((dat.edge_index, dat.edge_index[[1,0]]), dim = 1).size())
    #print(dat.edge_index.size())
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

def main():
    hpdict = {"dropout": 0.0, "dim": 256, "num_cells": 1, "num_pre": 1, "num_pro": 1, "lr": 0.01, "wd": 5e-4, "optimizer": "Adam", "num_epochs": 500}

    hp = HP()
    for key in hpdict:
        setattr(hp, key, hpdict[key])
    dname = "ogbn-proteins"
    dataset = get_dataset(dname)

    i = int(sys.argv[1])
    hostname = 't' + str(i)
    start_fac(hostname, i, hp, dataset, dname)

if __name__ == '__main__':
    main()

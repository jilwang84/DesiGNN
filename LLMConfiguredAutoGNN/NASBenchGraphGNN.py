import torch
import torch.nn.functional as F
from torch_geometric.datasets import Actor
from torch_geometric.data import DataLoader
from torch_geometric.transforms import NormalizeFeatures
from .NASBenchGraphSource.net import Net
from .NASBenchGraphSource.hpo import HP, Arch
from .NASBenchGraphSource.worker import Worker


def run_gnn_experiment(dataset_name, data, dag, ops, hp=None):

    # Setup the architecture
    arch = Arch(dag, ops)

    # Setup the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Initialize the worker to train and evaluate the model
    if hp is None:
        if dataset_name == 'CitationFull:DBLP':
            hp = HP()
            hp.num_cells = 1
            hp.num_pre = 1
            hp.num_pro = 1
            hp.dim = 256
            hp.dropout = 0.5
            hp.optimizer = 'SGD'
            hp.lr = 0.1
            hp.wd = 0.0005
            hp.num_epochs = 300
        elif dataset_name == 'Flickr':
            hp = HP()
            hp.num_cells = 1
            hp.num_pre = 1
            hp.num_pro = 1
            hp.dim = 128
            hp.dropout = 0.5
            hp.optimizer = 'Adam'
            hp.lr = 0.001
            hp.wd = 0.0005
            hp.num_epochs = 300
        elif dataset_name == 'Actor':
            hp = HP()
            hp.num_cells = 1
            hp.num_pre = 1
            hp.num_pro = 1
            hp.dim = 128
            hp.dropout = 0.5
            hp.optimizer = 'Adam'
            hp.lr = 0.005
            hp.wd = 0.0005
            hp.num_epochs = 400
        elif dataset_name == 'Planetoid:Cora':
            hp = HP()
            hp.num_cells = 1
            hp.num_pre = 0
            hp.num_pro = 1
            hp.dim = 256
            hp.dropout = 0.7
            hp.optimizer = 'SGD'
            hp.lr = 0.1
            hp.wd = 0.0005
            hp.num_epochs = 400
        elif dataset_name == 'Planetoid:CiteSeer':
            hp = HP()
            hp.num_cells = 1
            hp.num_pre = 0
            hp.num_pro = 1
            hp.dim = 256
            hp.dropout = 0.7
            hp.optimizer = 'SGD'
            hp.lr = 0.2
            hp.wd = 0.0005
            hp.num_epochs = 400
        elif dataset_name == 'Planetoid:PubMed':
            hp = HP()
            hp.num_cells = 1
            hp.num_pre = 0
            hp.num_pro = 0
            hp.dim = 128
            hp.dropout = 0.3
            hp.optimizer = 'SGD'
            hp.lr = 0.2
            hp.wd = 0.0005
            hp.num_epochs = 500
        elif dataset_name == 'Coauthor:CS':
            hp = HP()
            hp.num_cells = 1
            hp.num_pre = 1
            hp.num_pro = 0
            hp.dim = 128
            hp.dropout = 0.6
            hp.optimizer = 'SGD'
            hp.lr = 0.5
            hp.wd = 0.0005
            hp.num_epochs = 400
        elif dataset_name == 'Coauthor:Physics':
            hp = HP()
            hp.num_cells = 1
            hp.num_pre = 1
            hp.num_pro = 1
            hp.dim = 256
            hp.dropout = 0.4
            hp.optimizer = 'SGD'
            hp.lr = 0.01
            hp.wd = 0
            hp.num_epochs = 200
        elif dataset_name == 'Amazon:Photo':
            hp = HP()
            hp.num_cells = 1
            hp.num_pre = 1
            hp.num_pro = 0
            hp.dim = 128
            hp.dropout = 0.7
            hp.optimizer = 'Adam'
            hp.lr = 0.0002
            hp.wd = 0.0005
            hp.num_epochs = 500
        elif dataset_name == 'Amazon:Computers':
            hp = HP()
            hp.num_cells = 1
            hp.num_pre = 1
            hp.num_pro = 1
            hp.dim = 64
            hp.dropout = 0.1
            hp.optimizer = 'Adam'
            hp.lr = 0.005
            hp.wd = 0.0005
            hp.num_epochs = 500
        elif dataset_name == 'ogbn-arxiv':
            hp = HP()
            hp.num_cells = 1
            hp.num_pre = 0
            hp.num_pro = 1
            hp.dim = 128
            hp.dropout = 0.2
            hp.optimizer = 'Adam'
            hp.lr = 0.002
            hp.wd = 0
            hp.num_epochs = 500
        elif dataset_name == 'ogbn-proteins':
            hp = HP()
            hp.num_cells = 1
            hp.num_pre = 1
            hp.num_pro = 1
            hp.dim = 256
            hp.dropout = 0
            hp.optimizer = 'Adam'
            hp.lr = 0.01
            hp.wd = 0.0005
            hp.num_epochs = 500
        else:
            raise NotImplementedError

    worker = Worker(hp, data, device, dataset_name)

    # Run training and evaluation
    detailed_infos = worker.run(arch, return_all_info=True)

    return detailed_infos


'''
# Load the Actor dataset with feature normalization
dataset = Actor(root='/tmp/Actor', transform=NormalizeFeatures())
data = dataset[0]

# Define a simple GNN architecture and hyperparameters
dag = [0, 1, 1, 2]
ops = ['gcn', 'gcn', 'gin', 'gcn']
arch = Arch(dag, ops)

# Define hyperparameters
hp = HP()
hp.dropout = 0.5
hp.dim = 64
hp.num_cells = 1
hp.num_pre = 1
hp.num_pro = 0
hp.lr = 0.01
hp.wd = 5e-4
hp.optimizer = 'Adam'
hp.num_epochs = 200

# Setup the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Initialize the worker to train and evaluate the model
worker = Worker(hp, data, device, 'Actor')

# Run training and evaluation
performance = worker.run(arch, return_all_info=True)

# Output the performance
print(f"Test accuracy: {performance['perf']}")
print(f"Total params: {performance['para']} MB")
print(f"Average latency per epoch: {performance['latency']} seconds")
'''


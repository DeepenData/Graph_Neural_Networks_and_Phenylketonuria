from tabnanny import check
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB 
from sklearn.neural_network import MLPClassifier
import matplotlib.gridspec as gridspec
import umap
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from cobra.io import load_json_model
import networkx as nx 
import pandas as pd
import warnings 
import torch
from torch_geometric.data import Data
from torch.optim.swa_utils import SWALR
from datetime import datetime
warnings.filterwarnings("ignore")
from cobra import Model, Reaction, Metabolite
import cobra
import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
import itertools
import copy
from cobra.util.array import create_stoichiometric_matrix
import networkx as nx
from networkx.algorithms import bipartite
import torch
#from custom_clases import GINo
import copy
import networkx as nx
import pickle
from torch.utils.data import RandomSampler
from torch.nn import Linear, LeakyReLU
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import numpy as np
from torch_geometric.nn.models import GIN
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import numpy as np
from torch_geometric.nn.models import GIN
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB 
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_classification
import umap
from sklearn import preprocessing
import gc
import tqdm
from torch_scatter import scatter
import pandas as pd
def plot_classsifiers(X, y):
    

    #clf1 = LogisticRegression()
    #clf2 = DecisionTreeClassifier()
    clf3 = RandomForestClassifier()
    clf4 = SVC(gamma='auto')
    clf5 = GaussianNB()
    clf6 = MLPClassifier()

    gs = gridspec.GridSpec(3, 2)
    fig = plt.figure(figsize=(14,7))
    labels = ['Random Forest', 'SVM', 'Naive Bayes', 'Neural Network']
    for clf, lab, grd in zip([clf3, clf4, clf5, clf6],
                            labels,
                            [(0,0), (0,1), (1,0), (1,1)]):
        clf.fit(X, y)
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
        plt.title(lab)
    plt.show()
    
def cobra_to_networkx(model, undirected: bool = True):
  S_matrix = create_stoichiometric_matrix(model)

  n_mets, n_rxns = S_matrix.shape
  assert (
      n_rxns > n_mets
  ), f"Usualmente tienes mas metabolitos ({n_mets}) que reacciones ({n_rxns})"

  # Constructo raro de matrices cero
  # fmt: off
  S_projected = np.vstack(
      (
          np.hstack( # 0_mets, S_matrix
              (np.zeros((n_mets, n_mets)), S_matrix)
          ),  
          np.hstack( # S_trns, 0_rxns
              (S_matrix.T * -1, np.zeros((n_rxns, n_rxns)))
          ),
      )
  )
  S_projected_directionality = np.sign(S_projected).astype(int)
  G = nx.from_numpy_matrix(
      S_projected_directionality, 
      create_using=nx.DiGraph, 
      parallel_edges=False
  )

  # Cosas sorprendentemente no cursed
  # fmt: off
  #formulas: list[str] = [recon2.reactions[i].reaction for i in range(recon2.reactions.__len__())]
  rxn_list: list[str] = [model.reactions[i].id       for i in range(model.reactions.__len__())]
  met_list: list[str] = [model.metabolites[i].id     for i in range(model.metabolites.__len__())]

  assert n_rxns == rxn_list.__len__()
  assert n_mets == met_list.__len__()

  node_list : list[str] = met_list + rxn_list 
  part_list : list[dict[str, int]] = [{"bipartite": 0} for _ in range(n_rxns)] + [{"bipartite": 1} for _ in range(n_mets)]

  nx.set_node_attributes(G, dict(enumerate(part_list)))
  G = nx.relabel_nodes(G, dict(enumerate(node_list)))
  assert G.is_directed() 


  largest_wcc = max(nx.connected_components(nx.Graph(G)), key=len)


  # Create a subgraph SG based on G
  SG = G.__class__()
  SG.add_nodes_from((n, G.nodes[n]) for n in largest_wcc)


  SG.add_edges_from((n, nbr, d)
      for n, nbrs in G.adj.items() if n in largest_wcc
      for nbr, d in nbrs.items() if nbr in largest_wcc)

  SG.graph.update(G.graph)

  assert G.nodes.__len__() > SG.nodes.__len__()
  assert G.edges.__len__() > SG.edges.__len__()
  assert SG.nodes.__len__() == len(largest_wcc)
  assert SG.is_directed() 
  assert nx.is_connected(nx.Graph(SG))

  grafo_nx                     =copy.deepcopy(SG)
  first_partition , second_partition,   = bipartite.sets(grafo_nx)
  
  
  if first_partition.__len__() > second_partition.__len__():
      rxn_partition = first_partition
      met_partition = second_partition
  else:
      rxn_partition = second_partition 
      met_partition = first_partition
      
      
  
  assert set(rxn_partition).issubset(set(rxn_list)) and set(met_partition).issubset(set(met_list))
  if undirected:
      
      
      return   copy.deepcopy(nx.Graph(grafo_nx))
  
  
  return None


def add_metabolite_concentration_features(grafo_nx_input,feature_data,feature_names ):
    grafo_nx = copy.deepcopy(grafo_nx_input)
    node_list = list(grafo_nx.nodes)
    feature_data.rename(
        columns=feature_names.set_index("Simbolo_traductor")["Recon3_ID"].to_dict(), 
        inplace=True
    )

    number_to_fill_in = 1e-5#int(1)

    feature_data[
        [item for item in node_list if item not in feature_data.columns.tolist()]
    ] = number_to_fill_in

    # Re-empaca todo este vector como un diccionario de atributos
    feature_dict = feature_data.to_dict(orient="list")  # desde Pandas
    feature_dict = {key: {"x": feature_dict[key]} for key in feature_dict}

    nx.set_node_attributes(grafo_nx, feature_dict)

    assert feature_data['phe_L_c'].tolist() == grafo_nx.nodes['phe_L_c']['x']
    assert np.unique(grafo_nx.nodes(data=True)['r0399']['x']) == number_to_fill_in
    assert len(grafo_nx.nodes(data=True)['r0399']['x']) == len(grafo_nx.nodes(data=True)['phe_L_c']['x'])
    return grafo_nx

def graph_data_check(nx_G, pyg_graph, target_node):

    producto_idx      = list(nx_G.nodes()).index(target_node)
    producto_features = nx_G.nodes()[target_node]['x']

    if np.allclose(pyg_graph.x[producto_idx,:].numpy()[0:producto_features.__len__()], np.array(producto_features), 1e-7, 1e-10):
        return True
    else:
        False
    


    
def make_PYG_graph_from_grafo_nx(nx_G_in):

    nx_G               = copy.deepcopy(nx_G_in)
    pyg_graph          = from_networkx(nx_G)
    
    x_attribute     = nx.get_node_attributes(nx_G, "x")
    longest_feature = max(len(v) for k,v in x_attribute.items())
    assert pyg_graph.x.shape[1]  == pyg_graph.num_features == longest_feature # == flux_samples.shape[0]
    assert graph_data_check(nx_G, pyg_graph, target_node = 'phe_L_c')
    assert graph_data_check(nx_G, pyg_graph, target_node = 'r0399')
    assert not pyg_graph.is_directed()
    assert not pyg_graph.has_isolated_nodes()  
    
    
    return pyg_graph

def add_flux_features(nx_G_in,flux_samples, feature_data):

    nx_G = copy.deepcopy(nx_G_in)

    feature_length = len(nx_G.nodes(data=True)['r0399']['x']) 
    flux_dict = flux_samples.sample(feature_length, replace=True).to_dict(orient = 'list')

    x_attribute     = nx.get_node_attributes(nx_G, "x")
    x_attribute.update(flux_dict)
    nx.set_node_attributes(nx_G, x_attribute, 'x')
    assert nx_G.nodes(data=True)['r0399']['x'] == flux_dict['r0399']
    assert len(nx_G.nodes(data=True)['r0399']['x']) == len(nx_G.nodes(data=True)['phe_L_c']['x'])
    assert nx_G.nodes(data=True)['phe_L_c']['x']  == feature_data['phe_L_c'].tolist() 
    return nx_G

from torch.utils.data import RandomSampler

def train_classifiers_with_dataloader(loader):   
    
    contatenated = torch.Tensor()
    labels_from_loader       = torch.Tensor()

    for data in loader:  # Iterate in batches over the training dataset.
            #data.to('cuda')
            #out = model(data.x, data.edge_index, data.batch)
            reshaped_batch     = data.x.reshape(data.y.shape[0], -1)
            contatenated       = torch.cat((contatenated,reshaped_batch),0)        
            labels_from_loader = torch.cat((labels_from_loader, data.y),0)
            

    #non_zero_cols =  np.sum(contatenated.numpy() , 0) !=0
    X_from_loader = contatenated#[:,non_zero_cols]

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(X_from_loader)
    plot_classsifiers(embedding, labels_from_loader.to(int).numpy())
    plt.show()
    
import torch.nn.functional as F
class GIN_classifier(torch.nn.Module):
    
    def __init__(
        self, 
        target_node_idx: int,
        n_nodes : int, 
        num_features : int, 
        out_channels: int = 8,
        dropout : float = 0.05, 
        hidden_dim : int = 8, 
        #heads : int = 5,
        LeakyReLU_slope : float = 0.01,

        num_layers: int = 4
    ):
        super(GIN_classifier, self).__init__() # TODO: why SUPER gato? 
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.num_features = num_features
        self.target_node_idx = target_node_idx
        self.out_channels = out_channels
        
        self.GIN_layers =  GIN(in_channels= num_features, hidden_channels= hidden_dim, num_layers= num_layers, 
                               out_channels= out_channels, dropout=dropout,  jk=None, act='LeakyReLU', act_first = False)      
        
        self.FC1        = Linear(in_features=out_channels, out_features=1, bias=True)
        self.FC2        = Linear(in_features=self.n_nodes, out_features=2, bias=True)      
        self.leakyrelu  = LeakyReLU(LeakyReLU_slope)#.to('cuda')
    def forward(self, x):
        data       = x.x 
        edge_index = x.edge_index
        batch_size = x.y.shape[0]

        x     = self.GIN_layers(data, edge_index)
        x     = x.reshape(batch_size, self.n_nodes, self.out_channels)
        x     = self.FC1(self.leakyrelu(x))
        x     = x.reshape(batch_size,  self.n_nodes)              
        x     = self.FC2(self.leakyrelu(x))
        x     = x.reshape(batch_size, 2)
        return   torch.nn.functional.log_softmax(x, dim=1).squeeze()
    
from sklearn.model_selection import train_test_split

    
    
    
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GIN
from torch_geometric.nn import global_mean_pool
import torch
class batch_loader():
    
    def __init__(self, graphs: list, batch_size: int  =1*32, num_samples:int = None, validation_percent:float = .3):
        self.graphs = graphs
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.validation_percent = validation_percent
        
        self.train_idxs, self.sub_idxs = train_test_split(range(len(self.graphs)), test_size = self.validation_percent)
        self.val_idxs,   self.test_idxs = train_test_split(self.sub_idxs, test_size = self.validation_percent)     

        
    def get_train_loader(self):
        train_subset = [self.graphs[i] for i in self.train_idxs]
        sampler      = RandomSampler(
        train_subset,
        #num_samples= self.num_samples, 
        replacement=False)   
        
        return  DataLoader(train_subset, batch_size= self.batch_size, sampler = sampler,  drop_last=True)
    
    def get_validation_loader(self):
        validation_subset = [self.graphs[i] for i in self.val_idxs]
        sampler      = RandomSampler(
        validation_subset,
        #num_samples= self.num_samples, 
        replacement=False)   
        
        return  DataLoader(validation_subset, batch_size= self.batch_size, sampler = sampler,  drop_last=True)
    
    def get_test_loader(self):
        validation_subset = [self.graphs[i] for i in self.test_idxs]
        sampler      = RandomSampler(
        validation_subset,
        #num_samples= self.num_samples, 
        replacement=False)   
        
        return  DataLoader(validation_subset, batch_size= self.batch_size, sampler = sampler,  drop_last=True)
    
    
class GIN_classifier_to_explain(torch.nn.Module):
    
    def __init__(
        self, 
        batch_size,
        #target_node_idx: int,
        n_nodes : int, 
        num_features : int, 
        out_channels: int = 8,
        dropout : float = 0.05, 
        hidden_dim : int = 8, 
        LeakyReLU_slope : float = 0.01,

        num_layers: int = 4
    ):
        super(GIN_classifier_to_explain, self).__init__() # TODO: why SUPER gato? 
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.num_features = num_features
        #self.target_node_idx = target_node_idx
        self.out_channels = out_channels
        self.batch_size   = batch_size
        
        self.GIN_layers =  GIN(in_channels= num_features, hidden_channels= hidden_dim, num_layers= num_layers, 
                               out_channels= out_channels, dropout=dropout,  jk=None, act='LeakyReLU', act_first = False)      
        
        self.FC1        = Linear(in_features=out_channels, out_features=1, bias=True)
        self.FC2        = Linear(in_features=self.n_nodes, out_features=2, bias=True)      
        self.leakyrelu  = LeakyReLU(LeakyReLU_slope)#.to('cuda')
    def forward(self, x, edge_index, batch):


        x     = self.GIN_layers(x, edge_index)
        x     = x.reshape(self.batch_size, self.n_nodes, self.out_channels)
        x     = self.FC1(self.leakyrelu(x))
        x     = x.reshape(self.batch_size,  self.n_nodes)              
        x     = self.FC2(self.leakyrelu(x))
        x     = x.reshape(self.batch_size, 2)
        return   torch.nn.functional.log_softmax(x, dim=1).squeeze()

class GIN_classifier_to_explain_v2(torch.nn.Module):
    
    def __init__(
        self, 
        n_classes: int,
        n_nodes : int, 
        num_features : int, 
        out_channels: int = 8,
        dropout : float = 0.05, 
        hidden_dim : int = 8, 
        LeakyReLU_slope : float = 0.01,
        num_layers: int = 2
    ):
        super(GIN_classifier_to_explain_v2, self).__init__() # TODO: why SUPER gato? 
        self.n_nodes = n_nodes
        self.n_classes = n_classes
        self.dropout = dropout
        self.num_features = num_features
        self.out_channels = out_channels
        self.GIN_layers =  GIN(in_channels= num_features, hidden_channels= hidden_dim, num_layers= num_layers, 
                               out_channels= out_channels, dropout=dropout,  jk=None, act='LeakyReLU', act_first = False)              
        self.FC1          = Linear(in_features=out_channels, out_features=1, bias=True)
        self.FC2          = Linear(in_features= self.n_nodes, out_features=self.n_classes, bias=True)
        #self.FC          = Linear(in_features=out_channels, out_features=1, bias=True)           
           
        self.leakyrelu  = LeakyReLU(LeakyReLU_slope)#.to('cuda')
    def forward(self, x, edge_index, batch):
        batch_size = batch.unique().__len__()

        x     = self.GIN_layers(x, edge_index)
        x     = x.reshape(batch_size, self.n_nodes, self.out_channels)
        x     = self.FC1(self.leakyrelu(x))
        x     = x.reshape(batch_size,  self.n_nodes)       
        x     = self.FC2(self.leakyrelu(x))    

        return torch.nn.functional.log_softmax(x, dim=1)
    
from torch_geometric.nn.models import GIN
from torch_geometric.nn.models import GAT
from torch_geometric.nn.models import GCN
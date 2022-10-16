import networkx as nx 
import pandas as pd
import warnings 
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

def cobra_a_networkx(model, undirected: bool = True):
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
  
  
  return 

def set_flux_attributes_in_nx(flux_samples, grafo_nx, target_node):

    flux_dict =  flux_samples.to_dict(orient = 'list')
    attrs = dict(zip(grafo_nx.nodes, itertools.repeat([])))
    attrs.update(flux_dict)
    nx.set_node_attributes(grafo_nx, attrs, "x")
    nx_G         = copy.deepcopy(grafo_nx)
    x_attribute     = nx.get_node_attributes(nx_G, "x")
    longest_feature = max(len(v) for k,v in x_attribute.items())
    equal_len_attrs = {k:(longest_feature*[0] if len(v) == 0 else v) for k,v in x_attribute.items()}
    nx.set_node_attributes(nx_G, equal_len_attrs, 'x')
    assert len(set(len(v) for k,v in nx.get_node_attributes(nx_G, "x").items())) == 1
    pyg_graph       = from_networkx(nx.Graph(nx_G))
    assert pyg_graph.x.shape[1]  == pyg_graph.num_features == longest_feature == flux_samples.shape[0]
    device='cpu'
    #target_node = 'r0399'
    producto_idx = list(nx_G.nodes()).index(target_node)
    producto_features = nx_G.nodes()[target_node]['x']


    assert grafo_nx.nodes[target_node]['x'] == flux_dict[target_node]
    assert np.allclose(pyg_graph.x[producto_idx,:].numpy()[0:producto_features.__len__()], np.array(producto_features), 1e-7, 1e-10)
    assert np.allclose(flux_samples[target_node].tolist(), pyg_graph.x[producto_idx,:].numpy()[0:producto_features.__len__()], 1e-7, 1e-10)
    
    return copy.deepcopy(nx_G), pyg_graph, producto_idx


def generate_graph_list(pyg_graph, producto_idx, device: str = 'cpu'):
    
    
    def sliding_window(ts, features,producto_idx, target_len = 1):
        X, Y = [], []
        for i in range(features + target_len, ts.shape[1] + 1):  #en este caso sería de 14+1 hasta el final de la serie 
                
                
            X.append(ts[:,i - (features + target_len):i - target_len]) #15 - 15, posicion 0 : 15-1 = 14 valores
            Y.append(ts[producto_idx, i - target_len:i]) #15-1 = 14 : 15  1 valor [14,15]
                
        return X, Y


    X, y =sliding_window(ts = pyg_graph.x, features = 1, producto_idx = producto_idx, target_len = 1)
    lista_de_grafos = []

    for graph_x, target in  copy.deepcopy(zip(X, y)):
        graph_x[producto_idx,:] = 0
            
        nuevo = copy.deepcopy(pyg_graph)
        nuevo.x = torch.tensor(graph_x).float()
        nuevo.y =  torch.tensor(target).float()
        lista_de_grafos.append(nuevo.to(device))
        
    return lista_de_grafos

import gc
import torch

def make_loader(graphs: list, batch_size: int  =250, num_samples: int = 10):

    sampler_train_set = RandomSampler(
        graphs,
        num_samples= num_samples, #params["training"]["sampler_num_samples"],  # Genera un muestreo del grafo
        replacement=True,  # con repeticion de muestras
    )
    return DataLoader(graphs, batch_size=batch_size, sampler = sampler_train_set,  drop_last=True)

class regresor_GIN(torch.nn.Module):
    
    def __init__(
        self, 
        target_node_idx: int,
        n_nodes : int, 
        num_features : int, 
        out_channels: int,
        dropout : float = 0.09, 
        hidden_dim : int = 5, 
        #heads : int = 5,
        LeakyReLU_slope : float = 0.01,

        num_layers: int = 3
    ):
        super(regresor_GIN, self).__init__() # TODO: why SUPER gato? 
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.num_features = num_features
        self.target_node_idx = target_node_idx
        self.out_channels = out_channels
        
        self.GIN_layers =  GIN(in_channels= num_features, hidden_channels= hidden_dim, num_layers= num_layers, out_channels= out_channels, dropout=dropout,  jk=None, act='LeakyReLU', act_first = True)#.to('cuda')        
        self.FC1        = Linear(in_features=out_channels, out_features=1, bias=True)#.to('cuda')
        #self.FC2        = Linear(in_features=n_nodes, out_features=1, bias=True)#.to('cuda')        #self.leakyrelu = LeakyReLU(LeakyReLU_slope).to('cuda')
        self.leakyrelu = LeakyReLU(LeakyReLU_slope)#.to('cuda')
    def forward(
        self, x, 
        edge_index, # 
        batch_size
    ):

     x     = self.GIN_layers(x, edge_index)
     x     = x.reshape(batch_size, self.n_nodes, self.out_channels)
     x     = self.FC1(self.leakyrelu(x))
        
     return  x[:,self.target_node_idx,:].squeeze()
 
 

def evaluate(modelo: regresor_GIN, loss_fun: torch.nn, loader: DataLoader,  total_eval_loss: float = 0, batch_size:int = 250):
    #total_eval_loss: float = 0
    #modelo.to('cuda')

    modelo.eval()
    for data in loader:
        with torch.cuda.device('cuda'):
            modelo.to('cuda')            
            data.to('cuda')
            prediction = modelo(data.x, data.edge_index, batch_size = batch_size)
            loss_eval       = loss_fun(prediction, data.y)
            total_eval_loss += loss_eval.item()
    return total_eval_loss    
from tqdm import tqdm

from torch.optim.lr_scheduler import *

def train(optimizer: torch.optim, loss_fun: torch.nn, modelo: regresor_GIN,loader: DataLoader, batch_size:int = 250):
    #modelo.to('cuda')
    check_seen_y = []
    modelo.train()
    
    for data in loader:
        
        with torch.cuda.device('cuda'):
            
             modelo.to('cuda')            
             data.to('cuda')
             prediction = modelo(data.x, data.edge_index, batch_size = batch_size)
             loss       = loss_fun(prediction, data.y)
             check_seen_y.extend(data.y.squeeze().tolist())
             loss.backward()  # Derive gradients.
             optimizer.step()  # Update parameters based on gradients.
             optimizer.zero_grad()  # Clear gradients.
        #print(loss)
    return check_seen_y

from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_and_evaluate(optimizer,loss_fun, modelo, train_loader,test_loader,save_state_dict: bool = False, epochs: int = 100, min_total_loss_val: float= 1e10, verbose: bool = False, batch_size:int = 250 ):
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience =7, threshold = 1e-3)
    for epoch in tqdm(range(epochs)):
        gc.collect()
        torch.cuda.empty_cache()          
        #total_eval_loss: float = 0
        check_seen_y = train(optimizer,loss_fun, modelo,train_loader, batch_size=batch_size)
        
        if epoch % 1 == 0:
            total_eval_loss = evaluate(modelo,loss_fun,test_loader,total_eval_loss=0 ,batch_size=batch_size)
            scheduler.step(total_eval_loss)       
            if total_eval_loss     < min_total_loss_val:
                min_total_loss_val = total_eval_loss
                best_eval_weights  = copy.deepcopy(modelo.state_dict())
                best_model =  copy.deepcopy(modelo)
                if verbose:
                    print(f"NEW best min_total_loss_val {min_total_loss_val} epoch: {epoch}")
                if save_state_dict:
                    torch.save(best_eval_weights, "results/state_dicts/state_dict_best_evaluated_model.pth")
                    torch.save(best_model, "results/state_dicts/best_evaluated_model.pt")
                    if verbose:
                        print(f"best_evaluated_model.pt and state_dict_best_evaluated_model.pth overwritten")
            
    return best_eval_weights, check_seen_y
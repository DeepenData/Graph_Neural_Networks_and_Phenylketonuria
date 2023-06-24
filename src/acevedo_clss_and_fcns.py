#from tabnanny import check
#from mlxtend.plotting import plot_decision_regions
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB 
# from sklearn.neural_network import MLPClassifier
# import matplotlib.gridspec as gridspec
import umap
#from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
#from cobra.io import load_json_model
import networkx as nx 
import pandas as pd
import warnings 
import torch
from torch_geometric.data import Data
#from torch.optim.swa_utils import SWALR
from datetime import datetime
warnings.filterwarnings("ignore")
# import cobra
# from cobra import Model, Reaction, Metabolite
#
import numpy as np
from tqdm import tqdm
#from scipy.integrate import solve_ivp
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
#import pickle
from torch.utils.data import RandomSampler
from torch.nn import Linear, LeakyReLU
import torch_geometric
from torch_geometric.loader import DataLoader
#from torch_geometric.nn import global_mean_pool
import numpy as np
from torch_geometric.nn.models import GIN
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
#from torch_geometric.nn import global_mean_pool
import numpy as np
from torch_geometric.nn.models import GIN
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB 
# from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_decision_regions
# import matplotlib.gridspec as gridspec
# from sklearn.datasets import make_classification
#import umap
#from sklearn import preprocessing
import gc
import tqdm
#from torch_scatter import scatter
import pandas as pd



def get_averages(loader):    
    """
    This function takes a data loader and returns the mean value of a specific attribute for two categories of data: control and PKU.

    Args:
        loader (DataLoader): The data loader containing the dataset to process.

    Returns:
        tuple: A tuple containing the mean value of the attribute for control data and PKU data respectively.
    """

    # Initialize lists to hold control and PKU data
    control_list = []
    pku_list = []

    # Iterate over a subset of the dataset contained in the loader
    for graph in loader.dataset[0:1000]:
        
        # Check the category of the data point, add its attribute to the corresponding list
        if graph.y.item() == 0:   # If the data point is control
            control_list.append(graph.x) # Add its attribute x to the control list
            
        elif graph.y.item() == 1: # If the data point is PKU
            pku_list.append(graph.x) # Add its attribute x to the PKU list
    
    # Compute and return the mean values of the attributes for control and PKU data
    # Concatenates the tensors along dimension 1 (columns if 2D), then computes the mean along dimension 1
    return torch.cat(control_list, dim=1).mean(axis = 1),  torch.cat(pku_list, dim=1).mean(axis = 1)



def check_Concen_plus_Fluxes(a_graph, mask_mets, mask_rxns):
    """
    This function performs several assertions on a graph's attributes, ensuring specific conditions are met. 

    Args:
        a_graph (Graph object): The graph containing attribute 'x' to be checked.
        mask_mets (array-like): A mask (boolean array) for metabolite data within the graph attributes.
        mask_rxns (array-like): A mask (boolean array) for reaction data within the graph attributes.

    Returns:
        None. If all assertions pass, function completes successfully. If any assertion fails, an AssertionError is raised.

    Note:
        Assertions are used to make sure the input data fits expected format or constraints. 
        If the assertion condition is not met, the program will stop and give an AssertionError.
    """
    # Get a reshape version of the attribute 'x' in a_graph according to the length of mask_mets
    reshaped_attribute = a_graph.x.reshape(len(mask_mets))

    # Check that there are more than 3 unique values in the filtered (masked) attributes which are greater than 1e-10
    assert np.unique(reshaped_attribute[mask_mets][reshaped_attribute[mask_mets] > 1e-10]).__len__() > 3

    # Check that there are at most 2 unique values in the filtered (masked) attributes which are less than or equal to 1e-10
    assert np.unique(reshaped_attribute[mask_mets][np.invert(reshaped_attribute[mask_mets] > 1e-10)]).__len__() <=2

    # Check that the sum of unique values in the filtered (masked) attributes which are less than or equal to 1e-10 is less than 1e-9
    assert np.unique(reshaped_attribute[mask_mets][np.invert(reshaped_attribute[mask_mets] > 1e-10)]).sum() < 1e-9

    # Check that there are more than 3 unique values in the filtered (masked) attributes which are greater than 1e-10
    assert np.unique(reshaped_attribute[mask_mets][mask_rxns][reshaped_attribute[mask_mets][mask_rxns] > 1e-10]).__len__() > 3


def get_a_graph_from_loader(loader):
    """
    Function to extract a graph from a loader object.

    Args:
        loader: Loader object that generates batches of data.

    Returns:
        Batch: First batch generated by the loader object.
    """
    # Get the first batch from the loader
    a_batch = next(iter(loader.get_train_loader()))

    # Return the first graph in the batch
    return a_batch[0]


def make_graphs_list(pyg_graph_in, target_list, mask_target=False, mask_number=1e-10):
    """
    Function to create a list of PyTorch geometric data objects (graphs) 
    from the input graph, with an option to mask certain target nodes.

    Args:
        pyg_graph_in: Input PyTorch Geometric Data object.
        target_list: List of target nodes to mask.
        mask_target: If True, the target nodes will be masked.
        mask_number: The number with which to replace the target nodes 
                     if mask_target is True. 

    Returns:
        graphs_list: List of PyTorch Geometric Data objects.
    """
    import copy
    from torch_geometric.data import Data
    
    # Make a deep copy of the input graph to prevent modifications to the original
    pyg_graph = copy.deepcopy(pyg_graph_in)
    graphs_list = []

    # Iterate over the feature dimension of the node feature matrix 
    for i in range(pyg_graph.x.shape[1]):
        # Create a new PyTorch Geometric Data object for each feature
        new_pyg_data = Data(x=pyg_graph.x[:, i].reshape(pyg_graph.num_nodes, 1), 
                            y=pyg_graph.y[i], 
                            edge_index=pyg_graph.edge_index)
        new_pyg_data.num_classes = 2

        # If mask_target is True, mask the target nodes
        if mask_target:
            for n in target_list:
                new_pyg_data.x[n, :] = mask_number
                
        # Append the new graph to the list
        graphs_list.append(new_pyg_data)
        
    return graphs_list


def get_sample_subset(full_samples, concentration_data, label): 
    """
    Function to get a subset of samples from the full samples according to a given label.

    Args:
        full_samples: DataFrame of full samples.
        concentration_data: DataFrame of concentration data with a 'label' column.
        label: The label to filter the samples.

    Returns:
        sample_subset: The subset of samples corresponding to the given label.
    """
    # Count the frequency of each label in the concentration_data
    s = concentration_data.label.value_counts()
    
    # Sample from the full_samples according to the frequency of the given label
    sample_subset = full_samples.sample(s.loc[label], replace=True).reset_index(drop=True)
    
    # Assign the given label to the sample subset
    sample_subset["label"] = label

    return sample_subset


    
import copy
import networkx as nx

def get_largest_cc(G):
    """
    Function to extract the largest connected component (CC) from a graph.

    Args:
        G: A NetworkX graph object.

    Returns:
        SG: A subgraph of G that represents the largest connected component.
    """

    # Find the largest connected component in the graph G. 
    # The function nx.connected_components returns a list of sets, where each set represents a connected component.
    # The function max with key=len is used to find the largest connected component.
    largest_wcc = max(nx.connected_components(nx.Graph(G)), key=len)

    # Create a new graph object of the same type as G. This graph will eventually be the subgraph of G that represents the largest connected component.
    SG = G.__class__()

    # Add the nodes of the largest connected component to SG, preserving their attributes.
    SG.add_nodes_from((n, G.nodes[n]) for n in largest_wcc)

    # Add the edges of the largest connected component to SG, preserving their attributes.
    SG.add_edges_from((n, nbr, d)
        for n, nbrs in G.adj.items() if n in largest_wcc
        for nbr, d in nbrs.items() if nbr in largest_wcc)

    # Copy the graph attributes of G to SG.
    SG.graph.update(G.graph)

    # Check the correctness of the function by asserting some conditions.
    # G should have more or equal number of nodes than SG.
    assert G.nodes.__len__() >= SG.nodes.__len__()
    # G should have more or equal number of edges than SG.
    assert G.edges.__len__() >= SG.edges.__len__()
    # The number of nodes in SG should be equal to the size of the largest connected component.
    assert SG.nodes.__len__() == len(largest_wcc)
    # SG should not be a directed graph, because the function nx.connected_components only works correctly for undirected graphs.
    assert not SG.is_directed()
    # SG should be a connected graph.
    assert nx.is_connected(nx.Graph(SG))

    # Return a deep copy of SG to prevent modifications to the original object.
    return copy.deepcopy(SG)

    
def cobra_to_networkx(model, undirected: bool = True):
    """
    Convert a COBRA model to a NetworkX graph representation.

    Args:
        model: A COBRA model object.
        undirected: Boolean specifying if the NetworkX graph should be undirected.

    Returns:
        A NetworkX graph representing the COBRA model or None if undirected is False and graph couldn't be converted.
    """

    # Create the stoichiometric matrix for the COBRA model
    # The stoichiometric matrix is a 2D matrix where the rows correspond to metabolites 
    # and the columns correspond to reactions in the model.
    S_matrix = create_stoichiometric_matrix(model)

    n_mets, n_rxns = S_matrix.shape

    # Ensuring that there are more reactions than metabolites. In most metabolic models, 
    # this is the case because a reaction can involve multiple metabolites.
    assert (n_rxns > n_mets), f"Usually you have more metabolites ({n_mets}) than reactions ({n_rxns})"

    # Projecting the stoichiometric matrix into a new matrix that distinguishes 
    # between metabolites and reactions, with zeros in the off-diagonal blocks.
    S_projected = np.vstack(
        (
            np.hstack((np.zeros((n_mets, n_mets)), S_matrix)),  
            np.hstack((S_matrix.T * -1, np.zeros((n_rxns, n_rxns))))
        )
    )
    
    # Indicate the directionality of reactions based on the sign of the stoichiometric 
    # coefficients. A positive sign indicates products and a negative sign indicates reactants.
    S_projected_directionality = np.sign(S_projected).astype(int)
    
    # Convert the projected stoichiometric matrix into a directed graph.
    G = nx.from_numpy_matrix(
        S_projected_directionality, 
        create_using=nx.DiGraph, 
        parallel_edges=False
    )

    # Extract the reaction IDs and metabolite IDs from the model.
    rxn_list: list[str] = [model.reactions[i].id for i in range(model.reactions.__len__())]
    met_list: list[str] = [model.metabolites[i].id for i in range(model.metabolites.__len__())]

    # Ensuring that the number of reactions and metabolites in the list is same as in the model.
    assert n_rxns == rxn_list.__len__()
    assert n_mets == met_list.__len__()

    # Combine metabolites and reactions into a single list of nodes for the graph.
    node_list : list[str] = met_list + rxn_list 

    # Define bipartite attribute for the nodes in the graph.
    part_list : list[dict[str, int]] = [{"bipartite": 0} for _ in range(n_rxns)] + [{"bipartite": 1} for _ in range(n_mets)]

    # Set the bipartite attribute for the nodes in the graph.
    nx.set_node_attributes(G, dict(enumerate(part_list)))

    # Rename the nodes of the graph using the node_list.
    G = nx.relabel_nodes(G, dict(enumerate(node_list)))
    assert G.is_directed() 

    # Find the largest connected component of the graph.
    largest_wcc = max(nx.connected_components(nx.Graph(G)), key=len)

    # Create a subgraph SG of G based only on the nodes in the largest connected component.
    SG = G.__class__()
    SG.add_nodes_from((n, G.nodes[n]) for n in largest_wcc)
    SG.add_edges_from((n, nbr, d)
        for n, nbrs in G.adj.items() if n in largest_wcc
        for nbr, d in nbrs.items() if nbr in largest_wcc)
    SG.graph.update(G.graph)

    # The number of nodes and edges in the original graph G should be greater than or equal 
    # to the number of nodes and edges in the subgraph SG.
    assert G.nodes.__len__() > SG.nodes.__len__()
    assert G.edges.__len__() > SG.edges.__len__()
    assert SG.nodes.__len__() == len(largest_wcc)

    # The subgraph SG is a directed graph and it is a connected graph when 
    # considered as an undirected graph.
    assert SG.is_directed() 
    assert nx.is_connected(nx.Graph(SG))

    # Copy the subgraph SG and distinguish between reaction and metabolite nodes.
    grafo_nx = copy.deepcopy(SG)
    first_partition , second_partition = bipartite.sets(grafo_nx)
    
    # Ensure that the first_partition has more elements than the second_partition.
    if first_partition.__len__() > second_partition.__len__():
        rxn_partition = first_partition
        met_partition = second_partition
    else:
        rxn_partition = second_partition 
        met_partition = first_partition

    # Ensure that the reaction partition only contains reaction IDs and 
    # the metabolite partition only contains metabolite IDs.
    assert set(rxn_partition).issubset(set(rxn_list)) and set(met_partition).issubset(set(met_list))
    
    # Assign node attributes to the graph nodes, with 0 for metabolites and 1 for reactions.
    feature_dict = dict(zip(met_partition, itertools.repeat(0)))
    feature_dict.update(dict(zip(rxn_partition, itertools.repeat(1))))
    features_to_set = {key: {"bipartite": feature_dict[key]} for key in feature_dict}
    nx.set_node_attributes(grafo_nx, features_to_set)

    # If the undirected flag is True, convert the directed graph to an undirected graph.
    if undirected:
        return copy.deepcopy(nx.Graph(grafo_nx))    
    return None
import copy
import numpy as np
from torch_geometric.utils import from_networkx

def add_metabolite_concentration_features(grafo_nx_input, feature_data, feature_names):
    """
    Add metabolite concentration features to nodes in the graph.

    Args:
        grafo_nx_input: The input graph.
        feature_data: A DataFrame containing the concentration data for metabolites.
        feature_names: The names of the features to be added.

    Returns:
        The graph with added features.
    """
    
    # Create a deep copy of the input graph to prevent altering the original graph.
    grafo_nx = copy.deepcopy(grafo_nx_input)
    node_list = list(grafo_nx.nodes)
    
    # Rename the columns in the feature_data DataFrame to match the node names in the graph.
    feature_data.rename(columns=feature_names.set_index("Simbolo_traductor")["Recon3_ID"].to_dict(), inplace=True)

    # Set a default value for the metabolite concentrations. 
    number_to_fill_in = 1e-5 # The default concentration value.

    # For nodes in the graph that do not have corresponding columns in the feature_data DataFrame, 
    # fill in with the default concentration value.
    feature_data[[item for item in node_list if item not in feature_data.columns.tolist()]] = number_to_fill_in

    # Convert the DataFrame into a dictionary of attributes.
    feature_dict = feature_data.to_dict(orient="list")  # Convert from Pandas DataFrame to a dictionary.

    # Reformat the dictionary so that it can be used to set node attributes.
    feature_dict = {key: {"x": feature_dict[key]} for key in feature_dict}

    # Add the attributes to the nodes in the graph.
    nx.set_node_attributes(grafo_nx, feature_dict)

    # Assert statements for correctness checks
    assert feature_data['phe_L_c'].tolist() == grafo_nx.nodes['phe_L_c']['x']
    assert np.unique(grafo_nx.nodes(data=True)['r0399']['x']) == number_to_fill_in
    assert len(grafo_nx.nodes(data=True)['r0399']['x']) == len(grafo_nx.nodes(data=True)['phe_L_c']['x'])
    
    return grafo_nx


def graph_data_check(nx_G, pyg_graph, target_node):
    """
    Check if the data of a specific node is the same in both the NetworkX and PyG graphs.

    Args:
        nx_G: The NetworkX graph.
        pyg_graph: The PyG graph.
        target_node: The node to check the data for.

    Returns:
        True if the data is the same, False otherwise.
    """
    
    producto_idx = list(nx_G.nodes()).index(target_node)
    producto_features = nx_G.nodes()[target_node]['x']

    if np.allclose(pyg_graph.x[producto_idx,:].numpy()[0:producto_features.__len__()], 
                   np.array(producto_features), 1e-7, 1e-10):
        return True
    else:
        return False


def make_PYG_graph_from_grafo_nx(nx_G_in):
    """
    Convert a NetworkX graph into a PyG graph.

    Args:
        nx_G_in: The input NetworkX graph.

    Returns:
        The converted PyG graph.
    """
    
    # Create a deep copy of the input graph to prevent altering the original graph.
    nx_G = copy.deepcopy(nx_G_in)
    
    # Convert the NetworkX graph into a PyG graph.
    pyg_graph = from_networkx(nx_G)
    
    # Assert statements for correctness checks.
    longest_feature = max(len(v) for v in nx.get_node_attributes(nx_G, "x").values())
    assert pyg_graph.x.shape[1] == pyg_graph.num_features == longest_feature 
    assert graph_data_check(nx_G, pyg_graph, target_node = 'phe_L_c')
    assert graph_data_check(nx_G, pyg_graph, target_node = 'r0399')
    assert not pyg_graph.is_directed()
    assert not pyg_graph.has_isolated_nodes()  
    
    return pyg_graph


def add_flux_features(nx_G_in, flux_samples, feature_data):
    """
    Add flux features to nodes in the graph.

    Args:
        nx_G_in: The input graph.
        flux_samples: A DataFrame containing the flux data for nodes.
        feature_data: A DataFrame containing the concentration data for nodes.

    Returns:
        The graph with added flux features.
    """
    
    # Create a deep copy of the input graph to prevent altering the original graph.
    nx_G = copy.deepcopy(nx_G_in)

    # Determine the length of features for a node in the graph.
    feature_length = len(nx_G.nodes(data=True)['r0399']['x'])

    # Sample from the flux_samples DataFrame to get a number of samples equal to feature_length.
    flux_dict = flux_samples.sample(feature_length, replace=True).to_dict(orient='list')

    # Get current node attributes.
    x_attribute = nx.get_node_attributes(nx_G, "x")

    # Update the node attributes with the flux_dict data.
    x_attribute.update(flux_dict)

    # Set the node attributes in the graph.
    nx.set_node_attributes(nx_G, x_attribute, 'x')

    # Check that the flux features have been added correctly.
    assert nx_G.nodes(data=True)['r0399']['x'] == flux_dict['r0399']
    assert len(nx_G.nodes(data=True)['r0399']['x']) == len(nx_G.nodes(data=True)['phe_L_c']['x'])
    assert nx_G.nodes(data=True)['phe_L_c']['x']  == feature_data['phe_L_c'].tolist() 

    return nx_G



# def train_classifiers_with_dataloader(loader):   
    
#     contatenated = torch.Tensor()
#     labels_from_loader       = torch.Tensor()

#     for data in loader:  # Iterate in batches over the training dataset.
#             #data.to('cuda')
#             #out = model(data.x, data.edge_index, data.batch)
#             reshaped_batch     = data.x.reshape(data.y.shape[0], -1)
#             contatenated       = torch.cat((contatenated,reshaped_batch),0)        
#             labels_from_loader = torch.cat((labels_from_loader, data.y),0)
            

#     #non_zero_cols =  np.sum(contatenated.numpy() , 0) !=0
#     X_from_loader = contatenated#[:,non_zero_cols]

#     reducer = umap.UMAP()
#     embedding = reducer.fit_transform(X_from_loader)
#     plot_classsifiers(embedding, labels_from_loader.to(int).numpy())
#     plt.show()
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.functional import LeakyReLU
from torch.nn import Linear
from sklearn.model_selection import train_test_split
from torch_geometric.nn.models import GIN, GAT, GCN

class batch_loader():
    def __init__(self, graphs: list, batch_size: int = 32, num_samples: int = None, validation_percent: float = .3):
        """Initialize the batch loader with graphs, batch size, num_samples, and validation percentage."""
        self.graphs = graphs
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.validation_percent = validation_percent

        # Split data into train, validation, and test indices
        self.train_idxs, self.sub_idxs = train_test_split(range(len(self.graphs)), test_size=self.validation_percent)
        self.val_idxs, self.test_idxs = train_test_split(self.sub_idxs, test_size=self.validation_percent)

    def _get_dataloader(self, idxs):
        """Helper function to create a data loader."""
        subset = [self.graphs[i] for i in idxs]
        sampler = RandomSampler(subset, replacement=False)
        return DataLoader(subset, batch_size=self.batch_size, sampler=sampler, drop_last=True)

    def get_train_loader(self):
        """Return a data loader for training data."""
        return self._get_dataloader(self.train_idxs)

    def get_validation_loader(self):
        """Return a data loader for validation data."""
        return self._get_dataloader(self.val_idxs)

    def get_test_loader(self):
        """Return a data loader for test data."""
        return self._get_dataloader(self.test_idxs)


class my_GNN(torch.nn.Module):
    def __init__(self, model: str, n_classes: int, n_nodes: int, num_features: int, out_channels: int = 8,
                 dropout: float = 0.05, hidden_dim: int = 8, LeakyReLU_slope: float = 0.01, num_layers: int = 1):
        """Initialize the GNN model."""
        super(my_GNN, self).__init__()
        self.n_nodes = n_nodes
        self.n_classes = n_classes
        self.dropout = dropout
        self.num_features = num_features
        self.out_channels = out_channels
        self.model = model

        # Create the GNN layers based on the specified model type
        if self.model == "GCN":
            self.GNN_layers = GCN(in_channels=num_features, hidden_channels=hidden_dim, num_layers=num_layers,
                                  out_channels=out_channels, dropout=dropout, jk=None, act='LeakyReLU', act_first=False)
        elif self.model == "GAT":
            self.GNN_layers = GAT(in_channels=num_features, hidden_channels=hidden_dim, num_layers=num_layers,
                                  out_channels=out_channels, dropout=dropout, jk=None, act='LeakyReLU', act_first=False)
        elif self.model == "GIN":
            self.GNN_layers = GIN(in_channels=num_features, hidden_channels=hidden_dim, num_layers=num_layers,
                                  out_channels=out_channels, dropout=dropout, jk=None, act='LeakyReLU', act_first=False)

        # Create linear layers for output
        self.FC1 = Linear(in_features=out_channels, out_features=1, bias=True)
        self.FC2 = Linear(in_features=self.n_nodes, out_features=self.n_classes, bias=True)

        # Create activation function
        self.leakyrelu = LeakyReLU(LeakyReLU_slope)

    def forward(self, x, edge_index, batch):
        """Define the forward pass of the model."""
        batch_size = batch.unique().__len__()

        # Pass data through GNN layers
        x = self.GNN_layers(x, edge_index)

        # Reshape and pass through first linear layer
        x = x.reshape(batch_size, self.n_nodes, self.out_channels)
        x = self.FC1(self.leakyrelu(x))

        # Reshape and pass through second linear layer
        x = x.reshape(batch_size, self.n_nodes)
        x = self.FC2(self.leakyrelu(x))

        # Apply log softmax activation to the output
        return torch.nn.functional.log_softmax(x, dim=1)

def train_one_epoch(model,
                    optimizer, 
                    train_loader: torch_geometric.loader.dataloader.DataLoader,
                    loss_fun: torch.nn.modules.loss,
                    device:str='cpu' ):

    """Train the model for one epoch.
    
    Args:
    model: The model to be trained.
    optimizer: The optimizer to use for training.
    train_loader: The DataLoader that provides the training data.
    loss_fun: The loss function to use for training.
    device: The device on which to run the computations.

    Returns:
    The accuracy of the model on the training data for this epoch.
    """

    correct = 0
    for i, data in enumerate(train_loader):
        # Make sure data is not already on GPU
        assert not data.is_cuda

        # Move data to GPU if necessary
        if device in ['cuda:0', 'cuda']:
            data.to(device, non_blocking=True)
            # Check that data was successfully moved to GPU
            assert data.is_cuda

        optimizer.zero_grad(set_to_none=True)  # Reset gradients
        predictions = model(data.x, data.edge_index,  data.batch)  # Make predictions
        loss = loss_fun(predictions, data.y)  # Calculate loss
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        pred = predictions.argmax(dim=1)  # Get class with highest probability
        correct += int((pred == data.y).sum())  # Compare with ground-truth labels

    return correct / len(train_loader.dataset)  # Return accuracy

def validate(model, loader: DataLoader, device: str = 'cpu'):

    """Validate the model.
    
    Args:
    model: The model to validate.
    loader: The DataLoader that provides the validation data.
    device: The device on which to run the computations.

    Returns:
    The accuracy of the model on the validation data.
    """

    model.eval()  # Set model to evaluation mode
    correct = 0
    for i, val_data in enumerate(loader):
        # Make sure data is not already on GPU
        assert not val_data.is_cuda

        # Move data to GPU if necessary
        if device in ['cuda:0', 'cuda']:
            val_data.to(device, non_blocking=True)
            # Check that data was successfully moved to GPU
            assert val_data.is_cuda

        val_predictions = model(val_data.x, val_data.edge_index,  val_data.batch)  # Make predictions
        pred = val_predictions.argmax(dim=1)  # Get class with highest probability
        correct += int((pred == val_data.y).sum())  # Compare with ground-truth labels

    return correct / len(loader.dataset)  # Return accuracy


import os

def train_and_validate(gnn_type, mask, flux, concentration, loader_path , EPOCHS, save, verbose, saving_folder, device:str="cuda"):

    """Train and validate the model.
    
    Args:
    gnn_type: The type of GNN to use.
    mask, flux, concentration: Flags to specify what data to use.
    loader_path: Path to the DataLoader.
    EPOCHS: The number of epochs to train for.
    save: Whether to save the model.
    verbose: Whether to print detailed output.
    saving_folder: The folder in which to save the model.
    device: The device on which to run the computations.

    Returns:
    The path of the saved model, the training accuracies for each epoch, and the validation accuracies for each epoch.
    """

    # Check that loader_path corresponds to the right data
    if mask and flux and not concentration:
        assert "MASKED_loader_only_Fluxes.pt" == os.path.basename(loader_path)
    elif mask and not flux and concentration:
        assert "MASKED_loader_only_Concen.pt" == os.path.basename(loader_path)
    elif mask and  flux and concentration:
        assert "MASKED_loader_Concen_plus_Fluxes.pt" == os.path.basename(loader_path)  

    # Load data
    loader = torch.load(loader_path)

    a_batch = next(iter(loader.get_train_loader()))
    a_graph = a_batch[0]

    # Initialize model
    model = my_GNN(model=gnn_type,
                   n_nodes=a_graph.num_nodes,
                   num_features=a_graph.num_node_features,
                   n_classes=a_graph.num_classes,
                   hidden_dim=8,
                   num_layers=1).to(device, non_blocking=True)

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = torch.nn.NLLLoss()

    # Free up memory
    gc.collect()
    torch.cuda.empty_cache() 

    # Keep track of accuracies and best model
    all_train_accuracy_ = []
    all_validation_accuracy_ = []
    best_validation_accuracy = 1e-10
    best_val_model = None

    for epoch in tqdm.tqdm(range(EPOCHS)):

        # Train and validate for one epoch
        train_accuracy = train_one_epoch(model, optimizer, loader.get_train_loader(), loss_fun=loss_function, device=device)
        validation_accuracy = validate(model, loader.get_validation_loader(), device=device)
        
        # Keep track of accuracies
        all_train_accuracy_.append(train_accuracy)
        all_validation_accuracy_.append(validation_accuracy)

        # Check if this is the best model so far
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_val_model = copy.deepcopy(model)  # Update best model

        # Print and save, if necessary
        if verbose:
            print(f'Epoch: {epoch:03d}, train_accuracy: {train_accuracy:.4f}, best_validation_accuracy: {best_validation_accuracy:.4f}')
            if save and validation_accuracy == best_validation_accuracy:
                model_path = os.path.join(saving_folder, f'Model_{epoch}_best_ValAcc_{best_validation_accuracy:.4f}.pt')
                torch.save(best_val_model, model_path)
                print(f"Model saved as {model_path}")

    return model_path, all_train_accuracy_, all_validation_accuracy_



from sklearn.metrics import roc_curve, auc



def get_ROC_parameters(model, test_loader, device:str="cuda"):

    tprs            = []
    aucs = []
    base_fpr = np.linspace(0, 1, 101)
    for i, test_data in enumerate(test_loader):
            
        assert not test_data.is_cuda
        if (device == 'cuda:0') | (device == 'cuda'):
            test_data.to(device, non_blocking=True) 
            assert test_data.is_cuda                          

        test_predictions = model(test_data.x, test_data.edge_index,  test_data.batch)# Make predictions for this batch
        pred            = test_predictions.argmax(dim=1)
        y_batch         = test_data.y
        
        y_pred_tag = pred.squeeze().cpu().int().tolist()
        y_true     = y_batch.squeeze().cpu().int().tolist()
        fpr, tpr, _ = roc_curve(y_true, y_pred_tag)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)


    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    mean_auc = auc(base_fpr, mean_tprs)
    std_auc = np.std(aucs)


    tprs_upper = np.minimum(mean_tprs + tprs.std(axis=0), 1)
    tprs_lower = mean_tprs - tprs.std(axis=0)
    
    return base_fpr, mean_tprs, tprs_lower, tprs_upper, mean_auc, std_auc

def put_ROC_in_subplot(base_fpr, mean_tprs, tprs_lower,
                   tprs_upper, mean_auc, std_auc, AX, xlabel:str='', letter:str=''):
    
    AX.plot(base_fpr, mean_tprs, 'b', alpha = 0.8, label=r'Test set ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),)
    AX.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'blue', alpha = 0.2)
    AX.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'r', label = 'Random', alpha= 0.8)

    #ax1.plot(fpr, tpr, lw=1, alpha=0.6, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc), c = colors[i])

    AX.legend(loc="lower right", fontsize=7.5)
    AX.set_ylabel('True Positive Rate')
    AX.set_xlabel(xlabel)
    AX.set_title(letter, fontsize = 12,  fontweight ="bold", loc='left')
    
    
def put_Learning_curve(all_train_accuracy, all_validation_accuracy, AX, letter):
    AX.plot(all_train_accuracy,  label = "Train set", linestyle="--")
    AX.plot(all_validation_accuracy,  label = "Validation set", linestyle="-")
    AX.legend(loc="lower right", fontsize=11)
    AX.set_ylabel('Accuracy (%)')
    AX.set_xlabel("Epochs")
    AX.set_ylim(0.4, 1.1)
    AX.set_title(letter, fontsize = 12,  fontweight ="bold", loc='left')
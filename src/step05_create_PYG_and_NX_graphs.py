#%%
# Step 05: Create graph data structures for Pytorch Geometric from Networkx.
#name of the script: step05_create_PYG_and_NX_graphs.py
from   cobra.io.mat import *
import copy
import networkx as nx
import pandas as pd
from networkx.algorithms import bipartite
import itertools

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

def update_df_features(base: dict, new:dict):
    
    common_vars       = list(set(base.keys()).intersection(set(new)))
    base_updated      = copy.deepcopy(base)
    base_updated.update({key: new[key] for key in common_vars})
    
    return base_updated

def new_nx_from_dict(nx_G_in, feature_dict):
    
    nx_G        = copy.deepcopy(nx_G_in)    
    x_attribute = feature_dict #nx.get_node_attributes(nx_G, "x")
     
    nx.set_node_attributes(nx_G, x_attribute, 'x')
    
    len(nx_G.nodes(data=True)['r0399']['x']) == len(nx_G.nodes(data=True)['phe_L_c']['x'])
    assert nx_G.nodes(data=True)['phe_L_c']['x']  == feature_dict['phe_L_c']#.tolist() 
    assert nx_G.nodes(data=True)['r0399']['x']  == feature_dict['r0399']#.tolist() 

    
    
    return nx_G

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
    
from torch_geometric.utils.convert import from_networkx


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

model                 = load_matlab_model("./COBRA_models/GEM_Recon3_thermocurated_redHUMAN_AA.mat")
flux_samples_CONTROL  = pd.read_parquet("./results/fluxes/CLEANED_flux_samples_CONTROL_20_000.parquet.gzip")#.abs()
flux_samples_PKU      = pd.read_parquet("./results/fluxes/CLEANED_flux_samples_PKU_20_000.parquet.gzip")#.abs()


concentration_data = pd.read_parquet("./processed_data/augmented_balanced_metabolite_data.parquet.gzip").abs()
feature_names       = pd.read_csv("./metabolite_raw_data/metabolite_names.csv")
grafo_nx            = cobra_to_networkx(model)

concentration_data['Leu'] = concentration_data['Leu.Ile']
concentration_data['Ile'] = concentration_data['Leu.Ile']
concentration_data.drop('Leu.Ile', axis=1, inplace=True)

concentration_data['C3DC'] = concentration_data['C4OH.C3DC']
concentration_data['C4OH'] = concentration_data['C4OH.C3DC']
concentration_data.drop('C4OH.C3DC', axis=1, inplace=True)

concentration_data['C4DC'] = concentration_data['C5.OH.C4DC']
concentration_data['C5OH'] = concentration_data['C5.OH.C4DC']
concentration_data.drop('C5.OH.C4DC', axis=1, inplace=True)

assert len(set(feature_names.Simbolo_traductor)-set(['SA']) - set(concentration_data.columns)) == 0



concentration_data.rename(
        columns=feature_names.set_index("Simbolo_traductor")["Recon3_ID"].to_dict(), 
        inplace=True
    )

assert set(set(concentration_data.columns)-set(["label"])).issubset(set(list(grafo_nx.nodes)))

w  = dict(zip(grafo_nx.edges() , itertools.repeat(1)))

nx.set_edge_attributes(grafo_nx, w, "weight")

assert 1 == np.unique(list(nx.get_edge_attributes(grafo_nx, "weight").values())).__len__()




flux_samples_CONTROL = get_sample_subset(flux_samples_CONTROL, concentration_data, 0)
flux_samples_PKU     = get_sample_subset(flux_samples_PKU,     concentration_data, 1)
assert flux_samples_CONTROL.r0399.max() > 20 
assert flux_samples_PKU.r0399.max() < 4
assert all(flux_samples_CONTROL.columns == flux_samples_PKU.columns)

flux_samples_CONTROL = flux_samples_CONTROL.reindex(columns=flux_samples_PKU.columns)
flux_samples         = pd.concat([flux_samples_CONTROL, flux_samples_PKU], axis=0)
flux_samples         = flux_samples.reset_index(drop=True, inplace=False)

assert len(flux_samples.columns) == len(flux_samples_CONTROL.columns)
assert len(concentration_data) == len(flux_samples)
assert flux_samples.r0399.loc[flux_samples.label == 0].mean() > 20

assert flux_samples.r0399.loc[flux_samples.label == 1].mean() < 4

mets = ['nad_', 'nadh_', "nadp_", "nadph_", "adp_", "atp_", "gdt_", "gtp_",
        "pi_", "ppi_", "pppi_", "co2_", "hco3_", "h2o_", "h2o2_", "h_", "o2_", "oh1_",
        "o2s_", "fad_",  "fadh2_", "nh4_", "so3_", "so4_", "cl_", "k_", "na1_",
        "i_", "fe2_", "fe3_", "mg2_", "ca2", "zn2_", "M02382_"]

to_remove = []
for m in mets:

    to_remove.extend(["".join(l) for l in list(zip(itertools.repeat(m), list(model.compartments.keys())))])
    
grafo_nx.remove_nodes_from(to_remove)


grafo_nx = get_largest_cc(grafo_nx)
    
nx.write_gpickle(grafo_nx, "./results/graphs/NX_recon_graph.gpickle")

concentration_data = concentration_data.sort_values(by=['label'])
concentration_data.reset_index(drop=True, inplace=True)
flux_samples = flux_samples.sort_values(by=['label'])
flux_samples.reset_index(drop=True, inplace=True)
assert all(concentration_data.label == flux_samples.label)

Labels = flux_samples.label

flux_samples.drop("label", axis=1, inplace=True)
concentration_data.drop("label", axis=1, inplace=True)



from sklearn.preprocessing import MinMaxScaler

scaler             = MinMaxScaler(feature_range=(0, 1))
concentration_data_array = scaler.fit_transform(concentration_data.drop( ["phe_L_c","tyr_L_c"], axis=1))
concentration_data_scaled       = pd.DataFrame(concentration_data_array, columns=concentration_data.drop(["phe_L_c","tyr_L_c"], axis=1).columns)
concentration_data_scaled["phe_L_c"]   = concentration_data.phe_L_c
concentration_data_scaled["tyr_L_c"]   = concentration_data.tyr_L_c



scaler             = MinMaxScaler(feature_range=(0, 1))
flux_samples_array = scaler.fit_transform(flux_samples.drop( ["r0399","PHETHPTOX2"], axis=1))
flux_samples_scaled       = pd.DataFrame(flux_samples_array, columns=flux_samples.drop(["r0399","PHETHPTOX2"], axis=1).columns)
flux_samples_scaled["r0399"]   = flux_samples.r0399
flux_samples_scaled["PHETHPTOX2"]   = flux_samples.PHETHPTOX2



blank_features = pd.DataFrame(
                    np.full((len(concentration_data_scaled), list(grafo_nx.nodes).__len__()), 1e-10),  columns=list(grafo_nx.nodes)
                    )
blank_features.reset_index(drop=True, inplace=True)
assert len(blank_features) == len(concentration_data_scaled) == len(flux_samples_scaled)
assert set(concentration_data_scaled.columns).issubset(set(list(grafo_nx.nodes)))

blank_features_dict = blank_features.to_dict(orient="list")  
flux_samples_dict   = flux_samples_scaled.to_dict(orient="list")  
concentrations_dict = concentration_data_scaled.to_dict(orient="list")  

import copy



features_only_concentrations_dict = update_df_features(blank_features_dict, concentrations_dict)
features_only_fluxes_dict         = update_df_features(blank_features_dict, flux_samples_dict)

full_features_dict = copy.deepcopy(concentrations_dict)
full_features_dict.update(flux_samples_dict)  # 
features_completed_dict = update_df_features(blank_features_dict, full_features_dict)
nx_features_only_concentrations = new_nx_from_dict(grafo_nx, features_only_concentrations_dict)  

conc_df = pd.DataFrame(
nx.get_node_attributes(nx_features_only_concentrations, 'x'))

assert set(conc_df.sum().loc[lambda x: abs(x)>=2e-6].index.tolist()) ==  set([k for k in concentrations_dict])


nx_features_only_fluxes         = new_nx_from_dict(grafo_nx, features_only_fluxes_dict)  

flux_df = pd.DataFrame(
nx.get_node_attributes(nx_features_only_fluxes, 'x'))

rxn_list_recon: list[str] = [model.reactions[i].id       for i in range(model.reactions.__len__())]
met_list_recon: list[str] = [model.metabolites[i].id     for i in range(model.metabolites.__len__())]
first_partition , second_partition = bipartite.sets(grafo_nx)

if first_partition.__len__() > second_partition.__len__():
    rxn_partition = first_partition
    met_partition = second_partition
else:
    rxn_partition = second_partition 
    met_partition = first_partition
    
assert set(rxn_partition).issubset(set(rxn_list_recon)) and set(met_partition).issubset(set(met_list_recon))
assert len(set(rxn_partition) - set(rxn_list_recon)) == 0
assert len(set(met_partition) - set(met_list_recon)) == 0

partition_list =  np.array(list(nx.get_node_attributes(grafo_nx, "bipartite").values()))
mask_rxns      =  partition_list.astype(bool)
mask_mets      =  np.invert(partition_list.astype(bool))
assert flux_df.loc[:,mask_mets].sum().unique().__len__() == 1
assert flux_df.loc[:,mask_rxns].sum().unique().__len__() > 1

nx_full_features_completed  = new_nx_from_dict(grafo_nx, features_completed_dict)  
full_features_df = pd.DataFrame(nx.get_node_attributes(nx_full_features_completed, 'x'))

assert set(full_features_df.loc[:,mask_mets].sum().loc[lambda x: abs(x)>=2e-6].index.tolist()) ==  set([k for k in concentrations_dict])
assert full_features_df.loc[:,mask_mets].sum().loc[lambda x:  abs(x)<2e-6].unique().__len__() == 1
assert full_features_df.loc[:,mask_rxns].sum().unique().__len__() > 1
assert nx_features_only_concentrations.nodes(data=True)['r0399']['x']    != nx_features_only_fluxes.nodes(data=True)['r0399']['x'] 
assert nx_features_only_concentrations.nodes(data=True)['phe_L_c']['x']  != nx_features_only_fluxes.nodes(data=True)['phe_L_c']['x'] 
assert nx_full_features_completed.nodes(data=True)['phe_L_c']['x']  == nx_features_only_concentrations.nodes(data=True)['phe_L_c']['x'] 
assert nx_full_features_completed.nodes(data=True)['r0399']['x']    == nx_features_only_fluxes.nodes(data=True)['r0399']['x'] 




pyg_graph_onlyConcen         = make_PYG_graph_from_grafo_nx(nx_features_only_concentrations)
pyg_graph_onlyFluxes         = make_PYG_graph_from_grafo_nx(nx_features_only_fluxes)
pyg_graph_Concen_plus_Fluxes = make_PYG_graph_from_grafo_nx(nx_full_features_completed)



node_list = list(grafo_nx.nodes)

assert pyg_graph_onlyConcen.x.shape[0] == len(node_list) == len(nx_full_features_completed.nodes) == len(nx_features_only_fluxes.nodes)== len(nx_features_only_concentrations.nodes)
assert pyg_graph_onlyFluxes.x.shape[0] == len(node_list) #== len(nx_full_features_completed.nodes) == len(nx_features_only_fluxes.nodes)== len(nx_features_only_concentrations.nodes)
assert pyg_graph_Concen_plus_Fluxes.x.shape[0] == len(node_list) #== len(nx_full_features_completed.nodes) == len(nx_features_only_fluxes.nodes)== len(nx_features_only_concentrations.nodes)
import torch
import numpy as np
pyg_graph_onlyConcen.y               = torch.tensor(Labels).reshape(len(Labels),1)
pyg_graph_onlyFluxes.y               = torch.tensor(Labels).reshape(len(Labels),1)
pyg_graph_Concen_plus_Fluxes.y       = torch.tensor(Labels).reshape(len(Labels),1)

from itertools import compress

assert set(compress(compress(node_list, mask_mets), pyg_graph_onlyConcen.x.numpy()[mask_mets,:].sum(axis=1)>=2e-6)) == set([k for k in concentrations_dict])
assert np.unique(
                pyg_graph_onlyConcen.x.numpy()[mask_mets,:].sum(axis=1)[
                np.invert(pyg_graph_onlyConcen.x.numpy()[mask_mets,:].sum(axis=1)>=2e-6)
                ]).__len__() == 1

assert np.unique(
                pyg_graph_onlyConcen.x.numpy()[mask_rxns,:].sum(axis=1)
                ).__len__() == 1

assert np.unique(
                pyg_graph_onlyFluxes.x.numpy()[mask_mets,:].sum(axis=1)
                ).__len__() == 1

assert np.unique(
                pyg_graph_onlyFluxes.x.numpy()[mask_rxns,:].sum(axis=1)
                ).__len__() > 1
#pyg_graph_Concen_plus_Fluxes
assert np.unique(
                pyg_graph_Concen_plus_Fluxes.x.numpy()[mask_mets,:].sum(axis=1)[
                                                                                np.invert(
                                                                                    pyg_graph_Concen_plus_Fluxes.x.numpy()[mask_mets,:].sum(axis=1)  >=2e-6   
                                                                                )
                                                                                ]).__len__() ==1


assert np.unique(
                pyg_graph_Concen_plus_Fluxes.x.numpy()[mask_mets,:].sum(axis=1)[
                                                                                #np.invert(
                                                                                    pyg_graph_Concen_plus_Fluxes.x.numpy()[mask_mets,:].sum(axis=1)  >=2e-6   
                                                                                #)
                                                                                ]).__len__() > 1

assert np.unique(
                pyg_graph_Concen_plus_Fluxes.x.numpy()[mask_rxns,:].sum(axis=1)[
                                                                                #np.invert(
                                                                                    pyg_graph_Concen_plus_Fluxes.x.numpy()[mask_rxns,:].sum(axis=1)  >=2e-6   
                                                                                #)
                                                                                ]).__len__() > 1



torch.save(pyg_graph_onlyConcen, "./results/graphs/PYG_graph_only_Concen.pt")
torch.save(pyg_graph_onlyFluxes, "./results/graphs/PYG_graph_only_Fluxes.pt")
torch.save(pyg_graph_Concen_plus_Fluxes, "./results/graphs/PYG_graph_Concen_plus_Fluxes.pt")



# %%

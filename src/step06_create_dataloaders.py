#%%
from acevedo_clss_and_fcns import *

grafo_nx   = nx.read_gpickle( "./results/graphs/NX_recon_graph.gpickle")
phe_L_c    = list(grafo_nx.nodes).index('phe_L_c')
tyr_L_c    = list(grafo_nx.nodes).index('tyr_L_c')
r0399      = list(grafo_nx.nodes).index('r0399')
PHETHPTOX2 = list(grafo_nx.nodes).index('PHETHPTOX2')
PYG_graph_only_Concen         = torch.load("./results/graphs/PYG_graph_only_Concen.pt")
PYG_graph_only_Fluxes          = torch.load("./results/graphs/PYG_graph_only_Fluxes.pt")
PYG_graph_Concen_plus_Fluxes  = torch.load("./results/graphs/PYG_graph_Concen_plus_Fluxes.pt") 


first_partition , second_partition = bipartite.sets(grafo_nx)

if first_partition.__len__() > second_partition.__len__():
    rxn_partition = first_partition
    met_partition = second_partition
else:
    rxn_partition = second_partition 
    met_partition = first_partition


partition_list =  np.array(list(nx.get_node_attributes(grafo_nx, "bipartite").values()))
mask_rxns      =  partition_list.astype(bool)
mask_mets      =  np.invert(partition_list.astype(bool))
graphs_list_PYG_graph_only_Concen        = make_graphs_list(PYG_graph_only_Concen, target_list = [phe_L_c, tyr_L_c], mask_target = True)
graphs_list_PYG_graph_only_Fluxes        = make_graphs_list(PYG_graph_only_Fluxes, target_list = [r0399, PHETHPTOX2], mask_target = True)
graphs_list_PYG_graph_Concen_plus_Fluxes = make_graphs_list(PYG_graph_Concen_plus_Fluxes, 
                                                            target_list = [phe_L_c, tyr_L_c, r0399, PHETHPTOX2], mask_target = True)

loader_only_Concen                 = batch_loader(graphs_list_PYG_graph_only_Concen, batch_size= 4*32, validation_percent = .4)
loader_only_Fluxes                 = batch_loader(graphs_list_PYG_graph_only_Fluxes, batch_size= 4*32, validation_percent = .4)
loader_Concen_plus_Fluxes          = batch_loader(graphs_list_PYG_graph_Concen_plus_Fluxes, batch_size= 4*32, validation_percent = .4)




x_loader_only_Concen = get_a_graph_from_loader(loader_only_Concen)


assert np.unique(
    x_loader_only_Concen.x.reshape(len(mask_mets))[mask_mets][
    x_loader_only_Concen.x.reshape(len(mask_mets))[mask_mets] > 1e-10]).__len__() > 2

assert np.unique(
    x_loader_only_Concen.x.reshape(len(mask_mets))[mask_mets][
    x_loader_only_Concen.x.reshape(len(mask_mets))[mask_mets] <= 1e-10]).__len__() >= 1

#np.array([1.e-10]) in 
assert np.unique(
    x_loader_only_Concen.x.reshape(len(mask_mets))[mask_mets][
    x_loader_only_Concen.x.reshape(len(mask_mets))[mask_mets] <= 1e-10]).__len__() <=2


assert np.unique(
x_loader_only_Concen.x.reshape(len(mask_mets))[mask_mets][
    x_loader_only_Concen.x.reshape(len(mask_mets))[mask_mets] <= 1e-10]).sum() < 1e-9


assert np.unique(x_loader_only_Concen.x.reshape(len(mask_rxns))[mask_rxns]).__len__() == 1



x_loader_only_Fluxes = get_a_graph_from_loader(loader_only_Fluxes)


assert np.unique(x_loader_only_Fluxes.x.reshape(len(mask_mets))[mask_mets]).__len__() == 1
assert np.unique(x_loader_only_Fluxes.x.reshape(len(mask_mets))[mask_rxns]).__len__() > 2



x_loader_Concen_plus_Fluxes = get_a_graph_from_loader(loader_Concen_plus_Fluxes)


assert np.unique(
    x_loader_Concen_plus_Fluxes.x.reshape(len(mask_mets))[mask_mets][
    x_loader_Concen_plus_Fluxes.x.reshape(len(mask_mets))[mask_mets] > 1e-10]).__len__() > 3


assert np.unique(
    x_loader_Concen_plus_Fluxes.x.reshape(len(mask_mets))[mask_mets][
        np.invert(
    x_loader_Concen_plus_Fluxes.x.reshape(len(mask_mets))[mask_mets] > 1e-10)]
    ).__len__() <=2


assert np.unique(
    x_loader_Concen_plus_Fluxes.x.reshape(len(mask_mets))[mask_mets][
        np.invert(
    x_loader_Concen_plus_Fluxes.x.reshape(len(mask_mets))[mask_mets] > 1e-10)]
    ).sum() < 1e-9


assert np.unique(
    x_loader_Concen_plus_Fluxes.x.reshape(len(mask_mets))[mask_rxns][
    x_loader_Concen_plus_Fluxes.x.reshape(len(mask_mets))[mask_rxns] > 1e-10]).__len__() > 3

torch.save(loader_only_Concen, "./results/dataloaders/MASKED_loader_only_Concen.pt")
torch.save(loader_only_Fluxes, "./results/dataloaders/MASKED_loader_only_Fluxes.pt")
torch.save(loader_Concen_plus_Fluxes, "./results/dataloaders/MASKED_loader_Concen_plus_Fluxes.pt")
# %%

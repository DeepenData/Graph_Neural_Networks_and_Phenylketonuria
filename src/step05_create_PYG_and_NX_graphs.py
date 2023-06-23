#%%
from    acevedo_clss_and_fcns import * 
from    cobra.io.mat import *


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

def update_df_features(base: dict, new:dict):
    
    common_vars       = list(set(base.keys()).intersection(set(new)))
    base_updated      = copy.deepcopy(base)
    base_updated.update({key: new[key] for key in common_vars})
    
    return base_updated


features_only_concentrations_dict = update_df_features(blank_features_dict, concentrations_dict)
features_only_fluxes_dict         = update_df_features(blank_features_dict, flux_samples_dict)

full_features_dict = copy.deepcopy(concentrations_dict)
full_features_dict.update(flux_samples_dict)  # 
features_completed_dict = update_df_features(blank_features_dict, full_features_dict)

def new_nx_from_dict(nx_G_in, feature_dict):
    
    nx_G        = copy.deepcopy(nx_G_in)    
    x_attribute = feature_dict #nx.get_node_attributes(nx_G, "x")
     
    nx.set_node_attributes(nx_G, x_attribute, 'x')
    
    len(nx_G.nodes(data=True)['r0399']['x']) == len(nx_G.nodes(data=True)['phe_L_c']['x'])
    assert nx_G.nodes(data=True)['phe_L_c']['x']  == feature_dict['phe_L_c']#.tolist() 
    assert nx_G.nodes(data=True)['r0399']['x']  == feature_dict['r0399']#.tolist() 

    
    
    return nx_G


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

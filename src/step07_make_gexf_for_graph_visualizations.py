# %%
from acevedo_clss_and_fcns import * 
from    cobra.io.mat import *
from networkx.algorithms import bipartite
def check_Concen_plus_Fluxes(a_graph, mask_mets, mask_rxns):
    

    assert np.unique(
        a_graph.x.reshape(len(mask_mets))[mask_mets][
        a_graph.x.reshape(len(mask_mets))[mask_mets] > 1e-10]).__len__() > 3

    assert np.unique(
        a_graph.x.reshape(len(mask_mets))[mask_mets][
            np.invert(
        a_graph.x.reshape(len(mask_mets))[mask_mets] > 1e-10)]
        ).__len__() <=2

    assert np.unique(
        a_graph.x.reshape(len(mask_mets))[mask_mets][
            np.invert(
        a_graph.x.reshape(len(mask_mets))[mask_mets] > 1e-10)]
        ).sum() < 1e-9

    assert np.unique(
        a_graph.x.reshape(len(mask_mets))[mask_rxns][
        a_graph.x.reshape(len(mask_mets))[mask_rxns] > 1e-10]).__len__() > 3


model = load_matlab_model("./COBRA_models/GEM_Recon3_thermocurated_redHUMAN_AA.mat")
rxn_list_recon: list[str] = [model.reactions[i].id       for i in range(model.reactions.__len__())]
met_list_recon: list[str] = [model.metabolites[i].id     for i in range(model.metabolites.__len__())]

G = nx.read_gpickle("./results/graphs/NX_recon_graph.gpickle")


partition_list =  np.array(list(nx.get_node_attributes(G, "bipartite").values()))
mask_rxns      =  partition_list.astype(bool)
mask_mets      =  np.invert(partition_list.astype(bool))

def get_a_graph_from_loader(loader):
    
    #loader   = loader_only_Concen #torch.load(loader_path)
    a_batch  = next(iter(loader.get_train_loader()))
    return a_batch[0]

Concen_plus_Fluxes = get_a_graph_from_loader(torch.load('./results/dataloaders/MASKED_loader_Concen_plus_Fluxes.pt'))

check_Concen_plus_Fluxes(Concen_plus_Fluxes, mask_mets, mask_rxns)

def get_averages(loader):    

    control_list = []
    pku_list     = []

    for graph in loader.dataset[0:1000]:
        
        if graph.y.item() == 0:   
            control_list.append(graph.x) 
            
        elif graph.y.item() == 1:
            pku_list.append(graph.x) 
    
    return torch.cat(control_list, dim=1).mean(axis = 1),  torch.cat(pku_list, dim=1).mean(axis = 1)


CONTROL_only_Concen, PKU_only_Concen = get_averages( torch.load('./results/dataloaders/MASKED_loader_only_Concen.pt').get_train_loader())
CONTROL_only_Fluxes, PKU_only_Fluxes = get_averages( torch.load('./results/dataloaders/MASKED_loader_only_Fluxes.pt').get_train_loader())
CONTROL_Concen_plus_Fluxes, PKU_Concen_plus_Fluxes = get_averages( torch.load('./results/dataloaders/MASKED_loader_Concen_plus_Fluxes.pt').get_train_loader())

nx_Conc    = copy.deepcopy(nx.read_gpickle("./results/graphs/NX_recon_graph.gpickle"))
nx_Flux    = copy.deepcopy(nx.read_gpickle("./results/graphs/NX_recon_graph.gpickle"))
nx_ConFlux = copy.deepcopy(nx.read_gpickle("./results/graphs/NX_recon_graph.gpickle"))

def set_control_pku_features(G, control_tensor, pku_tensor):
    

    control_dict   = dict(zip(G,  control_tensor.tolist()))
    pku_dict       = dict(zip(G,  pku_tensor.tolist()))


    nx.set_node_attributes(G, control_dict, "control")
    nx.set_node_attributes(G, pku_dict, "pku")


set_control_pku_features(nx_Conc, CONTROL_only_Concen, PKU_only_Concen)
set_control_pku_features(nx_Flux, CONTROL_only_Fluxes, PKU_only_Fluxes)
set_control_pku_features(nx_ConFlux, CONTROL_Concen_plus_Fluxes, PKU_Concen_plus_Fluxes)

metabolites = pd.read_csv("./metabolite_raw_data/metabolite_names.csv").Recon3_ID
metabolites = metabolites[[m in list(G.nodes) for m in metabolites]].tolist()

metabolites = list(set(metabolites) - set(['sucaceto_c']))



assert set(metabolites).issubset(set(list(G.nodes)))

mets_bool = [bool(re.search("crn_c|\\d+dc_c", s)) for s in metabolites]


AAs       = list(itertools.compress(metabolites, np.invert(mets_bool)))
ACs       = list(itertools.compress(metabolites, mets_bool))

base_dict =  dict(zip(G,   itertools.repeat(0)))
AAs_dict  =  dict(zip(AAs, itertools.repeat(1)))
ACs_dict  =  dict(zip(ACs, itertools.repeat(2)))

colors_dict = copy.deepcopy(base_dict)

colors_dict.update(AAs_dict)
colors_dict.update(ACs_dict)

nx.set_node_attributes(nx_Conc, colors_dict, "color")
nx.write_gexf(nx_Conc, "./results/graphs/for_visualizations/nx_Conc.gexf")



model = load_matlab_model("./COBRA_models/GEM_Recon3_thermocurated_redHUMAN_AA.mat")
import re
rxns        = [ r.id for r in model.reactions]
subsystems  = [s.subsystem for s in model.reactions]
subsys_bool = [bool(re.search("Phenyla|phenyla|Tetrahydrobiopterin", s)) for s in subsystems[0:10600]]

subsys_rxns = list(itertools.compress(rxns, subsys_bool))

Phe_THBPT_rxns = ['r0399',"DHPR", "DHPR2", "THBPT4ACAMDASE", "HMR_6728", "r0403", "r0398", 'DHPR2',
 'PHETA1m', 'PHYCBOXL', 'PPOR', 'PTHPS', 'THBPT4ACAMDASE', 'r0403', 'r0545', 'r0547', 'PHLAC', 'DHPR', 'r0398', 'PHETA1', 'HMR_6770',  'HMR_6854', 'HMR_6874'
                  ]


ACYL_rxns = ['FAOXC14C12m', 'FAOXC14C14OHm', 'FAOXC162C142m', 'LNLCCPT2', 'C181OHc', 'C40CPT1', 'FAOXC3DC',
          'CSNATr', 'C30CPT1', 'C140CPT1', 'C141CPT1', 'FAOXC12DCc', 'C121CPT1', 'ADRNCPT1', 'ARACHCPT1', 'C160CPT1', 'C161CPT1', 'C161CPT12', 
          'C180CPT1', 'C181CPT1', 'C204CPT1', 'C226CPT1', 'CLPNDCPT1', 'DMNONCOACRNCPT1', 'DMNONCOACRNCPT1', 'EICOSTETCPT1', 'OCTDECCPT1',
          'OCD11COACPT1', 'C81CPT1', 'C80CPT1', 'C60CPT1', 'C51CPT1', 'C50CPT1']

base_dict            =  dict(zip(G,   itertools.repeat(0)))
Phe_THBPT_rxns_dict  =  dict(zip(Phe_THBPT_rxns, itertools.repeat(1)))
ACYL_rxns_dict       =  dict(zip(ACYL_rxns, itertools.repeat(2)))

rxns_colors_dict = copy.deepcopy(base_dict)
rxns_colors_dict.update(Phe_THBPT_rxns_dict)
rxns_colors_dict.update(ACYL_rxns_dict)

nx.set_node_attributes(nx_Flux, rxns_colors_dict, "rxns_colors")

def mask_fluxes(G,patient_group, attribute_name):

    nx_Flux_control_dict = nx.get_node_attributes(G, patient_group)
    visible_rxns         = list(set(Phe_THBPT_rxns).union(set(ACYL_rxns)))




    visible_subset_dict = {
        key: nx_Flux_control_dict[key]

        for key in visible_rxns
    }

    base_dict            =  dict(zip(G,   itertools.repeat(0.0)))
    flux_dict            = copy.deepcopy(base_dict)
    flux_dict.update(visible_subset_dict)
    nx.set_node_attributes(G, flux_dict, attribute_name)


mask_fluxes(nx_Flux,"control", "control_FLUX_node_sizes")
mask_fluxes(nx_Flux,"pku",      "pku_FLUX_node_sizes")

nx.write_gexf(nx_Flux, "./results/graphs/for_visualizations/nx_Flux.gexf")


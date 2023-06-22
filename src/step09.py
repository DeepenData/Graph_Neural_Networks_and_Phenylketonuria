#%%
from acevedo_clss_and_fcns import * 
device = 'cpu'
if torch.cuda.is_available():
    torch.cuda.init()
    if torch.cuda.is_initialized():
        device = 'cuda:0'
print(f"{device = }")

epochs = 35


GCN_masked_flux                          =   train_and_validate(gnn_type = "GCN", mask = True, flux = True, concentration = False,
                                                  loader_path = "./results/dataloaders/MASKED_loader_only_Fluxes.pt" ,
                                                    EPOCHS = epochs, save = True, verbose = True, saving_folder =  "./results/saved_GNNs/")

GCN_masked_concen                        =   train_and_validate(gnn_type = "GCN", mask = True, flux = False, concentration = True,
                                                  loader_path = "./results/dataloaders/MASKED_loader_only_Concen.pt" ,
                                                    EPOCHS = epochs, save = True, verbose = True, saving_folder =  "./results/saved_GNNs/")

GCN_masked_concen_plus_flux              =   train_and_validate(gnn_type = "GCN", mask = True, flux = True, concentration = True,
                                                  loader_path = "./results/dataloaders/MASKED_loader_Concen_plus_Fluxes.pt" ,
                                                    EPOCHS = epochs, save = True, verbose = True, saving_folder =  "./results/saved_GNNs/")
########################################################################################################################################################
GAT_masked_flux                          =   train_and_validate(gnn_type = "GAT", mask = True, flux = True, concentration = False,
                                                  loader_path = "./results/dataloaders/MASKED_loader_only_Fluxes.pt" ,
                                                    EPOCHS = epochs, save = True, verbose = True, saving_folder =  "./results/saved_GNNs/")

GAT_masked_concen                        =   train_and_validate(gnn_type = "GAT", mask = True, flux = False, concentration = True,
                                                  loader_path = "./results/dataloaders/MASKED_loader_only_Concen.pt" ,
                                                    EPOCHS = epochs, save = True, verbose = True, saving_folder =  "./results/saved_GNNs/")

GAT_masked_concen_plus_flux              =   train_and_validate(gnn_type = "GAT", mask = True, flux = True, concentration = True,
                                                  loader_path = "./results/dataloaders/MASKED_loader_Concen_plus_Fluxes.pt" ,
                                                    EPOCHS = epochs, save = True, verbose = True, saving_folder =  "./results/saved_GNNs/")
########################################################################################################################################################
GIN_masked_flux                          =   train_and_validate(gnn_type = "GIN", mask = True, flux = True, concentration = False,
                                                  loader_path = "./results/dataloaders/MASKED_loader_only_Fluxes.pt" ,
                                                    EPOCHS = epochs, save = True, verbose = True, saving_folder =  "./results/saved_GNNs/")

GIN_masked_concen                        =   train_and_validate(gnn_type = "GIN", mask = True, flux = False, concentration = True,
                                                  loader_path = "./results/dataloaders/MASKED_loader_only_Concen.pt" ,
                                                    EPOCHS = epochs, save = True, verbose = True, saving_folder =  "./results/saved_GNNs/")

GIN_masked_concen_plus_flux              =   train_and_validate(gnn_type = "GIN", mask = True, flux = True, concentration = True,
                                                  loader_path = "./results/dataloaders/MASKED_loader_Concen_plus_Fluxes.pt" ,
                                                    EPOCHS = epochs, save = True, verbose = True, saving_folder =  "./results/saved_GNNs/")

training_results = {
    'GCN_masked_flux' : GCN_masked_flux,
    'GCN_masked_concen': GCN_masked_concen,
    'GCN_masked_concen_plus_flux': GCN_masked_concen_plus_flux,
    
    'GAT_masked_flux': GAT_masked_flux,
    'GAT_masked_concen': GAT_masked_concen,
    'GAT_masked_concen_plus_flux': GAT_masked_concen_plus_flux,
    
    'GIN_masked_flux': GIN_masked_flux,
    'GIN_masked_concen': GIN_masked_concen,
    'GIN_masked_concen_plus_flux': GIN_masked_concen_plus_flux   }


import pickle

#a = learning_results

with open('./results/training_validation_best_models_paths/training_results.pickle', 'wb') as handle:
    pickle.dump(training_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
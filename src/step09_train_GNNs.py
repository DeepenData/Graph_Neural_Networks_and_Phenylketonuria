#%%
#from acevedo_clss_and_fcns import * 
import torch
device = 'cpu'
if torch.cuda.is_available():
    torch.cuda.init()
    if torch.cuda.is_initialized():
        device = 'cuda:0'
print(f"{device = }")
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch.utils.data import RandomSampler
from torch.nn import Linear, LeakyReLU
import torch_geometric
import gc
import tqdm
import copy
from datetime import datetime    
from torch_geometric.nn.models import GIN
from torch_geometric.nn.models import GAT
from torch_geometric.nn.models import GCN
class BatchLoader():
    """
    BatchLoader is a class that helps to generate DataLoader for train, validation, and test sets.

    Attributes:
    ----------
    graphs: list
        The list of input graphs.
    batch_size: int
        The size of each batch of graphs. Default is 32.
    num_samples: int, optional
        The number of samples to draw from the graphs to form a batch. 
        If not provided, all graphs are used.
    validation_percent: float
        The percentage of the dataset to be used for validation and testing. Default is 0.3 (30%).
        
    train_idxs: list
        List of indexes for the training data split.
    val_idxs: list
        List of indexes for the validation data split.
    test_idxs: list
        List of indexes for the test data split.

    Methods:
    -------
    get_train_loader():
        Returns a DataLoader for the training data.
    get_validation_loader():
        Returns a DataLoader for the validation data.
    get_test_loader():
        Returns a DataLoader for the testing data.
    """

    def __init__(self, graphs: list, batch_size: int  =1*32, num_samples:int = None, validation_percent:float = .3):
        self.graphs = graphs
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.validation_percent = validation_percent

        # Splitting the data into train, validation, and test sets.
        self.train_idxs, self.sub_idxs = train_test_split(range(len(self.graphs)), test_size = self.validation_percent)
        self.val_idxs,   self.test_idxs = train_test_split(self.sub_idxs, test_size = self.validation_percent)     

    def get_train_loader(self):
        """
        Generates and returns a DataLoader for the training data.
        """
        train_subset = [self.graphs[i] for i in self.train_idxs]
        sampler      = RandomSampler(train_subset, replacement=False)   
        return  DataLoader(train_subset, batch_size= self.batch_size, sampler = sampler,  drop_last=True)

    def get_validation_loader(self):
        """
        Generates and returns a DataLoader for the validation data.
        """
        validation_subset = [self.graphs[i] for i in self.val_idxs]
        sampler      = RandomSampler(validation_subset, replacement=False)   
        return  DataLoader(validation_subset, batch_size= self.batch_size, sampler = sampler,  drop_last=True)

    def get_test_loader(self):
        """
        Generates and returns a DataLoader for the test data.
        """
        test_subset = [self.graphs[i] for i in self.test_idxs]
        sampler      = RandomSampler(test_subset, replacement=False)   
        return  DataLoader(test_subset, batch_size= self.batch_size, sampler = sampler,  drop_last=True)
class my_GNN(torch.nn.Module):
    
    def __init__(
        self, 
        model:str ,
        n_classes: int,
        n_nodes : int, 
        num_features : int, 
        out_channels: int = 8,
        dropout : float = 0.05, 
        hidden_dim : int = 8, 
        LeakyReLU_slope : float = 0.01,
        num_layers: int = 1,
        
        
    ):
        super(my_GNN, self).__init__() # TODO: why SUPER gato? 
        self.n_nodes = n_nodes
        self.n_classes = n_classes
        self.dropout = dropout
        self.num_features = num_features
        self.out_channels = out_channels
        self.model        = model
        
        if self.model == "GCN":    
            self.GNN_layers =  GCN(in_channels= num_features, hidden_channels= hidden_dim, num_layers= num_layers, 
                                out_channels= out_channels, dropout=dropout,  jk=None, act='LeakyReLU', act_first = False) 
    
        elif self.model == "GAT":        
             self.GNN_layers =  GAT(in_channels= num_features, hidden_channels= hidden_dim, num_layers= num_layers, 
                                    out_channels= out_channels, dropout=dropout,  jk=None, act='LeakyReLU', act_first = False) 

        elif self.model == "GIN":
             self.GNN_layers =  GIN(in_channels= num_features, hidden_channels= hidden_dim, num_layers= num_layers, 
                                    out_channels= out_channels, dropout=dropout,  jk=None, act='LeakyReLU', act_first = False)
        
        #self.GIN_layers =  GIN(in_channels= num_features, hidden_channels= hidden_dim, num_layers= num_layers, 
        #                       out_channels= out_channels, dropout=dropout,  jk=None, act='LeakyReLU', act_first = False)              
        self.FC1          = Linear(in_features=out_channels, out_features=1, bias=True)
        self.FC2          = Linear(in_features= self.n_nodes, out_features=self.n_classes, bias=True)
        #self.FC          = Linear(in_features=out_channels, out_features=1, bias=True)           
           
        self.leakyrelu  = LeakyReLU(LeakyReLU_slope)#.to('cuda')
    def forward(self, x, edge_index, batch):
        batch_size = batch.unique().__len__()

        x     = self.GNN_layers(x, edge_index)
        x     = x.reshape(batch_size, self.n_nodes, self.out_channels)
        x     = self.FC1(self.leakyrelu(x))
        x     = x.reshape(batch_size,  self.n_nodes)       
        x     = self.FC2(self.leakyrelu(x))    

        return torch.nn.functional.log_softmax(x, dim=1)
    
    
def train_one_epoch(modelo,
                    optimizer, 
                    train_loader: torch_geometric.loader.dataloader.DataLoader,
                    loss_fun: torch.nn.modules.loss,
                    device:str='cpu' ):

    correct = 0
    for i, data in enumerate(train_loader):
        assert not data.is_cuda   
        if (device == 'cuda:0') | (device == 'cuda'):                            
            data.to(device, non_blocking=True) 
            assert data.is_cuda       
                
        optimizer.zero_grad(set_to_none=True) # Zero your gradients for every batch        
        if (device == 'cuda:0') | (device == 'cuda'):
            #with torch.cuda.amp.autocast():      
            predictions = modelo(data.x, data.edge_index,  data.batch)# Make predictions for this batch
            loss        = loss_fun(predictions, data.y)
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.        
            pred     = predictions.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.

    return correct / len(train_loader.dataset)

def validate(modelo, loader: DataLoader, device: str = 'cpu'):
    modelo.eval()
    correct = 0
    for i, val_data in enumerate(loader):
        
        assert not val_data.is_cuda
        if (device == 'cuda:0') | (device == 'cuda'):
            val_data.to(device, non_blocking=True) 
            assert val_data.is_cuda                          

        val_predictions = modelo(val_data.x, val_data.edge_index,  val_data.batch)# Make predictions for this batch
        pred            = val_predictions.argmax(dim=1)

        correct += int((pred == val_data.y).sum())
        

    return correct / len(loader.dataset)   


import os

def train_and_validate(gnn_type, mask, flux, concentration, loader_path , EPOCHS, save, verbose, saving_folder, device:str="cuda"):
    
    
    if mask and flux and not concentration:
        assert "MASKED_loader_only_Fluxes.pt" == os.path.basename(loader_path)
        
    elif mask and not flux and concentration:
        assert "MASKED_loader_only_Concen.pt" == os.path.basename(loader_path)

    elif mask and  flux and concentration:
        assert "MASKED_loader_Concen_plus_Fluxes.pt" == os.path.basename(loader_path)  
    
    loader = torch.load(loader_path)

    a_batch         = next(iter(loader.get_train_loader()))
    a_graph         = a_batch[0]
    
    model           = my_GNN( model=gnn_type,
                                            n_nodes = a_graph.num_nodes, 
                                            num_features = a_graph.num_node_features, 
                                            n_classes = a_graph.num_classes,
                                            hidden_dim=8,
                                            num_layers=1).to(device, non_blocking=True).to(device)
    
    
    optimizer       = torch.optim.Adam(model.parameters())
    loss_function   = torch.nn.NLLLoss()
    gc.collect()
    torch.cuda.empty_cache() 
    model_type = model.GNN_layers.__class__.__name__


    
    all_train_accuracy_ = []
    all_validation_accuracy_ = []
    best_validation_accuracy = 1e-10
    for epoch in tqdm.tqdm(range(EPOCHS)):
        
        train_accuracy = train_one_epoch(model,
                            optimizer=optimizer, 
                            train_loader=loader.get_train_loader(),
                            loss_fun=loss_function,
                            device = device)

        validation_accuracy = validate(model, loader.get_validation_loader(), device)
        
        all_train_accuracy_.extend([train_accuracy])
        all_validation_accuracy_.extend([validation_accuracy])
        
        
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            del validation_accuracy
            best_val_state_dict   = copy.deepcopy(model.state_dict())
            best_val_model        = copy.deepcopy(model)
            
            if verbose:
                timestamp     = datetime.now().strftime('%d-%m-%Y_%Hh_%Mmin')              
                print(f'{model_type = } {flux = } {mask = } Epoch: {epoch:03d}, train_accuracy: {train_accuracy:.4f}, best_validation_accuracy: {best_validation_accuracy:.4f}')
                
                if save:
                    #path = f"{saving_folder}Flux/" if flux else f"{saving_folder}Non_flux/"
                    #path = f"{path}Masked/{model_type}" if mask else f"{path}Non_masked/{model_type}" 
                    path = f"{saving_folder}Masked/{model_type}" if mask else f"{saving_folder}Non_masked/{model_type}" 

                    
                    if "MASKED_loader_only_Fluxes.pt" == os.path.basename(loader_path):
                        path = f"{path}/Fluxes"
                        
                    elif "MASKED_loader_only_Concen.pt" == os.path.basename(loader_path):
                        path = f"{path}/Concentration"
                        
                    elif "MASKED_loader_Concen_plus_Fluxes.pt" == os.path.basename(loader_path):
                        path = f"{path}/Concen_plus_Fluxes"
                    
                    
                               
                    model_path = path +'/Model_{}_{}_best_ValAcc_{}_epoch_{}.pt'.format(model_type,timestamp, best_validation_accuracy, epoch)
                    torch.save(best_val_model, model_path)
                    print(f"saved as {model_path}")
                    
    return model_path, all_train_accuracy_, all_validation_accuracy_
epochs = 3 #35


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
# %%

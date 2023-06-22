# import required libraries
import os
import cobra
from cobra.io.mat import *
import warnings
warnings.simplefilter('ignore')
import itertools
from cobra.sampling import OptGPSampler, ACHRSampler
import os
from cobra.sampling import OptGPSampler
import numpy as np

cpus = os.cpu_count()
from cobra.io.json import load_json_model

def read_mat_file():
    # get the parent directory of the current working directory
    parent_dir = os.path.dirname(os.getcwd())

    # construct the full path of the file
    file_path = os.path.join(parent_dir, "COBRA_models/GEM_Recon3_thermocurated_redHUMAN_AA.mat")

    # load the .mat file
    model = load_matlab_model("./COBRA_models/GEM_Recon3_thermocurated_redHUMAN_AA.mat")

    # return the loaded data
    return model

def main():
    model = read_mat_file()
    print(f"{            type(model)            }")
    print(f'Number of reactions: {model.reactions.__len__()}')
    print(f'Number of metabolites: {model.metabolites.__len__()}')
    print(f'Number of compartments: {model.compartments.__len__()}')

    model.reactions.get_by_id('biomass_maintenance').bounds = 0, 0
    model.reactions.get_by_id('biomass').bounds             = .5*model.optimize().objective_value, 100
    model.objective = 'r0399'
    model.reactions.get_by_id("r0399").bounds = (0, 1e5)
    model.reactions.get_by_id("PHETHPTOX2").bounds = (0, 0.0)   
    def turn_off_rxn(rxn_id):
        model.reactions.get_by_id(rxn_id).bounds       = (0, 0)
   
        return model.reactions.get_by_id(rxn_id).bounds 
    
    list(map(turn_off_rxn, [r.id for r in model.metabolites.get_by_id("tyr_L_e").reactions]))
    list(map(turn_off_rxn, [r.id for r in model.metabolites.get_by_id("dhbpt_e").reactions]))
    list(map(turn_off_rxn, [r.id for r in model.metabolites.get_by_id("thbpt_e").reactions]))
    
    regulatory_reactions = model.optimize().reduced_costs.loc[lambda x: abs(x)>1e-12].index.tolist()
    regulatory_reactions = list(
    itertools.compress(regulatory_reactions, 
                    np.invert(abs(np.array([list(model.reactions.get_by_id(rr).bounds) for rr in regulatory_reactions]
    )).sum(axis=1) == 0)))
    
    def set_rxn_bounds(rxn_id):
        model.reactions.get_by_id(rxn_id).bounds       = (0, 33.5)
        
        return model.reactions.get_by_id(rxn_id).bounds 


    list(map(set_rxn_bounds, regulatory_reactions))
    
    ##HEALTHY
    model.reactions.get_by_id("r0399").bounds       = (99, model.optimize().objective_value)
    model.reactions.get_by_id("PHETHPTOX2").bounds  = (0, 0)



    optgp = OptGPSampler(model, processes=os.cpu_count(), thinning=1)
    samples_healthy = ''
    samples_healthy = optgp.sample(10)
    print(f"{type(samples_healthy)}")
    
    def save_to_parquet(df, file_name):
        # get the parent directory of the current working directory
        parent_dir = os.path.dirname(os.getcwd())

        # construct the full path of the file
        file_path = os.path.join(parent_dir, f"results/fluxes/{file_name}.parquet.gzip")

        # Ensure the directory exists before writing to it
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # write DataFrame to a parquet file with gzip compression
        #df.to_parquet(file_path, compression='gzip')
        print(file_path)
        
    save_to_parquet(samples_healthy, 'aa1')
    
# call the main function
if __name__ == "__main__":
    main()

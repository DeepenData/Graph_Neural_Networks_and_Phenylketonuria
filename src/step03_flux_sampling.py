# Import necessary libraries
import os
import cobra
from cobra.io.mat import load_matlab_model
import warnings
import itertools
from cobra.sampling import OptGPSampler
import numpy as np

# Filter out warning messages
warnings.simplefilter('ignore')

# Function to read .mat file and load the model from the file
def read_mat_file():
    # Load the .mat file
    model = load_matlab_model("./COBRA_models/GEM_Recon3_thermocurated_redHUMAN_AA.mat")

    # Return the loaded data
    return model

# Function to disable a reaction in the model
def turn_off_rxn(model, rxn_id):
    model.reactions.get_by_id(rxn_id).bounds = (0, 0)
    return model.reactions.get_by_id(rxn_id).bounds

# Function to set the bounds of a reaction in the model
def set_rxn_bounds(model, rxn_id):
    model.reactions.get_by_id(rxn_id).bounds = (0, 33.5)
    return model.reactions.get_by_id(rxn_id).bounds

# Function to save a DataFrame to a parquet file with gzip compression
def save_to_parquet(df, file_name):
    file_path = os.path.join(os.getcwd(), f"results/fluxes/{file_name}.parquet.gzip")

    # Ensure the directory exists before writing to it
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Write DataFrame to a parquet file with gzip compression
    df.to_parquet(file_path, compression='gzip')
    print(file_path)

def main():
    # Read the .mat file and load the model
    model = read_mat_file()

    # Print information about the model
    print(f"Type of model: {type(model)}")
    print(f'Number of reactions: {len(model.reactions)}')
    print(f'Number of metabolites: {len(model.metabolites)}')
    print(f'Number of compartments: {len(model.compartments)}')

    # Adjust the bounds of some reactions in the model
    model.reactions.get_by_id('biomass_maintenance').bounds = 0, 0
    model.reactions.get_by_id('biomass').bounds = .5*model.optimize().objective_value, 100

    # Set the objective of the model
    model.objective = 'r0399'
    model.reactions.get_by_id("r0399").bounds = (0, 1e5)
    model.reactions.get_by_id("PHETHPTOX2").bounds = (0, 0.0)

    # Turn off some reactions in the model
    metabolites_to_turn_off = ["tyr_L_e", "dhbpt_e", "thbpt_e"]
    for metabolite in metabolites_to_turn_off:
        list(map(lambda rxn: turn_off_rxn(model, rxn), [r.id for r in model.metabolites.get_by_id(metabolite).reactions]))

    # Obtain regulatory reactions and set their bounds
    regulatory_reactions = model.optimize().reduced_costs.loc[lambda x: abs(x)>1e-12].index.tolist()
    regulatory_reactions = list(itertools.compress(regulatory_reactions, np.invert(abs(np.array([list(model.reactions.get_by_id(rr).bounds) for rr in regulatory_reactions])).sum(axis=1) == 0)))
    list(map(lambda rxn: set_rxn_bounds(model, rxn), regulatory_reactions))

    # Set up the sampler
    optgp = OptGPSampler(model, processes=os.cpu_count(), thinning=500)

    # Generate samples for healthy condition
    model.reactions.get_by_id("r0399").bounds = (99, model.optimize().objective_value)
    model.reactions.get_by_id("PHETHPTOX2").bounds = (0, 0)
    samples_healthy = optgp.sample(20_000)
    save_to_parquet(samples_healthy, 'flux_samples_CONTROL_20_000b')

    # Generate samples for PKU condition
    model.reactions.get_by_id("r0399").bounds = (0, 5)
    model.reactions.get_by_id("PHETHPTOX2").bounds = (0, 0)
    samples_pku = optgp.sample(20_000)
    save_to_parquet(samples_pku, 'flux_samples_PKU_20_000b')

# Call the main function
if __name__ == "__main__":
    main()




r3 = load('COBRA_models/GEM_Recon3_thermocurated_redHUMAN.mat')

r2 = load('COBRA_models/GEM_Recon2_thermocurated_redHUMAN.mat')



%outmodel = writeCbModel(r3.model)

sub_re3   = struct()
sub_re3.S = r3.model.S
sub_re3.mets = r3.model.mets
sub_re3.b = r3.model.b
sub_re3.csense = r3.model.csense
sub_re3.rxns = r3.model.rxns
sub_re3.lb = r3.model.lb
sub_re3.ub = r3.model.ub
sub_re3.c = r3.model.c
sub_re3.osense = r3.model.osense
sub_re3.genes = r3.model.genes
sub_re3.rules = r3.model.rules


sub_re3.metNames = r3.model.metNames
sub_re3.grRules = r3.model.grRules
sub_re3.rxnGeneMat = r3.model.rxnGeneMat
sub_re3.rxnNames = r3.model.rxnNames


sub_re3.subSystems = r3.model.subSystems




%sub_re3. = r3.model.


model = sub_re3

save COBRA_models/GEM_Recon3_thermocurated_redHUMAN_AA.mat model



r3AA = load('COBRA_models/GEM_Recon3_thermocurated_redHUMAN_AA.mat')


optimizeCbModel(r3AA.model)


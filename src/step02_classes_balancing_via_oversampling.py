
#%%
import pandas as pd
from  sklearn import preprocessing 
import os
parent_dir = os.path.dirname(os.getcwd())
#file_path = os.path.join(parent_dir, "./processed_data/metabolites_outliers_removed_and_imputed.parquet.gzip")

df        = pd.read_parquet('./processed_data/metabolites_outliers_removed_and_imputed.parquet.gzip')

col_names = df.select_dtypes(include=['float']).columns.__array__()
X         = df.select_dtypes(include=['float']).__array__()
le        = preprocessing.LabelEncoder()
y         = le.fit_transform(df.Group.__array__())
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(k_neighbors =40, n_jobs=-1 ).fit_resample(X, y)

aumented_data = pd.concat([ pd.DataFrame(y_resampled.astype(int), columns=['label']), pd.DataFrame(X_resampled, columns=col_names)], axis=1)
aumented_data.to_parquet('./processed_data/augmented_balanced_metabolite_data.parquet.gzip', compression='gzip')



# %%

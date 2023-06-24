
#%%
# Import necessary libraries.

# pandas is a data manipulation library. It lets us work with data in table format, 
# similar to Excel spreadsheets.
import pandas as pd

# sklearn.preprocessing provides a number of utility functions to convert text labels into 
# numeric form, making it easier for machine learning algorithms to work with such data.
from sklearn import preprocessing

# imblearn (Imbalanced-learn) is a Python library offering a number of resampling techniques 
# to adjust the class distribution of a dataset. Here we use SMOTE for oversampling the minority class.
from imblearn.over_sampling import SMOTE


# Define the file path for the input file.
# A parquet file is a columnar storage file format. The format is optimized for speed 
# and efficiency, hence preferred in Big Data environments.
read_file_path = './processed_data/metabolites_outliers_removed_and_imputed.parquet.gzip'

# Read the input DataFrame from the parquet file.
df = pd.read_parquet(read_file_path)


# Ensure the DataFrame contains the 'Group' column. If not, it's a critical error and the script should stop.
if 'Group' not in df.columns:
    print("'Group' column not found in the DataFrame.")
    exit()


# Identify all feature column names that are of type float.
# These will be the features used for our machine learning model.
col_names = df.select_dtypes(include=['float64']).columns.tolist()


# Create the feature matrix, X.
# We convert the DataFrame into a numpy array using the .values attribute.
# X is the standard notation for the feature matrix.
X = df[col_names].values


# Initialize an instance of LabelEncoder.
# LabelEncoder is a utility class to convert categorical labels into numerical form.
le = preprocessing.LabelEncoder()


# Encode the 'Group' column into numerical labels.
# The fit_transform function both fits the encoder to the data (i.e., learns the categories)
# and transforms the data in a single step.
# We store these numerical labels in y, the standard notation for the target vector.
y = le.fit_transform(df['Group'])


# Initialize an instance of SMOTE for oversampling the minority class.
# k_neighbors is the number of nearest neighbors to use for creating synthetic samples.
# n_jobs=-1 means using all available CPU cores for this task.
sm = SMOTE(k_neighbors=40, n_jobs=-1)


# Use SMOTE to balance the class distribution in our dataset.
# fit_resample is a function that fits the SMOTE instance to the data and returns a resampled version of the data.
X_resampled, y_resampled = sm.fit_resample(X, y)


# Create a new DataFrame from the resampled data.

# First, create a DataFrame from y_resampled and name its column 'label'.
# Then, create a DataFrame from X_resampled using the original column names from the input DataFrame.
# Finally, concatenate these two DataFrames along the column axis (axis=1).
augmented_data = pd.concat([pd.DataFrame(y_resampled, columns=['label']), pd.DataFrame(X_resampled, columns=col_names)], axis=1)


# Define the file path for the output file.
write_file_path = './processed_data/augmented_balanced_metabolite_data.parquet.gzip'

# Save the DataFrame as a parquet file using gzip compression.
# Gzip is a file format and a software application used for file compression and decompression.
augmented_data.to_parquet(write_file_path, compression='gzip')

# %%

#%%
# Imports
import os # For file and saving shit.
import tarfile # for the data (tar file crap).
from six.moves import urllib # For the http requests.

import pandas as pd # For the csv.
from pandas.plotting import scatter_matrix # For some graphs.
import matplotlib.pyplot as plt # For displaying the graphs.
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit # Sampling.

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# FETCH THE DATA
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data()
housing = load_housing_data()



# CREATE TRAINING DATA:
# Create a new income category based on the median_income:
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)

# Merge all the categories greater than 5:
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, text_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[text_index]

strat_test_set["income_cat"].value_counts() / len(strat_test_set)

# Remove the income category from the data sets.
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_vat", axis=1, inplace=True)
# %%

# Create a copy of the training set -> this is so we can play with it without harming the training set.
housing_test = strat_train_set.copy()

#Display the data in a scatter graph and change the alpha to see which areas are more concentrated.


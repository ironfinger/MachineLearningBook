#%%
import pandas as pd
import matplotlib.pyplot as plt
import os

data_path = os.path.join('Workshops')

train = pd.read_csv(os.path.join(data_path, 'regression_train(1).csv'))

train.head()

train_x = train['x']
train_y = train['y']


# %%

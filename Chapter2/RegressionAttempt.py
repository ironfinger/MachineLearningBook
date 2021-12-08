#%%
import pandas as pd
import matplotlib.pyplot as plt
import os

data_path = os.path.join('Workshops')

train = pd.read_csv('student-mat.csv')

train.head()

# We are looking for the fare they paid based on their age and see if there a correlation.


# %%

going_out = train['goout']
grade3 = train['freetime']

plt.plot(going_out, grade3, 'b.')
plt.show()
# %%

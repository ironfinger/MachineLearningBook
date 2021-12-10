#%%
import pandas as pd
import matplotlib.pyplot as plt
import os

data_path = os.path.join('Workshops')

train = pd.read_csv('Salary_Data.csv')

train.head()

# We are looking for the fare they paid based on their age and see if there a correlation.


# %%

years_experience = train['YearsExperience']
salary = train['Salary']

plt.plot(years_experience, salary, 'b.')
plt.show()
# %%
# Import lingal:
from numpy import linalg

theta_best = linalg.inv(years_experience.dot(years_experience)).dot(years_experience.dot(salary))

print('theta_best: ', theta_best)
# %%


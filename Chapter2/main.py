"""
Linear regression
"""
#%%

""" Imports """
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

#%%
# Lets generate some linear looking data:
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

plt.plot(X, y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()

# %%
"""
- Now we need to compute the beta using the normal equation.
- We will need to use the inv() function to compute the inversoe of a matrix,
    and then the dot method for matrix multiplication.
"""

X_b = np.c_[np.ones((100, 1)), X] # Add x0 = 1 to each instance.
theta_best = linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)
# %%

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance.
y_predict = X_new_b.dot(theta_best)
print(y_predict)

# Lets plot the prediction:
plt.plot(X_new, y_predict, 'r-')
plt.plot(X, y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()

# %%

# %%

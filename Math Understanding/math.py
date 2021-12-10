"""
The book says we need to understand what the:
- Vectors.
- Matrices.
- Dot product.
- Impartial Differentiation.
"""

#%%

"""
VECTORS:

A vector is quantity defined by a magnitude and direction.
For example a rockets velocity is a 3 dimensional vector: its magnitude is the speed of the rocket,
and its difrection is (hopefully) up. A vector can be represented by an array of numbers called scalars.
Each scalar corresponds to the magnitude of the vector and with regards to each dimension.
"""

"""
Examples:

1. Lets say a rocket is going up at a slight angle: it has a vertical speed of 5,000 m/s, 
   and also a slight speed towards the east at 10m/s and a slight speed towards the north at 50m/s.
   The rockets velocity could be represented as so:

   velocity = {
       10,
       50,
       5000
   }

   NOTE: by convention vectors are generally presented in a form of columns.
         vector names are also lower case to distinguish them from matrices.
"""


"""
Their purpose:

Vectors have many purposes in Machine learning, most notably to represent observations and predictions.

For example: lets say we wanted to build a system to classify videos based on 3 categories.
             This could be represented like so.

    video = {
        10.5,
        5.2,
        3.25,
        7.0
    }

    This could represent the playtime as 10.5 mins, 5.2% viewers watch longer than a minute, gets 3.25 views ect.
"""

#%%
"""
Vectors in python:
"""

# This is the simplest way to represent a vector in python:
simple_vector = [10.5, 5.2, 3.25, 7.0]

# However if we plan to do a lot of scientific calculations, it is much better to use NumPy arrays.
# These provide you with a lot of convenient and optimized implementations of essential mathematical operations on vectors.
from matplotlib import lines
import numpy as np
video = np.array([10.5 , 5.2, 3.25, 7.0])
print('Numpy video vector: ', video)

# The size of the vector can be obtained using the size attribute:
video.size

# In order to access the 'i'th element of a vector 'v' is noted 'vi' (the i is on the bottom right).

# %%
"""
Plotting vectors:
"""

# Imports
import matplotlib.pyplot as plt

# Lets create two bery simple 2D vectors to plot.
v_a = np.array([2, 5])
v_b = np.array([3, 1])

x_coords, y_coords = zip(v_a, v_b)
plt.scatter(x_coords, y_coords, color=['r', 'b'])
plt.axis([0, 9, 0, 6])
plt.grid()
plt.show()

# %%
"""
Vectors are arrows:
"""

def plt_vector2d(vector2d, origin=[0, 0], **options):
    return plt.arrow(origin[0], origin[1], vector2d[0], vector2d[1],
              head_width=0.2, head_length=0.3, length_includes_head=True,
              **options)

# Now lets draw the vectors
plt_vector2d(v_a, color='r')
plt_vector2d(v_b, color='b')
plt.axis([0, 9, 0, 6])
plt.grid()
plt.show()

# %%
"""
3D vector
"""

a = np.array([1, 2, 8])
b = np.array([5, 6, 3])

from mpl_toolkits.mplot3d import Axes3D

subplot3d = plt.subplot(111, projection='3d')
x_coords, y_coords, z_coords = zip(a,b)
subplot3d.scatter(x_coords, y_coords, z_coords)
subplot3d.set_zlim3d([0, 9])
plt.show()

def plot_vectors3d(ax, vectors3d, z0, **options):
    for v in vectors3d:
        x, y, z = v
        ax.plot([x,x], [y,y], [z0, z], color="gray", linestyle='dotted', marker=".")
    x_coords, y_coords, z_coords = zip(*vectors3d)
    ax.scatter(x_coords, y_coords, z_coords, **options)

subplot3d = plt.subplot(111, projection='3d')
subplot3d.set_zlim([0, 9])
plot_vectors3d(subplot3d, [a,b], 0, color=("r","b"))
plt.show()
# %%
"""
Vector Norm:

This is the measure of the length (a.k.a the magnitude) of the vector. We will be using the euclidian norm

sqrRoot(Sum(u^2))

"""

def vector_norm(vector):
    squares = [element**2 for element in vector]
    return sum(squares)**0.5

print("||", 'u', "|| =", vector_norm(a))

# However it is much more efficient to use Numpys norm function, available in linalg:
import numpy.linalg as LA
LA.norm(v_a)

"""
Lets plot a graph to confirm that the length of vector is indeed 5.4:
"""

radius = LA.norm(v_a)
plt.gca().add_artist(plt.Circle((0,0), radius, color="#DDDDDD"))
plt_vector2d(v_a, color="red")
plt.axis([0, 8.7, 0, 6])
plt.grid()
plt.show() # Dunno why the arrow isn't showing.
# %%
"""
ADDITION:

Vectors of the same size can be added together, addition is performed elementwise:
"""

print('v_a: ', v_a)
print('+')
print('v_b: ', v_b)
print('-'*10)
print(v_a + v_b)

# Lets look at vector addision graphically:
plt_vector2d(v_a, color='r')
plt_vector2d(v_b, color='b')
plt_vector2d(v_b, origin=v_a, color='b', linestyle='dotted')
plt_vector2d(v_a, origin=v_b, color='r', linestyle='dotted')
plt_vector2d(v_a + v_b, color='g')
plt.axis([0, 9, 0, 7])
plt.text(0.7, 3, "v_a", color="r", fontsize=18)
plt.text(4, 3, "v_a", color="r", fontsize=18)
plt.text(1.8, 0.2, "v_b", color="b", fontsize=18)
plt.text(3.1, 5.6, "v_b", color="b", fontsize=18)
plt.text(2.4, 2.5, "v_a + v_b", color="g", fontsize=18)
plt.grid()
plt.show()

# %%

"""
Vector Addition is commutative, meaning that u + v = v + u. 
Vector addition is also associative meaning that u + (v + w) = (u + v) + w

If you have a shape defined by a number of (vectors), and you add a vector v to all of these points, 
then the whole shape gets shifted by v. This is called a geometric translaion.
"""
import numpy as np

t1 = np.array([2, 0.25])
t2 = np.array([2.5, 3.5])
t3 = np.array([1, 2])

x_coords, y_coord = zip(t1, t2, t3, t1)
plt.plot(x_coords, y_coords, "c--", x_coords, y_coords, "co") # Hmmmmmmmmm
plt_vector2d(v_b, t1, color='r', linestyle=':')
plt_vector2d(v_b, t2, color='r', linestyle=':')
plt_vector2d(v_b, t3, color='r', linestyle=':')

t1b = t1 + v_b
t2b = t2 + v_b
t3b = t3 + v_b

x_coord_b, y_coords_b = zip(t1b, t2b, t3b, t1b)
plt.plot(x_coord_b, y_coords_b, 'b-', x_coord_b, y_coords_b, 'bo')
plt.text(4, 4.2, "v", color="r", fontsize=18)
plt.text(3, 2.3, "v", color="r", fontsize=18)
plt.text(3.5, 0.4, "v", color="r", fontsize=18)

plt.axis([0, 6, 0, 5])
plt.grid()
plt.show()

# %%
"""
Multiplication by scalar.

Vectors can be multiplied by scalars. All elements in a vector are multiplied by that number.

"""

print("1.5 *", v_a, "=")
print(1.5 * v_a)

k = 2.5
t1c = k * t1
t2c = k * t2
t3c = k * t3


plt_vector2d(t1, color="r")
plt_vector2d(t2, color="r")
plt_vector2d(t3, color="r")

x_coords_c, y_coords_c = zip(t1c, t2c, t3c, t1c)
plt.plot(x_coords_c, y_coords_c, "b-", x_coords_c, y_coords_c, "bo")

plt_vector2d(k * t1, color="b", linestyle=":")
plt_vector2d(k * t2, color="b", linestyle=":")
plt_vector2d(k * t3, color="b", linestyle=":")

plt.axis([0, 9, 0, 9])
plt.grid()
plt.show()

# %%

"""

Zero, unit and normalized vectors
* A Zero-vector is a vector full of 0s
* A unit vector is a vector with a norm equal to 1.
* The normalized vector of a non-null vector u. noted u^ is the uni vector that points in the same direction as u.
"""
plt.gca().add_artist(plt.Circle((0,0),1,color='c'))
plt.plot(0, 0, "ko")
plt_vector2d(v_b / LA.norm(v_b), color="k")
plt_vector2d(v_b, color="b", linestyle=":")
plt.text(0.3, 0.3, "$\hat{u}$", color="k", fontsize=18)
plt.text(1.5, 0.7, "$u$", color="b", fontsize=18)
plt.axis([-1.5, 5.5, -1.5, 3.5])
plt.grid()
plt.show()

# %%
"""
Dot Product

The dot product (also called scalar product or inner product in the context of the Euclidian space)

The dot product of two vectors u and v is a useful operation that comes up fairly often in linear algebra.
"""

def dot_product(v1, v2):
    return sum(v1i * v2i for v1i, v2i in zip(v1, v2))

dot_product(v_a, v_b)

# A more effienct way is:
np.dot(v_a, v_b)

# Or

v_a.dot(v_b)

# Caution the * operator will perform and element wise multiplication, NOT a dot product.

"""
Main properties:
* The dot product is commutative: u . v = v . u
* The dot product is only defined between two vectors, not between a scalar and a vector.
  This means that we cannoed chain the dot products: v . u . w 

* This also means that the dot product is NOT associative.
* However, the dot product is associative with regards to scalar multiplication.
* Finaly the dot product is distributive over addition of vectors.
"""

# %%

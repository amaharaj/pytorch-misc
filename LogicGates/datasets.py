import torch
import matplotlib.pyplot as plt
import numpy as np


# Generate random samples on interval [a,b) x [a,b)

# Sampling parameters
nsamples = 100  # Number of samples to be drawn
a = -5.0        # Left bound
b = 5.0         # Right bound

# Draw the numbers
X = (b-a)*torch.rand(nsamples,2) + a

# Assign a value of +1 or -1 based on XNOR rules
value = torch.sign(X[:,0]*X[:,1])
value = value.reshape(nsamples,1) # Reshape from (nsamples,) to (nsamples,1) 
X = torch.cat((X,value), dim=1) 

# Plot the results
mapping = {-1:"red", 1:"blue", 0 :"green"} # Values of 0 are unlikely but possible
my_cmap = [mapping[xi] for xi in X[:,2].numpy()] # assign colours to values for each data point
plt.scatter(X[:,0].numpy(), X[:,1].numpy(), c=my_cmap ) 
plt.show()


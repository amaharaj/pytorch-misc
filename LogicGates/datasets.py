import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

class Dataset:

   def __init__(self, nsamples, bounds, kind):

      self.nsamples = nsamples
      self.a = bounds[0]
      self.b = bounds[1]
      self.kind = kind

   def X_data(self):
      """ 
      Generates random samples on interval [a,b) x [a,b) 
      """
      X = (self.b-self.a) * torch.rand(self.nsamples, 2) + self.a  # Draw the numbers
      return X

   def Y_data(self, X):
      """
      Generates ground truth output values for dataset
     """
     # X = self.X_data()  # Generate inputs

      # Simplify logical comparisions by taking the sign of each input
      A = torch.sign(X[:,0])
      B = torch.sign(X[:,1])

      value = torch.empty((self.nsamples,1), dtype=torch.int32) # initialize tensor for truth values

      # Assign a value of +1 or -1 based on gate type
      
      if self.kind == "XNOR":
         for n in range(self.nsamples):
            value[n] = int(((A[n]>0) and (B[n]>0)) or (not(A[n]>0) and not(B[n]>0)) )*2 - 1

      elif self.kind == "XOR":
         for n in range(self.nsamples):
            value[n] = int(((A[n]>0) and (B[n]<0)) or ((A[n]<0) and (B[n]>0)))*2 - 1

      elif self.kind == "NOR":
         for n in range(self.nsamples):
            value[n] = int((not (A[n]>0) and not (B[n]>0)))*2 - 1

      elif self.kind == "OR":
         for n in range(self.nsamples):
            value[n] = int((not (A[n]>0) and not (B[n]>0)) or (not (A[n]>0) and (B[n]>0) ) or ( (A[n]>0) and not (B[n]>0) ) )*2 - 1

      elif self.kind == "NAND":
         for n in range(self.nsamples):
            value[n] = int((not (A[n]>0) and not (B[n]>0)) or (not (A[n]>0) and (B[n]>0) ) or ( (A[n]>0) and not (B[n]>0) ) )*2 - 1

      elif self.kind == "AND":
         for n in range(self.nsamples):
            value[n] = int( (A[n]>0) and (B[n]>0)) *2 - 1 

      return value

   def plot(self):
      """
      Plot datapoints and assign a colour to output values -1 or 1 (0 in the unlikely case that this datapoint 
      was selected during random initialization)  
      """
      X = self.X_data()
      value = self.Y_data(X)

      # Plot the results
      mapping = {-1:"red", 1:"blue", 0 :"green"}             # Values of 0 are unlikely but possible
      my_cmap = [mapping[xi] for xi in value[:,0].numpy()]   # Assign colours to values for each data point
      plt.scatter(X[:,0].numpy(), X[:,1].numpy(), c=my_cmap)
      plt.title(self.kind)
      plt.xlabel("X1")
      plt.ylabel("X2")
      plt.axhline(0, c='gray', ls='dashed')  # Add lines on axes to make it easier to see quadrants:
      plt.axvline(0, c='gray', ls='dashed') 

      labels = [Line2D([0], [0], marker = 'o', c='blue', label='True'),  # Add custom legend 
                Line2D([0], [0], marker = 'o', c='red', label='False')]

      plt.legend(handles=labels, loc='upper right')
      plt.show()

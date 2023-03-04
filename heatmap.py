import numpy as np
import matplotlib.pyplot as plt

# Define the data as a list
data = [1, 2, 3, 4, 5, 6, 7, 8, 10]

# Convert the list to a 2D numpy array
data_array = np.reshape(data, (3, 3))
heatmap = plt.imshow(data_array, cmap='Spectral')
plt.colorbar()
plt.show()
plt.savefig('plot1.png')

b = data_array.transpose()
heatmap = plt.imshow(b, cmap='Spectral')
plt.colorbar()
plt.savefig('plot2.png')

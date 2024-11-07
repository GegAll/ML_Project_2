import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import FastICA

#  load a data example
data = loadmat("data/000.mat")

# Get the values from the mat file
values = data['val']

# Create the X and y datasets
y = values[0][0]
X = np.linspace(0, len(y), len(y))


# Plot them
plt.figure()
plt.plot(X, y)
plt.show()


for i in range(0, 4):
    plt.figure(figsize=(14, 6))
    plt.plot(X, values[0][i])
    plt.title(f"Channel: {i+1}")
    plt.xlim([0, 400])
    plt.show()

# apply ICA on data
ica = FastICA()
result = np.transpose(ica.fit_transform(np.transpose(values[0])))
print(result.shape)

for i in range(0, 4):
    plt.figure(figsize=(14, 6))
    plt.plot(X, result[i])
    plt.title(f"Channel: {i+1}")
    plt.xlim([0, 1400])
    plt.show()
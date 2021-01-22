from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

iris = load_iris()
X = iris.data
# print(X)
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X)

print(X.shape)			# (150, 4)
print(reduced_X.shape) 	# (150, 2)
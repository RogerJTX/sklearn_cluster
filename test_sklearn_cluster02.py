from sklearn import datasets
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


iris = datasets.load_iris()
# target______1 = iris.target
# print(target______1)

print(iris)
original_x = iris.data
print(original_x)

# 特征转换，使用加和的方式将样本空间从4维转换为2维（以方便在2维平面上展示结果）
datas = original_x[:, :2] + original_x[:, 2:]
print(datas)

# 设定聚类模型参数，并进行训练
kmeans = KMeans(init="k-means++", n_clusters = 3)
kmeans.fit(datas)  
print(kmeans)

# 在kmeans中就包含了K均值聚类的结果：聚类中心点和每个样本的类别
labels = kmeans.labels_ 
centers = kmeans.cluster_centers_

print('--------------------------------------------------------')



#计算每一类到其中心距离的平均值，作为绘图时绘制圆圈的依据
distances_for_labels = []

for label in range(kmeans.n_clusters):
    distances_for_labels.append([])

for i, data in enumerate(datas):
    label = kmeans.labels_[i]
    center = kmeans.cluster_centers_[label]
    distance = np.sqrt(np.sum(np.power(data - center, 2)))
    distances_for_labels[label].append(distance)

ave_distances = [np.average(distances_for_label) for distances_for_label in distances_for_labels]

#绘图
fig, ax = plt.subplots()
ax.set_aspect('equal')
#设置坐标范围
x_min = 1
x_max = 20
y_min = 1
y_max = 20
ax.set_xlim((x_min, x_max))
ax.set_ylim((y_min, y_max))

#绘制每个Cluster
for label, center in enumerate(kmeans.cluster_centers_):
    radius = ave_distances[label] * 1.5
    ax.add_artist(plt.Circle(center, radius = radius, color = "r", fill = False))

#根据每个数据的真实label来选择数据点的颜色
plt.scatter(datas[:, 0], datas[:, 1], c = iris.target)
plt.show()

print(kmeans.cluster_centers_)  # 查看质心
inertia=kmeans.inertia_
print(inertia)




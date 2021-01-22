import numpy as np
import matplotlib.pyplot as mp
import sklearn.cluster as sc

# 读取数据，绘制图像
x = np.loadtxt('123.txt', unpack=False, dtype=int)
print(x.shape)

# 基于Agglomerativeclustering完成聚类
model = sc.AgglomerativeClustering(n_clusters=4)
pred_y = model.fit_predict(x)
print(pred_y)

# 画图显示样本数据
mp.figure('Agglomerativeclustering', facecolor='lightgray')
mp.title('Agglomerativeclustering', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
mp.scatter(x[:, 0], x[:, 1], s=80, c=pred_y, cmap='brg', label='Samples')
mp.legend()
mp.show()

import re
from bs4 import BeautifulSoup
import logging
import pymongo
import base64
from goose3 import Goose
from goose3.text import StopWordsChinese
import urllib
import time, requests
import datetime, random
import chardet
import pandas as pd
import matplotlib
# import matplotlib.pyplot as plt
import jieba as jb
import re
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import chardet
import urllib.request
# from boilerpipe.extract import Extractor
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn import datasets
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import operator
from functools import reduce
import jieba
import gensim
from gensim.models.doc2vec import Doc2Vec
list_test = []
with open('D:\Python\python_code\Liangzhi\TianPengTrans-tmp\etl\pytorch/test2.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if len(line) > 2 :
            list_test.append(line.replace('\n','').replace('\r','').replace(' ','').replace('   ',''))
            print(line)
            # dict[c] = line
            # c += 1
        else:
            pass

TaggededDocument = gensim.models.doc2vec.TaggedDocument

# list_line = tag.find_all('p')

# c = 0


list_test_indexed = []
for i, text in enumerate(list_test):
    document = TaggededDocument(text, tags=[i])
    list_test_indexed.append(document)
    print(document)
# c = 0
dict = {}



# print(list_test)
# print(dict)
print('-----------------------------------------------')
count_vect = CountVectorizer()

def myPredict(sec):
    # format_sec = " ".join([w for w in list(jb.cut(sec))])
    # print(format_sec)
    # TaggededDocument = gensim.models.doc2vec.TaggedDocument


    # def ceshi(sec):
    #     model_dm = Doc2Vec.load("model_word2vec")
    #     ##此处需要读入你所需要进行提取出句子向量的文本   此处代码需要自己稍加修改一下
    #     ##你需要进行得到句子向量的文本，如果是分好词的，则不需要再调用结巴分词
    #     test_text = [w for w in list(jieba.cut(sec))]
    #
    #     inferred_vector_dm = model_dm.infer_vector(test_text)  ##得到文本的向量
    #     print(inferred_vector_dm)
    #
    #     return inferred_vector_dm
    #
    #
    # doc_2_vec = ceshi('2018年03月23日晚上大概十一点多钟我和张三骑着摩托车从住处出门想看看有什么能吃的东西.')
    # print(type(doc_2_vec))
    # print(doc_2_vec.shape)


    vector = count_vect.fit_transform(sec)
    print(type(vector))
    output1 = vector.toarray()
    print(type(output1))


    # 特征转换，使用加和的方式将样本空间从4维转换为2维（以方便在2维平面上展示结果）
    # output = output1[:, :2] + output1[:, 2:]
    pca = PCA(n_components=2)
    output = pca.fit_transform(output1)


    return output

output = myPredict(list_test)
print(output)

# 设定聚类模型参数，并进行训练
kmeans = KMeans(init="k-means++", n_clusters=3)
kmeans.fit(output)
print(kmeans)

# 在kmeans中就包含了K均值聚类的结果：聚类中心点和每个样本的类别
labels = kmeans.labels_
centers = kmeans.cluster_centers_   # 聚类中心点
print(2222)
print(centers)
print(2222)

print('--------------------------------------------------------')

# 计算每一类到其中心距离的平均值，作为绘图时绘制圆圈的依据
distances_for_labels = []

for label in range(kmeans.n_clusters):
    distances_for_labels.append([])

m = {}

for i, data in enumerate(output):
    # print(i)
    list(enumerate(output))
    print(list(enumerate(output))[i])
    label = kmeans.labels_[i]


    center = kmeans.cluster_centers_[label]
    distance = np.sqrt(np.sum(np.power(data - center, 2)))
    distances_for_labels[label].append(distance)

ave_distances = [np.average(distances_for_label) for distances_for_label in distances_for_labels]
print(1111)
print(ave_distances)
print(1111)


# 绘图
fig, ax = plt.subplots()
ax.set_aspect('equal')
# 设置坐标范围
x_min = -10
x_max = 10
y_min = -10
y_max = 10
ax.set_xlim((x_min, x_max))
ax.set_ylim((y_min, y_max))


# 绘制每个Cluster
for label, center in enumerate(kmeans.cluster_centers_):
    radius = ave_distances[label] * 1.5

    ax.add_artist(plt.Circle(center, radius=radius, color="r", fill=False))


plt.scatter(output[:, 0], output[:, 1], c='b')
plt.show()


inertia = kmeans.inertia_
print(inertia)

    




# df_cat = pd.DataFrame(data=list_test).reset_index(drop=True)
# print(df_cat)
# print(type(df_cat))





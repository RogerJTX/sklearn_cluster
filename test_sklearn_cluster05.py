from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

vecizer =  CountVectorizer()

corpus = [   'This is the first document.','This is the second second document.','And the third one.','Is this the first document?']

X = vecizer.fit_transform(corpus)

output = X.toarray()

print(output)


print('----------------------------')

corpus = ['你好',

             '大家一起去吃饭']

X = vecizer.fit_transform(corpus)

output = X.toarray()

print(output)

print('----------------------------')

corpus = ['2011.01 2011年广东省科学技术奖二等奖：《蓝盾网络安全云关键技术》 赵淦森（5） ',
          '2011.01 2011年广州市科学技术奖二等奖：《蓝盾网络安全云关键技术》 赵淦森（5）',
          '2004 IFIP通信与多媒体安全学术会议，任组织委员会委员，2004（英国）',
          '2009.12-2009.12,第一届云计算国际会议(The First International Conference on Cloud Computing, 2009, Beijing China)大会程序主席']

X = vecizer.fit_transform(corpus)

output = X.toarray()

print(output)

print('----------------------------')

corpus = ['hello word']

X = vecizer.fit_transform(corpus)

output = X.toarray()

print(output)

print('----------------------------')

corpus = ['hello word']

X = vecizer.fit_transform(corpus)

output = X.toarray()

print(output)
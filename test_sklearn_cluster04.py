# 现在有一个任务，需要判断一句话是否为脏话。假定训练集和测试集如下：
#
# 训练集
# 是脏话：fuck you
# 是脏话：fuck you all
# 不是脏话：hello everyone
#
# 测试集：
# fuck boy
# hello girl

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
X = []

# 输入训练样本
X.append("傻子")
X.append("笨蛋")
X.append("你好")
print(X)
# y为训练样本标注
y = [1,1,0]

# 输入测试样本
T = ["你真傻","你好"]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X)
T_test = vectorizer.transform(T)

# 用逻辑回归模型做训练
classifier = LogisticRegression()
classifier.fit(X_train, y)

# 做测试样例的预测
predictions = classifier.predict(T_test)
print(predictions)  # [1,0]
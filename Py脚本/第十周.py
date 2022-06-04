import json
import re

import jieba
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


class Manipulate_File(object):
    def __init__(self):
        self.file = "../data/all_data.json"
        #  初始化各变量
        self.txt = []
        self.label = []
        self.real_url = []
        self.fake_url = []
        '''
        self.num1 = 0  # 开始读取文件的开始行号
        self.num2 = 0  # 结束读取文件的行号
        '''
        self.counter_real = 0
        self.counter_fake = 0

        self.data = self.read_file()  # 用json解析文件
        self.clean_and_add()  # 简化文件，并更新各变量

        # self.test_size= self.best_parameter()
        # self.stop_words = []

    def read_file(self):  # 读取文件
        with open(self.file, mode='r', encoding="UTF-8") as data:
            return json.load(data)

    def clean_and_add(self):  # 将数据转化成词袋模型
        for line in self.data:
            clean_text = re.sub('\W*', '', line['text'])  # 数据清理
            token = jieba.cut(clean_text)  # 中⽂分词
            final_token = " ".join(token)
            self.txt.append(final_token)
            length = len(line['pic_url'])
            self.label.append(line['label'])
            if line['label'] == 'fake':
                self.fake_url.append(length)
                self.counter_fake += 1
            else:
                self.real_url.append(length)
                self.counter_real += 1

        """
        print(self.txt)
        print(self.label)
        """

    def pancake(self):  # 画出饼图
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        fig, ax = plt.subplots(1, 1)
        y = ['real-data', 'fake-data']
        x = [self.counter_real, self.counter_fake]
        fig.set_size_inches(5, 5)
        ax.pie(x, labels=y)
        plt.legend()
        plt.show()

    def bar(self):  # 画出直方图

        fig1, ax1 = plt.subplots(1, 2, sharex=True, tight_layout=True)
        fig1.set_size_inches(6, 6)
        colors = ['red', 'green']
        ax1[0].hist(self.real_url, color=colors[0])
        ax1[1].hist(self.fake_url, color=colors[1])

        ax1[0].set_title("real-data")
        ax1[0].set_xlabel('urls of real')
        ax1[0].set_ylabel('amount of real')

        ax1[1].set_title("fake-data")
        ax1[1].set_xlabel('urls of fake')
        ax1[1].set_ylabel('amount of fake')
        plt.show()

    def best_parameter(self):  # 尝试找出最好的test_size，并将其返回
        dic = {}

        # np.arange(0.1, 1.0,0.1)建立一个从0.1到1.0，步长为0.1的列表
        for decimal in np.arange(start=0.1, stop=1.0, step=0.1):
            x_train, x_test, y_train, y_test = train_test_split(self.txt, self.label, test_size=decimal,
                                                                random_state=2023)
            pipe = Pipeline([('vect', CountVectorizer()),
                             ('model', LinearSVC())])
            # 模型选择为LinearSVC
            model = pipe.fit(x_train, y_train)
            score = round(model.score(x_test, y_test) * 100, 2)
            dic[decimal] = score
        dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)  # 经典集合排序
        return dic[0][0]

    def all_models(self):
        models = ["svm", "nb", "dt", "sgd", "rf", "gb", "kn", "lg"]
        for i in models:
            x_train, x_test, y_train, y_test = train_test_split(
                self.txt, self.label, test_size=.2, random_state=2023)
            print(i)
            if i == "svm":
                pipe = Pipeline(
                    [('vect', TfidfVectorizer()), ('model', LinearSVC())])
            elif i == "nb":
                pipe = Pipeline([('vect', TfidfVectorizer()),
                                 ('model', MultinomialNB())])
            elif i == "dt":
                pipe = Pipeline([('vect', TfidfVectorizer()),
                                 ('model', DecisionTreeClassifier())])
            elif i == "sgd":
                pipe = Pipeline([('vect', TfidfVectorizer()),
                                 ('model', SGDClassifier())])
            elif i == "rf":
                pipe = Pipeline([('vect', TfidfVectorizer()),
                                 ('model', RandomForestClassifier())])
            elif i == "gb":
                pipe = Pipeline([('vect', TfidfVectorizer()),
                                 ('model', GradientBoostingClassifier())])
            elif i == "kn":
                pipe = Pipeline([('vect', TfidfVectorizer()),
                                 ('model', KNeighborsClassifier())])
            else:
                pipe = Pipeline([('vect', TfidfVectorizer()),
                                 ('model', LogisticRegression())])
            model = pipe.fit(x_train, y_train)
            score = round(model.score(x_test, y_test) * 100, 2)
            print('模型精准度 : {}%'.format(score))

    def TF_IDF(self):  # 两种模型拟合数据
        new_txt = self.txt[40:60]
        print(new_txt)
        new_label = self.label[40:60]
        print(new_label)

        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(new_txt)
        vectorizers = ["tf-idf", "bow"]
        for i in vectorizers:
            x_train, x_test, y_train, y_test = train_test_split(
                self.txt, self.label, test_size=0.2, random_state=2023)
            print(i)
            if i == "tf-idf":
                pipe = Pipeline(
                    [('vect', TfidfVectorizer()), ('model', LinearSVC())])
            else:
                pipe = Pipeline(
                    [('vect', CountVectorizer()), ('model', LinearSVC())])
            model = pipe.fit(x_train, y_train)
            score = round(model.score(x_test, y_test) * 100, 2)
            print('模型精准度:{}'.format(score))


# 有待优化

Manipulate_File().TF_IDF()

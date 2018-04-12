%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")
plt.rcParams['figure.figsize'] = 16, 12
from tqdm import tqdm_notebook
import pandas as pd
from collections import defaultdict


DS_FILE_NAME = '../../data/stackoverflow_sample_125k.tsv'
TAGS_FILE_NAME = '../../data/top10_tags.tsv'

top_tags = []
with open(TAGS_FILE_NAME, 'r') as f:
    for line in f:
        top_tags.append(line.strip())
top_tags = set(top_tags)
print(top_tags)

class LogRegressor():

    """Конструктор

    Параметры
    ----------
    tags_top : list of string, default=tags_top
        список тегов
    """
    def __init__(self, tags=top_tags):

        self._vocab = {}

        # параметры модели: веса
        # для каждого класса/тега нам необходимо хранить собственный вектор весов
        # по умолчанию у нас все веса будут равны нулю
        # мы заранее не знаем сколько весов нам понадобится
        # поэтому для каждого класса мы сосздаем словарь изменяемого размера со значением по умолчанию 0
        # пример: self._w['java'][self._vocab['exception']]  содержит вес для слова exception тега java
        self._w = dict([(t, defaultdict(int)) for t in tags])

        # параметры модели: смещения или вес w_0
        self._b = dict([(t, 0) for t in tags])

        self._tags = set(tags)


    def iterate_file(self,
                     fname=DS_FILE_NAME,
                     top_n_train=100000,
                     total=125000,
                     learning_rate=0.1,
                     tolerance=1e-16):

        self._loss = []
        n = 0

        # откроем файл
        with open(fname, 'r') as f:

            # прогуляемся по строкам файла
            for line in tqdm_notebook(f, total=total, mininterval=1):
                pair = line.strip().split('\t')
                if len(pair) != 2:
                    continue
                sentence, tags = pair
                # слова вопроса, это как раз признаки x
                sentence = sentence.split(' ')
                # теги вопроса, это y
                tags = set(tags.split(' '))

                # значение функции потерь для текущего примера
                sample_loss = 0

                # прокидываем градиенты для каждого тега
                for tag in self._tags:
                    # целевая переменная равна 1 если текущий тег есть у текущего примера
                    y = int(tag in tags)

                    # расчитываем значение линейной комбинации весов и признаков объекта

                    z = self._b[tag]
                    for word in sentence:
                        # если в режиме тестирования появляется слово которого нет в словаре, то мы его игнорируем
                        if n >= top_n_train and word not in self._vocab:
                            continue
                        if word not in self._vocab:
                            self._vocab[word] = len(self._vocab)
                        # z += ...
                        z += self._w[tag][self._vocab[word]]

                    # вычисляем вероятность наличия тега

                    sigma = 1 / (1 + np.exp(-z)) if z >= 0 else 1 - 1 / (1 + np.exp(z))

                    # обновляем значение функции потерь для текущего примера


                    sample_loss += -y * np.log(np.max([tolerance, sigma])) if y == 1 else \
                                   -(1 - y) * np.log(1 - np.min([1 - tolerance, sigma]))

                    # если мы все еще в тренировочной части, то обновим параметры
                    if n < top_n_train:
                        # вычисляем производную логарифмического правдоподобия по весу

                        # dLdw = ...
                        dLdw = y - sigma

                        # делаем градиентный шаг
                        # мы минимизируем отрицательное логарифмическое правдоподобие (второй знак минус)
                        # поэтому мы идем в обратную сторону градиента для минимизации (первый знак минус)
                        for word in sentence:
                            self._w[tag][self._vocab[word]] -= -learning_rate * dLdw
                        self._b[tag] -= -learning_rate * dLdw

                n += 1

                self._loss.append(sample_loss)

# создадим эксемпляр модели и пройдемся по датасету
model = LogRegressor()
model.iterate_file()
plt.plot(pd.Series(model._loss[:-25000]).rolling(10000).mean());


#прогноз тегов новых вопросов
class LogRegressor():

    def __init__(self, tags=top_tags):
        self._vocab = {}
        self._w = dict([(t, defaultdict(int)) for t in tags])
        self._b = dict([(t, 0) for t in tags])
        self._tags = set(tags)
        self._word_stats = defaultdict(int)

    def iterate_file(self,
                     fname=DS_FILE_NAME,
                     top_n_train=100000,
                     total=125000,
                     learning_rate=0.1,
                     tolerance=1e-16,
                     accuracy_level=0.9,
                     lmbda=0.0002,
                     gamma=0.1,
                     update_vocab=True):

        self._loss = []
        n = 0
        accuracy = []
        with open(fname, 'r') as f:
            for line in tqdm_notebook(f, total=total, mininterval=1):
                pair = line.strip().split('\t')
                if len(pair) != 2:
                    continue
                sentence, tags = pair
                sentence = sentence.split(' ')
                tags = set(tags.split(' '))

                sample_loss = 0
                predicted_tags = None

                for ix_tag, tag in enumerate(self._tags):
                    y = int(tag in tags)

                    z = self._b[tag]
                    for word in sentence:
                        if n >= top_n_train and word not in self._vocab:
                            continue
                        if word not in self._vocab and update_vocab:
                            self._vocab[word] = len(self._vocab)
                        if word not in self._vocab:
                            continue
                        if update_vocab and ix_tag == 0 and n < top_n_train:
                            self._word_stats[self._vocab[word]] += 1
                        z += self._w[tag][self._vocab[word]]

                    sigma = 1/(1 + np.exp(-z)) if z >= 0 else 1 - 1/(1 + np.exp(z))

                    sample_loss += -y*np.log(np.max([tolerance, sigma])) if y == 1 else \
                                   -(1 - y)*np.log(1 - np.min([1 - tolerance, sigma]))

                    if n < top_n_train:
                        dLdw = y - sigma

                        for word in sentence:
                            if word not in self._vocab:
                                continue
                            self._w[tag][self._vocab[word]] -= -learning_rate * dLdw \
                                + 2 * learning_rate * lmbda * gamma * self._w[tag][self._vocab[word]] \
                                + learning_rate * lmbda *(1 - gamma) * np.sign(self._w[tag][self._vocab[word]])
                        self._b[tag] -= -learning_rate * dLdw
                    else:
                        if predicted_tags is None:
                            predicted_tags = []
                        if sigma > accuracy_level:
                            predicted_tags.append(tag)

                n += 1

                self._loss.append(sample_loss)
                if predicted_tags is not None:
                    accuracy.append(len(tags.intersection(predicted_tags))/len(tags.union(predicted_tags)))

        return(np.mean(accuracy))

    def filter_vocab(self, n=10000):
        keep_words = set([wid for (wid, wn) in sorted(self._word_stats.items(),
                                                      key=lambda t: t[1], reverse=True)[:n]])
        self._vocab = dict([(k, v) for (k, v) in self._vocab.items() if v in keep_words])
        for tag in self._tags:
            self._w[tag] = dict([(k, v) for (k, v) in self._w[tag].items() if k in keep_words])


    def predict_proba(self, sentence):
        p = {}
        sentence = sentence.split(' ')
        for tag in self._tags:
            z = self._b[tag]
            for word in sentence:
                if word not in self._vocab:
                    continue
                z += self._w[tag][self._vocab[word]]
            sigma = 1 / (1 + np.exp(-z)) if z >= 0 else 1 - 1 / (1 + np.exp(z))
            p[tag] = sigma
        return p

model = LogRegressor()
acc = model.iterate_file(update_vocab=True)
print('%0.2f' % acc)
model.filter_vocab(n=10000)
acc = model.iterate_file(update_vocab=False, learning_rate=0.01)
print('%0.2f' % acc)

sentence = ("I want to improve my coding skills, so I have planned write " +
            "a Mobile Application.need to choose between Apple's iOS or Google's Android." +
            " my background: I have done basic programming in .Net,C/C++,Python and PHP " +
            "in college, so got OOP concepts covered. about my skill level, I just know " +
            "concepts and basic syntax. But can't write complex applications, if asked :(" +
            " So decided to hone my skills, And I wanted to know which is easier to " +
            "learn for a programming n00b. A) iOS which uses Objective C B) Android " +
            "which uses Java. I want to decide based on difficulty " +
            "level").lower().replace(',', '')

sorted(model.predict_proba(sentence).items(),
       key=lambda t: t[1], reverse=True)


#регуляризация ElasticNet
class LogRegressor():

    def __init__(self, tags=top_tags):
        self._vocab = {}
        self._w = dict([(t, defaultdict(int)) for t in tags])
        self._b = dict([(t, 0) for t in tags])
        self._tags = set(tags)

    def iterate_file(self,
                     fname=DS_FILE_NAME,
                     top_n_train=100000,
                     total=125000,
                     learning_rate=0.1,
                     tolerance=1e-16,
                     accuracy_level=0.9,
                     lmbda=0.0002,
                     gamma=0.1):

        self._loss = []
        n = 0
        accuracy = []
        with open(fname, 'r') as f:
            for line in tqdm_notebook(f, total=total, mininterval=1):
                pair = line.strip().split('\t')
                if len(pair) != 2:
                    continue
                sentence, tags = pair
                sentence = sentence.split(' ')
                tags = set(tags.split(' '))

                sample_loss = 0
                predicted_tags = None

                for tag in self._tags:
                    y = int(tag in tags)

                    z = self._b[tag]
                    for word in sentence:
                        if n >= top_n_train and word not in self._vocab:
                            continue
                        if word not in self._vocab:
                            self._vocab[word] = len(self._vocab)
                        z += self._w[tag][self._vocab[word]]

                    sigma = 1/(1 + np.exp(-z)) if z >= 0 else 1 - 1/(1 + np.exp(z))

                    sample_loss += -y * np.log(np.max([tolerance, sigma])) if y == 1 else \
                                   -(1 - y) * np.log(1 - np.min([1 - tolerance, sigma]))

                    if n < top_n_train:
                        dLdw = y - sigma

                        r_buf = {}
                        for word in sentence:
                            if word not in r_buf:
                                r = 2 * learning_rate * lmbda * gamma * self._w[tag][self._vocab[word]] + \
                                    learning_rate * lmbda*(1 - gamma) * np.sign(self._w[tag][self._vocab[word]])
                                r_buf[word] = True
                            else:
                                r = 0

                            self._w[tag][self._vocab[word]] -= -learning_rate * dLdw + r
                        self._b[tag] -= -learning_rate * dLdw
                    else:
                        if predicted_tags is None:
                            predicted_tags = []
                        if sigma > accuracy_level:
                            predicted_tags.append(tag)

                n += 1

                self._loss.append(sample_loss)
                if predicted_tags is not None:
                    accuracy.append(len(tags.intersection(predicted_tags))/len(tags.union(predicted_tags)))

        return(np.mean(accuracy))

model = LogRegressor()
acc = model.iterate_file()
print('%0.2f' % acc)
plt.plot(pd.Series(model._loss[:-25000]).rolling(10000).mean());

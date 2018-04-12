import pandas
import numpy as np
from sklearn import datasets, grid_search
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold



newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
X = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(X)

# 3. Подберите минимальный лучший параметр C из множества [10^-5, 10^-4, ... 10^4, 10^5] для SVM с
# линейным ядром (kernel='linear') при помощи кросс-валидации по 5 блокам. Укажите параметр random_state=241 и для SVM,
# и для KFold. В качестве меры качества используйте долю верных ответов (accuracy).

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
model = SVC(kernel='linear', random_state=241)
gs = grid_search.GridSearchCV(model, grid, scoring='accuracy', cv=cv)
gs.fit(vectorizer.transform(X), y)

score = 0
C = 0
for attempt in gs.grid_scores_:
    if attempt.mean_validation_score > score:
        score = attempt.mean_validation_score
        C = attempt.parameters['C']

# 4. Обучите SVM по всей выборке с оптимальным параметром C, найденным на предыдущем шаге.

model = SVC(kernel='linear', random_state=241, C=C)
model.fit(vectorizer.transform(X), y)

# 5. Найдите 10 слов с наибольшим по модулю весом. Они являются ответом на это задание. Укажите их через запятую или
# пробел, в нижнем регистре, в лексикографическом порядке.

words = vectorizer.get_feature_names()
coef = pandas.DataFrame(model.coef_.data, model.coef_.indices)
top_words = coef[0].map(lambda w: abs(w)).sort_values(ascending=False).head(10).index.map(lambda i: words[i])
print (top_words)

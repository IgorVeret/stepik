# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mM4Ua7g4cswrRf_EM7Q44ngMm_AGsqJ8

<p style="align: center;"><img align=center src="https://s8.hostingkartinok.com/uploads/images/2018/08/308b49fcfbc619d629fe4604bceb67ac.jpg" width=500 height=450/></p>

<h3 style="text-align: center;"><b>Школа глубокого обучения ФПМИ МФТИ</b></h3>

<h3 style="text-align: center;"><b>Домашнее задание. Продвинутый поток. Весна 2021</b></h3>

Это домашнее задание будет посвящено полноценному решению задачи машинного обучения.

Есть две части этого домашнего задания: 
* Сделать полноценный отчет о вашей работе: как вы обработали данные, какие модели попробовали и какие результаты получились (максимум 10 баллов). За каждую выполненную часть будет начислено определенное количество баллов.
* Лучшее решение отправить в соревнование на [kaggle](https://www.kaggle.com/c/advanced-dls-spring-2021/) (максимум 5 баллов). За прохождение определенного порогов будут начисляться баллы.


**Обе части будут проверяться в формате peer-review. Т.е. вашу посылку на степик будут проверять несколько других студентов и аггрегация их оценок будет выставлена. В то же время вам тоже нужно будет проверить несколько других учеников.**

**Пожалуйста, делайте свою работу чистой и понятной, чтобы облегчить проверку. Если у вас будут проблемы с решением или хочется совета, то пишите в наш чат в телеграме или в лс @runfme. Если вы захотите проаппелировать оценку, то пипшите в лс @runfme.**

**Во всех пунктах указания это минимальный набор вещей, которые стоит сделать. Если вы можете сделать какой-то шаг лучше или добавить что-то свое - дерзайте!**

# Как проверять?

Ставьте полный балл, если выполнены все рекомендации или сделано что-то более интересное и сложное. За каждый отсустствующий пункт из рекомендация снижайте 1 балл.

# Метрика

Перед решением любой задачи важно понимать, как будет оцениваться ваше решение. В данном случае мы используем стандартную для задачи классификации метрику ROC-AUC. Ее можно вычислить используя только предсказанные вероятности и истинные классы без конкретного порога классификации + она раотает даже если классы в данных сильно несбалансированны (примеров одного класса в десятки раз больше примеров длугого). Именно поэтому она очень удобна для соревнований.

Посчитать ее легко:
"""

from sklearn.metrics import roc_auc_score

y_true = [
    0,
    1,
    1,
    0,
    1
]

y_predictions = [
    0.1,
    0.9,
    0.4,
    0.6,
    0.61
]

roc_auc_score(y_true, y_predictions)

"""# Первая часть. Исследование"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


"""## Загрузка данных (2 балла)

1) Посмотрите на случайные строчки. 

2) Посмотрите, есть ли в датасете незаполненные значения (nan'ы) с помощью data.isna() или data.info() и, если нужно, замените их на что-то. Будет хорошо, если вы построите табличку с количеством nan в каждой колонке.
"""

data = pd.read_csv('./train.csv')

# Для вашего удобства списки с именами разных колонок

# Числовые признаки
num_cols = [
    'ClientPeriod',
    'MonthlySpending',
    'TotalSpent'
]

# Категориальные признаки
cat_cols = [
    'Sex',
    'IsSeniorCitizen',
    'HasPartner',
    'HasChild',
    'HasPhoneService',
    'HasMultiplePhoneNumbers',
    'HasInternetService',
    'HasOnlineSecurityService',
    'HasOnlineBackup',
    'HasDeviceProtection',
    'HasTechSupportAccess',
    'HasOnlineTV',
    'HasMovieSubscription',
    'HasContractPhone',
    'IsBillingPaperless',
    'PaymentMethod'
]

feature_cols = num_cols + cat_cols
target_col = 'Churn' # Целевой столбец

# Посчитаем количество строк  столбцов
data.shape

# Выведем случайные данные
data.sample(10)

# Просмотрим информацию об объекте
data.info()

# Колонка TotalSpent тип object. Преобразуем в float64. Если есть значения в виде пробелов, заменим их на NaN
data['TotalSpent'] = data.TotalSpent.replace(' ', np.nan)
# Преобразуем данные в тип float64
data['TotalSpent'] = data.TotalSpent.astype('float64')

# Просмотрим информацию о незаполненных значениях
data.isna().sum()

# Заменим nan'ы в колонке TotalSpent (9 NAN) на среднее значение по этой колонке
data['TotalSpent'].fillna(value=data.TotalSpent.mean(), inplace=True)

"""## Анализ данных (3 балла)

1) Для численных призанков постройте гистограмму (*plt.hist(...)*) или boxplot (*plt.boxplot(...)*). Для категориальных посчитайте количество каждого значения для каждого признака. Для каждой колонки надо сделать *data.value_counts()* и построить bar диаграммы *plt.bar(...)* или круговые диаграммы *plt.pie(...)* (хорошо, елси вы сможете это сделать на одном гарфике с помощью *plt.subplots(...)*). 

2) Посмотрите на распределение целевой переменной и скажите, являются ли классы несбалансированными.

3) (Если будет желание) Поиграйте с разными библиотеками для визуализации - *sns*, *pandas_visual_analysis*, etc.

Второй пункт очень важен, потому что существуют задачи классификации с несбалансированными классами. Например, это может значить, что в датасете намного больше примеров 0 класса. В таких случаях нужно 1) не использовать accuracy как метрику 2) использовать методы борьбы с imbalanced dataset (обычно если датасет сильно несбалансирован, т.е. класса 1 в 20 раз меньше класса 0).
"""

# Посчитаем количество  столбцов с числовыми и категориальными прзнаками
len(num_cols), len(cat_cols)

# Для численных призанков построим гистограмму
fig, axes = plt.subplots(3, figsize=(15, 15),)
for j in range(3):
    axes[j].set_title(f'{num_cols[j]}')
    axes[j].hist(data[num_cols[j]], bins=300)

"""**ВЫВОД: сильных выбросов на гистограммах не  наблюдается.**"""

# Для каждой колонки надо сделать data.value_counts() и построить bar диаграммы plt.bar(...) 
# или круговые диаграммы plt.pie(...) 

fig, axes = plt.subplots(4, 4, figsize=(15, 15))

for i in range(4):
    for j in range(4):
        axes[i, j].set_title(f'{cat_cols[4 * i + j]}')
        axes[i, j].pie(data[cat_cols[4 * i + j]].value_counts(), 
                       labels=data[cat_cols[4 * i + j]].value_counts())

#Распределение целевой переменной
data.Churn.value_counts(bins=None).plot(kind='pie');

"""**Имеется некритичный дисбаланс**

(Дополнительно) Если вы нашли какие-то ошибки в данных или выбросы, то можете их убрать. Тут можно поэксперементировать с обработкой данных как угодно, но не за баллы.
"""

# Замечено, что тип 'IsSeniorCitizen' является числовым, однако по названию он может быть как string так и Bool
#Переведем его в string или (bool)
data.loc[:, 'IsSeniorCitizen'] = data.IsSeniorCitizen.astype('bool')

"""## Применение линейных моделей (3 балла)

1) Обработайте данные для того, чтобы к ним можно было применить LogisticRegression. Т.е. отнормируйте числовые признаки, а категориальные закодируйте с помощью one-hot-encoding'а. 

2) С помощью кроссвалидации или разделения на train/valid выборку протестируйте разные значения гиперпараметра C и выберите лучший (можно тестировать С=100, 10, 1, 0.1, 0.01, 0.001) по метрике ROC-AUC. 

Если вы разделяете на train/valid, то используйте LogisticRegressionCV. Он сам при вызове .fit() подберет параметр С. (не забудьте передать scroing='roc_auc', чтобы при кроссвалидации сравнивались значения этой метрики, и refit=True, чтобы при потом модель обучилась на всем датасете с лучшим параметром C). 


(более сложный вариант) Если вы будете использовать кроссвалидацию, то преобразования данных и LogisticRegression нужно соединить в один Pipeline с помощью make_pipeline, как это делалось во втором семинаре. Потом pipeline надо передать в GridSearchCV. Для one-hot-encoding'a можно испльзовать комбинацию LabelEncoder + OneHotEncoder (сначала превращаем строчки в числа, а потом числа првращаем в one-hot вектора.)
"""

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline

# Создадим переменные с колонками.
numeric_data = data[num_cols]
categorial_data = data[cat_cols]

#Ранее видели на Гистограмме. Количество уникальных записей в наборе данных или в столбце, мы должны использовать метод .nunique (). 
categorial_data.nunique()

#Считаем количество one-hot-encoding 
categorial_data.nunique().sum()

# Применяем One-Hot-Encoding
dummy_features = pd.get_dummies(categorial_data)

# Создаем датафрейм из числовых и категориальных признаков
X = pd.concat([numeric_data, dummy_features], axis=1)
X.head()

# Вектор ответов
y = data['Churn']
y.head()

X.shape



# При разбиении 69% тренировочные данные и 31% тестовые
# Почему то получается лучшее качество
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, 
                                                    train_size=0.69,
                                                    random_state=42)

# Стандартизация данных, иначе получим STOP: TOTAL NO. of ITERATIONS REACHED LIMIT
# Не отнормированные данные приведут к низкому результату. 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#GridSearchCV находит наилучшие параметры, путем обычного перебора: он создает 
# модель для каждой возможной комбинации параметров
model = GridSearchCV(LogisticRegression(), 
                   [{'C': [100, 10, 1, 0.1, 0.01, 0.001]}],
                   cv=5,
                   scoring='roc_auc',
                   refit=True)
model.fit(X_train_scaled, y_train)

# Гиперпараметры лучшей модели
model.best_params_

y_pred = model.predict_proba(X_test_scaled)[:, 1]
roc_auc_score(y_test, y_pred)

"""**Получено качество 0.8324 при C = 100**

Выпишите какое лучшее качество и с какими параметрами вам удалось получить

## Применение градиентного бустинга (2 балла)

Если вы хотите получить баллы за точный ответ, то стоит попробовать градиентный бустинг. Часто градиентный бустинг с дефолтными параметрами даст вам 80% результата за 0% усилий.

Мы будем использовать catboost, поэтому нам не надо кодировать категориальные признаки. catboost сделает это сам (в .fit() надо передать cat_features=cat_cols). А численные признаки нормировать для моделей, основанных на деревьях не нужно.

1) Разделите выборку на train/valid. Протестируйте catboost cо стандартными параметрами.

2) Протестируйте разные занчения параметроа количества деревьев и learning_rate'а и выберите лучшую по метрике ROC-AUC комбинацию. 

(Дополнительно) Есть некоторые сложности с тем, чтобы использовать CatBoostClassifier вместе с GridSearchCV, поэтому мы не просим использовать кроссвалидацию. Но можете попробовать)
"""

#!pip install catboost
import catboost

# Объединим данные числовые 
X = pd.concat([data[num_cols], data[cat_cols]], axis=1)
y = data[target_col]


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

# Теструем со стандартными параметрами
standart_boosting_model = catboost.CatBoostClassifier(cat_features=cat_cols)
standart_boosting_model.fit(X_train, y_train)

roc_auc_score(y_test, standart_boosting_model.predict_proba(X_test)[:,1])

boosting_model = catboost.CatBoostClassifier(silent=True,
                                             cat_features= np.arange(3, 19),
                                             eval_metric='AUC')
grid = {'learning_rate': [0.03, 0.1, 0.3],
        'depth': [1, 2, 4],
        'l2_leaf_reg': [2, 4, 8],
        'iterations': [200, 400, 600]}

boosting_model.grid_search(grid, 
                           X_train, 
                           y_train, plot=True, refit=True)

roc_auc_score(y_test, boosting_model.predict_proba(X_test)[:,1])

"""Выпишите какое лучшее качество и с какими параметрами вам удалось получить

**Лучший результат** - 0.8566
 **При данных параметрах** 'params': {'depth': 2,  'l2_leaf_reg': 4,  'iterations': 200,'learning_rate': 0.1}

# Предсказания
"""

best_model = boosting_model

# От меня. Ни в коем случаее не предобрабатывайте в этом модуле данные, иначе низкий балл гарантирован.
X_test = pd.read_csv('./test.csv')
submission = pd.read_csv('./submission.csv')
#Протестируем
submission['Churn'] =best_model.predict_proba(X_test)[:,1]
# Запишем полученный результат в файл, для отправки на kaggle
submission.to_csv('./my_submission.csv', index=False)

# Для проверки корректности данных 
XX = pd.read_csv('./my_submission.csv')
XX.head(10)

"""# Kaggle (5 баллов)

Как выставить баллы:

1) 1 >= roc auc > 0.84 это 5 баллов

2) 0.84 >= roc auc > 0.7 это 3 балла

3) 0.7 >= roc auc > 0.6 это 1 балл

4) 0.6 >= roc auc это 0 баллов


Для выполнения задания необходимо выполнить следующие шаги.
* Зарегистрироваться на платформе [kaggle.com](kaggle.com). Процесс выставления оценок будет проходить при подведении итогового рейтинга. Пожалуйста, укажите во вкладке Team -> Team name свои имя и фамилию в формате Имя_Фамилия (важно, чтобы имя и фамилия совпадали с данными на Stepik).
* Обучить модель, получить файл с ответами в формате .csv и сдать его в конкурс. Пробуйте и экспериментируйте. Обратите внимание, что вы можете выполнять до 20 попыток сдачи на kaggle в день.
* После окончания соревнования отправить в итоговый ноутбук с решением на степик. 
* После дедлайна проверьте посылки других участников по критериям. Для этого надо зайти на степик, скачать их ноутбук и проверить скор в соревновании.
"""
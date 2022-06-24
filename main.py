import seaborn as sns
import matplotlib.pyplot as plt

from diagnostic_plots import diagnostic_plots

import pandas as pd
import numpy as np

import statsmodels.api as sm
import scipy.stats as sts

sns.set()

"""Нужно подгрузить датасеты для того, чтобы можно прокликать ячейки в заданиях 2-3.
* ```softdrin.txt``` 
* ```sydhob.txt```

# Test for ```diagnostic_plots```.
"""

softdrin = pd.read_csv('softdrin.txt', sep='\t')

y = softdrin.Time
X = softdrin.Cases

X = sm.add_constant(X)

model = sm.OLS(y, X)
fitted = model.fit()

gamma_model = sm.GLM(y, X, family=sm.families.Gamma())
gamma_results = gamma_model.fit()

"""Все работает."""

diagnostic_plots(X, y, model='LM', model_fit=fitted, figsize=(15, 15));
# diagnostic_plots(X, y, model='GLM', model_fit=gamma_results, figsize=(15, 15));

"""# Пример (для себя).

Иллюстративный пример. Если гетероскедастичность сужающаяся, то нужно применить выпуклое преобразование $\lambda > 1$ Бокса-Кокса (и наоборот, если 
есть расширение).
"""

# сужающаяся гетероскедастичность

x = sts.norm.rvs(size=1000)
y = x + sts.norm.rvs(scale=0.1, size=1000) * np.exp(-x)

y = y - min(y) + 0.01
y_transformed = sts.boxcox(y, lmbda=3)

fig, axs = plt.subplots(1, 2, figsize=(14, 7))

axs[0].scatter(x, y)
axs[0].set_xlabel('x');
axs[0].set_ylabel('y');

axs[1].scatter(x, y_transformed)
axs[1].set_xlabel('x');
axs[1].set_ylabel('$Box-Cox\ (\lambda=2)$ y');

# растущая гетероскедастичность

x = sts.norm.rvs(size=100)
y = x + sts.norm.rvs(scale=0.1, size=100) * np.exp(x)

y = y - min(y) + 0.01
y_transformed = sts.boxcox(y, lmbda=0)

fig, axs = plt.subplots(1, 2, figsize=(14, 7))

axs[0].scatter(x, y)
axs[0].set_xlabel('x');
axs[0].set_ylabel('y');

axs[1].scatter(x, y_transformed)
axs[1].set_xlabel('x');
axs[1].set_ylabel('$Box-Cox\ (\lambda=0)$ y');

"""# Задание 1.
В файле ```diamonds``` пакета ```seaborn``` содержатся данные о бриллиантах. Построить модель зависимости ```price``` от ```carat```, подобрав нужное преобразование переменных.

**Необязательный пункт**. Разбейте данные по категориальной переменной ```cut```. Улучшается ли регрессионная модель, если ее производить в отдельной категории? 
"""

diamonds = sns.load_dataset('diamonds')

diamonds.head()

"""Можно заметить:
* гетероскедастичность,
* неадекватность линейной модели.
"""

fig, ax = plt.subplots(1, 1, figsize=(7, 7))

sns.scatterplot(data=diamonds.sample(1000), x="carat", y="price")

"""Решение: т.к. обе переменные положительны, то
* применим Бокса-Кокса к ```price```= $y$, чтобы убрать гетероскедастичность;
* после применим Бокса-Кокса к ```carat```=$x$, чтобы сделать линейную модель адекватной.

Подберем $\lambda$ для преобразования Бокса-Кокса $y$ на сетке. Оценим визуально с помощью ```scatterplot``` для каждого $\lambda$ насколько у нас стала гомоскедастичнее модель.

Визуально предпочтение падает на значения $-1/2, -1/4$.
"""

lambdas = [-4, -2, -1, -1 / 2, -1 / 4, 0, 1 / 4, 1 / 2, 1, 2, 4]

data = diamonds.sample(1000).reset_index(drop=True)
y = data.price.values
X = data.carat.values
gmean = (y ** (1. / len(y))).prod()

fig, axs = plt.subplots(3, 5, figsize=(25, 17))

for idx, lmbda in enumerate(lambdas):
    y_transformed = sts.boxcox(y, lmbda=lmbda) * gmean ** (1 - lmbda)
    sns.scatterplot(x=X, y=y_transformed, ax=axs[idx // 5, idx % 5])
    axs[idx // 5, idx % 5].set_title(f"Lambda = {lmbda}")

# возьмем lambda = -1/4
lmbda_opt = -0.25
diamonds.price = sts.boxcox(diamonds.price.values, lmbda=lmbda_opt) * gmean ** (1 - lmbda_opt)

"""Теперь сделаем преобразование Бокса-Кокса для $x$. Тут уже мы стремимся к адекватности модели и можно использовать $RSS$, как некоторую меру качества для преобразования Бокса-Кокса."""

lambdas = np.linspace(-4, 4, 100)
RSS = []

data = diamonds.sample(10000).reset_index(drop=True)
y = data.price.values.reshape(-1, 1)
X = data.carat.values
gmean = np.prod(X ** (1 / len(X)))

for lmbda in lambdas:
    X_transformed = sts.boxcox(X, lmbda=lmbda) * gmean ** (1 - lmbda)
    X_transformed = sm.add_constant(X_transformed.reshape(-1, 1))

    model = sm.OLS(y, X_transformed)
    fitted = model.fit()

    RSS.append(fitted.ssr)

RSS = np.array(RSS)

# optimal values
idx = RSS.argmin()
lmbda_opt = lambdas[idx]
RSS_opt = RSS[idx]

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.scatter(x=[lmbda_opt], y=[RSS_opt], marker='*', color='r', s=200, zorder=5)
ax.annotate(f"({lmbda_opt}, {RSS_opt})", (lmbda_opt, RSS_opt))
ax.plot(lambdas, RSS);

ax.set_title("Optimal Lambda Value");
ax.set_xlabel("Lambda");
ax.set_ylabel("RSS");

diamonds.carat = sts.boxcox(diamonds.carat.values, lmbda=lmbda_opt) * gmean ** (1 - lmbda_opt)

"""Наконец, построим линейную модель."""

y = diamonds.price
X = diamonds.carat

X = sm.add_constant(X)

model = sm.OLS(y, X)
fitted = model.fit()

print(fitted.summary())

"""Диагностические графики:

* ```Residuals vs Fitted``` - ок.
* ```Normal Q-Q``` - у остатков потяжелее хвосты.
* ```Scale-Location``` - не очень (?).
* ```Leverage``` - сложно сказать, вроде выбросов нет, но тут график не очень хорошо отрисовывается.
"""

diagnostic_plots(X, y, model='LM', model_fit=fitted, figsize=(15, 15));

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(data=diamonds, x="carat", y="price", ax=ax)
ax.plot(diamonds.carat, fitted.predict(sm.add_constant(diamonds.carat.values.reshape(-1, 1))), color='r', zorder=5)

"""## Улучшат ли ситуацию переменные ```depth``` и ```table```?"""

diamonds = sns.load_dataset('diamonds')

diamonds.head()

"""Сначала просто попробуем построить линейную модель."""

data = diamonds.sample(10000).reset_index(drop=True)
y = data.price
X = data[['carat', 'depth', 'table']]

X = sm.add_constant(X)

model = sm.OLS(y, X)
fitted = model.fit()

print(fitted.summary())

"""Какие-то проблемы с мультиколлинеарностью:
```
The condition number is large, 4.97e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
```

Посмотрим на корреляции.
"""

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

corrs_pearson = diamonds[['carat', 'depth', 'table']].corr(method="pearson")
sns.heatmap(corrs_pearson, vmin=-1, vmax=1, annot=True, ax=ax, square=True, cmap="seismic")

"""Уберем ```table```, который коррелирует с остальными предикторами. Убрав его, получим уже более-менее некоррелированные предикторы.

Повторим процедуру.
"""

data = diamonds.sample(10000).reset_index(drop=True)
y = data.price
X = data[['carat', 'depth']]

X = sm.add_constant(X)

model = sm.OLS(y, X)
fitted = model.fit()

print(fitted.summary())

"""Отрисуем на всякий случай диагностические графики. Они покажут нам, что у нас большие проблемы с гетероскедастичностью и нормальностью остатков."""

diagnostic_plots(X, y, model='LM', model_fit=fitted, figsize=(15, 15));

"""Попробуем исправить ситуацию, сначала преобразовав $y$. 

Построим **Inverse Response Plot**. Так мы поймем, как может выглядеть $g^{-1}(\cdot)$. Сразу же построим ее аппроксимацию локально-линейной регрессией (нужно прощелкать код с реализацией, который ее реализует: через ```Ctrl+F``` можно вбить **a(x, X, Y, h)** и найти).

Здесь видно, как растет дисперсия с ростом $y$ (и $\hat{y}$). Возможно, что получившееся вогнутое преобразование $g^{-1}$ как раз поможет с этим справиться.
"""

x_axis = y.values
y_axis = fitted.predict(X).values

zipped = sorted(zip(x_axis, y_axis), key=lambda x: x[0])
x_axis = np.array([tup[0] for tup in zipped])
y_axis = np.array([tup[1] for tup in zipped])

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.set_xlabel("y")
ax.set_ylabel("$\hat{y}$")
ax.plot(x_axis, y_axis)

# nonparametric linear-local regression.
x_axis_mesh = np.linspace(x_axis.min(), x_axis.max(), 500)
inv_g_approx = [a(x, x_axis, y_axis, h=250) for x in x_axis_mesh]

ax.plot(x_axis_mesh, inv_g_approx, color='r', label='$g^{-1}$ approximation')
ax.legend()

"""Применим это преобразование к $y$ (надо (не)много подождать)."""

from multiprocessing import Pool
from tqdm import tqdm
import time


def wrapped_regression(y_value):
    '''
    Обертка над функцией ядерной регрессии для того, чтобы можно было использовать imap.
    '''
    return a(y_value, x_axis, y_axis, h=250)


with Pool(4) as p:
    y_transformed_raw = list(tqdm(p.imap(wrapped_regression, diamonds.price.values), total=len(diamonds.price)))

# можно сохранить.
np.savetxt('y_transformed_by_inv_g.txt', y_transformed_raw, delimiter=',')

# и загрузить.
# y_transformed_raw = np.loadtxt('y_transformed_by_inv_g.txt')

diamonds.price = y_transformed_raw

"""Теперь применим 'двумерного' Бокса-Кокса к ```carat``` и ```depth```. Мера качества - $RSS$."""

lambdas = np.linspace(-3, 3, 20)
RSS = np.zeros((20, 20))

y = diamonds.price
X1 = diamonds.carat
X2 = diamonds.depth

gmean1 = np.prod(X1 ** (1 / len(X1)))
gmean2 = np.prod(X2 ** (1 / len(X2)))

for i, lmbda1 in enumerate(lambdas):
    for j, lmbda2 in enumerate(lambdas):
        X1_transformed = sts.boxcox(X1, lmbda=lmbda1) * gmean1 ** (1 - lmbda1)
        X2_transformed = sts.boxcox(X2, lmbda=lmbda2) * gmean2 ** (1 - lmbda2)

        X_transformed = np.hstack((X1_transformed.reshape(-1, 1), X2_transformed.reshape(-1, 1)))
        X_transformed = sm.add_constant(X_transformed)

        model = sm.OLS(y, X_transformed)
        fitted = model.fit()

        RSS[i, j] = fitted.ssr

# optimal values
idx = np.unravel_index(RSS.argmin(), (20, 20))
lmbda1_opt = lambdas[idx[0]]
lmbda2_opt = lambdas[idx[1]]

print('Оптимальные значения для лямбд в преобразованиях Бокса-Кокса.')
print(lmbda1_opt, lmbda2_opt)

"""Кажется, ```depth``` ни на что не влияет: видно, что тут все меняется только вдоль одной оси (это параметр Бокса-Кокса для ```carat```)."""

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# что-то типа линий уровня.
sns.heatmap(RSS, ax=ax)

diamonds.carat = sts.boxcox(diamonds.carat.values, lmbda=lmbda1_opt) * gmean1 ** (1 - lmbda1_opt)
diamonds.depth = sts.boxcox(diamonds.depth.values, lmbda=lmbda2_opt) * gmean2 ** (1 - lmbda2_opt)

"""Попытка №2.
Строим линейную модель с преобразованным $y$.
"""

data = diamonds.sample(10000).reset_index(drop=True)
y = data.price
X = data[['carat', 'depth']]

X = sm.add_constant(X)

model = sm.OLS(y, X)
fitted = model.fit()

print(fitted.summary())

diagnostic_plots(X, y, model='LM', model_fit=fitted, figsize=(15, 15));

"""В общем, как-то не получается нормально построить модель:
* остатки не нормальные,
* дисперсия intercept большая.

Возможно, лучше было попреобразовывать $y$ Боксом-Коксом.
А еще лучше вообще отказаться от лишних предикторов и строить модель с одним, преобразовывая данные Боксом-Коксом, благо что хорошо получается.

# Задание 2. 

В файле ```softdrin.txt``` содержится информация об обслуживании автоматов по продаже напитков в торговом центре. Интересует предсказание переменной ```Time``` в зависимости от числа товаров ```Cases```, которые нужно добавить в автомат и расстояния ```Distance```, которое нужно пройти до автомата. Построить модель и сравнить $GLM$-Gamma модель и нормальную $LM$ модель. При необходимости удалите часть выбросов.
"""

softdrin = pd.read_csv('softdrin.txt', sep='\t')

softdrin.head()

"""Предикторов мало, можно сразу проверить визуально, как они все коррелируют между собой и с response."""

sns.pairplot(data=softdrin)

"""Похоже, что все коррелирует между собой очень хорошо:
* предикторы хорошо коррелируют с response,
* и между собой.

Это говорит о том, что следует избавиться от одного из предикторов (н-р, ```Distance```), чтобы избежать мультиколлинеарности. Есть надежда на то, что получившаяся линейная модель будет хороша.

* Плоха ли мультиколлинеарность предикторов для GLM? - Видимо, нет, в GLM модели можно и оставить.
"""

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

corrs_pearson = softdrin.corr(method="pearson")
sns.heatmap(corrs_pearson, vmin=-1, vmax=1, annot=True, ax=ax, square=True, cmap="seismic")

"""## GLM-Gamma.

Для гамма-распределения $Y \sim \Gamma(a, b)$:
* Матожидание: $\mathbb{E} Y = \mu = ab$.
* Плотность:
$$
f(x) = \frac{x^{a-1} e^{-\frac{x}{b}}}{\Gamma(a) b^{a}} = \exp \left\{ (a-1) \ln x - \frac{x}{b} - \ln\left( \Gamma(a) \right) - a \ln b \right\} \\
= \exp \left\{ x \cdot \left( -\frac{1}{b} \right) + a \ln \left( -\left(-\frac{1}{b}\right) \right) + \dots \right\}
$$
т.е. гамма-распределение - часть экспоненциального семейства с параметрами:
$$
\begin{aligned}
\phi &=1,\\
\theta &= -\frac{1}{b},\\
d(\theta) &= -a \ln(-\theta)\\
\end{aligned}
$$

При канонической функции связи верно последнее равенство:
$$
\mathbb{E}(Y|X=x) = \mu = d'(\theta) = -\frac{a}{\theta} = -\frac{a}{\langle X, \vec{\beta} \rangle}
$$

Попробуем просто настроить модель.

Вопросы актуальные и не очень: 
* Какой должен быть Deviance?
* Вредна ли мультиколлинеарность? - **Нет**.
* Зачем link function respect the domain of the Gamma family? - **Непонятно, ну да и ладно.**
* Что делать с параметром $a$, который выскочил в link function? - **```statsmodels``` как-то сам справляется. Видимо, там вообще какая-то другая параметризация гамма-распределения.**

## GLM-Gamma с одним предиктором ```Cases```.

Сначала попробуем использовать один предиктор, который кажется более удачным. Так можно будет посмотреть, как справилась наша модель, рисуя ```scatterplot```.
"""

softdrin = pd.read_csv('softdrin.txt', sep='\t')

y = softdrin.Time
X = softdrin.Cases
X = sm.add_constant(X)

"""Попробуем сразу построить модель без всяких предобработок."""

gamma_model = sm.GLM(y, X, family=sm.families.Gamma())
gamma_results = gamma_model.fit()

print(gamma_results.summary())

"""```Residuals vs Fitted``` и ```Scale-Location``` говорят, что все не очень хорошо."""

diagnostic_plots(X, y, model='GLM', model_fit=gamma_results, figsize=(15, 15));

"""```scatterplot``` подтверждает наши опасения."""

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(data=softdrin, x="Cases", y="Time", ax=ax)

for num, (x_, y_) in enumerate(zip(softdrin.Cases, softdrin.Time)):
    ax.annotate(num, (x_, y_))

x = np.sort(softdrin.Cases.values)

ax.plot(x, gamma_results.predict(sm.add_constant(x.reshape(-1, 1))), color='r')

"""Попробуем исправить ситуацию, как и ранее, преобразованием Бокса-Кокса предиктора. Мера качества - RSS."""

lambdas = np.linspace(-4, 4, 100)
RSS = []

y = softdrin.Time
X = softdrin.Cases
gmean = np.prod(X ** (1 / len(X)))

for lmbda in lambdas:
    X_transformed = sts.boxcox(X, lmbda=lmbda) * gmean ** (1 - lmbda)
    X_transformed = sm.add_constant(X_transformed.reshape(-1, 1))

    gamma_model = sm.GLM(y, X_transformed, family=sm.families.Gamma());
    gamma_results = gamma_model.fit()

    RSS.append(np.sum((gamma_results.resid_response) ** 2))

RSS = np.array(RSS)

# optimal values
idx = RSS.argmin()
lmbda_opt = lambdas[idx]
RSS_opt = RSS[idx]

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.scatter(x=[lmbda_opt], y=[RSS_opt], marker='*', color='r', s=200, zorder=5)
ax.annotate(f"({lmbda_opt}, {RSS_opt})", (lmbda_opt, RSS_opt))
ax.plot(lambdas, RSS);

ax.set_title("Optimal Lambda Value");
ax.set_xlabel("Lambda");
ax.set_ylabel("RSS");

softdrin.Cases = sts.boxcox(softdrin.Cases.values, lmbda=lmbda_opt) * gmean ** (1 - lmbda_opt)

y = softdrin.Time
X = softdrin.Cases
X = sm.add_constant(X)

gamma_model = sm.GLM(y, X, family=sm.families.Gamma())
gamma_results = gamma_model.fit()

print(gamma_results.summary())

"""Графики не такие хорошие, но скеттер показывает, что все более-менее удачно."""

diagnostic_plots(X, y, model='GLM', model_fit=gamma_results, figsize=(15, 15));

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(data=softdrin, x="Cases", y="Time", ax=ax)

for num, (x_, y_) in enumerate(zip(softdrin.Cases, softdrin.Time)):
    ax.annotate(num, (x_, y_))

x = np.sort(softdrin.Cases.values)

ax.plot(x, gamma_results.predict(sm.add_constant(x.reshape(-1, 1))), color='r')

"""## GLM-Gamma с двумя предикторами ```Cases``` и ```Distance```."""

softdrin = pd.read_csv('softdrin.txt', sep='\t')

y = softdrin.Time
X = softdrin[['Cases', 'Distance']]
X = sm.add_constant(X)

"""Попробуем сразу построить модель без всяких предобработок."""

gamma_model = sm.GLM(y, X, family=sm.families.Gamma())
gamma_results = gamma_model.fit()

print(gamma_results.summary())

"""```Residuals vs Fitted``` и ```Scale-Location``` снова говорят, что все не очень хорошо.


"""

diagnostic_plots(X, y, model='GLM', model_fit=gamma_results, figsize=(15, 15));

"""Сделаем одновременно двух Боксов-Коксов для ```Cases``` и ```Distance```."""

lambdas = np.linspace(-3, 3, 20)
RSS = np.zeros((20, 20))

y = softdrin.Time
X1 = softdrin.Cases
X2 = softdrin.Distance

gmean1 = np.prod(X1 ** (1 / len(X1)))
gmean2 = np.prod(X2 ** (1 / len(X2)))

for i, lmbda1 in enumerate(lambdas):
    for j, lmbda2 in enumerate(lambdas):
        X1_transformed = sts.boxcox(X1, lmbda=lmbda1) * gmean1 ** (1 - lmbda1)
        X2_transformed = sts.boxcox(X2, lmbda=lmbda2) * gmean2 ** (1 - lmbda2)

        X_transformed = np.hstack((X1_transformed.reshape(-1, 1), X2_transformed.reshape(-1, 1)))
        X_transformed = sm.add_constant(X_transformed)

        gamma_model = sm.GLM(y, X_transformed, family=sm.families.Gamma());
        gamma_results = gamma_model.fit()

        RSS[i, j] = np.sum((gamma_results.resid_response) ** 2)

# optimal values
idx = np.unravel_index(RSS.argmin(), (20, 20))
lmbda1_opt = lambdas[idx[0]]
lmbda2_opt = lambdas[idx[1]]

print('Оптимальные значения для лямбд в преобразованиях Бокса-Кокса.')
print(lmbda1_opt, lmbda2_opt)

fig, ax = plt.subplots(1, 1, figsize=(25, 15))

sns.heatmap(RSS, annot=True, fmt='.2f', ax=ax)

softdrin.Cases = sts.boxcox(softdrin.Cases.values, lmbda=lmbda1_opt) * gmean1 ** (1 - lmbda1_opt)
softdrin.Distance = sts.boxcox(softdrin.Distance.values, lmbda=lmbda2_opt) * gmean2 ** (1 - lmbda2_opt)

y = softdrin.Time
X = softdrin[['Cases', 'Distance']]
X = sm.add_constant(X)

gamma_model = sm.GLM(y, X, family=sm.families.Gamma())
gamma_results = gamma_model.fit()

print(gamma_results.summary())

"""Тут смущает ```Scale-Location```, хотя ```Residual vs Fitted``` уже лучше."""

diagnostic_plots(X, y, model='GLM', model_fit=gamma_results, figsize=(15, 15));

"""## LM."""

softdrin = pd.read_csv('softdrin.txt', sep='\t')

"""Пользуемся одномерностью предиктора и рисуем ```scatter```.

* Видим, что у 8-ой точки скорее всего будет большой рычаг => избавляемся от нее. 
* Изначально линейная модель адекватна, но есть расширяющаяся гетероскедастичность.
"""

fig, axs = plt.subplots(1, 2, figsize=(15, 7))

y = softdrin.Time
X = softdrin.Cases

axs[0].scatter(X, y)
axs[0].set_title('With 8th point')
axs[0].set_xlabel('Cases')
axs[0].set_ylabel('Time')

for num, (x_, y_) in enumerate(zip(X, y)):
    axs[0].annotate(num, (x_, y_))

softdrin = softdrin.drop(index=[8]).reset_index(drop=True)

y = softdrin.Time
X = softdrin.Cases

axs[1].scatter(X, y)
axs[1].set_title('Without 8th point')
axs[1].set_xlabel('Cases')
axs[1].set_ylabel('Time')

for num, (x_, y_) in enumerate(zip(X, y)):
    axs[1].annotate(num, (x_, y_))

"""Попробуем все-таки построить модель и посмотреть на результаты."""

y = softdrin.Time
X = softdrin.Cases

X = sm.add_constant(X)

model = sm.OLS(y, X)
fitted = model.fit()

print(fitted.summary())

"""Диагностические графики (тут возможен эффект того, что выборка небольшая):

* ```Residuals vs Fitted``` - тут лучше посмотреть на ```Scale-Location```, хотя видно, что есть улетевшие точки.
* ```Normal Q-Q``` - не нормально.
* ```Scale-Location``` - гетероскедастичность.
* ```Leverage``` - выбросов нет.
"""

diagnostic_plots(X, y, model='LM', model_fit=fitted, figsize=(15, 15));

"""Попробуем сделать преобразования Бокса-Кокса для $x$ и $y$, как в задании 1."""

lambdas = [-3, -2, -1, -0.5, -0.25, 0]

y = softdrin.Time
X = softdrin.Cases
gmean = (y ** (1. / len(y))).prod()

fig, axs = plt.subplots(2, 3, figsize=(20, 15))

for idx, lmbda in enumerate(lambdas):
    y_transformed = sts.boxcox(y, lmbda=lmbda) * gmean ** (1 - lmbda)
    sns.scatterplot(x=X, y=y_transformed, ax=axs[idx // 3, idx % 3])
    axs[idx // 3, idx % 3].set_title(f"Lambda = {lmbda}")

# возьмем lambda = -0.5
lmbda_opt = -0.5
softdrin.Time = sts.boxcox(softdrin.Time, lmbda=lmbda_opt) * gmean ** (1 - lmbda_opt)

lambdas = np.linspace(-4, 4, 100)
RSS = []

y = softdrin.Time
X = softdrin.Cases
gmean = (X ** (1. / len(X))).prod()

for lmbda in lambdas:
    X_transformed = sts.boxcox(X, lmbda=lmbda) * gmean ** (1 - lmbda)
    X_transformed = sm.add_constant(X_transformed.reshape(-1, 1))

    model = sm.OLS(y, X_transformed)
    fitted = model.fit()

    RSS.append(fitted.ssr)

RSS = np.array(RSS)

# optimal values
idx = RSS.argmin()
lmbda_opt = lambdas[idx]
RSS_opt = RSS[idx]

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.scatter(x=[lmbda_opt], y=[RSS_opt], marker='*', color='r', s=200, zorder=5)
ax.annotate(f"({lmbda_opt}, {RSS_opt})", (lmbda_opt, RSS_opt))
ax.plot(lambdas, RSS);

ax.set_title("Optimal Lambda Value");
ax.set_xlabel("Lambda");
ax.set_ylabel("RSS");

softdrin.Cases = sts.boxcox(softdrin.Cases.values, lmbda=lmbda_opt) * gmean ** (1 - lmbda_opt)

"""Кажется, что можно еще убрать точку 8 (уже другую), которая далековато улетела."""

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

y = softdrin.Time
X = softdrin.Cases

ax.scatter(X, y)
ax.set_xlabel('Cases')
ax.set_ylabel('Time')

for num, (x_, y_) in enumerate(zip(X, y)):
    ax.annotate(num, (x_, y_))

softdrin = softdrin.drop(index=[8]).reset_index(drop=True)

y = softdrin.Time
X = softdrin.Cases

X = sm.add_constant(X)

model = sm.OLS(y, X)
fitted = model.fit()

print(fitted.summary())

"""Диагностические графики показывают, что ситуация улучшилась, хотя ```Scale-Location``` еще не очень хорошо (но лучше, чем раньше)."""

diagnostic_plots(X, y, model='LM', model_fit=fitted, figsize=(15, 15));

"""Получили такую прямую."""

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(data=softdrin, x="Cases", y="Time", ax=ax)

for num, (x_, y_) in enumerate(zip(softdrin.Cases, softdrin.Time)):
    ax.annotate(num, (x_, y_))

ax.plot(softdrin.Cases, fitted.predict(sm.add_constant(softdrin.Cases.values.reshape(-1, 1))), color='r')

"""# Задание 3. 

В файле ```sybhob.txt``` содержится информация о времени победителя в гонках и параметрах. Рассмотреть модель времени от года. Преобразуйте зависимую переменную так, чтобы добиться гомоскедастичности и постройте непараметрическую модель.
"""

sydhob = pd.read_csv('sydhob.txt', sep='\t')

sydhob.head()

"""Воспользуемся одномерностью предиктора и построим ```scatter```, чтобы оценить ситуацию."""

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
sns.scatterplot(data=sydhob, x='Year', y='Time')

"""Видно, что тут довольно заметный шум (сложно сказать гетероскедастичный ли). Все-таки попробуем покрутить $y$ преобразованием Бокса-Кокса.

Возьмем небольшую сетку по $\lambda$ и построим соотв. преобразования Бокса-Кокса для $y$.

Видно, что визуально все несильно лучше становится. Масштаб сильно меняется, поэтому я добавлю нормировку на среднее геометрическое.
"""

lambdas = [-1, -1 / 2, -1 / 4, 0, 1 / 4, 1 / 2, 1, 2, 3, 5]

y = sydhob.Time.values
X = sydhob.Year.values
gmean = y.prod() ** (1. / len(y))  # geometric mean.

fig, axs = plt.subplots(3, 4, figsize=(25, 17))

for idx, lmbda in enumerate(lambdas):
    y_transformed = sts.boxcox(y, lmbda=lmbda) * gmean ** (1 - lmbda)
    sns.scatterplot(x=X, y=y_transformed, ax=axs[idx // 4, idx % 4])
    axs[idx // 4, idx % 4].set_title(f"Lambda = {lmbda}")

    for num, (x_, y_) in enumerate(zip(X, y_transformed)):
        axs[idx // 4, idx % 4].annotate(num, (x_, y_))

"""Выберем $\lambda=2$. Кажется, что это не слишком хорошее преобразование с точки зрения гомоскедастичности, но относительно других выглядит получше.

Заодно выбросим точки 0, 7, 23, 48, которые не очень хорошо ложатся.
"""

sydhob = sydhob.drop(index=[0, 7, 23, 48]).reset_index(drop=True)
sydhob.Time = sts.boxcox(sydhob.Time.values, lmbda=2) / gmean

"""## Локально-линейная регрессия by ```statsmodels```."""

from statsmodels.nonparametric.kernel_regression import KernelReg

X = sydhob.Year.values
Y = sydhob.Time.values

model = KernelReg(Y, X, var_type='c')

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.scatter(X, Y)

x = np.linspace(X.min(), X.max(), 100)
ax.plot(x, model.fit(x)[0], color='r', label='Local Linear Regression')

ax.legend()

"""## Локально-линейная регрессия by me.

Теперь построим **локально-линейную модель**:
$$
\mathbb{E}(Y|X=x) = a(x)
$$

Значение $a(x)$ находится, как решение оптимизационной задачи.
$$
\forall x: \sum_{i=1}^{n} w_{i}(x) \left( y_i - a - b (X_i-x) \right) \to \max_{a, b}.
$$

У этой модели (в каждой точке) есть аналитическое решение:
$$
\begin{aligned}
& a = \frac{\sum_{i=1}^{n} w_{i}(x) \left( y_i - b (X_i -x) \right)}{\sum_{i=1}^{n} w_{i}(x)} \\
& b = \frac{ \sum_{j=1}^{n} w_{j}(x) \cdot \sum_{i=1}^{n} w_{i}(x) (X_i-x) y_i - \sum_{j=1}^{n} w_{j}(x) (X_j-x) \cdot \sum_{i=1}^{n} w_{i}(x) y_i }{ \sum_{j=1}^{n} w_{j}(x) \cdot \sum_{i=1}^{n} w_{i}(x) (X_i-x)^2 - \left( \sum_{j=1}^{n} w_{j}(x) (X_j-x) \right)^2  }
\end{aligned}
$$

В качестве весов можно использовать (нормировка не сказывается на формуле для $a, b$, поэтому веса нормировать веса не будем):
$$
w_{i}(x) = K \left( \frac{x-x_i}{h} \right)
$$

Ядро возьмем гауссовское (просто так), а с $h$ выясним позднее.
"""


def K(x):
    return sts.norm.pdf(x)


def w(x, X, h):
    '''
    x - argument,
    X - sample element,
    h - bandwidth.
    '''
    return K((x - X) / h)


def a(x, X, Y, h):
    '''
    x - argument,
    X, Y - sample.
    '''
    b_num = np.sum(w(x, X, h)) * np.sum(w(x, X, h) * (X - x) * Y) - np.sum(w(x, X, h) * (X - x)) * np.sum(
        w(x, X, h) * Y)
    b_denom = np.sum(w(x, X, h)) * np.sum(w(x, X, h) * (X - x) ** 2) - np.sum((w(x, X, h) * (X - x)) ** 2)

    b = b_num / b_denom

    a = np.sum(w(x, X, h) * (Y - b * (X - x))) / np.sum(w(x, X, h))
    return a


"""Посмотрим несколько значений $h$, выберем на свой визуальный вкус лучшее."""

fig, axs = plt.subplots(2, 2, figsize=(20, 10))

X = sydhob.Year.values
Y = sydhob.Time.values

h_vals = [0.5, 1, 2, 5]
for idx, h in enumerate(h_vals):
    years = np.linspace(X.min(), X.max(), 1000)
    times = [a(x, X, Y, h) for x in years]

    sns.scatterplot(x=X, y=Y, ax=axs[idx // 2, idx % 2])
    axs[idx // 2, idx % 2].plot(years, times, color='r')
    axs[idx // 2, idx % 2].set_title(f"h={h}")

"""Ширина $h=2$ или $h=5$ уже более менее пристойно себя ведет. Давайте посмотрим на экстраполяцию."""

fig, axs = plt.subplots(1, 2, figsize=(15, 7))

X = sydhob.Year.values
Y = sydhob.Time.values

h_vals = [2, 5]
for idx, h in enumerate(h_vals):
    years = np.linspace(1900, 2050, 600)
    times = [a(x, X, Y, h) for x in years]

    sns.scatterplot(x=X, y=Y, ax=axs[idx])
    axs[idx].plot(years, times, color='r')
    axs[idx].set_title(f"h={h}")

"""Экстраполяция неудачная (можно было ожидать от локальной регрессии такого исхода)."""
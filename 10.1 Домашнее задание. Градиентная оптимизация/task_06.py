'''Это задание чуть сложнее. Если раньше Вам нужно было просто найти минимум у довольно
 хорошей функции, то сейчас в тестах будут плохие. У них может быть несколько локальных минимумов,
 вам же нужно найти глобальный минимум у каждой функции.
В общем случае такая задача невыполнима, но у вас будут одномерные функции и все самое интересное
будет сосредоточено в районе нуля. А именно, известно что глобальный минимум лежит в пределах
(low, high) (параметры алгоритма). Вам нужно модифицировать градиентный спуск, который вы написали
в предыдущем задании, чтобы он работал и в таком случае.
Сначала запустите градиентный спуск из прошлого пункта на тестах из ноутбука. Скорее всего,
некоторые из них не пройдут. Подумайте, как исправить ситуацию.
И снова не забывайте вызывать callback(x, f(x)) на каждом шаге алгоритма!
Возможное решение Если вы хотите поэкспериментировать и ощутить всю боль от оптимизации таких функций,
 сначала подумайте сами, не пытаясь следовать нашим указаниям. Тем не менее, для тех из вас, у кого таких
 наклонностей нет, мы выписали одно из возможных решений, которое приводит к успеху.
Сделайте шаг обучения не константным, а зависящим от номера итерации. Неплохая эвристика --- домножать lr на
1iteration√ .
В этой задаче в функциях могут после первого же шага градиентного спуска появляться очень большие значения.
Для того, чтобы не вылезать за пределы отрезка, на котором ищется минимум, после каждого шага
спуска используйте np.clip к очередной точке x_n.
Разбейте весь отрезок на несколько (3-6) подотрезков и найдите минимум на каждом из отрезков
(на каждом отрезке, кстати, можно сделать больше одного запуска). Затем из всех найденных результатов
выберите минимальный.
Авторское решение использует параметры iters = 5000 и lr = 0.05'''
from copy import copy, deepcopy
import numpy as np
import matplotlib.pyplot as plt


def grad_descent_v2(f, df, low=None, high=None, callback=None):
    """
    Реализация градиентного спуска для функций с несколькими локальным минимумами,
    но с известной окрестностью глобального минимума.
    Все тесты будут иметь такую природу.
    :param func: float -> float — функция
    :param deriv: float -> float — её производная
    :param low: float — левая граница окрестности
    :param high: float — правая граница окрестности
    :param callback: callalbe -- функция логирования
    """

    def find_local_min(f, df, low_local, high_local, iters=5000, lr=0.05):
        x0 = np.random.uniform(low_local, high_local)
        x = x0

        for i in range(iters):
            x -= lr * df(x) / (i + 1) ** 0.5
            x = np.clip(x, low_local, high_local)

        callback(x, f(x))
        return x

    x_min = []
    f_min = []

    num = 7
    t = 10
    local = np.linspace(low, high, num)

    for i in range(num - 1):
        low_local = local[i]
        high_local = local[i + 1]
        local_x_min = []
        local_f_min = []
        for j in range(t):
            local_x_min.append(find_local_min(f, df, low_local, high_local))
            local_f_min.append(f(local_x_min[j]))
        ind_local_x_min = np.argmin(local_f_min)
        x_min.append(local_x_min[ind_local_x_min])
        f_min.append(f(x_min[i]))

    ind = np.argmin(f_min)
    best_estimate = x_min[ind]

    return best_estimate


# Проверка________________________________________
def plot_convergence_1d(func, x_steps, y_steps, ax, grid=None, title=""):
    """
    Функция отрисовки шагов градиентного спуска.
    Не меняйте её код без необходимости!
    :param func: функция, которая минимизируется градиентным спуском
    :param x_steps: np.array(float) — шаги алгоритма по оси Ox
    :param y_steps: np.array(float) — шаги алгоритма по оси Оу
    :param ax: холст для отрисовки графика
    :param grid: np.array(float) — точки отрисовки функции func
    :param title: str — заголовок графика
    """
    ax.set_title(title, fontsize=16, fontweight="bold")

    if grid is None:
        grid = np.linspace(np.min(x_steps), np.max(x_steps), 100)

    fgrid = [func(item) for item in grid]
    ax.plot(grid, fgrid)
    yrange = np.max(fgrid) - np.min(fgrid)

    arrow_kwargs = dict(linestyle="--", color="grey", alpha=0.4)
    for i, _ in enumerate(x_steps):
        if i + 1 < len(x_steps):
            ax.arrow(
                x_steps[i], y_steps[i],
                x_steps[i + 1] - x_steps[i],
                y_steps[i + 1] - y_steps[i],
                **arrow_kwargs
            )

    n = len(x_steps)
    color_list = [(i / n, 0, 0, 1 - i / n) for i in range(n)]
    ax.scatter(x_steps, y_steps, c=color_list)
    ax.scatter(x_steps[-1], y_steps[-1], c="red")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")


class LoggingCallback:
    """
    Класс для логирования шагов градиентного спуска.
    Сохраняет точку (x, f(x)) на каждом шаге.
    Пример использования в коде: callback(x, f(x))
    """

    def __init__(self):
        self.x_steps = []
        self.y_steps = []

    def __call__(self, x, y):
        self.x_steps.append(x)
        self.y_steps.append(y)


def test_convergence_1d(grad_descent, test_cases, tol=1e-2, axes=None, grid=None):
    """
    Функция для проверки корректности вашего решения в одномерном случае.
    Она же используется в тестах на Stepik, так что не меняйте её код!
    :param grad_descent: ваша реализация градиентного спуска
    :param test_cases: dict(dict), тесты в формате dict с такими ключами:
        - "func" — функция (обязательно)
        - "deriv" — её производная (обязательно)
        - "start" — начальная точка start (м.б. None) (опционально)
        - "low", "high" — диапазон для выбора начальной точки (опционально)
        - "answer" — ответ (обязательно)
    При желании вы можете придумать и свои тесты.
    :param tol: предельное допустимое отклонение найденного ответа от истинного
    :param axes: матрица холстов для отрисовки, по ячейке на тест
    :param grid: np.array(float), точки на оси Ох для отрисовки тестов
    :return: флаг, корректно ли пройдены тесты, и дебажный вывод в случае неудачи
    """
    right_flag = True
    debug_log = []
    for i, key in enumerate(test_cases.keys()):
        # Формируем входные данные и ответ для алгоритма.
        answer = test_cases[key]["answer"]
        test_input = deepcopy(test_cases[key])
        del test_input["answer"]
        # Запускаем сам алгоритм.
        callback = LoggingCallback()  # Не забываем про логирование
        res_point = grad_descent(*test_input.values(), callback=callback)
        # Отрисовываем результаты.
        if axes is not None:
            ax = axes[np.unravel_index(i, shape=axes.shape)]
            x_steps = np.array(callback.x_steps)
            y_steps = np.array(callback.y_steps)
            plot_convergence_1d(
                test_input["func"], x_steps, y_steps,
                ax, grid, key
            )
            ax.axvline(answer, 0, linestyle="--", c="red",
                       label=f"true answer = {answer}")
            ax.axvline(res_point, 0, linestyle="--", c="xkcd:tangerine",
                       label=f"estimate = {np.round(res_point, 3)}")
            ax.legend(fontsize=16)
        # Проверяем, что найденая точка достаточно близко к истинной
        if abs(answer - res_point) > tol or np.isnan(res_point):
            debug_log.append(
                f"Тест '{key}':\n"
                f"\t- ответ: {answer}\n"
                f"\t- вывод алгоритма: {res_point}"
            )
            right_flag = False
    return right_flag, debug_log


test_cases = {
    "poly1": {
        "func": lambda x: x ** 4 + 3 * x ** 3 + x ** 2 - 1.5 * x + 1,
        "deriv": lambda x: 4 * x ** 3 + 9 * x ** 2 + 2 * x - 1.5,
        "low": -4, "high": 2, "answer": -1.88
    },
    "poly2": {
        "func": lambda x: x ** 4 + 3 * x ** 3 + x ** 2 - 2 * x + 1.0,
        "deriv": lambda x: 4 * x ** 3 + 9 * x ** 2 + 2 * x - 2.0,
        "low": -3, "high": 3, "answer": 0.352
    },
    "another yet poly": {
        "func": lambda x: x ** 6 + x ** 4 - 10 * x ** 2 - x,
        "deriv": lambda x: 6 * x ** 5 + 4 * x ** 3 - 20 * x - 1,
        "low": -2, "high": 2, "answer": 1.24829
    },
    "and another yet poly": {
        "func": lambda x: x ** 20 + x ** 2 - 20 * x + 10,
        "deriv": lambda x: 20 * x ** 19 + 2 * x - 20,
        "low": -0, "high": 2, "answer": 0.994502
    },
    "|x|/x^2 - x + sqrt(-x) + (even polynom)": {
        "func": lambda x: 5 * np.abs(x) / x ** 2 - 0.5 * x + 0.1 * np.sqrt(-x) + 0.01 * x ** 2,
        "deriv": lambda x: -0.5 - 0.05 / np.sqrt(-x) + 0.02 * x + 5 / (x * np.abs(x)) - (10 * np.abs(x)) / x ** 3,
        "low": -4, "high": -2, "answer": -2.91701
    },
}

tol = 1e-2  # желаемая точность

fig, axes = plt.subplots(2, 4, figsize=(24, 8))
fig.suptitle("Градиентный спуск, версия 2", fontweight="bold", fontsize=20)
grid = np.linspace(-3, 3, 100)

is_correct, debug_log = test_convergence_1d(
    grad_descent_v2, test_cases, tol,
    axes, grid
)
plt.show()
if not is_correct:
    print("Не сошлось. Дебажный вывод:")
    for log_entry in debug_log:
        print(log_entry)

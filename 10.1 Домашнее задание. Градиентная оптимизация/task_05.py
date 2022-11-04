'''В этом задании Вы должны найти минимум функций с помощью градиентного спуска.

Вам на вход подаются функция func, ее производная deriv (*), а также начальная точка start, на выходе -
точка локального минимума. Для вашего удобства мы написали функцию для отрисовки траектории градиентного спуска

(*) - вам не нужно будет ее вычислять. То, что вы написали в предыдущем задании, вам пригодится чуть позже.

В первой реализации градиентного спуска можете предполагать, что на вход подаются функции с единственным,
глобальным минимумом. Перед тем, как писать код, ответьте себе на следующие вопросы:

Как понять, что пора остановиться? Это может зависеть от градиента или расстояния между двумя соседними
шагами алгоритма, так и от числа уже выполненных итераций.
Как правильно менять величину шага (learning rate) от итерации к итерации?
В этом пункте гарантируется, что существует решение, использующее обычный градиентный спуск с
фиксированным learning rate и наперёд заданным количеством итераций.'''
from copy import copy, deepcopy
import numpy as np
import matplotlib.pyplot as plt


def grad_descent_v1(f, deriv, x0=None, lr=0.1, iters=1000, callback=None):
    """
    Реализация градиентного спуска для функций с одним локальным минимумом,
    совпадающим с глобальным. Все тесты будут иметь такую природу.
    :param func: float -> float — функция
    :param deriv: float -> float — её производная
    :param x0: float — начальная точка
    :param lr: float — learning rate
    :param iters: int — количество итераций
    :param callback: callable — функция логирования
    """

    if x0 is None:
        # Если точка не дана, сгенерируем случайную
        # из равномерного распределения.
        # При таком подходе начальная точка может быть
        # любой, а не только из какого-то ограниченного диапазона
        # np.random.seed(179)
        x0 = np.random.uniform()

    x = x0

    callback(x, f(x))  # не забывайте логировать
    i = 0
    while (i < iters) or (abs(lr * deriv(x)) > lr):
        x = x - lr * deriv(x)
        callback(x, f(x))
        i = i + 1

    # YOUR CODE. Сделайте итерации градиентного спуска для x

    return x


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
    "square": {
        "func": lambda x: x * x,
        "deriv": lambda x: 2 * x,
        "start": 2,
        "answer": 0.0
    },
    "module": {
        "func": lambda x: abs(x),
        "deriv": lambda x: 1 if x > 0 else -1,
        "start": 2,
        "answer": 0.0
    },
    "third_power": {
        "func": lambda x: abs((x - 1) ** 3),
        "deriv": lambda x: 3 * (x - 1) ** 2 * np.sign(x - 1),
        "start": -1,
        "answer": 1.0
    },
    "ln_x2_1": {
        "func": lambda x: np.log((x + 1) ** 2 + 1),
        "deriv": lambda x: 2 * (x + 1) / (x ** 2 + 1),
        "start": 1,
        "answer": -1.0
    }
}

tol = 1e-2  # желаемая точность
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Градиентный спуск, версия 1", fontweight="bold", fontsize=20)
grid = np.linspace(-2, 2, 100)

is_correct, debug_log = test_convergence_1d(
    grad_descent_v1, test_cases, tol,
    axes, grid
)
plt.show()
if not is_correct:
    print("Не сошлось. Дебажный вывод:")
    for log_entry in debug_log:
        print(log_entry)

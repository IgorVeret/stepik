'''Собственный класс "Нейрон"

Реализуйте класс "Нейрон", у которого будет несколько методов: '''

class Neuron:

    def __init__(self, w, f = lambda x: x):
        self.w=w
        self.f=f

    def forward(self, x):
        self.x=x
        result = sum(map(lambda x, y: x * y, self.w, self.x))
        result=self.f(result)
        return result

    def backlog(self):
        return self.x
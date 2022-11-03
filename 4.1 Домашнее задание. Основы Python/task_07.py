''' Хитрая сортировка
Пусть у нас есть следующий список, в котором элементы -- tuple из строк:
items = [('one', 'two'), ('three', 'four'), ('five', 'six'), ('string', 'a')]
Мы хотим отсортировать этот список по последней букве второго элемента каждого tuple, т.е. получить такой список:
sorted_items = [ ('string', 'a'), ('one', 'two'), ('three', 'four'), ('five', 'six'),]
Напишите код вместо "<YOUR CODE>" в следующем выражении, чтобы получить верную сортировку.
sorted_items = sorted(items, key=lambda x: <YOUR CODE>)
P.S.: в ответе не должно фигурировать слово len .
В этом задании функция print вам не понадобится. Необходимо лишь заполнить пропуск.'''

# items объявлять не нужно
items = [('one', 'two'), ('three', 'four'), ('five', 'six'), ('string', 'a')]
sorted_items = sorted(items, key=lambda x:  x[1][-1])
print(sorted_items)
'''Дан список текстов, слова в которых разделены пробелами
(можно считать, что знаков препинания нет). Часть слов является "мусорными":
в них присутствуют цифры и спецсимволы. Отфильтруйте такие слова из каждого текста.
['1 thousand devils', 'My name is 9Pasha', 'Room #125 costs $100', '888']
Пример вывода:
['thousand devils', 'My name is', 'Room costs', '']
В этом задании функция print вам не понадобится. Результаты выполнения функций нужно возвращать, а не печатать!
Если в тексте все слова являются мусорными, текст должен преобразоваться в пустую строку.'''
def process(sentences):
    dict = []
    for i in sentences:
        words = i.split(' ')
        word_str = ''
        for j in words:
            words_sort = str.isalpha(j)
            if words_sort == True:
                word_str+= j
                word_str += ' '

        dict.append(word_str.strip())

    result = dict
    return result
stroka=['1 thousand devils', 'My name is 9Pasha', 'Room #125 costs $100', '888']
res=process(stroka)
print(res)
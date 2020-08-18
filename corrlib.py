"""
Предобработка данных, подсчёт лагов, корреляция и взаимная информация.
"""
import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
import csv
import random

sns.set(font_scale=1.5)

def corrPirson(a, b):
    """ Корреляция Пирсона.
    Аргументы:
        a, b - NDArray - Исследуемые массивы переменных.
    Возврат:
        float - Коэффициент корреляции.
    """
    a = a - a.mean()
    b = b - b.mean()
    cr = np.dot(a, b) / a.std() / b.std() / a.size
    return cr

def correlation(a, b, method='Pirson', shold=0.05):
    """ Function gets two vectors "a" and "b" as inputs.
    If "method" is 'Pirson' this function returns Pirson correlation and Spearman correlation in others.
    Аргументы:
        a, b - NDArray - Исследуемые массивы переменных.
        method - str - Используемый метод расчёта корреляции Pirson/Spearman (по-умолчанию 'Pirson').
    Возврат:
        float - Коэффициент корреляции.
    """
    if method == 'Pirson':
        cr = sps.pearsonr(a, b)
        if cr[1] <= shold: return cr[0]
    elif method == 'Spearman':
        cr = sps.spearmanr(a, b)
        if cr[1] <= shold: return cr[0]
    return 0.0

def Entropy(x, n):
    """ Расчёт энтропии.
    Аргументы:
        x - NDArray - Исследуемый массив переменных.
        n - NDArray - Массив отсчётов для гистограммы.
    Возврат:
        float - Энтропия.
    """
    Px = np.histogram(x, n)[0]
    Hx = 0
    nrm = Px.sum()
    if nrm > 0:
        Px = Px / nrm
        Hx = -np.dot(Px[Px > 0], np.log(Px[Px > 0]))
    return Hx

def CrossEntropy(x, y, n1, n2):
    """ Расчёт кросс-энтропии.
    Аргументы:
        x, y - NDArray - Исследуемые массивы переменных.
        n1, n2 - NDArray - Массивы отсчётов для гистограммы.
    Возврат:
        float - Кросс-энтропия.
    """
    Pxy = np.histogram2d(x, y, [n1, n2])[0]
    Hxy = 0
    nrm = Pxy.sum()
    if nrm > 0:
        Pxy = Pxy / nrm
        Pxy[Pxy > 0] = -Pxy[Pxy > 0] * np.log(Pxy[Pxy > 0])
        Hxy = Pxy.sum()
    return Hxy

def MutualInformation(x, y, nx, ny):
    """ Расчёт взаимной информации.
    Аргументы:
        x, y - NDArray - Исследуемые массивы переменных.
        n1, n2 - NDArray - Массивы отсчётов для гистограммы.
    Возврат:
        float - Взаимная информация.
    """
    Hx = Entropy(x, nx)
    Hy = Entropy(y, ny)
    Hxy = CrossEntropy(x, y, nx, ny)
    return Hx + Hy - Hxy

def sns_heatmap(data, name_file, cmap="YlGnBu", dpi=600):
    """ Построение тепловой карты.
    Аргументы:
        data - NDArray - Данные 2D для построения.
        name_file - str - Имя сохраняемого файла картинки.
        cmap - Str - Карта цветов для раскраски.
        dpi - int - Качество картинки.
    """
    ax = sns.heatmap(data, cmap=cmap)
    for item in ax.get_yticklabels():
        item.set_rotation(0)
    for item in ax.get_xticklabels():
        item.set_rotation(90)
    plt.savefig(name_file, dpi=dpi)
    plt.show()

def save_csv(data, name_file):
    """ Сохранение CSV файла с данными.
    Аргументы:
        data - NDArray - Данные 2D для сохранения.
        name_file - str - Имя сохраняемого файла.
    """
    with open(name_file, 'wt', newline='') as fout:
        wRt = csv.writer(fout, delimiter=';')
        wRt.writerows(data)

def HeadIndicators(matrix, crr_inds_abs, norma=0.85):
    """ Отбор значимых параметров на основе кросс-корреляций.
    Аргументы:
        matrix - NDArray - Матрица взаимных корреляций.
        crr_inds_abs - NDArray - Массив с отсортированными индексами показателей.
        norma - float - Норма отсечения взаимной корреляции показателей.
    Возврат:
        NDArray - Список индексов значимых параметров.
    """
    blist = np.array([])
    wlist = np.array([])
    for ind in range(matrix.shape[0]):
        if not (crr_inds_abs[ind] in blist):
            wlist = np.append(wlist, crr_inds_abs[ind])
            for jnd in range(matrix.shape[1]):
                if matrix[ind, jnd] > norma:
                    blist = np.append(blist, crr_inds_abs[jnd])
    return wlist

def pred_error_sqr(original, pred):
    """ Подсчёт стандартной среднеквадратичной ошибки.
    Аргументы:
        original - NDArray - Оригинальный вектор значений.
        pred - NDArray - Рассчитанный вектор значений.
    Возврат:
        float - Стандартная среднеквадратичная ошибка.
    """
    return ((pred - original)**2).mean()**0.5 / original.std()

def pred_error_delta(original, pred, delta):
    """ Подсчёт средней абсолютной ошибки в зависимости от размера допустимого диапазона delta.
    Аргументы:
        original - NDArray - Оригинальный вектор значений.
        pred - NDArray - Рассчитанный вектор значений.
        delta - int - Разность между максимальным и минимальным значениями показателя.
    Возврат:
        float - Средняя абсолютная ошибка.
    """
    return np.abs(pred - original).mean() / delta

def pred_error_rmse(original, pred):
    """ Подсчёт среднеквадратического отклонения.
    Аргументы:
        original - NDArray - Оригинальный вектор значений.
        pred - NDArray - Рассчитанный вектор значений.
    Возврат:
        float - Среднеквадратическое отклонение.
    """
    return ((pred - original)**2).mean()**0.5

def calc_short_input(xdata, names, shortNames):
    """ Отбор данных по укороченному списку показателей.
    Аргументы:
        xdata - NDArray - Исходная матрица данных.
        names - NDArray - Весь список колонок данных xdata.
        shortNames - NDArray - Короткий список выбираемых из данных колонок.
    Возврат:
        NDArray - Полученная матрица с урезанными данными.
    """
    indx = []
    for nm in shortNames:
        indx.append(names.index(nm))
    inp = xdata[:,indx]
    return inp

def brute_names(AllNames, num):
    """ Создание случайного набора из указанного количества показателей.
    Аргументы:
        AllNames - NDArray - Весь список показателей.
        num - int - Количество отбираемых показателей.
    Возврат:
        NDArray - Результирующий набор показателей.
    """
    shortNames = np.array([])
    for ind in range(num):
        name = random.choice(AllNames)
        while name in shortNames:
            name = random.choice(AllNames)
        shortNames = np.append(shortNames, name)
    return shortNames

def valcopy(pinp, indx):
    """ Заменяет указанные индексы indx списка inp на ближайшие значения ряда.
    Аргументы:
        inp - NDArray - 1D список значений.
        indx - NDArray - Список индексов для inp.
    Возврат:
        NDArray - Результирующий список с заменёнными значениями.
    """
    inp = np.copy(pinp)
    #if indx == None:
    #    indx = np.where(inp == 0)[0]
    if indx.shape[0] == 0:
        return inp
    beg = 0
    while indx[beg] == beg:
        beg += 1
    if beg < inp.shape[0]:
        for ind in range(beg):
            inp[ind] = inp[beg]
        for ind in indx[beg:]:
            inp[ind] = inp[ind-1]
    return inp

def arrmean(pinp, sigma=3, is_null=False):
    """ Заменяет большие отклонения в списке inp на ближайшие значения ряда.
    Аргументы:
        inp - NDArray - 2D список значений.
        sigma - float - Допустимое отклонение в сигмах.
        is_null - bool - Для расчёта матожидания и отклонения не учитывать нулевые значения данных?
    Возврат:
        NDArray - Результирующий список с заменёнными значениями.
    """
    inp = np.copy(pinp)
    for ind in range(inp.shape[1]):
        inp0 = inp[:,ind]
        inpd = inp0[inp0 != 0] if is_null else inp0
        ma = np.mean(inpd)
        std = sigma * np.std(inpd)
        inp0 = valcopy(inp0, np.where((inp0 < ma - std) | (inp0 > ma + std))[0])
        inp.T[ind] = inp0
    return inp

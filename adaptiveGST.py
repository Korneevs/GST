import numpy as np
import math
import scipy.stats as stats
from scipy.stats import norm
from statsmodels.stats.power import zt_ind_solve_power
from collections.abc import Iterable
import matplotlib.pyplot as plt


class AdaptiveGST(object):
    def __init__(self, p0, lift, alpha=0.05, power=0.8, alternative="two-sided",
                 control=[], test=[], decision_check_flg=False, stop_flg=False, peeking_array=[],
                 iuse=3, phi=1):

        """
        :param p0: значение вероятности при нулевой гипотезе
        :param alpha: ошибка 1 рода
        :param power: мощность теста ( 1 - ошибка второго рода)
        :param lift: минимальный ожидаемый uplift/downlift на тесте
        :param alternative: альтернатива
        :param control: Бинарный массив контрольной вариации
        :param test: Бинарный массив тестовой вариации
        :param stop_flg: Флаг прекращения сэмплирования
        :param peeking_array: Массив долей выборки, в которых происходило подглядывание
        :param iuse: Выбор alpha-spending функции для построения границ
        """

        self.p0 = p0
        self.lift = np.abs(lift)
        self.alpha = alpha
        self.power = power
        self.alternative = alternative
        self.t = np.array([])
        self.decision_check_flg = decision_check_flg
        self.decision = "Тест продолжается"  # Нужно ли?

        self.stop_flg = stop_flg

        self.control = control
        self.test = test
        self.peeking_array = peeking_array
        self.iuse = iuse
        self.phi = phi

    def rep(self, value, length):
        repeated_array = np.full(length, value)
        return repeated_array

    # Функция length нужна только для того, чтобы не обрабатывать отдельно разные кейсы и ошибки
    def length(self, x):
        if x is None:
            return 0
        elif isinstance(x, float) and math.isnan(x):
            return 0
        elif isinstance(x, (int, float, str)):
            return 1
        elif isinstance(x, (list, tuple, np.ndarray)):
            return len(x)
        elif isinstance(x, dict):
            return len(x)
        elif isinstance(x, set):
            return len(x)
        elif isinstance(x, Iterable):
            try:
                return len(x)
            except TypeError:
                raise ValueError("Object is iterable but has no defined length")
        else:
            raise ValueError("Object should be list, tuple, array, string, number, set, or dict")

    # Рассчитываем границы
    def ldBounds(self, t2=None, iuse=3, asf=None, alpha=0.05, phi=np.full(1, 3), sides=2, ztrun=None):
        """
        Расчёт границ в зависимости от выбора

        :param t2: Вторая временная шкала. По дефолту равна первой
        :param iuse: Выбор alpha-spending функции, по дефолу используется power-family
        :param asf: Забейте вообще
        :param alpha: Ошибка первого рода, по дефолту равна 0.05
        :param phi: Степень, используется при iuse = 3 или iuse = 4
        :param sides: Количество сторон
        :param ztrun: Недостижимые границы

        return {
            'bounds_type': Тип границ, 'spending_type': Тип alpha-spending, 'time': Временная шкала, 'time2': Вторая временная шкала,
            'alpha': Заданная ошибка первого рода, 'overall_alpha': Суммарная ошибка первого рода,
            'lower_bounds': [Массив нижних границ, игнорируется при односторонней гипотезе],
            'upper_bounds': [Массив верхних границ], 'exit_pr': [Вектор камулятивных вероятностей пересечения границ],
            'diff_pr': [Вектор вероятностей пересечения границ], 'nom_alpha': [Массив номинальных ошибок первого рода на каждой границе]
            }
        """
        tol = np.sqrt(np.finfo(float).eps)
        t = np.array(self.peeking_array)

        if ztrun is None:
            ztrun = self.rep(np.inf, 1)

        if not np.isscalar(t) and not isinstance(t, (list, tuple, np.ndarray)):
            raise ValueError("'t' must be a vector of analysis times or the number of analysis times")

        if self.length(t) == 1 and isinstance(t, Iterable) == False:
            if abs(t - round(t)) < tol:
                t = np.arange(1, t + 1) / t
            elif t > 1:
                raise ValueError("t must be an integer or in (0,1]")

        if t2 is None:
            t2 = t
        if self.length(t) != self.length(t2):
            raise ValueError("Original and second time scales must be vectors of the same length.")

        if np.min(t) < tol or np.max(t) > 1 + tol or np.min(t2) < tol:
            raise ValueError("Analysis times must be in (0,1]. Second time scale values must be positive.")

        t3 = t2
        t2 = t2 / np.max(t2)

        if np.any(np.diff(t) < tol) or np.any(np.diff(t2) < tol):
            raise ValueError("Analysis times must be ordered from smallest to largest.")

        if np.any(alpha < tol) or np.sum(alpha) > 1 + tol:
            raise ValueError("Each component of alpha must be positive and their sum cannot exceed 1.")

        if sides not in [1, 2]:
            raise ValueError("Sides must be 1 or 2.")

        type_ = 0

        if sides == 1:
            if iuse == 5:
                if callable(asf):
                    asf = list(asf)
                elif isinstance(asf, list):
                    if not callable(asf[0]):
                        raise ValueError("If iuse==5, asf must be a function or list of functions.")
                alpha = asf[0](1)
            type_ = 1
        else:
            # sym = lambda x: len(x) == 1 or x[0] == x[1]
            def sym(x):
                if isinstance(x, (int, float)):
                    return True
                elif isinstance(x, (list, tuple, np.ndarray)):
                    if self.length(x) == 1 or (self.length(x) > 1 and x[0] == x[1]):
                        return True
                return False

            # type_ = 2 if (sym(iuse) and (sym(phi) or 3 not in iuse) and (len(asf) == 1 or iuse[0] != 5) and (sym(np.array(alpha)) or iuse[0] == 5)) else 3
            if type_ == 1:
                type_ = 1
            elif (isinstance(iuse, (list, tuple, np.ndarray)) == True):
                if (sym(iuse) and (sym(phi) or not (iuse in range(3, 5))) and (self.length(asf) == 1 or iuse != 5) and (
                        sym(alpha) or iuse == 5)):
                    type_ = 2
            elif (isinstance(iuse, (int, float)) == True):
                if (sym(iuse) and (sym(phi) or (iuse != 3 and iuse != 4)) and (self.length(asf) == 1 or iuse != 5) and (
                        sym(alpha) or iuse == 5)):
                    type_ = 2
            else:
                type_ = 3

            if (self.length(alpha) == 1 and type_ == 3):
                print("Warning: Asymmetric boundary with alpha of length 1.")
            if type_ == 2:
                if iuse == 5:  # Почему просто не написать iuse == 5
                    if callable(asf):
                        asf = [asf]
                    elif isinstance(asf, list):
                        if not callable(asf[0]):
                            raise ValueError("If iuse==5, asf must be a function or list of functions.")
                    alpha = asf[0](1)

            else:
                iuse = (self.rep(iuse, 2))
                phi = (self.rep(phi, 2))
                ztrun = np.array(self.rep(ztrun, 2))  # Не уверен, нужен ли np.array
                asfTmp = [None, None]
                alphaTmp = [np.nan, np.nan]
                for i in range(2):
                    if iuse[i] == 5:
                        if isinstance(asf, list) and callable(asf[i]):
                            asfTmp[i] = asf[i]
                        elif callable(asf):
                            asfTmp[i] = asf
                        else:
                            raise ValueError("asf must be a function or list of functions.")
                        alphaTmp[i] = asfTmp[i](1)
                        if self.length(alpha) == 2:
                            print(f"Warning: alpha for {['lower', 'upper'][i]} boundary ignored.")
                    else:
                        alphaTmp[i] = alpha / 2 if self.length(alpha) == 1 else alpha[i]
                        if iuse[i] == 3 and phi[i] <= 0:
                            raise ValueError("For power family (iuse=3), phi must be positive.")
                        elif iuse[i] == 4 and phi[i] == 0:
                            raise ValueError("For Hwang-Shih-DeCani family (iuse=4), phi cannot be 0.")
                alpha = alphaTmp
                asf = asfTmp

        if type_ <= 2:
            ld = self.landem(t, t2, sides, iuse, asf, np.sum(alpha), int(phi[0]), ztrun[0])
            ubnd = ld['upper_bounds']
            lbnd = ld['lower_bounds']
            epr = ld['exit_pr']
            dpr = ld['diff_pr']
            spend = ld['spend']
        else:
            ld1 = self.landem(t, t2, 1, iuse, asf[0], alpha[0], phi[0], ztrun[0])
            ld2 = self.landem(t, t2, 1, iuse, asf[1], alpha[1], phi[1], ztrun[1])
            lbnd = -ld1['upper_bounds']
            ubnd = ld2['upper_bounds']
            epr = ld1.exit_pr + ld2.exit_pr
            dpr = ld1.diff_pr + ld2.diff_pr
            spend = np.concatenate([ld1.spend, ld2.spend])

        nom_alpha = 1 - norm.cdf(ubnd) + norm.cdf(lbnd)
        if type_ == 3:
            nom_alpha = self.rep(np.nan, self.length(nom_alpha))

        return {
            "bounds_type": type_,
            "spending_type": spend,
            "time": t,
            "time2": t3,
            "alpha": alpha,
            "overall_alpha": np.sum(alpha),
            "lower_bounds": lbnd,
            "upper_bounds": ubnd,
            "exit_pr": epr,
            "diff_pr": dpr,
            "nom_alpha": nom_alpha
        }

    def fcab(self, last, nints, yam1, h, x, stdv):
        """
        Вычисление площади под кривой

        :param last: Вектор значений функции плотности вероятности для предыдущего интервала
        :param nints: Количество интервалов разбиения по сетке.
        :param yam1: Нижняя граница сетки, на которой будет рассчитываться площадь.
        :param h: Шаг сетки, используется для создания равномерных интервалов разбиения.
        :param x: Массив значений, для которых рассчитываются значения функции плотности вероятности на каждом шаге анализа
        :param stdv: Стандартное отклонение, используется для масштабирования распределения

        return Площадь под кривой плотности вероятности

        """
        matrix = np.stack([x] * (int(nints) + 1), axis=1)
        f = last * norm.pdf(h * np.arange(nints + 1) + yam1, loc=matrix, scale=stdv)
        area = 0.5 * h * (2 * np.sum(f, axis=1) - f[:, 0] - f[:, -1])
        return area

    def qp(self, xq, last, nints, yam1, ybm1, stdv):

        """
        Оценка типичной вероятности первого рода

        :param xq: Значение, по которому будет оцениваться функция распределения.
        :param last: Вектор значений функции плотности вероятности для предыдущего интервала
        :param nints: Количество интервалов разбиения по сетке. Используется для построения сетки значений, по которым будет рассчитываться площадь.
        :param yam1: Нижняя граница сетки, на которой будет рассчитываться площадь.
        :param ybm1: Верхняя граница сетки, на которой будет рассчитываться площадь
        :param stdv: Стандартное отклонение, используется для масштабирования распределения

        return Численное значение интеграла функции распределения на заданном интервале, что позволяет оценить вероятность типичной ошибки первого рода для данного анализа

        """

        hlast = (ybm1 - yam1) / nints
        grid = np.linspace(yam1, ybm1, int(nints) + 1)
        fun = last * stats.norm.cdf(grid, loc=xq, scale=stdv)
        qp = 0.5 * hlast * (2 * np.sum(fun) - fun[0] - fun[-1])  # This is "trap"
        return qp

    def bsearch(self, last, nints, i, pd, stdv, ya, yb):

        """
        Бинарный поиск верхней границы

        :param last: Вектор значений функции плотности вероятности для предыдущего интервала
        :param nints: Количество интервалов разбиения по сетке.
        :param i: Индекс текущего интервала.
        :param pd: Значение вероятности типа I ошибки, которое нужно найти
        :param stdv: Стандартное отклонение, используется для масштабирования распределения
        :param ya: Нижняя граница предыдущего интервала
        :param yb: Верхняя граница предыдущего интервала

        return Верхняя граница, при которой функция интегральной вероятности достигает заданного уровня ошибки типа I.

        """

        tol = 10 ** (-7)
        del_ = 10

        # Проверка, является ли pd массивом или скаляром
        is_scalar = isinstance(pd, (float, int))
        if is_scalar:
            pd = [pd]

        ybvals = []

        for pd_val in pd:
            uppr = yb[i - 1]
            q = self.qp(uppr, last, int(nints[i - 1]), ya[i - 1], yb[i - 1], stdv)

            while abs(q - pd_val) > tol:
                del_ = del_ / 10
                incr = 2 * int(q > pd_val + tol) - 1
                j = 1
                while j <= 50:
                    uppr = uppr + incr * del_
                    q = self.qp(uppr, last, int(nints[i - 1]), ya[i - 1], yb[i - 1], stdv)
                    if abs(q - pd_val) > tol and j == 50:
                        print(f"Failed to converge for pd_val = {pd_val}, last uppr = {uppr}, last q = {q}")
                        raise ValueError("Error in search: not converging")
                    elif (incr == 1 and q <= pd_val + tol) or (incr == -1 and q >= pd_val - tol):
                        break
                    j += 1

            ybvals.append(uppr)

        # Если на входе был скаляр, вернуть скаляр
        if is_scalar:
            return ybvals[0]
        else:
            return ybvals

    def alphas(self, iuse, asf, alpha, phi, side, t, pe=None):

        """
        Определение, как изменяется вероятность ошибки I рода на различных этапах анализа, в зависимости от выбранной функции расходов

        :param iuse: Индекс выбора alpha-spending функции
        :param asf: Количество интервалов разбиения по сетке.
        :param alpha: Общая ошибка I рода.
        :param phi: Степень (только если iuse == 3 или iuse == 4)
        :param side: Количество сторон (1 или 2) для теста.
        :param t: Вектор временных точек анализа.
        :param pe: Опциональный параметр, если задано значение ошибки типа I.

        return {'pe': Камулятивная ошибка первого рода, 'pd': На каждом отдельном шаге, 'spend': Тип alpha-spanding функции}

        """

        tol = 1e-13
        if pe is not None:
            pe = pe
            spend = ""
        elif iuse == 1:
            pe = 2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - (alpha / side) / 2) / np.sqrt(t)))
            spend = "O'Brien-Fleming"
        elif iuse == 2:
            pe = (alpha / side) * np.log(1 + (np.exp(1) - 1) * t)
            spend = "Pocock"
        elif iuse == 3:
            pe = (alpha / side) * t ** phi
            if phi == 1:
                spend = "Power Family: alpha * t"
            else:
                spend = f"Power Family: alpha * t^{round(phi, 2)}"
        elif iuse == 4:
            pe = (alpha / side) * (1 - np.exp(-phi * t)) / (1 - np.exp(-phi))
            spend = "Hwang-Shih-DeCani Family"
        elif iuse == 5:
            if alpha is None:
                alpha = asf(1)
            if np.any(np.diff(asf(t)) <= 1e-7):
                raise ValueError("Alpha Spending function must an increasing function.")
            if asf(1) > 1:
                raise ValueError("Alpha Spending function must be less than or equal to 1.")
            spend = "User-specified spending function"
            pe = (1 / side) * asf(t)
        else:
            raise ValueError("Must choose 1, 2, 3, 4, or 5 as spending function.")

        pe = side * pe
        pd = np.diff(np.concatenate(([0], pe)))
        if np.sum((pd < 0.0000001 * (-1)) | (pd > 1.0000001)) >= 1:
            print("Warning: Spending function error")
            pd = np.clip(pd, 0, 1)

        for j, p in enumerate(pd):
            if p < tol:
                print(
                    f"Warning: Type I error spent too small for analysis #{j + 1}, using 0 as approximation for {p}")
                pd[j] = 0

        return {'pe': pe, 'pd': pd, 'spend': spend}

    # landem

    def landem(self, t, t2, side, iuse, asf, alpha, phi, ztrun, pe=None):

        """
        Вычисляет границы остановки и другие статистические показатели для последовательного анализа данных,
        используя заданное правило расходования ошибки I рода.

        :param t: Вектор временных точек анализа
        :param t2: Второй вектор временных точек анализа
        :param side: Количество сторон, зависящее от гипотезы.
        :param iuse: Индекс выбранного типа alpha-spending функции
        :param asf: Своя функция
        :param alpha: Заданная ошибка I рода
        :param phi: Степень (только если iuse == 3 или iuse == 4)
        :param ztrun: Значение, до которого можно обрезать границы
        :param pe: Опциональный параметр, если задано значение ошибки типа I.

        return {lower_bounds: [Вектор нижних границ], upper_bounds: [Вектор верхних границ], 'exit_pr': [Вектор камулятивных вероятностей пересечения границ],
                'diff_pr': [Вектор вероятностей пересечения границ], 'spend': Тип alpha-spending функции
            }

        """

        h = 0.05
        zninf = -8
        tol = np.sqrt(np.finfo(float).eps)
        stdv = np.sqrt([t2[0]] + np.diff(t2).tolist())  # These are subroutine "sd"
        sdproc = np.sqrt(t2)  # These are subroutine "sd"
        alph = self.alphas(iuse, asf, alpha, phi, side, t, pe)
        nints = self.rep(0.0, self.length(t))
        yb = self.rep(0.0, self.length(t))
        ya = self.rep(0.0, self.length(t))
        zb = self.rep(0.0, self.length(t))
        za = self.rep(0.0, self.length(t))
        pd = alph['pd']
        pe = alph['pe']

        if pd[0] == 0:
            zb[0] = -zninf
            if zb[0] > ztrun:
                zb[0] = ztrun
                pd[0] = side * (1 - norm.cdf(zb[0]))
                pe[0] = pd[0]
                if self.length(t) > 1:
                    pd[1] = pe[1] - pe[0]
            yb[0] = zb[0] * stdv[0]
        elif pd[0] < 1:
            zb[0] = norm.ppf(1 - pd[0] / side)
            if zb[0] > ztrun:
                zb[0] = ztrun
                pd[0] = side * (1 - norm.cdf(zb[0]))
                pe[0] = pd[0]
                if self.lenght(t) > 1:
                    pd[1] = pe[1] - pe[0]
            yb[0] = zb[0] * stdv[0]

        if side == 1:
            za[0] = zninf
            ya[0] = za[0] * stdv[0]
        elif side != 1:
            za[0] = -zb[0]
            ya[0] = -yb[0]

        nints[0] = np.ceil((yb[0] - ya[0]) / (h * stdv[0]))

        if self.length(t) >= 2:
            grid = np.linspace(ya[0], yb[0], int(nints[0]) + 1)  # These are "first"
            last = norm.pdf(grid, loc=0, scale=stdv[0])  # These are "first"
            for i in range(1, self.length(t)):
                if pd[i] < 0 or pd[i] > 1:
                    print("Possible error in spending function. May be due to truncation.")
                    pd[i] = np.clip(pd[i], 0, 1)
                if pd[i] < tol:
                    zb[i] = -zninf
                    if zb[i] > ztrun:
                        zb[i] = ztrun
                        pd[i] = side * self.qp(zb[i] * sdproc[i], last, nints[i - 1], ya[i - 1], yb[i - 1], stdv[i])
                        pe[i] = pd[i] + pe[i - 1]
                        if i < self.length(t) - 1:
                            pd[i + 1] = pe[i + 1] - pe[i]
                    yb[i] = zb[i] * sdproc[i]
                elif pd[i] == 1:
                    zb[i] = yb[i] = 0
                elif tol <= pd[i] < 1:
                    yb[i] = self.bsearch(last, nints, i, pd[i] / side, stdv[i], ya, yb)
                    zb[i] = yb[i] / sdproc[i]
                    if zb[i] > ztrun:
                        zb[i] = ztrun
                        pd[i] = side * self.qp(zb[i] * sdproc[i], last, nints[i - 1], ya[i - 1], yb[i - 1], stdv[i])
                        pe[i] = pd[i] + pe[i - 1]
                        if i < self.length(t) - 1:
                            pd[i + 1] = pe[i + 1] - pe[i]
                    yb[i] = zb[i] * sdproc[i]

                if side == 1:
                    ya[i] = zninf * sdproc[i]
                    za[i] = zninf
                elif side == 2:
                    ya[i] = -yb[i]
                    za[i] = -zb[i]

                nints[i] = np.ceil((yb[i] - ya[i]) / (h * stdv[i]))

                if i < self.length(t) - 1:
                    hlast = (yb[i - 1] - ya[i - 1]) / nints[i - 1]  # These are "other"
                    x = np.linspace(ya[i], yb[i], int(nints[i]) + 1)  # These are "other"
                    last = self.fcab(last, nints[i - 1], ya[i - 1], hlast, x, stdv[i])  # These are "other"

        # za[za < 7.9995 * (-1)] = -np.inf
        # zb[zb > 7.9995] = np.inf
        return {
            "lower_bounds": za,
            "upper_bounds": zb,
            "exit_pr": pe,
            "diff_pr": pd,
            "spend": alph['spend']
        }

    def z_statistics_camulative(self, test, control):

        """
        Вычисление камулятивной z-статистики

        :param test: Массив значений тестовой вариации
        :param control: Массив значений контрольной вариации

        return z-статистика

        """

        x_test = np.sum(test)
        x_control = np.sum(control)

        # Размеры выборок
        n_test = len(test)
        n_control = len(control)

        # Доли успехов
        p_test = x_test / n_test
        p_control = x_control / n_control

        # Общая доля успехов для объединённой выборки
        p_pool = (x_test + x_control) / (n_test + n_control)

        # Вычисление z-статистики
        z_value = (p_test - p_control) / np.sqrt(p_pool * (1 - p_pool) * (1 / n_test + 1 / n_control))

        return z_value

    def sample_size(self, ratio=1):

        """
        Вычисление необходимого размера выборки для классического z-теста

        :param ratio: соотношение размера выборок, по дефолту 1

        return Суммарный размер выборки на 2 вариации

        """

        effect = self.lift
        p0 = self.p0
        alpha = self.alpha
        power = self.power
        if self.alternative == "two-sided":
            multiplier = 1
        else:
            multiplier = 2

        if effect > 0:
            effect_size = (effect / 100) * p0 / np.sqrt(
                p0 * (1 - p0))  # Пропорциональный эффект в стандартных отклонениях
            nobs1 = zt_ind_solve_power(effect_size=effect_size, nobs1=None, alpha=alpha * multiplier, power=power,
                                       ratio=ratio)
            sample_size = round(nobs1 + nobs1 * ratio)

        else:
            sample_size = self.t * 1000

        return sample_size

    def sample_ratio(self, control, test):

        """
        Вычисление доли выборки от необходимого размера

        :param test: Массив значений в тестовой вариации
        :param control: Массив значений в контрольной вариации

        return Доля выборки в момент подглядывания

        """

        n_min = min(len(test), len(control))
        sample_ratio = n_min / (self.sample_size() / 2)

        return sample_ratio

    def check_result(self, test, control, sample_size=0):

        """
        Проверка результата теста

        :param test: Массив значений в тестовой вариации
        :param control: Массив значений в контрольной вариации
        :param sample_size: Необходимый размер выборки на две вариации

        return [Результат, Доля выборки, Вектор временных значений]

        """

        if sample_size == 0:
            sample_size = self.sample_size()

        ratio = self.sample_ratio(control, test)
        if ratio <= 1:
            self.peeking_array.append(ratio)
        else:
            self.peeking_array.append(1)

        if self.alternative == "two-sided":

            lower_bounds = self.ldBounds(iuse=self.iuse, phi=np.full(1, self.phi))['lower_bounds']
            upper_bounds = self.ldBounds(iuse=self.iuse, phi=np.full(1, self.phi))['upper_bounds']

            cur_test = test
            cur_control = control

            if self.z_statistics_camulative(cur_test, cur_control) > upper_bounds[-1]:
                result = 1
                self.stop_flg = True
            elif self.z_statistics_camulative(cur_test, cur_control) < lower_bounds[-1]:
                result = -1
                self.stop_flg = True
            else:
                if ratio >= 1:
                    result = 0
                    self.stop_flg = True
                else:
                    result = 42


        elif self.alternative == "right-sided":
            upper_bounds = self.ldBounds(iuse=self.iuse, phi=np.full(1, self.phi), sides=1)['upper_bounds']

            cur_test = test
            cur_control = control

            if self.z_statistics_camulative(cur_test, cur_control) > upper_bounds[-1]:
                result = 1
                self.stop_flg = True
            else:
                if ratio >= 1:
                    result = 0
                    self.stop_flg = True
                else:
                    result = 42

        elif self.alternative == "left-sided":
            lower_bounds = - self.ldBounds(iuse=self.iuse, phi=np.full(1, self.phi), sides=1)['upper_bounds']

            cur_test = test
            cur_control = control

            if self.z_statistics_camulative(cur_test, cur_control) < lower_bounds[-1]:
                result = -1
                self.stop_flg = True
            else:
                if ratio >= 1:
                    result = 0
                    self.stop_flg = True
                else:
                    result = 42

        else:
            raise ValueError("Alternative should be two-sides, right-sided or left-sided")

        if result == 1:
            desc = "Тестовая вариация победила, тест можно останавливать"
        elif result == 0:
            desc = "Нет стат.значимого эффекта, тест можно останавливать"
        elif result == -1:
            desc = "Контрольная вариация победила, тест можно останавливать"
        elif result == 42:
            desc = "Продолжаем сэмплирование"

        return desc, ratio, self.peeking_array

    def check_deadline(self, test, control, sample_size=0):

        if sample_size == 0:
            sample_size = self.sample_size()

        if self.sample_ratio(test, control) >= 1:
            self.decision_check_flg = True

    def describe_gst(self, peeking_array, upper_bounds, lower_bounds, z_array):
        # Создаем фигуру и оси
        fig, ax = plt.subplots()

        ax.plot(peeking_array, lower_bounds, marker='o', color='white', label='Lower Bounds')
        ax.plot(peeking_array, upper_bounds, marker='o', color='white', label='Upper Bounds')
        ax.plot(peeking_array, z_array, color='yellow', label='Z Array')

        # Фиксируем ось X от 0 до 1.2
        ax.set_xlim(0, 1.2)

        ax.axvline(x=1.0, color='black', linestyle='--', label='x = 1.0')

        ax.grid(True, which='both')
        ax.set_facecolor('gray')  # черный фон

        ax.legend()
        ax.set_xlabel('Peeking Array')
        ax.set_ylabel('Values')

        plt.show()







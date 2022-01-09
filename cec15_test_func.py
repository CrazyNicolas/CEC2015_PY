import math
import numpy as np
import random

eps = 1e-14
INF = 1e50


def rotatefunc(x, Mr):
    return np.matmul(Mr, x.transpose())


def sr_func(x, Os, Mr, sh):  # shift and rotate
    y = (x[:Os.shape[-1]] - Os) * sh
    return rotatefunc(y, Mr)


# Superclass for all basis problems, for the clarity and simplicity of code, and in the convenience of calling
class Problem:
    def __init__(self, dim, shift, rotate):
        self.dim = dim
        self.shift = shift
        self.rotate = rotate

    def func(self, x):
        return 0

    @staticmethod
    def read(problem_path):  # Read the problem data from files
        with open(problem_path, 'r') as fpt:
            if fpt is None:
                print("\n Error: Cannot open input file for reading \n")
            dim = int(fpt.readline().split()[0])
            dim = dim
            shift = np.zeros(dim)
            rotate = np.eye(dim)
            text = fpt.readline().split()
            for j in range(dim):
                shift[j] = float(text[j])
            for i in range(dim):
                text = fpt.readline().split()
                for j in range(dim):
                    rotate[i][j] = float(text[j])
            return dim, shift, rotate

    @staticmethod
    def generator(problem_type, dim):  # Generate an instance of type-assigned problem
        shift = np.random.random(dim) * 160 - 80
        H = Problem.rotate_gen(dim)
        return eval(problem_type)(dim, shift, H)

    @staticmethod
    def rotate_gen(dim):  # Generate a rotate matrix
        random_state = np.random
        H = np.eye(dim)
        D = np.ones((dim,))
        mat = np.eye(dim)
        for n in range(1, dim):
            x = random_state.normal(size=(dim - n + 1,))
            D[n - 1] = np.sign(x[0])
            x[0] -= D[n - 1] * np.sqrt((x * x).sum())
            # Householder transformation
            Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
            mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
        D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
        # Equivalent to np.dot(np.diag(D), H) but faster, apparently
        H = (D * H.T).T
        return H

    @staticmethod
    def store_instance(instance, filename):  # Store the problem instance into a file
        with open(filename, 'w') as fpt:
            fpt.write(str(instance.dim) + '\n')
            fpt.write(' '.join(str(i) for i in instance.shift))
            fpt.write('\n')
            for i in range(instance.dim):
                fpt.write(' '.join(str(j) for j in instance.rotate[i]))
                fpt.write('\n')


class Sphere(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 1

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        return np.sum(z ** 2)


class Ellipsoidal(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 1

    def func(self, x):
        nx = self.dim
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        i = np.arange(nx)
        return np.sum(np.power(10, 6 * i / (nx - 1)) * (z ** 2))


class Bent_cigar(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 1

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        return z[0] ** 2 + np.sum(np.power(10, 6) * (z[1:] ** 2))


class Discus(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 1

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        return np.power(10, 6) * (z[0] ** 2) + np.sum(z[1:] ** 2)


class Dif_powers(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 1

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        i = np.arange(self.dim)
        return np.power(np.sum(np.power(np.fabs(z), 2 + 4 * i / (self.dim - 1))), 0.5)


class Rosenbrock(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 2.048 / 100

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z_ = z[1:]
        z = z[:-1]
        tmp1 = (z + 1) ** 2 - z_ - 1
        return np.sum(100 * tmp1 * tmp1 + z * z)


class Ackley(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 1

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        sum1 = -0.2 * np.sqrt(np.sum(z ** 2) / self.dim)
        sum2 = np.sum(np.cos(2 * np.pi * z)) / self.dim
        return np.e + 20 - 20 * np.exp(sum1) - np.exp(sum2)


class Weierstrass(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 0.5 / 100

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        a, b, k_max = 0.5, 3.0, 20
        sum1, sum2 = 0, 0
        for k in range(k_max + 1):
            sum1 += np.sum(np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * (z + 0.5)))
            sum2 += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * 0.5)
        return sum1 - self.dim * sum2


class Griewank(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 6

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        s = np.sum(z ** 2)
        p = 1
        for i in range(z.shape[-1]):
            p *= np.cos(z[i] / np.sqrt(1 + i))
        return 1 + s / 4000 - p


class Rastrigin(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 5.12 / 100

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        return np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10)


class Schwefel(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 10

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        a = 4.209687462275036e+002
        b = 4.189828872724338e+002
        z += a
        res = 0
        for i in range(self.dim):
            if z[i] > 500:
                res -= (500 - z[i] % 500) * np.sin(np.power(500 - z[i] % 500, 0.5))
                tmp = (z[i] - 500) / 100
                res += tmp * tmp / self.dim
            elif z[i] < -500:
                res -= (-500.0 + np.fabs(z[i]) % 500) * np.sin(np.power(500.0 - np.fabs(z[i]) % 500, 0.5))
                tmp = (z[i] + 500.0) / 100
                res += tmp * tmp / self.dim
            else:
                res -= z[i] * np.sin(np.power(np.fabs(z[i]), 0.5))
        return res + b * self.dim


class Katsuura(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 5 / 100

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        tmp3 = np.power(self.dim, 1.2)
        res = 1
        for i in range(self.dim):
            temp = 0
            for j in range(32 + 1):
                tmp1 = np.power(2, j, dtype=np.float64)
                tmp2 = tmp1 * z[i]
                temp += np.fabs(tmp2 - np.floor(tmp2 + 0.5)) / tmp1
            res *= np.power(1 + (i + 1) * temp, 10 / tmp3)
        tmp = 10 / self.dim / self.dim
        return res * tmp - tmp


class Grie_rosen(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 5 / 100

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z_ = z[1:]
        _z = z[:-1]
        tmp1 = (_z + 1) ** 2 - z_ - 1
        temp = 100 * tmp1 * tmp1 + _z * _z
        res = np.sum(temp * temp / 4000 - np.cos(temp + 1))
        tmp1 = (z[-1] + 1) * (z[-1] + 1) - z[0] - 1
        temp = 100 * tmp1 * tmp1 + z[-1] * z[-1]
        return res + temp * temp / 4000 - np.cos(temp) + 1


class Escaffer6(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 1

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z_ = np.concatenate((z[1:], z[:1]))
        return np.sum(0.5 + (np.sin(np.sqrt(z ** 2 + z_ ** 2)) ** 2 - 0.5) / ((1 + 0.001 * (z ** 2 + z_ ** 2)) ** 2))


class Happycat(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 5 / 100

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        alp = 1 / 8
        sum_z = np.sum(z - 1)
        r2 = np.sum((z - 1) ** 2)
        return np.power(np.fabs(r2 - self.dim), 2 * alp) + (0.5 * r2 + sum_z) / self.dim + 0.5


class Hgbat(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 5 / 100

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        alp = 1 / 4
        sum_z = np.sum(z - 1)
        r2 = np.sum((z - 1) ** 2)
        return np.power(np.fabs(np.power(r2, 2) - np.power(sum_z, 2)), 2 * alp) + (0.5 * r2 + sum_z) / self.dim + 0.5


# Dictionary of supported problems, in the convenience of calling and composition
functions = {'Sphere': Sphere, 'Ellipsoidal': Ellipsoidal, 'Bent_cigar': Bent_cigar, 'Discus': Discus,
             'Dif_powers': Dif_powers, 'Rosenbrock': Rosenbrock, 'Ackley': Ackley, 'Weierstrass': Weierstrass,
             'Griewank': Griewank,'Rastrigin': Rastrigin, 'Schwefel': Schwefel, 'Katsuura': Katsuura,
             'Grie_rosen': Grie_rosen, 'Escaffer6': Escaffer6, 'Happycat': Happycat, 'Hgbat': Hgbat}


class Hybrid:
    pass


class Composition:
    def __init__(self, problems_path):
        with open(problems_path, 'r') as fpt:
            if fpt is None:
                print("\n Error: Cannot open input file for reading \n")
                return
            cf_num = int(fpt.readline().split()[0])
            self.cf_num = cf_num

            lamda = fpt.readline().split()
            self.lamda = np.zeros(cf_num)
            for i in range(cf_num):
                self.lamda[i] = float(lamda[i])

            sigma = fpt.readline().split()
            self.sigma = np.zeros(cf_num)
            for i in range(cf_num):
                self.sigma[i] = float(sigma[i])

            bias = fpt.readline().split()
            self.bias = np.zeros(cf_num)
            for i in range(cf_num):
                self.bias[i] = float(bias[i])

            self.F = float(fpt.readline().split()[0])

            self.problems = []
            self.dim = 0
            for i in range(cf_num):
                text = fpt.readline().split()
                name = text[0]
                dim = int(text[1])
                self.dim = max(self.dim, dim)
                shift = np.zeros(dim)
                rotate = np.eye(dim)
                text = fpt.readline().split()
                for j in range(dim):
                    shift[j] = float(text[j])
                for i in range(dim):
                    text = fpt.readline().split()
                    for j in range(dim):
                        rotate[i][j] = float(text[j])
                if functions.get(name) is None:
                    print("\n Error: No such problem function: {} \n".format(name))
                    return
                self.problems.append(functions.get(name)(dim, shift, rotate))

    def func(self, x):
        w = np.zeros(self.cf_num)
        for i in range(self.cf_num):
            a = np.sqrt(np.sum((x[:self.problems[i].dim] - self.problems[i].shift) ** 2))
            if a != 0:
                w[i] = 1 / a * np.exp(-np.sum((x[:self.problems[i].dim] - self.problems[i].shift) ** 2) / (2 * self.problems[i].dim * self.sigma[i] * self.sigma[i]))
            else:
                w[i] = INF
        if np.max(w) == 0:
            w = np.ones(self.cf_num)
        res = 0
        for i in range(self.cf_num):
            fit = self.lamda[i] * self.problems[i].func(x) + self.bias[i]
            res += w[i] / np.max(w) * fit
        return res + self.F

    @staticmethod
    def generator(filename, dim=0, cf_num=0, problem_names=None):  # Generate a composition problem and store in a file
        if cf_num <= 0:
            cf_num = np.random.randint(3, 11)  # The number of problems in the composition
        if dim <= 0:
            dim = np.random.randint(30, 101)
        lamda = np.random.random(cf_num)
        sigma = np.random.randint(1, cf_num, cf_num) * 10
        bias = np.random.permutation(cf_num) * 100
        F = np.random.randint(1, 16) * 100
        problems = []
        names = []
        for i in range(cf_num):
            if problem_names is None or len(problem_names) == 0:  # User doesn't assign the problems in the composition
                name = random.sample(list(functions.keys()), 1)[0]
            else:
                name = random.sample(problem_names, 1)[0]
            names.append(name)
            problems.append(Problem.generator(name, dim))
        with open(filename, 'w') as fpt:
            fpt.write(str(cf_num) + '\n')
            fpt.write(' '.join(str(i) for i in lamda))
            fpt.write('\n')
            fpt.write(' '.join(str(i) for i in sigma))
            fpt.write('\n')
            fpt.write(' '.join(str(i) for i in bias))
            fpt.write('\n')
            fpt.write(str(F))
            fpt.write('\n')
            for i in range(cf_num):
                fpt.write(names[i] + ' ' + str(problems[i].dim) + '\n')
                fpt.write(' '.join(str(i) for i in problems[i].shift))
                fpt.write('\n')
                for k in range(problems[i].dim):
                    fpt.write(' '.join(str(j) for j in problems[i].rotate[k]))
                    fpt.write('\n')
        return Composition(filename)


# C = Composition.generator('test.txt', 5, 5, ['Sphere'])
# x = C.problems[np.argmin(C.bias)].shift
# print(C.func(x))



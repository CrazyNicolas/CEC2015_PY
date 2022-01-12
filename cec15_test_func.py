import math
import numpy as np
import random

import torch

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
    def read(problem_path, problem_type, size=1):  # Read the problem data from file
        instances = []
        with open(problem_path, 'r') as fpt:
            if fpt is None:
                print("\n Error: Cannot open input file for reading \n")
                return
            for k in range(size):
                d = fpt.readline().split()
                if len(d) < 1:
                    print("\n Error: Not enough instances for reading \n")
                    return
                name = d[0]
                dim = int(d[1])
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
                instances.append([problem_type, dim, shift, rotate])
            return instances

    @staticmethod
    def generator(problem_type, dim, size=1):  # Generate a specified number(size) of instance data of type-assigned problem
        instances = []
        for i in range(size):
            shift = np.random.random(dim) * 160 - 80
            H = Problem.rotate_gen(dim)
            # instances.append(eval(problem_type)(dim, shift, H))
            instances.append([problem_type, dim, shift, H])
        return instances

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
    def store_instance(instances, filename):  # Store the problem instance data into a file
        size = len(instances)
        mode = 'w'
        for k in range(size):
            with open(filename, mode) as fpt:
                fpt.write(str(instances[k][0]) + " " + str(instances[k][1]) + '\n')
                fpt.write(' '.join(str(i) for i in instances[k][2]))
                fpt.write('\n')
                for i in range(instances[k][1]):
                    fpt.write(' '.join(str(j) for j in instances[k][3][i]))
                    fpt.write('\n')
            mode = 'a'

    @classmethod
    def get_instance(cls, batch_data):  # Transfer a batch of instance data to a batch of instance objects
        types, dims, shifts, Hs = batch_data
        if dims.dim() < 1:
            return eval(types)(np.array(dims), np.array(shifts), np.array(Hs))
        instances = []
        for i in range(len(dims)):
            type, dim, shift, H = types[i], dims[i], shifts[i], Hs[i]
            instances.append(eval(type)(np.array(dim), np.array(shift), np.array(H)))
        return instances


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
        return np.power(np.sum(np.power(np.fabs(z), 2 + 4 * i / max(1, self.dim - 1))), 0.5)


class Rosenbrock(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 2.048 / 100

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z_ = z[1:]
        z = z[:-1]
        tmp1 = z ** 2 - z_
        return np.sum(100 * tmp1 * tmp1 + (z - 1) ** 2)


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
        for i in range(self.dim):
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
                res += (500 - z[i] % 500) * np.sin(np.power(500 - z[i] % 500, 0.5))
                tmp = (z[i] - 500) / 100
                res -= tmp * tmp / self.dim
            elif z[i] < -500:
                res += (-500.0 + np.fabs(z[i]) % 500) * np.sin(np.sqrt(500.0 - np.fabs(z[i]) % 500))
                tmp = (z[i] + 500.0) / 100
                res -= tmp * tmp / self.dim
            else:
                res += z[i] * np.sin(np.sqrt(np.fabs(z[i])))
        return b * self.dim - res


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
        z_ = np.concatenate((z[1:], z[:1]))
        _z = z
        tmp1 = _z ** 2 - z_
        temp = 100 * tmp1 * tmp1 + (_z - 1) ** 2
        res = np.sum(temp * temp / 4000 - np.cos(temp)) + 1
        return res


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
        sum_z = np.sum(z)
        r2 = np.sum(z ** 2)
        return np.power(np.fabs(r2 - self.dim), 1 / 4) + (0.5 * r2 + sum_z) / self.dim + 0.5


class Hgbat(Problem):
    def __init__(self, dim, shift, rotate):
        Problem.__init__(self, dim, shift, rotate)
        self.shrink = 5 / 100

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        sum_z = np.sum(z)
        r2 = np.sum(z ** 2)
        return np.sqrt(np.fabs(np.power(r2, 2) - np.power(sum_z, 2))) + (0.5 * r2 + sum_z) / self.dim + 0.5


# Dictionary of supported problems, in the convenience of calling and composition
functions = {'Sphere': Sphere, 'Ellipsoidal': Ellipsoidal, 'Bent_cigar': Bent_cigar, 'Discus': Discus,
             'Dif_powers': Dif_powers, 'Rosenbrock': Rosenbrock, 'Ackley': Ackley, 'Weierstrass': Weierstrass,
             'Griewank': Griewank,'Rastrigin': Rastrigin, 'Schwefel': Schwefel, 'Katsuura': Katsuura,
             'Grie_rosen': Grie_rosen, 'Escaffer6': Escaffer6, 'Happycat': Happycat, 'Hgbat': Hgbat}


class Hybrid:
    def __init__(self, dim, cf_num, length, shuffle, problems):
        self.dim = dim
        self.cf_num = cf_num
        self.length = length
        self.shuffle = shuffle
        self.problems = problems

    @staticmethod
    def read(problems_path, size=1, align=True):  # Read a specified number of problem data from file
        with open(problems_path, 'r') as fpt:
            if fpt is None:
                print("\n Error: Cannot open input file for reading \n")
                return
            instances = []
            for i in range(size):
                tmp = fpt.readline().split()
                if len(tmp) < 1:
                    print("\n Error: Not enough instances for reading \n")
                    return
                dim = int(tmp[0])
                cf_num = int(tmp[1])

                length_str = fpt.readline().split()
                length = np.zeros(cf_num, dtype=int)
                for i in range(cf_num):
                    length[i] = int(length_str[i])

                shuffle_str = fpt.readline().split()
                shuffle = np.zeros(dim, dtype=int)
                for i in range(dim):
                    shuffle[i] = int(shuffle_str[i])

                problems = []
                for i in range(cf_num):
                    text = fpt.readline().split()
                    name = text[0]
                    d = int(text[1])
                    shift = np.zeros(d)
                    rotate = np.eye(d)
                    text = fpt.readline().split()
                    for j in range(d):
                        shift[j] = float(text[j])
                    for i in range(d):
                        text = fpt.readline().split()
                        for j in range(d):
                            rotate[i][j] = float(text[j])
                    problems.append([name, d, shift, rotate])
                if align:  # If user or subsequent treatments(such as DatasetLoader) requires aligned data
                    for i in range(cf_num):
                        name, d, shift, rotate = problems[i]
                        alg_shift = np.zeros(dim)
                        alg_shift[:d] = shift
                        alg_rotate = np.zeros((dim, dim))
                        alg_rotate[:d, :d] = rotate
                        problems[i] = [name, d, alg_shift, alg_rotate]
                instances.append([dim, cf_num, length, shuffle, problems])
        return instances

    def func(self, x):
        y = x[self.shuffle]
        index = 0
        res = 0
        for i in range(self.cf_num):
            yi = y[index:index + self.length[i]]
            index += self.length[i]
            res += self.problems[i].func(yi)
        return res

    @staticmethod
    def generator(filename, dim=0, cf_num=0, problem_names=None, size=1, store=True, align=True):  # Generate a specified number of hybrid problems and store in a file (or not)
        if cf_num <= 0:
            cf_num = np.random.randint(3, 6)  # The number of problems in the hybrid
        if dim <= 0:
            dim = np.random.randint(3, 11) * 10
        instances = []
        mode = 'w'
        for i in range(size):
            seg = np.random.uniform(0.1, 1, cf_num)
            seg /= np.sum(seg)
            length = np.array(np.round(dim * seg), dtype=int)
            length[length < 1] += 1
            length[-1] = dim - np.sum(length[:-1])
            shuffle = np.random.permutation(dim)
            names = []
            problems = []
            for i in range(cf_num):
                if problem_names is None or len(problem_names) == 0:  # User doesn't assign the problems in the hybrid
                    name = random.sample(list(functions.keys()), 1)[0]
                else:
                    name = random.sample(problem_names, 1)[0]
                names.append(name)
                problems.append(Problem.generator(name, int(length[i]))[0])
            if store:  # If user chooses to store problem set into a file
                with open(filename, mode) as fpt:
                    fpt.write(str(dim) + ' ' + str(cf_num) + '\n')
                    fpt.write(' '.join(str(int(i)) for i in length))
                    fpt.write('\n')
                    fpt.write(' '.join(str(int(i)) for i in shuffle))
                    fpt.write('\n')
                    for i in range(cf_num):
                        fpt.write(str(problems[i][0]) + ' ' + str(problems[i][1]) + '\n')
                        fpt.write(' '.join(str(j) for j in problems[i][2]))
                        fpt.write('\n')
                        for k in range(problems[i][1]):
                            fpt.write(' '.join(str(j) for j in problems[i][3][k]))
                            fpt.write('\n')
            if align:  # If user or subsequent treatments(such as DatasetLoader) requires aligned data
                for i in range(cf_num):
                    name, d, shift, rotate = problems[i]
                    alg_shift = np.zeros(dim)
                    alg_shift[:d] = shift
                    alg_rotate = np.zeros((dim, dim))
                    alg_rotate[:d, :d] = rotate
                    problems[i] = [name, d, alg_shift, alg_rotate]
            instances.append([dim, cf_num, length, shuffle, problems])
            mode = 'a'
        return instances

    @staticmethod
    def get_instance(batch_data, aligned=True):  # Transfer a batch of instance data to a batch of instance objects
        dims, cf_nums, lengths, shuffles, problems = batch_data
        instances = []
        for i in range(len(dims)):
            dim, cf_num, length, shuffle = dims[i], cf_nums[i], lengths[i], shuffles[i]
            problem = []
            for j in range(len(problems)):
                tmp = []
                for k in range(len(problems[j])):
                    tmp.append(problems[j][k][i])
                if aligned:
                    name, d, alg_shift, alg_rotate = tmp
                    shift = alg_shift[:d]
                    rotate = alg_rotate[:d, :d]
                    tmp = [name, d, shift, rotate]
                problem.append(Problem.get_instance(tmp))
            instances.append(eval('Hybrid')(dim, cf_num, length, shuffle, problem))
        return instances


class Composition:
    def __init__(self, dim, cf_num, lamda, sigma, bias, F, problems):
        self.dim = dim
        self.cf_num = cf_num
        self.lamda = lamda
        self.sigma = sigma
        self.bias = bias
        self.F = F
        self.problems = problems

    @staticmethod
    def read(problems_path, size=1):  # Read a specified number of problem data from file
        with open(problems_path, 'r') as fpt:
            if fpt is None:
                print("\n Error: Cannot open input file for reading \n")
                return
            instances = []
            for i in range(size):
                cf_str = fpt.readline().split()
                if len(cf_str) < 1:
                    print("\n Error: Not enough instances for reading \n")
                    return
                cf_num = int(cf_str[0])

                lamda_str = fpt.readline().split()
                lamda = np.zeros(cf_num)
                for i in range(cf_num):
                    lamda[i] = float(lamda_str[i])

                sigma_str = fpt.readline().split()
                sigma = np.zeros(cf_num)
                for i in range(cf_num):
                    sigma[i] = float(sigma_str[i])

                bias_str = fpt.readline().split()
                bias = np.zeros(cf_num)
                for i in range(cf_num):
                    bias[i] = float(bias_str[i])

                F = float(fpt.readline().split()[0])

                problems = []
                dim = 0
                for i in range(cf_num):
                    text = fpt.readline().split()
                    name = text[0]
                    d = int(text[1])
                    dim = max(dim, d)
                    shift = np.zeros(d)
                    rotate = np.eye(d)
                    text = fpt.readline().split()
                    for j in range(d):
                        shift[j] = float(text[j])
                    for i in range(d):
                        text = fpt.readline().split()
                        for j in range(d):
                            rotate[i][j] = float(text[j])
                    if functions.get(name) is None:
                        print("\n Error: No such problem function: {} \n".format(name))
                        return
                    problems.append([name, dim, shift, rotate])
                instances.append([dim, cf_num, lamda, sigma, bias, F, problems])
        return instances

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
    def generator(filename, dim=0, cf_num=0, problem_names=None, size=1, store=True):  # Generate a specified number of composition problems and store in a file (or not)
        if cf_num <= 0:
            cf_num = np.random.randint(3, 11)  # The number of problems in the composition
        if dim <= 0:
            dim = np.random.randint(3, 11) * 10
        instances = []
        mode = 'w'
        for i in range(size):
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
                problems.append(Problem.generator(name, dim)[0])
            if store:
                with open(filename, mode) as fpt:
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
                        fpt.write(names[i] + ' ' + str(problems[i][1]) + '\n')
                        fpt.write(' '.join(str(j) for j in problems[i][2]))
                        fpt.write('\n')
                        for k in range(problems[i][1]):
                            fpt.write(' '.join(str(j) for j in problems[i][3][k]))
                            fpt.write('\n')
            # instances.append(Composition(dim, cf_num, lamda, sigma, bias, F, problems))
            instances.append([dim, cf_num, lamda, sigma, bias, F, problems])
            mode = 'a'
        return instances

    @staticmethod
    def get_instance(batch_data):  # Transfer a batch of instance data to a batch of instance objects
        dims, cf_nums, lamdas, sigmas, biases, Fs, problems = batch_data
        instances = []
        for i in range(len(dims)):
            dim, cf_num, lamda, sigma, bias, F = dims[i], cf_nums[i], lamdas[i], sigmas[i], biases[i], Fs[i]
            problem = []
            for j in range(len(problems)):
                tmp = []
                for k in range(len(problems[j])):
                    tmp.append(problems[j][k][i])
                problem.append(Problem.get_instance(tmp))
            instances.append(eval('Composition')(np.array(dim), np.array(cf_num), np.array(lamda), np.array(sigma),
                                                 np.array(bias), np.array(F), problem))
        return instances


# dim = 5
# x = np.ones(dim)
# for p in list(functions.keys()):
#     ins = Problem.generator(p, dim)
#     print(p, ins.func(x))
# C = Composition.generator('test.txt', 5, 5, ['Sphere'], 2)[0]
# C = Composition.read('test.txt', 3)[0]
# x = C.problems[np.argmin(C.bias)].shift
# print(C.func(x))
# dim = 50
# x = np.zeros(dim)
# # H = Hybrid.generator('test.txt', 5, 5, ['Sphere'], 2)[0]
# H = Hybrid.read('test.txt', 2)[0]
# print(H.func(x))


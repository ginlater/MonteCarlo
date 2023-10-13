import os
import shutil

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号


class MonteCarlo:
    def __init__(self, save_path="./output"):
        # 设置随机变量的数量
        self.n = 10000
        # 设置蒙特卡洛模拟的次数
        self.num_simulations = 100
        # 随机采样方式
        self.random_type = None
        # 输出目录
        self.save_path = save_path
        self.prepare_dirs(save_path)
    def prepare_dirs(self, save_path):
        self.save_path = save_path
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path + "/normal"):
            os.makedirs(save_path + "/normal")
        if not os.path.exists(save_path + "/index"):
            os.makedirs(save_path + "/index")

    def normal_distribution(self):
        return np.random.normal(size=self.n)

    def index_distribution(self):
        return np.random.exponential(scale=1, size=(self.n))


    def print_quantile(self, p_values, statistics):
        # 创建CDF的逆函数
        inverse_cdf = interp1d(p_values, statistics, fill_value="extrapolate")

        # 计算满足P{T ≤ tα} = α的分位数tα
        alpha_values = [0.01, 0.05, 0.1]
        quantiles = inverse_cdf(alpha_values)

        # 打印结果
        for alpha, quantile in zip(alpha_values, quantiles):
            print(f"对于α={alpha}，满足P{{T ≤ tα}} = α的分位数tα为{quantile}")

    def __call__(self, *args, **kwargs): # input (n, num_simulations, [normal or index])
        self.n = args[0]
        self.num_simulations = args[1]
        self.random_type = args[2]
        self.save_flag = args[3]
        self.xmin = args[4]
        self.xmax = args[5]
        self.quantile_print = args[6]
        self.t0 = args[7]

        # 存储统计量的值
        statistics = []

        # 进行蒙特卡洛模拟
        for _ in range(self.num_simulations):
            # 生成n个独立同分布的正态随机变量
            Z = self.normal_distribution() if self.random_type == "normal" else self.index_distribution()

            # 计算均值
            ZM = np.mean(Z)

            # 计算最小值
            Z_min = np.min(Z)

            # 计算统计量
            statistic = np.sqrt(np.sum((Z - ZM) ** 2) / self.n) / (ZM - Z_min)

            # 将统计量的值添加到列表中
            statistics.append(statistic)

        # 对统计量进行排序
        statistics.sort()

        # 计算CDF值
        p_values = np.arange(len(statistics)) / float(len(statistics))

        # 绘制CDF曲线
        plt.plot(statistics, p_values, label='n = {}; ns = {}'.format(self.n, self.num_simulations))
        plt.title('P(统计量 < t)的分布曲线')
        plt.xlabel('统计量')
        plt.ylabel('P(统计量 < t)')
        plt.xlim(self.xmin, self.xmax)  # 限制x轴的范围在-5到5之间
        plt.legend()
        plt.grid(True)
        # plt.show()

        # 找到t=0时的CDF值
        if not self.quantile_print:
            p_at_t0 = np.interp(self.t0, statistics, p_values)
            with open("./temp.txt", "a") as f:
                f.write(str(p_at_t0) + " ")

        if self.save_flag:
            plt.savefig(self.save_path + "/normal/{}_{}.jpg".format(self.n, self.num_simulations)
                        if self.random_type == "normal" else self.save_path + "/index/{}_{}.jpg".format(self.n, self.num_simulations))

        if self.quantile_print:
            self.print_quantile(p_values, statistics)

def main(monteCarlo, random_type):

    # random_type = "normal"
    x_axiv_edge = [0, 0.5] if random_type == "normal" else [0.8, 1.4]

    t0 = 0.25 if random_type == "normal" else 0.975
    n_list = [100, 500, 1000, 5000, 8000]
    n_const = [5000]
    n_infinity = 10000
    num_simulations_list = [100, 500, 1000, 5000, 8000]
    num_simulations_const = [5000]
    num_simulations_infinity = 10000

    # 控制 n 不变， 讨论 num_simulations 对 结果的影响
    # for n in n_list:
    for n in range(500, 8001, 500):
        for num_simulations in num_simulations_const:
            monteCarlo(n, num_simulations, random_type, n == n_list[-1], x_axiv_edge[0], x_axiv_edge[1], False, t0)

    plt.close()
    with open("./temp.txt", "a") as f:
        f.write("\n")

    # 控制 num_simulations 不变， 讨论 n 对 结果的影响
    for n in n_const:
        # for num_simulations in num_simulations_list:
        for num_simulations in range(500, 8001, 500):
            monteCarlo(n, num_simulations, random_type, num_simulations == num_simulations_list[-1], x_axiv_edge[0],
                       x_axiv_edge[1], False, t0)

    plt.close()
    with open("./temp.txt", "a") as f:
        f.write("\n")

    print("n = {} ns = {} 时，曲线无限逼近真实值".format(n_infinity, num_simulations_infinity))
    monteCarlo(n_infinity, num_simulations_infinity, random_type, True, x_axiv_edge[0], x_axiv_edge[1], True, t0)

    plt.close()
    with open("./temp.txt", "a") as f:
        f.write("\n")


def error_rate():
    import numpy as np

    with open('./temp.txt') as f:
        for line in f:
            if len(line) < 2:
                continue
            nums = [float(x) for x in line.split()]
            gt = nums[-1]
            errors = []
            for num in nums[:-1]:
                try:
                    error = abs(num - gt) / gt
                except:
                    error = 0
                errors.append(error)

            with open("./errors_rate.txt", 'a') as f:
                for e in errors:
                    f.write('%.2f%% ' % (e * 100))
            with open("./errors_rate.txt", 'a') as f:
                f.write("\n")



if __name__ == "__main__":
    monteCarlo = MonteCarlo()
    for random_type in ["normal", "index"]:
        print("random_type = {}".format(random_type))
        main(monteCarlo, random_type)
        print("------------------------------------")
    error_rate()


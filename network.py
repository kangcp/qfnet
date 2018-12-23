import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import math

class Network:
    def __init__(self, B=1, expectBS=32, expectUE=200, BS_radius=15, BS_init_power=5,
                 subcarrier=64, net_size=(100, 100), step=2, max_velocity=1, N0=1e-7):
        '''
        :function: 初始化网络
        :param B:
        :param expectBS: 基站数
        :param expectUE: 用户数
        :param BS_radius: 基站通信半径
        :param BS_init_power: 基站初始化功率
        :param subcarrier: 子载波个数
        :param net_size: 网络大小
        :param step: 基站感知网络间隔
        :param max_velocity: 最大速率
        :param N0: 噪声干扰
        '''
        # np.random.seed(1)
        self.expectBS = expectBS
        self.UE_num = expectUE
        self.BS_radius = BS_radius
        self.net_size = net_size
        self.B = B
        self.N0 = N0
        self.subcarrier = subcarrier
        # self.BS_power = (BS_init_power * np.ones((1,expectBS))).squeeze()  # 功率数组
        self.BS_power = (BS_init_power * np.random.rand(1, expectBS)).squeeze()
        self.adjacent_matrix = []  # 邻接矩阵
        self.all_distance = []  # 基站和用户两两之间的距离
        self.BS_UE_distance = []  # 用户到基站的距离
        self.BS_BS_distance = []  # 基站到基站的距离
        self.BS_location = []  # 基站、用户位置
        self.UE_location = []  # 基站、用户位置
        self.UE_move_direction = []  # 用户移动方向
        self.UE_move_velocity = []  # 用户移动速度
        self.STEP = step  # 基站感知网络时间间隔
        self.MAX_VELOCITY = max_velocity  # 最大移动速度
        self.eachBS_number = []  # 每个基站连接的用户数
        self.CSI = []
        self.power_subcarrier = []

    def start(self):
        # UE_number = self.get_ue_number()
        UE_number = self.UE_num
        UE_location = self.get_ue_location(UE_number)
        BS_location = self.get_bs_location()

        # self.UE_num = UE_number
        self.BS_location = BS_location
        self.UE_location = UE_location
        self.UE_move_direction = np.zeros((UE_number,))  # 用户移动方向
        self.UE_move_velocity = np.zeros((UE_number,))  # 用户移动速度

        self.create_topology()

    def create_topology(self):
        '''
        创建拓扑结构
        return：adjacent_matrix：邻接矩阵，1表示相连，0表示不相连
                bs_ue_distance：距离矩阵。采用欧式距离
                bs_ue_location: 位置矩阵。记录基站、用户的位置
                all_distance: 所有的距离
                eachBS_number: 每个基站连接的用户数
                CSI：信道增益信息
                power_subcarrier：每个子载波上的功率
        '''
        BS_UE_location = np.vstack((self.()BS_location, self.UE_location))

        all_distance = squareform(pdist(BS_UE_location, 'euclidean'))  # 基站和用户两两之间的距离
        BS_UE_distance = all_distance[self.expectBS:, :self.expectBS]  # 得到用户到基站的距离
        BS_BS_distance = all_distance[:self.expectBS, :self.expectBS]  # 得到基站到基站的距离
        adjacent_matrix = [np.where(i == np.min(i), 1, 0) if np.min(i) < self.BS_radius else [0] * len(i) for i in
                           BS_UE_distance]  # 邻接矩阵，1表示连接，0表示不相连

        self.adjacent_matrix = np.vstack(adjacent_matrix)
        self.BS_UE_distance = BS_UE_distance
        self.BS_BS_distance = BS_BS_distance
        self.all_distance = all_distance
        '''网络拓扑一旦建立好，这个t时间间隔内基站连接用户数不变，信道不变，根据基站功率，载波功率进行重新分配'''
        self.eachBS_number = self.get_numbers()
        self.CSI = self._channel_model(self.UE_num)
        # print(self.CSI.shape)
        self.power_subcarrier = self._compute_power()


    def get_numbers(self):
        '''得到每个基站连接的用户数'''
        return np.sum(self.adjacent_matrix, axis=0)

    def update(self, t=1):
        '''
        更新网络连接
            更新用户位置
            更新基站用户连接
        '''
        new_location = self.batch_updat_ue_location(self.UE_location, self.UE_move_velocity, self.UE_move_direction)
        new_direction = self.update_ue_move_direction(self.UE_num)
        new_velocity = self.update_ue_velocity(self.UE_num)

        self.UE_move_direction = new_direction
        self.UE_location = new_location
        self.UE_move_velocity = new_velocity
        self.create_topology()

    def update_ue_move_direction(self, UE_num, keep_prob=0.6):
        '''更新用户移动方向,角度,只更新keep_prob'''
        return np.random.rand(UE_num) * 2 * math.pi
        # may_keep = np.random.rand(UE_num)
        # mask = may_keep > keep_prob
        # direction = self.UE_move_direction * (~mask)  + may_keep * mask # 不需要更新+需要更新
        # return direction * 2 * math.pi

    def update_ue_velocity(self, UE_num, keep_prob=0.6):
        '''更新用户移动速度'''
        velocity = np.random.rand(UE_num)
        # may_keep = np.random.rand(UE_num)
        # mask = may_keep > keep_prob
        # velocity = self.UE_move_velocity * (~mask) + may_keep * mask  # 不需要更新+需要更新
        return velocity * self.MAX_VELOCITY

    def batch_updat_ue_location(self, UE_location, UE_move_velocity, UE_move_direction, keep_prob=0.4):
        '''批量更新用户位置'''
        may_keep = np.random.rand(self.UE_num)
        mask = may_keep > keep_prob
        new_location = np.zeros(np.shape(UE_location))
        new_location[:, 0] = UE_location[:, 0] + UE_move_velocity * self.STEP * np.cos(
            UE_move_direction) * mask  # 批量更新x
        new_location[:, 1] = UE_location[:, 1] + UE_move_velocity * self.STEP * np.sin(
            UE_move_direction) * mask  # 批量更新y

        '''走出网络范围的节点从另一端出现'''
        new_location[new_location[:, 0] < 0, 0] = self.net_size[0]
        new_location[new_location[:, 0] > self.net_size[0], 0] = 0
        new_location[new_location[:, 1] < 0, 1] = self.net_size[1]
        new_location[new_location[:, 1] > self.net_size[1], 1] = 0
        return new_location

    def update_ue_location(self, location, velocity, move_direction, t):
        '''根据起始位置、速度、移动方向计算用户的位置'''
        new_x = location[0] + velocity * t * np.cos(move_direction)  # 极坐标转为直角坐标
        new_y = location[1] + velocity * t * np.sin(move_direction)
        new_x = 0 if new_x > self.net_size[0] else self.net_size[0] if new_x < 0 else new_x
        return [new_x, new_y]

    def compute_capacity_on_subcarrier(self, n, k):
        '''计算单个子载波吞吐量'''
        assert k >= 0 and k < self.subcarrier
        assert n >= 0 and n < self.expectBS
        if np.sum(self.adjacent_matrix[:, n]) == 0:  # 该基站没有用户
            return 0
        else:
            # power_subcarrier = self._compute_power()
            interfence = self._compute_interference_on_subcarrier(n)
            interfence_p = np.sum(interfence * self.power_subcarrier[:, k])
            s_p = np.power(10, -3 + self.power_subcarrier[n, k] / 10)
            # print(s_p)
            return np.log(1 + s_p / (interfence_p + self.N0))

    def _compute_interference_on_subcarrier(self, n):
        '''计算n基站的干扰'''
        assert n >= 0 and n < self.expectBS
        if np.sum(self.adjacent_matrix[:, n]) == 0:  # 没有用户，干扰为0
            return np.zeros((1, self.expectBS))
        nums = np.arange(0, self.UE_num, 1)
        nums = nums[self.adjacent_matrix[:, n] == 1]  # 得到用户序号
        d = np.mean(self.BS_UE_distance[nums, :], axis=0)  # 计算距离的平均值
        interfence = d ** (-4)
        interfence[n] = 0
        return interfence

    def compute_capacity(self, power_subcarrier, BS_power):
        '''计算总个网络的吞吐量'''
        interfence = self.compute_interference()
        # power_subcarrier = self._compute_power()
        assert interfence.shape == (self.expectBS, self.expectBS)
        assert power_subcarrier.shape == (self.expectBS, self.subcarrier)
        interfence_p = np.dot(interfence, power_subcarrier)  # 干扰和power_subcarrier对应上
        s_p = np.power(10, -3 + BS_power / 10)
        print('power: \n', s_p)
        return self.B * np.log(1 + np.abs(np.sum(s_p)) / (np.sum(interfence_p) + self.N0))

    def compute_interference(self):
        '''计算所有基站收到的干扰'''
        interfence = []
        for bs in np.arange(self.expectBS):
            interfence.append(self._compute_interference_on_subcarrier(bs))
        return np.vstack(interfence)

    def allocate_subcarrier(self):
        '''子载波分配'''
        pass

    def _compute_power(self):
        '''计算每个基站的子载波上的功率'''
        power_allocation = []
        for index, number in enumerate(self.get_numbers()):
            # CSI = self._channel_model(number)
            if number != 0:
                bs_allocation = self._water_filling(self.BS_power[index], self.CSI)  # 计算每个基站在每个子载波的上功率
                '''每个基站在每个子载波上的功率取平均值'''
                power_allocation.append(np.mean(bs_allocation, axis=0))
            else:
                power_allocation.append([0] * self.subcarrier)
        return np.array(power_allocation)

    def _water_filling(self, BS_power, CSI, B=10000000, N0=1e-7):
        '''
        注水算法进行功率分配
        :param BS_power: 基站的功率
        :param CSI:用户占用的子载波的CSI，没占用的子载波CSI=0
        :param B: 带宽
        :param N0:噪声功率
        :return:
        '''
        # CSI_not_zero = CSI != 0
        CSI_temp = CSI + 1e-10  # 防止分母出现0的情况
        NA = CSI_temp.shape[1]  # 得到每个用户占用的子载波数
        # NA = [len(user[user>0]) for user in CSI]  # 得到每个用户占用的子载波数
        H = CSI_temp / (B * N0)  # the parameter relate to SNR in subchannels
        # H_DAO = 1/H * CSI_not_zero
        power_allocation = (BS_power + np.sum(1 / H, axis=1, keepdims=True)) / NA - 1 / H
        while (len(power_allocation[power_allocation < 0]) > 0):
            indexN = power_allocation <= 0
            indexP = power_allocation > 0
            CSI[indexN] = 0
            MP = [len(user[user > 0]) for user in CSI]
            MP = np.reshape(MP, (CSI.shape[0], 1))  # bug: 没有考虑MP的值等于0的情况
            CSI_temp = CSI + 1e-20
            HT = CSI_temp / (B * N0)
            HT_DAO = 1 / HT * indexP
            power_allocation = ((BS_power + np.sum(HT_DAO, axis=1, keepdims=True)) / (MP + 1e-20) - HT_DAO) * indexP
            power_allocation[indexN] = 0
        return power_allocation

    def _channel_model(self, UE_num, sample_num=10, fm=30):
        '''
        信道模型，采用六径衰落信道模型,信道数为1
        :param UE_num:用户数
        :param sample_num: 采样数
        :param fm: 频率参数
        :return: CSI 返回信道增益状态信息CSI
        '''
        Ts = 1e-3  # Ts=50e-9*10000*2/20; time sample interval
        N = 200  # number of input waves
        n = np.arange(0, N)
        cita = 2 * math.pi * n / N
        alfa_normed = np.zeros((6, N))
        phase_init = np.zeros((6, N))
        subcarrier = self.subcarrier
        BER = 1e-3  # 误码率
        Gap = -np.log(5 * BER) / 1.6  # SNR gap

        I = np.zeros((UE_num, 6, sample_num))
        Q = np.zeros((UE_num, 6, sample_num))
        # env = np.zeros((self.UE_num,6,sample_num))
        CSI = np.zeros((UE_num, subcarrier))  # 信道增益状态信息
        for user in range(UE_num):
            alfa = np.random.rand(6, N)  # magnitudes of input waves
            alfa_normed = alfa / np.linalg.norm(alfa, ord=2, axis=1, keepdims=True)
            phase = np.random.rand(6, N)  # initial phases
            phase_init = phase / np.linalg.norm(phase, ord=2, axis=1, keepdims=True)
            # using one-sided exponential profile
            to = 1 * Ts
            tt = np.arange(0, 6) * Ts
            g = np.exp(-tt / to)
            # I[user,:,:] = g * np.sum(alfa_normed *
            #                             np.cos(2 * np.pi * fm * t * np.cos(cita) + phase_init),axis=1)
            # Q[user,:,:] = g * np.sum(alfa_normed *
            #                             np.sin(2 * np.pi * fm * t * np.cos(cita) + phase_init),axis=1)
            for j in range(0, 6):
                for i in range(0, sample_num):
                    t = i * Ts
                    iss = g[j] * np.sum(alfa_normed[j, :] *
                                        np.cos(2 * np.pi * fm * t * np.cos(cita) + phase_init[j, :]))
                    qss = g[j] * np.sum(alfa_normed[j, :] *
                                        np.sin(2 * np.pi * fm * t * np.cos(cita) + phase_init[j, :]))
                    I[user, j, i] = iss
                    Q[user, j, i] = qss
                    # envs = np.sqrt(qss**2 + iss**2)
                    # env[user,j,i] = envs

        # h = np.random.rand(1,UE_num)
        # gamma = np.sum(0.064 * h[h<0.5]) + np.sum(0.128*h[(h >= 0.5) & (h < 0.8)]) + np.sum(0.256*h[h>0.8])
        # CSI = np.abs(np.fft.fft(I + 1j * Q,subcarrier,axis=1)**2/Gap)
        for diffsamp in np.arange(0, sample_num):
            for i in np.arange(UE_num):
                user = I[i, :, diffsamp] + 1j * Q[i, :, diffsamp]
                CSI[i, :] = np.abs(np.fft.fft(user, subcarrier)) ** 2 / Gap
                # CSI[i,:] = 2 * CSI[i,:] / np.sum(CSI[i,:])  # 归一化
                CSI[i, :] = CSI[i, :] / np.linalg.norm(CSI[i, :], ord=2, keepdims=True)
        return CSI

    def get_ue_number(self):
        '''用户数量是变化的，但是符合泊松分布'''
        return np.random.poisson(lam=self.UE_num)

    def get_ue_location(self, UE_num):  # 用户数最开始是服从泊松分布的
        '''得到UE的位置信息'''
        width = self.net_size[0]
        length = self.net_size[1]
        locationx = np.random.rand(UE_num) * width
        locationy = np.random.rand(UE_num) * length
        UE_location = np.vstack((locationx, locationy))
        return UE_location.T

    def get_bs_location(self):
        '''得到BS的位置信息'''
        width = self.net_size[0]
        length = self.net_size[1]
        locationx = np.random.rand(self.expectBS) * width
        locationy = np.random.rand(self.expectBS) * length
        BS_location = np.vstack((locationx, locationy))
        return BS_location.T

    def draw_net(self, ue_location, bs_location, fig_num):
        '''画出网络图'''
        plt.subplot(fig_num)
        plt.plot(ue_location[:, 0], ue_location[:, 1], '.b')
        plt.hold(True)
        plt.plot(bs_location[:, 0], bs_location[:, 1], '^r')
        plt.show()
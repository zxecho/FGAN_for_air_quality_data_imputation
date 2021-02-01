import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from util_tools import get_time_stamp

torch.cuda.set_device(0)
# 算法参数设置
# 1. Mini batch size
mb_size = 128
# 2. Missing rate
p_miss = 0.1
# 3. Hint rate
p_hint = 0.9
# 4. Loss Hyperparameters
alpha = 10
# 5. Train Rate
train_rate = 0.8

# Hidden state dimensions
H_Dim = 16


# 载入数据
def load_datasets(AQS):
    # 载入气象站数据
    # 载入数据处理
    dataset_list = []
    for station in AQS:
        station_data_list = []
        for i in range(1, 3):
            data_file = station + '_' + str(i) + '.xlsx'
            print(data_file)
            data_file_path = './air_quality_datasets/' + data_file
            air_data_df = pd.read_excel(data_file_path)
            df_sel = air_data_df.loc[:, ['PM2_5', 'PM10', 'SO2', 'O3', 'NOX']]
            df_noNaN = df_sel[df_sel.notnull().sum(axis=1) == 5]
            Data = df_noNaN.values
            print('Data %d shape : \n' % i, Data.shape)
            station_data_list.append(Data)
        station_data = np.concatenate((station_data_list[0], station_data_list[1]), axis=0)
        print('station_data shape: ', station_data.shape)
        dataset_list.append(station_data)

    return dataset_list


def construct_missing_mask(data):
    L = data.shape[0]
    dim = data.shape[1]
    data_shape = data.shape

    line_miss_num = int(1 / 1000 * L)
    row_miss_num = int(1 / 1000 * L)
    p_miss = 0.2

    # 行缺失
    while True:
        rand_m1 = np.random.randint(0, L, line_miss_num)
        u = np.unique(rand_m1)
        if len(u) == line_miss_num:
            break
    line_missing_p_num = line_miss_num * dim  # 计算出行缺失的数量

    # 列缺失
    rand_m2_v = np.random.randint(0, dim, row_miss_num)
    rand_m2_l = np.random.randint(1, L, row_miss_num)
    # 每个列缺失的数据个数，在一定范围内随机
    missing_len = np.random.randint(2, 6, row_miss_num)
    # 计算出列缺失的数量
    row_missing_p_num = 0
    for i in range(len(missing_len)):
        row_missing_p_num += missing_len[i]

    # 点缺失
    miss_num = p_miss * L * dim
    miss_p_num = miss_num - row_missing_p_num - line_missing_p_num
    prob_missing_p = round(miss_p_num / (L * dim), 3)
    # 构造missing 矩阵
    Missing = np.zeros(data_shape)
    p_miss_vec = prob_missing_p * np.ones((L, 1))
    for i in range(dim):
        A = np.random.uniform(0., 1., size=[L, ])
        B = A > p_miss_vec[i]
        Missing[:, i] = 1. * B

    Missing[rand_m1, :] = Missing[rand_m1, :] * 0  # 行缺失赋值

    # 列缺失赋值
    for i in range(row_miss_num):
        Missing[rand_m2_l[i]: rand_m2_l[i] + missing_len[i], rand_m2_v[i]] = Missing[rand_m2_l[i]: rand_m2_l[i] + missing_len[i],
                                                                             rand_m2_v[i]] * 0
    print('Missing: \n', Missing)
    real_missing_num = 1 - Missing
    real_missing_num = real_missing_num.sum()
    print('real missing num :', real_missing_num, 'expect missing num: ', miss_num)

    return Missing


# 构造缺失数据
def construct_train_test_dataset(Data):
    # 数据参数
    No = len(Data)  # 数据总数
    Dim = len(Data[0, :])  #

    # Normalization (0 to 1)
    Min_Val = np.zeros(Dim)
    Max_Val = np.zeros(Dim)

    for i in range(Dim):
        Min_Val[i] = np.min(Data[:, i])
        Data[:, i] = Data[:, i] - np.min(Data[:, i])
        Max_Val[i] = np.max(Data[:, i])
        Data[:, i] = Data[:, i] / (np.max(Data[:, i]) + 1e-6)

    # Missing introducing
    # p_miss_vec = p_miss * np.ones((Dim, 1))
    #
    # Missing = np.zeros((No, Dim))
    #
    # for i in range(Dim):
    #     A = np.random.uniform(0., 1., size=[len(Data), ])
    #     B = A > p_miss_vec[i]
    #     Missing[:, i] = 1. * B

    # 分离出训练和验证的数据集

    idx = np.random.permutation(No)

    Train_No = int(No * train_rate)
    Test_No = No - Train_No

    # Train / Test Features
    trainX = Data[idx[:Train_No], :]
    testX = Data[idx[Train_No:], :]

    trainM = construct_missing_mask(trainX)
    testM = construct_missing_mask(testX)

    # Train / Test Missing Indicators
    # trainM = Missing[idx[:Train_No], :]
    # testM = Missing[idx[Train_No:], :]

    return Dim, trainX, testX, trainM, testM, Train_No, Test_No


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size=size, scale=xavier_stddev)


class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_dim):
        super(Discriminator, self).__init__()
        self.d = nn.Sequential(
            nn.Linear(input_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_size),
            nn.Sigmoid()
        )

    def forward(self, new_x, h):
        inputs = torch.cat(dim=1, tensors=[new_x, h])
        D_prob = self.d(inputs)
        return D_prob


class Generator(nn.Module):

    def __init__(self, input_size, hidden_dim):
        super(Generator, self).__init__()
        self.g = nn.Sequential(
            nn.Linear(input_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_size),
            nn.Sigmoid()
        )

    def forward(self, new_x, m):
        # Mask + Data Concatenate
        inputs = torch.cat(dim=1, tensors=[new_x, m])
        G_prob = self.g(inputs)
        return G_prob


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class GAIN:
    def __init__(self):
        # 构建数据集
        station_dataset = ['A']
        datasets = load_datasets(station_dataset)
        self.Dim, self.trainX, self.testX, self.trainM, self.testM, self.Train_No, self.Test_No = \
            construct_train_test_dataset(datasets[0])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 新建生成器网络G
        self.G = Generator(self.Dim, H_Dim).to(device)
        # 新建判别器网络D
        self.D = Discriminator(self.Dim, H_Dim).to(device)

        # 参数初始化
        self.G.apply(weights_init)
        self.D.apply(weights_init)

        self.optimizer_D = torch.optim.Adam(params=self.D.parameters())
        self.optimizer_G = torch.optim.Adam(params=self.G.parameters())

    def training(self):

        # Start Iterations
        for it in tqdm(range(5000)):

            # %% Inputs
            mb_idx = self.sample_idx(self.Train_No, mb_size)
            X_mb = self.trainX[mb_idx, :]

            Z_mb = self.sample_Z(mb_size, self.Dim)
            M_mb = self.trainM[mb_idx, :]
            H_mb1 = self.sample_M(mb_size, self.Dim, 1 - p_hint)
            H_mb = M_mb * H_mb1

            New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

            X_mb = torch.tensor(X_mb, device="cuda", dtype=torch.float32)
            M_mb = torch.tensor(M_mb, device="cuda", dtype=torch.float32)
            H_mb = torch.tensor(H_mb, device="cuda", dtype=torch.float32)
            New_X_mb = torch.tensor(New_X_mb, device="cuda", dtype=torch.float32)

            self.optimizer_D.zero_grad()
            D_loss_curr = self.comput_D_loss(M=M_mb, New_X=New_X_mb, H=H_mb)
            D_loss_curr.backward()
            self.optimizer_D.step()

            self.optimizer_G.zero_grad()
            G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = self.compute_G_loss(X=X_mb, M=M_mb, New_X=New_X_mb,
                                                                                       H=H_mb)
            G_loss_curr.backward()
            self.optimizer_G.step()

            # %% Intermediate Losses
            if it % 100 == 0:
                print('Iter: {}'.format(it), end='\t')
                print('G loss cur: {}'.format(G_loss_curr.item()), end='\t')
                print('D loss cur: {}'.format(D_loss_curr.item()))
                print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())), end='\t')
                print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr.item())))

    def comput_D_loss(self, M, New_X, H):
        """
        :arg
        M: mask
        New_X: 缺失后的数据
        H： hint
        """
        # 生成器
        G_sample = self.G(New_X, M)
        # 将生成的数据与原数据进行组合，相当于用生成的数据填充原来空缺的地方
        Hat_New_X = New_X * M + G_sample * (1 - M)

        # 判别器
        D_prob = self.D(Hat_New_X, H)

        # 损失函数
        D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1 - M) * torch.log(1. - D_prob + 1e-8))
        return D_loss

    def compute_G_loss(self, X, M, New_X, H):
        """
        :param X:  原数据
        :param M: Mask
        :param New_X: 原数据进行人为缺失和加上扰动
        :param H: Hint
        :return:
        """
        # %% Structure
        # Generator
        G_sample = self.G(New_X, M)

        # Combine with original data
        Hat_New_X = New_X * M + G_sample * (1 - M)

        # Discriminator
        D_prob = self.D(Hat_New_X, H)

        # %% Loss
        G_loss1 = -torch.mean((1 - M) * torch.log(D_prob + 1e-8))
        MSE_train_loss = torch.mean((M * New_X - M * G_sample) ** 2) / torch.mean(M)

        G_loss = G_loss1 + alpha * MSE_train_loss

        # %% MSE Performance metric
        MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / torch.mean(1 - M)
        return G_loss, MSE_train_loss, MSE_test_loss

    def Algo_test(self, X, M, New_X):
        """
        用于测试模型
        :param X: 原数据
        :param M: mask
        :param New_X: 构造出的包含Missing的数据
        :return:
        """
        # %% Structure
        # Generator
        G_sample = self.G(New_X, M)

        # %% MSE Performance metric
        MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / torch.mean(1 - M)
        return MSE_test_loss, G_sample

    def testing(self):
        # 对算法进行测试
        Z_mb = self.sample_Z(self.Test_No, self.Dim)
        M_mb = self.testM
        X_mb = self.testX

        New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

        X_mb = torch.tensor(X_mb, device='cuda', dtype=torch.float32)
        M_mb = torch.tensor(M_mb, device='cuda', dtype=torch.float32)
        New_X_mb = torch.tensor(New_X_mb, device='cuda', dtype=torch.float32)

        MSE_final, Sample = self.Algo_test(X=X_mb, M=M_mb, New_X=New_X_mb)

        print('Final Test RMSE: ' + str(np.sqrt(MSE_final.item())))

        imputed_data = M_mb * X_mb + (1 - M_mb) * Sample
        print("Imputed test data:")
        np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})

        print(imputed_data.cpu().detach().numpy())

    def sample_M(self, m, n, p):
        """
        m: batch_size
        n: Dim 数据的属性维度
        p: 采样概率
        """
        # Hint vector的生成函数
        A = np.random.uniform(0., 1., size=[m, n])
        B = A > p
        C = 1. * B
        return C

    def sample_idx(self, m, n):
        """
        :arg
        n: 采样index的个数
        m: 样本总个数
        """
        A = np.random.permutation(m)
        idx = A[:n]
        return idx

    def sample_Z(self, m, n):
        """
        用于采样随机扰动
        :param m: batch_size
        :param n: Dim, 数据属性维度
        :return:
        """
        return np.random.uniform(0., 0.01, size=[m, n])

    def save_model(self):
        torch.save(self.D.state_dict(), './saved_model/D.pkl')
        torch.save(self.G.state_dict(), './saved_model/G.pkl')

    def load_model(self):
        self.G.load_state_dict(torch.load('./saved_model/G.pkl'))
        self.D.load_state_dict(torch.load('./saved_model/D.pkl'))


if __name__ == '__main__':
    algo = GAIN()
    algo.training()
    algo.testing()
    algo.save_model()

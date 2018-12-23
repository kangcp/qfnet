from qlearning import *
from dlmodel import *
from utils import *
from sklearn.model_selection import train_test_split


def evaluate(ql, dl, epoch=10):
    '''该函数用来判断模型的质量'''
    net = Network()
    net.start()
    model_best = dl.load_best_model()
    ql_only_best = ql.load_qlmodel(Tools.ql_model)
    qf_best = ql.load_qlmodel(Tools.ql_model_pro)
    cap = np.zeros((4, epoch))
    for i in range(epoch):
        net.update()
        epoch_data = Tools.get_epoch_data(net)
        print('正在计算普通情况')
        cap[0, i] = net.compute_capacity(net.power_subcarrier, net.BS_power)
        print('正在计算QFNet情况')
        outs_best = dl.predict(model_best, epoch_data)
        outs_best = np.reshape(outs_best, (net.expectBS, net.subcarrier))
        qfnet_capacity = net.compute_capacity(
            outs_best + net.power_subcarrier,
            net.BS_power + np.sum(outs_best, axis=1)
        )
        qfouts = ql.ql_evaluate(qf_best, net, iteration=1)
        qfouts = np.reshape(qfouts, (net.expectBS, net.subcarrier))
        qfouts_capacity = net.compute_capacity(
            qfouts + net.power_subcarrier,
            net.BS_power + np.sum(qfouts, axis=1)
        )
        cap[1, i] = qfouts_capacity
        print('正在计算ql情况')
        qlouts = ql.ql_evaluate(ql_only_best, net, iteration=1)
        qlouts = np.reshape(qlouts, (net.expectBS, net.subcarrier))
        qlouts_capacity = net.compute_capacity(
            qlouts + net.power_subcarrier,
            net.BS_power + np.sum(qlouts, axis=1)
        )
        cap[2, i] = qlouts_capacity
        cap[3, i] = qfnet_capacity
        print('epoch: ', i, 'normal: ', cap[0, i], 'qfouts: ', cap[1, i], 'ql: ', cap[2, i], 'qfnet: ', cap[3, i])
    dl.draw_capacity(cap, epoch)


def create_only_q():
    net = Network()
    net.start()
    ql = Qlearning()
    if len(os.listdir(Tools.ql_model)) == 0:
        print('从头开始训练')
        ql.start('1', epoch=20, iteration=64, only_q=True)


def draw_convergence():
    only_c = open(Tools.qfnet_data + '/' + '1' + '_only_c.txt', 'r')
    only_c = np.array(only_c.read().split(',')[:-1])
    only_c = str_map_float(only_c)
    dql = open(Tools.qfnet_data + '/' + '0' + '_dqn_c.txt', 'r')
    dql = np.array(dql.read().split(',')[:-1])
    dql = str_map_float(dql)
    n = range(min(len(dql), len(only_c)))
    only_c[50:] = np.array(only_c)[50:]
    dql[50:] = np.array(dql)[50:]
    plt.plot(n, only_c[0:len(n)], 'r', label='Q-learning')
    plt.plot(n, dql[0:len(n)], 'b', label='DQFCNet')
    plt.ylabel('Spectral-Efficiency(bps/Hz)')
    plt.xlabel('iterations')
    plt.legend()
    plt.savefig(Tools.qf_image + '/' + 'convergence.png')
    plt.close()


def start_net():
    net = Network()
    net.start()
    qfnet = DLModel()
    qlearning = Qlearning()
    for i in range(5):
        model_name = str(i)
        print('生成第' + model_name + '个model')
        qlearning.start(model_name, epoch=20, iteration=64, only_q=False)
        model = qfnet.build_model()
        x_train, y_train = qfnet.load_data(model_name)
        train_data, _, train_lab, _ = train_test_split(x_train, y_train, test_size=0.2, random_state=36)
        qfnet.train(model, train_data, train_lab, epoch=20)
        # qfnet.train(model, x_train, y_train, epoch=50)
        qfnet.evaluate(model, epoch=10)

        evaluate(qlearning, qfnet, epoch=10)
        # qlearning.remove_data()  # 删除训练数据
        # qfnet.model_summary()
        # qfnet.examples(net, 'QFNet_' + str(3) + '.jpg')


if __name__ == '__main__':

    Tools.create_dirs()
    create_only_q()
    start_net()
    # draw_convergence()
    '''画基站位置'''
    # net = Network()
    # net.start()
    # fig = plt.figure(1)
    # window1 = fig.add_subplot(111)
    #
    # '''画出基站'''
    # window1.scatter(net.BS_location[:, 0], net.BS_location[:, 1],marker='^', c='r')
    # # window2 = fig.add_subplot(111)
    # window1.scatter(net.UE_location[:, 0], net.UE_location[:, 1], marker='*', c='b')
    # plt.legend(['BS','UE'])
    # plt.show()
    # window1.scatter(net.BS_location[:, 0], net.BS_location[:, 1], marker='^', c='b')

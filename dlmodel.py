from qlearning import *
from utils import *
from network import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras import regularizers
from keras.utils import plot_model
from keras.utils import multi_gpu_model


set_gpu()


class DLModel:
    def __init__(self):
        self.layers_units = [14880, 10000, 5000, 2048]

    def build_model(self):
        model = Sequential()
        model.add(
            Dense(units=self.layers_units[1],
                  activation='relu',
                  input_dim=self.layers_units[0],
                  name='ds1_10000')
        )
        model.add(Dropout(rate=0.5))
        model.add(
            Dense(units=self.layers_units[2],
                  activation='relu',
                  kernel_regularizer=regularizers.l2(0.01),
                  name='ds2_5000')
        )
        model.add(Dropout(rate=0.5))
        model.add(
            Dense(units=self.layers_units[3],
                  activation='relu',
                  kernel_regularizer=regularizers.l2(0.01),
                  name='out_2048')
        )
        if is_gpu_available():
            gpus = get_available_gpus()
            model = multi_gpu_model(model, gpus)
        return model

    def generate_batch_data_random(self, data, labels, batch_size):
        ylen = len(labels)
        loopcount = ylen // batch_size
        while True:
            i = np.random.randint(0, loopcount)
            yield (data[i * batch_size: (i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size])

    def train(self, model, x_train, y_train, epoch=200):
        model.compile(
            optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999),
            loss=Tools.adjust_loss,
            metrics=[Tools.my_metrics]
        )
        model.fit_generator(self.generate_batch_data_random(x_train, y_train, 32),
                            steps_per_epoch=10, epochs=epoch)
#        model.fit(
#            x_train, y_train,
#            batch_size=32,
#            epochs=epoch
#        )

    def evaluate(self, model, epoch=100):
        """
        评估当前模型与最佳模型的好坏
        :param model: 当前模型
        :param net:  网络
        :param epoch: 评估次数
        :return: 更新最佳模型
        """
        model_best = self.load_best_model()
        net = Network()
        net.start()
        scores = [0, 0]
        #cap = [[], []]
        if model_best is not None:
            for i in range(epoch):
                '''随机产生网络拓扑，来评估两个网络的调节能力'''
                net.update()
                epoch_data = Tools.get_epoch_data(net)
                outs_best = self.predict(model_best, epoch_data)
                outs_best = np.reshape(outs_best, (net.BSs, net.subcarrier))
                outs_best_capacity = net.compute_capacity(
                    outs_best + net.power_subcarrier,
                    net.BS_power + np.sum(outs_best, axis=1)
                )
                outs = self.predict(model, epoch_data)
                outs = np.reshape(outs, (net.BSs, net.subcarrier))
                outs_capacity = net.compute_capacity(
                    outs + net.power_subcarrier,
                    net.BS_power + np.sum(outs, axis=1)
                )
                if outs_best_capacity > outs_capacity:
                    scores[0] += 1
                    print('最佳模型胜出...' + '得分：', scores[0])
                else:
                    scores[1] += 1
                    print('当前模型胜出...' + '得分：', scores[1])
                print('outs_capacity ', outs_capacity)
                print('outs_best_capacity', outs_best_capacity)
#                cap[0, i] = outs_best_capacity
#                cap[1, i] = outs_capacity
#            self.draw_capacity(cap, epoch, labels=['best model', 'model', ''])
        print('得分情况：best_model:normal', scores[0], ':', scores[1])
        self.save_weights(model)
        if scores[0] > scores[1]:  # 当前模型不是最佳模型
            print('当前模型不是最佳模型...')
            self.save_weights(model_best)


    def examples(self, epoch=20):
        '''该函数用来判断模型的质量'''
        model_best = self.load_best_model()

        net = Network()
        net.start()
        if model_best is not None:
            cap = np.zeros((2, epoch))
            for i in range(epoch):
                net.update()
                epoch_data = Tools.get_epoch_data(net)
                cap[0, i] = net.compute_capacity(net.power_subcarrier, net.BS_power)
                outs_best = self.predict(model_best, epoch_data)
                outs_best_capacity = net.compute_capacity(
                    np.reshape(outs_best + net.power_subcarrier.flatten(), (net.BSs, net.subcarrier)),
                    net.BS_power + np.sum(outs_best)
                )
                cap[1, i] = outs_best_capacity
                print('epoch: ', i, 'normal: ', cap[0, i], 'qfnet: ', cap[1, i])
            print(cap[1] - cap[0])
            self.draw_capacity(cap, epoch)


    def draw_capacity(self, cap, count, labels=['Water-filling', 'DQFCNet', 'Q-learning', 'Netouts'], c=['r','b','g','k']):
        for i in range(len(cap)):
            plt.plot(range(count), cap[i], c[i], label=labels[i])
        plt.xlabel('iterations')
        plt.ylabel('Spectral-Efficiency(bps/Hz)')
#        plt.plot(range(count), cap[1], 'b', label=labels[1])
#        if cap.shape[0] == 3:
#            plt.plot(range(count), cap[2], 'g', label=labels[2])
        plt.legend()
        # plt.show()
        tname = str(int(t.time()))
        filename = 'QFNet_' + tname + '_.png'
        plt.savefig(Tools.qf_image + '/' + filename)
        with open(Tools.qf_image + '/' + tname + '.txt', 'w') as fw:
            fw.write(str(cap.flatten()))
        plt.close()


    def model_summary(self):
        best_model = self.load_best_model()
        best_model.summary()
        plot_model(best_model, to_file=Tools.qf_image + '/Model.png')


    def predict(self, model, x_test):
        x_test.reshape((self.layers_units[0],))
        outs = model.predict(x_test)
        # print('outs: ', outs)
        return outs


    def save_weights(self, model):
        '''保留最新的2个模型，t最大的为best_model'''
        models = os.listdir(Tools.qf_model)
        if len(models) >= 2:
            tim = []
            for i in range(len(models)):
                tim.append(int(models[i].split('_')[1]))
            os.remove(Tools.qf_model + '/weights_' + str(np.min(tim)) + '_.h5')
        model.save_weights(Tools.qf_model + '/weights_' + str(int(t.time())) + '_.h5')


    def load_best_model(self):
        '''加载当前最佳model'''
        models = os.listdir(Tools.qf_model)
        if len(models) != 0:
            tim = []
            for i in range(len(models)):
                tim.append(int(models[i].split('_')[1]))
            best_model = self.build_model()
            best_model.load_weights(Tools.qf_model + '/weights_' + str(np.max(tim)) + '_.h5')
            best_model.compile(
                optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                loss=Tools.adjust_loss,
                metrics=[Tools.my_metrics]
            )
            return best_model
        else:
            print('当前没有模型...')
            return None


    def load_data(self, model_num):
        '''加载最新数据'''
        if len(os.listdir(Tools.qf_data)) == 0:
            print('没有数据可以加载...')
            return None
        datas = os.listdir(Tools.qf_data)
        tim = []
        for i in range(len(datas)):
            tim.append(int(datas[i].split('_')[2]))
        datas_x = shelve.open(Tools.qf_data + '/' + model_num + '_x_' + str(np.max(tim)) + '_.bd', writeback=True, flag='c')
        datas_y = shelve.open(Tools.qf_data + '/' + model_num + '_y_' + str(np.max(tim)) + '_.bd', writeback=True, flag='c')
        x_data = datas_x['epoch_data']
        y_label = datas_y['epoch_data']

        datas_x.close()
        datas_y.close()
        return x_data, y_label
from GSSFiltering.model import StateSpaceModel
from GSSFiltering.filtering import Extended_Kalman_Filter
from GSSFiltering.filtering import KalmanNet_Filter, Split_KalmanNet_Filter, KalmanNet_Filter_v2
from GSSFiltering.trainer import Trainer
from GSSFiltering.tester import Tester
import numpy as np
import configparser
import os


if not os.path.exists('./.results'):
    os.mkdir('./.results')

config = configparser.ConfigParser()
config.read('./config.ini')

TRAIN=True

if (config['StateSpace']['model'] == "Synthetic"):
    r2 = float(config['StateSpace']['r2_synthetic'])
else:
    r2 = float(config['StateSpace']['r2_pos'])

Trajectory_State = config['StateSpace']['Trajectory_State'].strip("'")
if TRAIN:
    StateSpaceModel(mode='train', knowledge=Trajectory_State, data_gen=True).generate_data()
    StateSpaceModel(mode='valid', knowledge=Trajectory_State, data_gen=True).generate_data()
StateSpaceModel(mode='test', knowledge=Trajectory_State, data_gen=True).generate_data()

train_iter = int(config['Train']['train_iter'])

# S_KalmanNet
test_list = ['best']

loss_list_Kalman = []
loss_list_Kalman_bnn = []
loss_list_Kalman_v2 = []
loss_list_Split = []
loss_ekf = []

valid_loss_Kalman = []
valid_loss_Kalman_bnn = []
valid_loss_Kalman_v2 = []
valid_loss_Split = []

knowledge = config['StateSpace']['model_knowledge'].strip("'")
if TRAIN:
    # KalmanNet (architecture 1)
    trainer_kalman = Trainer(
        dnn=KalmanNet_Filter(
            StateSpaceModel(mode='train', knowledge = knowledge)),
        data_path='./.data/StateSpace/train/',
        save_path='(StateSpace) KalmanNet.pt',
        mode=0)
    # trainer_kalman.batch_size = batch_size
    # trainer_kalman.alter_num = alter_num

    # KalmanNet (architecture 1) - Bayesian Neural Network
    trainer_kalman_bnn = Trainer(
        dnn=KalmanNet_Filter(
            StateSpaceModel(mode='train', knowledge=knowledge), isBNN=True),
        data_path='./.data/StateSpace/train/',
        save_path='(StateSpace) KalmanNet_bnn.pt',
        mode=0,
        isBNN=True)

    # KalmanNet (architecture 2)
    trainer_kalman_v2 = Trainer(
        dnn=KalmanNet_Filter_v2(
            StateSpaceModel(mode='train', knowledge = knowledge)),
        data_path='./.data/StateSpace/train/',
        save_path='(StateSpace, v2) KalmanNet.pt',
        mode=0)
    # trainer_kalman_v2.batch_size = batch_size
    # trainer_kalman_v2.alter_num = alter_num

    # S_KalmanNet
    trainer_split = Trainer(
        dnn=Split_KalmanNet_Filter(
            StateSpaceModel(mode='train', knowledge = knowledge)),
        data_path='./.data/StateSpace/train/',
        save_path='(StateSpace) Split_KalmanNet.pt',
        mode=1)
    # trainer_split.batch_size = batch_size
    # trainer_split.alter_num = alter_num


    for i in range(train_iter):

        # Split KalmanNet
        trainer_split.train_batch()
        trainer_split.dnn.reset(clean_history=True)
        if trainer_split.train_count % trainer_split.save_num == 0:
            trainer_split.validate(
                Tester(
                        filter = Split_KalmanNet_Filter(
                            StateSpaceModel(mode='valid', knowledge = knowledge)),
                        data_path = './.data/StateSpace/valid/',
                        model_path = './.model_saved/(StateSpace) Split_KalmanNet_' + str(trainer_split.train_count) + '.pt',
                        is_validation=True
                        )
            )
            valid_loss_Split += [trainer_split.valid_loss]

        # KalmanNet V1
        trainer_kalman.train_batch()
        trainer_kalman.dnn.reset(clean_history=True)
        if trainer_kalman.train_count % trainer_kalman.save_num == 0:
            trainer_kalman.validate(
                Tester(
                    filter=KalmanNet_Filter(
                        StateSpaceModel(mode='valid', knowledge=knowledge)),
                    data_path='./.data/StateSpace/valid/',
                    model_path='./.model_saved/(StateSpace) KalmanNet_' + str(trainer_kalman.train_count) + '.pt',
                    is_validation=True
                )
            )
            valid_loss_Kalman += [trainer_kalman.valid_loss]

        # KalmanNet V1 - BNN
        trainer_kalman_bnn.train_batch()
        trainer_kalman_bnn.dnn.reset(clean_history=True)
        if trainer_kalman_bnn.train_count % trainer_kalman_bnn.save_num == 0:
            trainer_kalman_bnn.validate(
                Tester(
                    filter=KalmanNet_Filter(
                        StateSpaceModel(mode='valid', knowledge=knowledge), isBNN=True),
                    data_path='./.data/StateSpace/valid/',
                    model_path='./.model_saved/(StateSpace) KalmanNet_bnn_' + str(trainer_kalman_bnn.train_count) + '.pt',
                    is_validation=True,
                    isBNN=True
                )
            )
            valid_loss_Kalman_bnn += [trainer_kalman_bnn.valid_loss]

        # KalmanNet V2
        trainer_kalman_v2.train_batch()
        trainer_kalman_v2.dnn.reset(clean_history=True)
        if trainer_kalman_v2.train_count % trainer_kalman_v2.save_num == 0:
            trainer_kalman_v2.validate(
                Tester(
                        filter = KalmanNet_Filter_v2(
                            StateSpaceModel(mode='valid', knowledge = knowledge)),
                        data_path = './.data/StateSpace/valid/',
                        model_path = './.model_saved/(StateSpace, v2) KalmanNet_' + str(trainer_kalman_v2.train_count) + '.pt',
                        is_validation=True
                        )
            )
            valid_loss_Kalman_v2 += [trainer_kalman_v2.valid_loss]


    validator_ekf = Tester(
                filter = Extended_Kalman_Filter(
                    StateSpaceModel(mode='valid', knowledge = knowledge)),
                data_path = './.data/StateSpace/valid/',
                model_path = 'EKF'
                )
    loss_ekf = [validator_ekf.loss.item()]

    np.save('./.results/valid_loss_ekf.npy', np.array(loss_ekf))
    np.save('./.results/valid_loss_kalman.npy', np.array(valid_loss_Kalman))
    np.save('./.results/valid_loss_kalman_bnn.npy', np.array(valid_loss_Kalman_bnn))
    np.save('./.results/valid_loss_kalman_v2.npy', np.array(valid_loss_Kalman_v2))
    np.save('./.results/valid_loss_split.npy', np.array(valid_loss_Split))


tester_ekf = Tester(
            filter = Extended_Kalman_Filter(
                StateSpaceModel(mode='test', knowledge = knowledge)),
            data_path = './.data/StateSpace/test/',
            model_path = 'EKF'
            )
loss_ekf = [tester_ekf.loss.item()]
print(loss_ekf)

for elem in test_list:

    tester_kf = Tester(
                filter = KalmanNet_Filter(
                    StateSpaceModel(mode='test', knowledge = knowledge)),
                data_path = './.data/StateSpace/test/',
                model_path = './.model_saved/(StateSpace) KalmanNet_' + elem + '.pt'
                )
    loss_list_Kalman += [tester_kf.loss.item()]

    tester_kf_bnn = Tester(
                filter = KalmanNet_Filter(
                    StateSpaceModel(mode='test', knowledge = knowledge), isBNN=True),
                data_path = './.data/StateSpace/test/',
                model_path = './.model_saved/(StateSpace) KalmanNet_bnn_' + elem + '.pt',
                isBNN=True)
    loss_list_Kalman_bnn += [tester_kf_bnn.loss.item()]

    tester_kf2 = Tester(
                filter = KalmanNet_Filter_v2(
                    StateSpaceModel(mode='test', knowledge = knowledge)),
                data_path = './.data/StateSpace/test/',
                model_path = './.model_saved/(StateSpace, v2) KalmanNet_' + elem + '.pt'
                )
    loss_list_Kalman_v2 += [tester_kf2.loss.item()]

    tester_skf = Tester(
                filter = Split_KalmanNet_Filter(
                    StateSpaceModel(mode='test', knowledge = knowledge)),
                data_path = './.data/StateSpace/test/',
                model_path = './.model_saved/(StateSpace) Split_KalmanNet_' + elem + '.pt'
                )
    loss_list_Split += [tester_skf.loss.item()]

# print(loss_ekf)
# print(loss_list_Kalman)
# print(loss_list_Kalman_bnn)
# print(loss_list_Kalman_v2)
# print(loss_list_Split)

print('EKF: ', loss_ekf)
print('KalmanNet: ', loss_list_Kalman)
print('KalmanNet_bnn: ', loss_list_Kalman_bnn)
print('KalmanNet_v2: ', loss_list_Kalman_v2)
print('Split_KalmanNet: ', loss_list_Split)

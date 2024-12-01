import math
import torch
from GSSFiltering.filtering import Extended_Kalman_Filter, KalmanNet_Filter, KalmanNet_Filter_v2, Split_KalmanNet_Filter
import time
from datetime import timedelta
import configparser
config = configparser.ConfigParser()

print_num = 25

class Tester():
    def __init__(self, filter, data_path, model_path, is_validation=False, is_mismatch=False, isBNN:bool=False):
        config.read('./config.ini')

        if isinstance(filter, Extended_Kalman_Filter):
            self.result_path = 'EKF '
        if isinstance(filter, KalmanNet_Filter):
            if isBNN:
                self.result_path = 'KF v1 bnn '
            else:
                self.result_path = 'KF v1 '
        if isinstance(filter, KalmanNet_Filter_v2):
            self.result_path = 'KF v2 '
        if isinstance(filter, Split_KalmanNet_Filter):
            self.result_path = 'SKF '


        self.filter = filter
        if not isinstance(filter, Extended_Kalman_Filter):
            self.filter.kf_net = torch.load(model_path)
            self.filter.kf_net.initialize_hidden()
        self.x_dim = self.filter.x_dim
        self.y_dim = self.filter.y_dim
        self.data_path = data_path
        self.model_path = model_path
        self.is_validation = is_validation
        self.is_mismatch = is_mismatch

        self.loss_fn = torch.nn.MSELoss()

        self.data_x = torch.load(data_path + 'state.pt')
        self.data_x = self.data_x[:,0:(self.x_dim),:]
        self.data_y = torch.load(data_path + 'obs.pt')
        self.data_num = self.data_x.shape[0]
        self.seq_len = self.data_x.shape[2]
        assert(self.x_dim == self.data_x.shape[1])
        assert(self.y_dim == self.data_y.shape[1])
        assert(self.seq_len == self.data_y.shape[2])
        assert(self.data_num == self.data_y.shape[0])
        if (config['StateSpace']['model'] == "Synthetic"):
            scz = 2  # State's Chopped Size
        elif (config['StateSpace']['model'] == "Linear_Canonical"):
            scz = self.x_dim  # State's Chopped Size
        elif (config['StateSpace']['model'] == "Pendulum"):
            scz = 1  # State's Chopped Size
        else:
            scz = 4  # State's Chopped Size

        x_hat = torch.zeros_like(self.data_x)

        estimation_error = torch.zeros_like(self.data_x)

        cov_pred = torch.zeros(x_hat.shape[0],x_hat.shape[1],x_hat.shape[1],x_hat.shape[2])
        cov_pred_byK_opt1 = torch.zeros(x_hat.shape[0], scz, scz, x_hat.shape[2])
        cov_pred_byK_opt2 = torch.zeros(x_hat.shape[0], scz, scz, x_hat.shape[2])

        cov_post = torch.zeros(x_hat.shape[0],x_hat.shape[1],x_hat.shape[1],x_hat.shape[2])
        cov_post_optA = torch.zeros(x_hat.shape[0],x_hat.shape[1],x_hat.shape[1],x_hat.shape[2])
        cov_post_optB = torch.zeros(x_hat.shape[0],x_hat.shape[1],x_hat.shape[1],x_hat.shape[2])
        cov_post_byK_optA = torch.zeros(x_hat.shape[0], scz, scz, x_hat.shape[2])
        cov_post_byK_optB = torch.zeros(x_hat.shape[0], scz, scz, x_hat.shape[2])

        # Bayesian NN
        self.isBNN = isBNN
        if self.isBNN:
            self.num_of_mc_iterations = int(config['Bayesian']['num_of_mc_iterations'])
            x_hat_temp = torch.zeros((x_hat.shape + (self.num_of_mc_iterations,)))
            cov_post_bnn = torch.zeros_like(x_hat)

        start_time = time.monotonic()

        with torch.no_grad():
            for i in range(self.data_num):

                last_iter = bool(i==self.data_num-1)

                if i % print_num == 0:
                    if self.is_validation:
                        print(f'Validating {i} / {self.data_num} of {self.model_path}')
                    else:
                        print(f'Testing {i} / {self.data_num} of {self.model_path}')

                if self.isBNN:
                    for mc_ind in range(self.num_of_mc_iterations):
                        self.filter.state_post = self.data_x[i, :, 0].reshape((-1, 1))
                        for ii in range(1, self.seq_len):
                            self.filter.filtering(self.data_y[i, :, ii].reshape((-1, 1)))
                        x_hat_temp[i, :, :, mc_ind] = self.filter.state_history[:,-self.seq_len:]
                        self.filter.reset(clean_history=False)
                    x_hat[i] += torch.mean(x_hat_temp[i], dim=-1)
                    cov_post_bnn[i] += torch.var(x_hat_temp[i], dim=-1)
                else:
                    self.filter.state_post = self.data_x[i, :, 0].reshape((-1, 1))
                    for ii in range(1, self.seq_len):
                        self.filter.filtering(self.data_y[i,:,ii].reshape((-1,1)))
                    x_hat[i] = self.filter.state_history[:,-self.seq_len:]
                    estimation_error[i] = x_hat[i] - self.data_x[i,:,:]

                if (isinstance(filter, Extended_Kalman_Filter)):
                    try:
                        # Sigma t|t-1
                        cov_pred[i] = self.filter.cov_pred_history[:, :, -self.seq_len:]
                        if last_iter: torch.save(cov_pred, data_path + self.result_path + 'cov_pred.pt')
                        cov_post_optA[i] = self.filter.cov_post_optA_history[:, :, -self.seq_len:]
                        cov_post_optB[i] = self.filter.cov_post_optB_history[:, :, -self.seq_len:]
                        if last_iter: torch.save(cov_post_optA, data_path + self.result_path + 'cov_post_optA.pt')
                        if last_iter: torch.save(cov_post_optB, data_path + self.result_path + 'cov_post_optB.pt')
                    except:
                        pass

                    # Sigma t|t
                    cov_post[i] = self.filter.cov_history[:,:, -self.seq_len:]
                    if last_iter: torch.save(cov_post, data_path + self.result_path + 'cov_post.pt')


                if (isinstance(filter, KalmanNet_Filter)):
                    try:
                        # Sigma t|t
                        cov_post_byK_optA[i] = self.filter.cov_post_byK_optA_history[:, :, -self.seq_len:]
                        cov_post_byK_optB[i] = self.filter.cov_post_byK_optB_history[:, :, -self.seq_len:]
                        if last_iter: torch.save(cov_post_byK_optA, data_path + self.result_path + 'cov_post_byK_optA.pt')
                        if last_iter: torch.save(cov_post_byK_optB, data_path + self.result_path + 'cov_post_byK_optB.pt')
                    except:
                        pass

                    if self.isBNN:
                        if last_iter: torch.save(cov_post_bnn, data_path + self.result_path + 'cov_post_bnn.pt')

                if (isinstance(filter, KalmanNet_Filter_v2)):
                    try:
                        # Sigma t|t
                        cov_post_byK_optA[i] = self.filter.cov_post_byK_optA_history[:, :, -self.seq_len:]
                        cov_post_byK_optB[i] = self.filter.cov_post_byK_optB_history[:, :, -self.seq_len:]
                        if last_iter: torch.save(cov_post_byK_optA, data_path + self.result_path + 'cov_post_byK_optA.pt')
                        if last_iter: torch.save(cov_post_byK_optB, data_path + self.result_path + 'cov_post_byK_optB.pt')
                    except:
                        pass

                if (isinstance(filter, Split_KalmanNet_Filter)):
                    try:
                        # Sigma t|t-1
                        cov_pred[i] = self.filter.cov_pred_history[:, :, -self.seq_len:]
                        if last_iter: torch.save(cov_pred, data_path + self.result_path + 'cov_pred.pt')

                        # Sigma t|t
                        cov_post_byK_optA[i] = self.filter.cov_post_byK_optA_history[:, :, -self.seq_len:]
                        cov_post_byK_optB[i] = self.filter.cov_post_byK_optB_history[:, :, -self.seq_len:]
                        if last_iter: torch.save(cov_post_byK_optA, data_path + self.result_path + 'cov_post_byK_optA.pt')
                        if last_iter: torch.save(cov_post_byK_optB, data_path + self.result_path + 'cov_post_byK_optB.pt')
                    except:
                        pass

                    # Sigma t|t
                    cov_post[i] = self.filter.cov_history[:,:, -self.seq_len:]
                    if last_iter: torch.save(cov_post, data_path + self.result_path + 'cov_post.pt')

                try:
                    cov_pred_byK_opt1[i] = self.filter.cov_pred_byK_opt1_history[:, :, -self.seq_len:]
                    cov_pred_byK_opt2[i] = self.filter.cov_pred_byK_opt2_history[:, :, -self.seq_len:]
                except:
                    pass

                self.filter.reset(clean_history=False)

            # Sigma t|t-1
            torch.save(cov_pred_byK_opt1, data_path + self.result_path + 'cov_pred_byK_opt1.pt')
            torch.save(cov_pred_byK_opt2, data_path + self.result_path + 'cov_pred_byK_opt2.pt')

            # EE
            torch.save(estimation_error, data_path + self.result_path + 'ee.pt')

            end_time = time.monotonic()
            # print(timedelta(seconds=end_time - start_time))

            torch.save(x_hat, data_path + self.result_path + 'x_hat.pt')

            loss = self.loss_fn(self.data_x[:,:,1:], x_hat[:,:,1:])
            # loss = self.loss_fn(self.data_x[:,[0,3],1:], x_hat[:,[0,3],1:])
            loss_dB = 10*torch.log10(loss)
            print(f'loss [dB] = {loss_dB:.4f}')


            # Compute loss at instantaneous time
            self.loss_instant = torch.zeros(self.data_x[:,:,1:].shape[-1])
            for i in range(self.data_x[:,:,1:].shape[-1]):
                self.loss_instant[i] = self.loss_fn(self.data_x[:, :, i+1], x_hat[:, :, i+1])
            self.loss_instant_dB = 10*torch.log10(self.loss_instant)

        self.loss = loss_dB
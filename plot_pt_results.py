import torch
import matplotlib.pyplot as plt

def plot_mc(KF_Empirical_STD=None,\
            KF_predicted_std=None,\
            KNetV1_Empirical_STD=None,\
            KNetV1_predicted_std=None,\
            KNetV1_bnn_Empirical_STD=None,\
            KNetV1_bnn_predicted_std=None, \
            KNetV2_Empirical_STD=None, \
            KNetV2_predicted_std=None, \
            SplitKnet_Empirical_STD=None, \
            SplitKnet_predicted_std=None, \
            dim=None, file_name=None):

    font_size = 14
    T_test = KF_Empirical_STD.size()[1]
    start_inx = 2

    x_plt = range(start_inx, T_test)

    if (KNetV1_Empirical_STD is not None): plt.plot(x_plt, KNetV1_Empirical_STD[dim, start_inx:].detach().numpy(), color='blue', label='KNetV1 Empirical', linestyle='dashed')
    if (KNetV1_predicted_std is not None): plt.plot(x_plt, KNetV1_predicted_std[dim, dim, start_inx:].detach().numpy(), color='blue', label='KNetV1 Predicted', linestyle='solid')

    if (KNetV1_bnn_Empirical_STD is not None): plt.plot(x_plt, KNetV1_bnn_Empirical_STD[dim, start_inx:].detach().numpy(), color='red', label='KNetV1 BNN Empirical', linestyle='dashed')
    if (KNetV1_bnn_predicted_std is not None): plt.plot(x_plt, KNetV1_bnn_predicted_std[dim, start_inx:].detach().numpy(), color='red', label='KNetV1 BNN Predicted', linestyle='solid')

    if (KF_Empirical_STD is not None): plt.plot(x_plt, KF_Empirical_STD[dim, start_inx:].detach().numpy(), color='green', label='EKF Empirical', linestyle='dashed')
    if (KF_predicted_std is not None): plt.plot(x_plt, KF_predicted_std[dim, dim, start_inx:].detach().numpy(), color='green', label='EKF Predicted', linestyle='solid')

    if (KNetV2_Empirical_STD is not None): plt.plot(x_plt, KNetV2_Empirical_STD[dim, start_inx:].detach().numpy(), color='purple', label='KNetV2 Empirical', linestyle='dashed')
    if (KNetV2_predicted_std is not None): plt.plot(x_plt, KNetV2_predicted_std[dim, dim, start_inx:].detach().numpy(), color='purple', label='KNetV2 Predicted', linestyle='solid')

    if (SplitKnet_Empirical_STD is not None): plt.plot(x_plt, SplitKnet_Empirical_STD[dim, start_inx:].detach().numpy(), color='orange', label='SplitKNet Empirical', linestyle='dashed')
    if (SplitKnet_predicted_std is not None): plt.plot(x_plt, SplitKnet_predicted_std[dim, dim, start_inx:].detach().numpy(), color='orange', label='SplitKNet Predicted', linestyle='solid')

    plt.legend(fontsize=font_size)
    plt.xlabel('t', fontsize=font_size)

    if dim == 0:
        plt.ylabel('position', fontsize=font_size)
        plt.title('Position X State')
    elif dim == 1:
        plt.ylabel('position', fontsize=font_size)
        plt.title('Position Y State')
    elif dim == 2:
        plt.ylabel('velocity', fontsize=font_size)
        plt.title('Velocity X State')
    elif dim == 3:
        plt.ylabel('velocity', fontsize=font_size)
        plt.title('Velocity Y State')
    else:
        print("invalid dimension")

    plt.savefig(file_name)
    plt.clf()

def plotTraj_CA(ground_truth=None,\
                EKF_x_hat=None,\
                KalmanNetV1_x_hat=None,\
                KalmanNetV1_bnn_x_hat=None,\
                KalmanNetV2_x_hat=None,\
                Split_KalmanNet_x_hat=None,\
                dim=None, file_name=None):

    legend = ["KNet V1 ","KNet V1 BNN", "EKF", "Ground Truth", "KNet V2", "Split KalmanNet"]
    font_size = 14
    T_test = EKF_x_hat[0].size()[1]
    x_plt = range(0, T_test)

    if (KalmanNetV1_x_hat is not None): plt.plot(x_plt, KalmanNetV1_x_hat[0][dim, :].detach().numpy(), label=legend[0], color='blue')
    if (KalmanNetV1_bnn_x_hat is not None): plt.plot(x_plt, KalmanNetV1_bnn_x_hat[0][dim, :].detach().numpy(), label=legend[1],  color='red')
    if (EKF_x_hat is not None): plt.plot(x_plt, EKF_x_hat[0][dim, :], label=legend[2], color='green')
    if (ground_truth is not None): plt.plot(x_plt, ground_truth[0][dim, :].detach().numpy(), label=legend[3], color='black')
    if (KalmanNetV2_x_hat is not None): plt.plot(x_plt, KalmanNetV2_x_hat[0][dim, :].detach().numpy(), label=legend[4], color='purple')
    if (Split_KalmanNet_x_hat is not None): plt.plot(x_plt, Split_KalmanNet_x_hat[0][dim, :].detach().numpy(), label=legend[5], color='orange')

    plt.legend(fontsize=font_size)
    plt.xlabel('t', fontsize=font_size)

    if dim == 0:
        plt.ylabel('position', fontsize=font_size)
        plt.title('Position X State')
    elif dim == 1:
        plt.ylabel('position', fontsize=font_size)
        plt.title('Position Y State')
    elif dim == 2:
        plt.ylabel('velocity', fontsize=font_size)
        plt.title('Velocity X State')
    elif dim == 3:
        plt.ylabel('velocity', fontsize=font_size)
        plt.title('Velocity Y State')
    else:
        print("invalid dimension")


    plt.savefig(file_name)
    plt.clf()

def dB(x):
    return 10*torch.log10(x)

print("\n \n \n \n \n \n")
print("Plotting Results")

results_path = '.data/StateSpace/test/'
PlotfolderName = '.figures/'

################
# test_length = ground_truth.size()[2]
test_length = 100 # [sec]
total_test_legnth = torch.load(results_path + 'state.pt').size()[2]
################

########################
### State Estimation ###
########################

ground_truth = torch.load(results_path + 'state.pt')[:, :, :test_length]
# ground_truth = None

EKF_estimation = torch.load(results_path + 'EKF x_hat.pt')[:, :, :test_length]
# EKF_estimation = None

Knet_v1_estimation = torch.load(results_path + 'KF v1 x_hat.pt')[:, :, :test_length]
# Knet_v1_estimation = None

Knet_v1_bnn_estimation = torch.load(results_path + 'KF v1 bnn x_hat.pt')[:, :, :test_length]
# Knet_v1_bnn_estimation = None

Knet_v2_estimation = torch.load(results_path + 'KF v2 x_hat.pt')[:, :, :test_length]
# Knet_v2_estimation = None

SKF_estimation = torch.load(results_path + 'SKF x_hat.pt')[:, :, :test_length]
# SKF_estimation = None

#############################
### Covariance Estimation ###
#############################

EKF_predicted_cov = torch.load(results_path + 'EKF cov_post.pt')[:, :, :, :test_length]
# EKF_predicted_cov = None

Knet_v1_predicted_cov = torch.load(results_path + 'KF v1 cov_post_byK_optA.pt')[:, :, :, :test_length]
# Knet_v1_predicted_cov = None

Knet_v1_bnn_predicted_cov = torch.load(results_path + 'KF v1 bnn cov_post_bnn.pt')[:, :, :test_length]
# Knet_v1_bnn_predicted_cov = None

Knet_v2_predicted_cov = torch.load(results_path + 'KF v2 cov_post_byK_optA.pt')[:, :, :, :test_length]
# Knet_v2_predicted_cov = None

SKF_predicted_cov = torch.load(results_path + 'SKF cov_post.pt')[:, :, :, :test_length]
# SKF_predicted_cov = torch.load(results_path + 'SKF cov_post_byK_optA.pt')[:, :, :, :test_length]
# SKF_predicted_cov = None

####################
### Plot results ###
####################
# Shapes
x_dim = EKF_estimation.size()[1]
ground_truth = ground_truth[:, :x_dim, :]

PlotfileName0 = "TrainPVA_positionX.png"
PlotfileName1 = "TrainPVA_positionY.png"

PlotfileName2 = "TrainPVA_velocityX.png"
PlotfileName3 = "TrainPVA_velocityY.png"

PlotfileName9 = "TrainPVA_positionX_mc.png"
PlotfileName10 = "TrainPVA_positionY_mc.png"

PlotfileName11 = "TrainPVA_velocityX_mc.png"
PlotfileName12 = "TrainPVA_velocityY_mc.png"


##################
### Single Run ###
##################

plotTraj_CA(ground_truth, EKF_estimation, Knet_v1_estimation, Knet_v1_bnn_estimation, Knet_v2_estimation, SKF_estimation, dim=0, file_name=PlotfolderName + PlotfileName0)  # Position X
plotTraj_CA(ground_truth, EKF_estimation, Knet_v1_estimation, Knet_v1_bnn_estimation, Knet_v2_estimation, SKF_estimation, dim=1, file_name=PlotfolderName + PlotfileName1)  # Position Y
# plotTraj_CA(ground_truth, EKF_estimation, Knet_v1_estimation, Knet_v1_bnn_estimation, Knet_v2_estimation, SKF_estimation, dim=2, file_name=PlotfolderName + PlotfileName2)  # Velocity X
# plotTraj_CA(ground_truth, EKF_estimation, Knet_v1_estimation, Knet_v1_bnn_estimation, Knet_v2_estimation, SKF_estimation, dim=3, file_name=PlotfolderName + PlotfileName3)  # Velocity Y

###################
### Monte Carlo ###
###################

KF_error = (ground_truth - EKF_estimation) if (EKF_estimation is not None) else None
KNetV1_error = (ground_truth - Knet_v1_estimation) if (Knet_v1_estimation is not None) else None
KNetV1_bnn_error = (ground_truth - Knet_v1_bnn_estimation) if (Knet_v1_bnn_estimation is not None) else None
KNetV2_error = (ground_truth - Knet_v2_estimation) if (Knet_v2_estimation is not None) else None
SKF_error = (ground_truth - SKF_estimation) if (SKF_estimation is not None) else None

KF_Empirical_STD = torch.std(KF_error, dim=0) if (KF_error is not None) else None
KNetV1_Empirical_STD = torch.std(KNetV1_error, dim=0) if (KNetV1_error is not None) else None
KNetV1_bnn_Empirical_STD = torch.std(KNetV1_bnn_error, dim=0) if (KNetV1_bnn_error is not None) else None
KNetV2_Empirical_STD = torch.std(KNetV2_error, dim=0) if (KNetV2_error is not None) else None
SKF_Empirical_STD = torch.std(SKF_error, dim=0) if (SKF_error is not None) else None

EKF_mean_std = torch.sqrt( torch.mean(EKF_predicted_cov, dim=0) ) if (EKF_predicted_cov is not None) else None
KNet_v1_mean_std = torch.sqrt( torch.mean(Knet_v1_predicted_cov, dim=0) ) if (Knet_v1_predicted_cov is not None) else None
KNet_v1_bnn_mean_std = torch.sqrt( torch.mean(Knet_v1_bnn_predicted_cov, dim=0) ) if (Knet_v1_bnn_predicted_cov is not None) else None
KNet_v2_mean_std = torch.sqrt( torch.mean(Knet_v2_predicted_cov, dim=0) ) if (Knet_v2_predicted_cov is not None) else None
SKF_mean_std = torch.sqrt( torch.mean(SKF_predicted_cov, dim=0) ) if (SKF_predicted_cov is not None) else None

plot_mc(KF_Empirical_STD, EKF_mean_std, KNetV1_Empirical_STD, KNet_v1_mean_std, KNetV1_bnn_Empirical_STD, KNet_v1_bnn_mean_std, KNetV2_Empirical_STD, KNet_v2_mean_std, SKF_Empirical_STD, SKF_mean_std, dim=0, file_name=PlotfolderName + 'MC_' + PlotfileName9)
plot_mc(KF_Empirical_STD, EKF_mean_std, KNetV1_Empirical_STD, KNet_v1_mean_std, KNetV1_bnn_Empirical_STD, KNet_v1_bnn_mean_std, KNetV2_Empirical_STD, KNet_v2_mean_std, SKF_Empirical_STD, SKF_mean_std, dim=1,  file_name=PlotfolderName + 'MC_' + PlotfileName10)
# plot_mc(KF_Empirical_STD, EKF_mean_std, KNetV1_Empirical_STD, KNet_v1_mean_std, KNetV1_bnn_Empirical_STD, KNet_v1_bnn_mean_std, KNetV2_Empirical_STD, KNet_v2_mean_std, SKF_Empirical_STD, SKF_mean_std, dim=2, file_name=PlotfolderName + 'MC_' + PlotfileName11)
# plot_mc(KF_Empirical_STD, EKF_mean_std, KNetV1_Empirical_STD, KNet_v1_mean_std, KNetV1_bnn_Empirical_STD, KNet_v1_bnn_mean_std, KNetV2_Empirical_STD, KNet_v2_mean_std, SKF_Empirical_STD, SKF_mean_std, dim=3,  file_name=PlotfolderName + 'MC_' + PlotfileName12)


# Summarize results
print('Test length taken for MSE: ', test_length, 'out of ', total_test_legnth)
print(' ')
print('MSE:')
print( dB(torch.mean(KF_error[:,:,:test_length]**2)), '[dB] (EKF)') if (KF_error is not None) else None
print( dB(torch.mean(KNetV1_error[:,:,:test_length]**2)), '[dB] (KNetV1)') if (KNetV1_error is not None) else None
print( dB(torch.mean(KNetV1_bnn_error[:,:,:test_length]**2)), '[dB] (KNetV1 BNN)') if (KNetV1_bnn_error is not None) else None
print( dB(torch.mean(KNetV2_error[:,:,:test_length]**2)), '[dB] (KNetV2)') if (KNetV2_error is not None) else None
print( dB(torch.mean(SKF_error[:,:,:test_length]**2)), '[dB] (SKF)') if (SKF_error is not None) else None
print(' ')
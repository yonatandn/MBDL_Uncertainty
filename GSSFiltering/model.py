import torch
import os
import numpy as np
import math, random
from math import pi
from scipy.linalg import sqrtm

import configparser
config = configparser.ConfigParser()

if not os.path.exists('./.data'):
    os.mkdir('./.data')


class GSSModel():
    def __init__(self):
        self.cov_q = None
        self.cov_r = None
        self.x_dim = None
        self.y_dim = None

    def generate_data(self):
        raise NotImplementedError()

    def f(self, current_state):
        raise NotImplementedError()
    def g(self, current_state):
        raise NotImplementedError()
    def Jacobian_f(self, x):
        raise NotImplementedError()
    def Jacobian_g(self, x):
        raise NotImplementedError()
    def next_state(self, current_state):
        config.read('./config.ini')

        if (self.config['model'] == "Synthetic"):
            noise = np.random.multivariate_normal(np.zeros(self.x_dim), self.cov_q)
        elif (self.config['model'] == "Navigation" or self.config['model'] == "Navigation_Linear"):
            dt = self.dt
            dt1, dt2, dt3, dt4, dt5 = dt, dt**2, dt**3, dt**4, dt**5
            if(self.knowledge == 'Full'):
                Q = np.array([[ (1/20)*dt5 , 0 , (1/8)*dt4 , 0 , (1/6)*dt3 , 0 ],
                    [ 0 , (1/20)*dt5 , 0 , (1/8)*dt4 , 0 , (1/6)*dt3 ],
                    [ (1/8)*dt4 , 0 , (1/3)*dt3 , 0 , (1/2)*dt2 , 0 ],
                    [ 0 , (1/8)*dt4 , 0 , (1/3)*dt3 , 0 , (1/2)*dt2 ],
                    [ (1/6)*dt3 , 0 , (1/2)*dt2 , 0 , dt1 , 0 ],
                    [ 0 , (1/6)*dt3 , 0 , (1/2)*dt2 , 0 , dt1]])
            elif(self.knowledge == 'Partial'):
                Q = np.array([[ (1/3)*dt3 , 0 , (1/2)*dt2 , 0],
                    [ 0 , (1/3)*dt3 , 0 , (1/2)*dt2],
                    [ (1/2)*dt2 , 0 , dt1 , 0],
                    [ 0 , (1/2)*dt2 , 0 , dt1]])
            else:
                raise NotImplementedError()
        elif(self.config['model'] == "Linear_Canonical"):
            Q = np.eye(self.x_dim)
            Q = Q * np.array(self.q2)
            noise = sqrtm(Q) @ np.random.normal(0, 1, size=(self.x_dim,))
        elif (config['StateSpace']['model'] == "Pendulum"):
            Q = np.eye(self.x_dim)
            Q = Q * np.array(self.q2)
            noise = sqrtm(Q) @ np.random.normal(0, 1, size=(self.x_dim,))
        else:
            raise NotImplementedError()
            
        noise = sqrtm(Q) @ np.random.normal(0, 1, size=(self.x_dim,))
        return self.f(current_state) + torch.tensor(noise, dtype=torch.float).reshape((-1,1))

    def observe(self,current_state):
        noise = np.random.multivariate_normal(np.zeros(self.y_dim), self.cov_r)
        return self.g(current_state) + torch.tensor(noise, dtype=torch.float).reshape((-1,1))

    def Create_Satellites(self, Nsatellites, RandSat = False, Close_sat=False):
        config.read('./config.ini')
        self.sats = []
        for i in range(Nsatellites):
            if RandSat:
                sat = Satellite(random.randint(-500, 500),random.randint(-500, 500),random.randint(50, 100))
            else:
                if Close_sat:
                    # Random option from :
                    # XsatOptions = torch.randint(-300, 300, [1, 30])
                    # YsatOptions = torch.randint(-300, 300, [1, 30])
                    # ZsatOptions = torch.randint(10, 100, [1, 30])

                    XsatOptions = torch.tensor([ 76,   12,  173,   19,  276,  229,  -47,  -10, -295,  273,  175,  -78,
                                                202,  292, -237, -106, -107,  175, -215,  271,  253,  298,  -23, -143,
                                                2,  -47, -241, -158,   28, -226])
                    YsatOptions = torch.tensor([254,  200,  200,   53,  188,  112, -268, -211, -244,  -86,  204,  166,
                                                   25,  167,  -22,  -13,  291,  -67,   50, -268,   32, -224,   98,  -14,
                                                 -187, -189, -289,    5,  -69, -210])
                    ZsatOptions = torch.tensor([67, 55, 30, 56, 24, 64, 86, 58, 40, 78, 11, 51, 35, 99, 75, 68, 34, 54,
                                                60, 41, 69, 53, 34, 21, 13, 77, 47, 16, 92, 87])

                    sat = Satellite(XsatOptions[i],YsatOptions[i],ZsatOptions[i])
                else:
                    sat = Satellite(-2500 + i*1000, -3500 + i*1000, 36000 + i*100)

            self.sats.append(sat)

        self.Nsatellites = Nsatellites
        return self.sats

    def Construct_R(self):
        config.read('./config.ini')
        if (self.config['diag_r'].strip('') == "True"):
            if(self.config['model'] == "Synthetic"):
                return float(self.config['r2_synthetic']) * torch.eye(self.y_dim)
            elif (self.config['model'] == "Navigation"):
                r2_PR = float(self.config['r2_PR'])
                r2_PRR = float(self.config['r2_PRR'])
                if self.knowledge=='Full':
                    return torch.diag(torch.squeeze( torch.cat((r2_PR * torch.ones((1,int(self.y_dim/2))),r2_PRR * torch.ones((1,int(self.y_dim/2)))),dim=1) ))
                elif self.knowledge=='Partial':
                    return torch.diag(torch.squeeze( torch.cat((r2_PR * torch.ones((1,int(self.y_dim/2))),r2_PRR * torch.ones((1,int(self.y_dim/2)))),dim=1) ))
            elif (self.config['model'] == "Navigation_Linear"):
                r2_pos = float(self.config['r2_pos'])
                r2_vel = float(self.config['r2_vel'])
                if (self.config['H_only_pos_when_Navigation_Linear'] == "False"):
                    if self.knowledge=='Full':
                        return torch.diag(torch.squeeze( torch.cat((r2_pos * torch.ones((1,2)),r2_vel * torch.ones((1,2))),dim=1) ))
                    elif self.knowledge=='Partial':
                        return torch.diag(torch.squeeze( torch.cat((r2_pos * torch.ones((1,2)),r2_vel * torch.ones((1,2))),dim=1) ))
                elif (self.config['H_only_pos_when_Navigation_Linear'] == "True"):
                    if self.knowledge=='Full':
                        return torch.diag(torch.squeeze( r2_pos * torch.ones((1,2)) ) )
                    elif self.knowledge=='Partial':
                        return torch.diag(torch.squeeze( r2_pos * torch.ones((1,2)) ) )
            elif(self.config['model'] == "Linear_Canonical"):
                return float(self.config['r2_synthetic']) * torch.eye(self.y_dim)
            elif (config['StateSpace']['model'] == "Pendulum"):
                return float(self.config['r2_synthetic']) * torch.eye(self.y_dim)
            else:
                raise NotImplementedError()
        else:
            if(self.config['model'] == "Synthetic"):
                raise NotImplementedError()
            elif (self.config['model'] == "Navigation"):
                r2_PR = float(self.config['r2_PR'])
                r2_PRR = float(self.config['r2_PRR'])
                if self.knowledge=='Full':
                    return torch.block_diag(r2_PR * torch.ones((int(self.y_dim/2),int(self.y_dim/2))), r2_PRR * torch.ones((int(self.y_dim/2),int(self.y_dim/2))))
                elif self.knowledge=='Partial':
                    return torch.block_diag(r2_PR * torch.ones((int(self.y_dim/2),int(self.y_dim/2))), r2_PRR * torch.ones((int(self.y_dim/2),int(self.y_dim/2))))
            elif (self.config['model'] == "Navigation_Linear"):
                r2_pos = float(self.config['r2_pos'])
                r2_vel = float(self.config['r2_vel'])
                if (self.config['H_only_pos_when_Navigation_Linear'] == "False"):
                    if self.knowledge=='Full':
                        return torch.diag(torch.squeeze( torch.cat((r2_pos * torch.ones((1,2)),r2_vel * torch.ones((1,2))),dim=1) ))
                    elif self.knowledge=='Partial':
                        return torch.diag(torch.squeeze( torch.cat((r2_pos * torch.ones((1,2)),r2_vel * torch.ones((1,2))),dim=1) ))
                elif (self.config['H_only_pos_when_Navigation_Linear'] == "True"):
                    if self.knowledge=='Full':
                        return torch.diag(torch.squeeze( r2_pos * torch.ones((1,2)) ) )
                    elif self.knowledge=='Partial':
                        return torch.diag(torch.squeeze( r2_pos * torch.ones((1,2)) ) )
            else:
                raise NotImplementedError()

class Satellite:
    # Initialize the object with the x, y, and z coordinates of the satellite
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    # Function to create the Pseudo-Range (PR) and Pseudo-Range Rate (PRR)
    def createPR_PRR(self, Ref):
        # Calculate the PR using the coordinates of the satellite and the reference station
        PR = np.sqrt((Ref[1,:]-self.x)**2 + (Ref[2,:]-self.y)**2 + (0-self.z)**2)
        # Calculate the PRR using the coordinates of the satellite and the reference station
        PRR = (Ref[3,:]*(Ref[1,:]-self.x) + Ref[4,:]*(Ref[2,:]-self.y) + 0*(0-self.z))/PR
        # Assign the calculated PR and PRR to the object
        self.PR = np.squeeze(PR)
        self.PRR = np.squeeze(PRR)


class StateSpaceModel(GSSModel):
    def __init__(self, mode='train', knowledge='Full', data_gen=False):
        config.read('./config.ini')
        super().__init__()
        self.config = config['StateSpace']
        # mode = 'train' or 'valid' or 'test'
        self.mode = mode
        self.knowledge = knowledge
        self.sats = None
        self.data_gen = data_gen

        if self.mode not in ['train', 'valid', 'test']:
            raise ValueError('Possible mode = ["train", "valid", "test"]')
        if(self.config['model'] == "Synthetic"):
            if self.knowledge =='Full':
                self.alpha = 0.9
                self.beta = 1.1
                self.phi = 0.1*pi
                self.delta = 0.01
                self.a = 1
                self.b = 1
                self.c = 0
            elif self.knowledge =='Partial':
                self.alpha = 1
                self.beta = 1
                self.phi = 0
                self.delta = 0
                self.a = 1
                self.b = 1
                self.c = 0
            else:
                raise NotImplementedError()
            self.x_dim = 2
            self.y_dim = 2
        elif (self.config['model'] == "Navigation"):
            self.y_dim = int(self.config['n_Sat']) * 2
            self.dt = float(self.config['dt'])
            r2_PR = float(self.config['r2_PR'])
            r2_PRR = float(self.config['r2_PRR'])
            if knowledge=='Full':
                self.x_dim = 6
            elif knowledge=='Partial':
                self.x_dim = 4 # no Acc_x Acc_y
        elif (self.config['model'] == "Navigation_Linear"):
            r2_pos = float(self.config['r2_pos'])
            r2_vel = float(self.config['r2_vel'])
            if knowledge=='Full':
                self.x_dim = 6
            elif knowledge=='Partial':
                self.x_dim = 4 # no Acc_x Acc_y
            if (self.config['H_only_pos_when_Navigation_Linear'] == "True"):
                self.y_dim = 2
            else:
                self.y_dim = 4 # Position, Velocity, X and Y
            self.dt = float(self.config['dt'])
        elif (self.config['model'] == "Linear_Canonical"):
            self.x_dim = 2
            self.y_dim = 2
            self.dt = float(self.config['dt'])
        elif (config['StateSpace']['model'] == "Pendulum"):
            self.x_dim = 2
            self.y_dim = 2
            self.dt = float(self.config['dt'])
            self.G = 9.81 # Gravity [m/s^2]
            self.L = 1 # Length of the pendulum [m]
        else:
            raise NotImplementedError()

        self.q2 = float(self.config['q2'])
        self.cov_q = self.q2 * torch.eye(self.x_dim)
        self.cov_r = GSSModel.Construct_R(self)
        # self.v = float(self.config['v'])
        # self.r2 = torch.mul(self.q2, self.v)

        if(self.config['model'] == "Synthetic"):
            # self.init_state = torch.tensor([1., 0.]).reshape((-1, 1))
            # self.init_cov = torch.zeros((self.x_dim, self.x_dim))
            self.init_state = torch.tensor([0., 0.]).reshape((-1, 1))
            self.init_cov = float(config['EKF']['P0']) * torch.eye(self.x_dim)
        elif (self.config['model'] == "Navigation" or self.config['model'] == "Navigation_Linear"):
            dt = self.dt
            if knowledge=='Full':
                self.F = torch.tensor([[1, 0, dt, 0, 0.5 * dt ** 2, 0],
                                       [0, 1, 0, dt, 0, 0.5 * dt ** 2],
                                       [0, 0, 1, 0, dt, 0],
                                       [0, 0, 0, 1, 0, dt],
                                       [0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 1]])
            elif knowledge=='Partial':
                self.F = torch.tensor([[1, 0, dt, 0],
                                       [0, 1, 0, dt],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])
            self.sats = GSSModel.Create_Satellites(self, Nsatellites = int(self.config['n_Sat']), RandSat=False, Close_sat=((self.config['Close_Sat'].strip(''))=="True")) # Max 30 sattelites
            # self.init_state = torch.tensor([0.01, 0.01, 1, 1, 1, 1]).reshape(-1,1)
            # self.init_state = (torch.rand(self.x_dim)*2-1).reshape(-1, 1)
            # self.init_cov = torch.zeros((self.x_dim, self.x_dim))
            self.init_state = (torch.zeros(self.x_dim)).reshape(-1, 1)
            self.init_cov = float(config['EKF']['P0']) * torch.eye(self.x_dim)
        elif (self.config['model'] == "Linear_Canonical"):
            alpha_degree = float(self.config['rotate_linear_states'])
            if(alpha_degree!=0 and self.x_dim!=2):
                raise NotImplementedError()
            rotate_alpha = torch.tensor([alpha_degree / 180 * torch.pi])
            cos_alpha = torch.cos(rotate_alpha)
            sin_alpha = torch.sin(rotate_alpha)
            if self.data_gen: # Checks if its in Generate Data or not
                rotate_matrix = torch.tensor([[cos_alpha, -sin_alpha],
                                              [sin_alpha, cos_alpha]])
            else:
                rotate_matrix = torch.eye(2)
            F = torch.eye(self.x_dim)
            F[0] = torch.ones(1, self.x_dim) # First Row of F is 1
            # F = torch.mm(F, rotate_matrix)
            F = torch.mm(rotate_matrix, F)
            self.F = F / torch.linalg.svd(F).S[0] # Normalize F by the largest singular value - to make it stable
            self.init_state = torch.tensor([0., 0.]).reshape((-1, 1))
            self.init_cov = float(config['EKF']['P0']) * torch.eye(self.x_dim)
        elif (config['StateSpace']['model'] == "Pendulum"):
            self.init_state = torch.tensor([0., 0.]).reshape((-1, 1))
            self.init_cov = float(config['EKF']['P0']) * torch.eye(self.x_dim)
        else:
            raise NotImplementedError()

    def generate_data(self):
        config.read('./config.ini')
        self.save_path = './.data/StateSpace/'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)      

        if self.mode == 'train':
            if (self.config['model'] == "Synthetic"):
                self.seq_len = int(self.config['train_seq_len'])
            elif (self.config['model'] == "Navigation" or self.config['model'] == "Navigation_Linear"):
                self.seq_len = int(int(self.config['train_seq_len']) / self.dt)
            elif (self.config['model'] == "Linear_Canonical"):
                self.seq_len = int(int(self.config['train_seq_len']) / self.dt)
            elif (config['StateSpace']['model'] == "Pendulum"):
                self.seq_len = int(int(self.config['train_seq_len']) / self.dt)
            else:
                raise NotImplementedError()
            self.num_data = int(self.config['train_seq_num'])
            self.save_path += 'train/'
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)      
        elif self.mode == 'valid':
            if (self.config['model'] == "Synthetic"):
                self.seq_len = int(self.config['valid_seq_len'])
            elif (self.config['model'] == "Navigation" or self.config['model'] == "Navigation_Linear"):
                self.seq_len = int(int(self.config['valid_seq_len']) / self.dt)
            elif (self.config['model'] == "Linear_Canonical"):
                self.seq_len = int(int(self.config['valid_seq_len']) / self.dt)
            elif (config['StateSpace']['model'] == "Pendulum"):
                self.seq_len = int(int(self.config['valid_seq_len']) / self.dt)
            else:
                raise NotImplementedError()
            self.num_data = int(self.config['valid_seq_num'])
            self.save_path += 'valid/'
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path) 
        elif self.mode == 'test':
            if (self.config['model'] == "Synthetic"):
                self.seq_len = int(self.config['test_seq_len'])
            elif (self.config['model'] == "Navigation" or self.config['model'] == "Navigation_Linear"):
                self.seq_len = int(int(self.config['test_seq_len']) / self.dt)
            elif (self.config['model'] == "Linear_Canonical"):
                self.seq_len = int(int(self.config['test_seq_len']) / self.dt)
            elif (config['StateSpace']['model'] == "Pendulum"):
                self.seq_len = int(int(self.config['test_seq_len']) / self.dt)
            else:
                raise NotImplementedError()
            self.num_data = int(self.config['test_seq_num'])
            self.save_path += 'test/'
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path) 
        else:
            raise NotImplementedError()

        state_mtx = torch.zeros((self.num_data, self.x_dim, self.seq_len))
        obs_mtx = torch.zeros((self.num_data, self.y_dim, self.seq_len))

        with torch.no_grad():
            for i in range(self.num_data):

                if i % 100 == 0:
                    print(f'Saving {i} / {self.num_data} at {self.save_path}')
                state_tmp = torch.zeros((self.x_dim, self.seq_len))
                obs_tmp = torch.zeros((self.y_dim, self.seq_len))
                state_last = torch.clone(self.init_state)

                for j in range(self.seq_len):
                    x = self.next_state(state_last)
                    state_last = torch.clone(x)
                    y = self.observe(x)
                    state_tmp[:,j] = x.reshape(-1)
                    obs_tmp[:,j] = y.reshape(-1)
                
                state_mtx[i] = state_tmp
                obs_mtx[i] = obs_tmp
        
        torch.save(state_mtx, self.save_path + 'state.pt')
        torch.save(obs_mtx, self.save_path + 'obs.pt')

    def f(self, x):
        config.read('./config.ini')
        if (self.config['model'] == "Synthetic"):
            return self.alpha*torch.sin(self.beta*x+self.phi)+self.delta
        elif (config['StateSpace']['model'] == "Pendulum"):
            return torch.tensor([x[0]+x[1]*self.dt, x[1]-self.G/self.L*torch.sin(x[0])*self.dt]).reshape((-1,1))
        else:
            return torch.matmul(self.F, x)
    
    def g(self, x):
        config.read('./config.ini')

        if (self.config['model'] == "Synthetic"):
            return self.a*(self.b*x+self.c)**2
        elif (self.config['model'] == "Navigation"):
            x = x.reshape(-1)
            meas = []
            # Psuedo-Range
            for sat in self.sats:
                # Calculate the PR using the coordinates of the satellite and the reference station
                PR = np.sqrt((x[0] - sat.x) ** 2 + (x[1] - sat.y) ** 2 + (0 - sat.z) ** 2)
                # Stack the measurements
                meas.append(PR)
            # Psuedo-Range Rate
            for sat in self.sats:
                # Calculate the PRR using the coordinates of the satellite and the reference station
                PRR = (x[2] * (x[0] - sat.x) + x[3] * (x[1] - sat.y) + 0 * (0 - sat.z)) / PR
                # Stack the measurements
                meas.append(PRR)
            return torch.tensor([meas]).reshape((-1, 1))
        elif (self.config['model'] == "Navigation_Linear"):
            x = x.reshape(-1)
            if(self.data_gen): # Checks if its in Generate Data or not
                theta = float(self.config['rotate_linear_measurements']) * pi / 180
                rotate = torch.tensor(
                    [[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])
            else:
                rotate = torch.eye(2)
            pos_xy = rotate @ torch.tensor([[x[0]],[x[1]]]) # Rotated Position
            vel_xy = rotate @ torch.tensor([[x[2]], [x[3]]])  # Rotated Position
            meas = []
            meas.append(pos_xy[0]) # Position X
            meas.append(pos_xy[1]) # Position Y
            if (self.config['H_only_pos_when_Navigation_Linear'] == "False"):
                meas.append(vel_xy[0]) # Velocity X
                meas.append(vel_xy[1]) # Velocity Y
            return torch.tensor([meas]).reshape((-1, 1))
        elif (self.config['model'] == "Linear_Canonical"):
            if (self.y_dim==2):
                # H = I
                H = torch.eye(2)
            else:
                # H in reverse canonical form
                H = torch.zeros(self.x_dim, self.x_dim)
                H[0] = torch.ones(1, self.x_dim)
                for i in range(self.x_dim):
                    H[i, self.x_dim - 1 - i] = 1
            if (self.data_gen):  # Checks if its in Generate Data or not
                theta = float(self.config['rotate_linear_measurements']) * pi / 180
                rotate = torch.tensor(
                    [[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])
            else:
                rotate = torch.eye(2)
            self.H = torch.mm(H,rotate) * float(self.config['scale_linear_measurements'])
            return torch.matmul(self.H, x)
        elif (config['StateSpace']['model'] == "Pendulum"):
            theta = x[0]

            x_pos = self.L * torch.sin(theta)
            y_pos = -self.L * torch.cos(theta)

            return torch.tensor([x_pos, y_pos]).reshape((-1,1))
        else:
            raise NotImplementedError()

    def Jacobian_f(self, x):
        config.read('./config.ini')
        if (self.config['model'] == "Synthetic"):
            return torch.diag(torch.squeeze( self.alpha * self.beta * torch.cos(self.beta * x + self.phi) ))
        elif (config['StateSpace']['model'] == "Pendulum"):
            theta = x[0]

            F_matrix = torch.tensor([
                [1, self.dt],
                [-self.G/self.L*torch.cos(theta)*self.dt, 1]
            ])

            return F_matrix
        else:
            return self.F
        
    def Jacobian_g(self, x):
        config.read('./config.ini')

        if (self.config['model'] == "Synthetic"):
            return torch.diag(torch.squeeze(2*self.a*self.b*(self.b*x+self.c)))
        elif (self.config['model'] == "Navigation"):
            # Prepare Structures:
            nSat = len(self.sats)
            H = torch.zeros([nSat * 2, x.shape[0]])
            Pos_x = x[0]
            Pos_y = x[1]
            Pos_z = torch.zeros(Pos_x.shape)
            # Block Jacobian
            for ind, sat in enumerate(self.sats):
                Xsat = sat.x
                Ysat = sat.y
                Zsat = sat.z
                Sat_Distance = torch.sqrt((Xsat - Pos_x) ** 2 + (Ysat - Pos_y) ** 2 + (Zsat - Pos_z) ** 2)
                x_hat = (Pos_x - Xsat) / Sat_Distance
                y_hat = (Pos_y - Ysat) / Sat_Distance
                zero_padd2 = torch.zeros(1,2)  # For dh/dx Equation
                if self.knowledge == 'Partial':
                    H[ind + 0, :] = torch.hstack((x_hat.reshape(-1,1), y_hat.reshape(-1,1), zero_padd2))
                    H[ind + nSat, :] = torch.hstack((zero_padd2, x_hat.reshape(-1,1), y_hat.reshape(-1,1)))
                elif self.knowledge == 'Full':
                    H[ind + 0, :] = torch.hstack((x_hat.reshape(-1,1), y_hat.reshape(-1,1), zero_padd2, zero_padd2))
                    H[ind + nSat, :] = torch.hstack((zero_padd2, x_hat.reshape(-1,1), y_hat.reshape(-1,1), zero_padd2))
            return H
        elif(self.config['model'] == "Navigation_Linear"):
            if (self.config['H_only_pos_when_Navigation_Linear'] == "True"):
                if self.knowledge == 'Partial':
                    H = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
                elif self.knowledge == 'Full':
                    H = torch.tensor([[1., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.]])
            else:
                if self.knowledge == 'Partial':
                    H = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
                elif self.knowledge == 'Full':
                    H = torch.tensor([[1., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0.], [0., 0., 0., 1., 0., 0.]])
            return H
        elif(self.config['model'] == "Linear_Canonical"):
            return self.H
        elif (config['StateSpace']['model'] == "Pendulum"):
            theta = x[0]

            H_matrix = torch.tensor([
                [self.L * torch.cos(theta), 0],
                [self.L * torch.sin(theta), 0]
            ])

            return H_matrix
        else:
            raise NotImplementedError()

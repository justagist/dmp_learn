import os
import numpy as np
import matplotlib.pyplot as plt
from discrete_dmp import DiscreteDMP
from config import discrete_dmp_config


def plot_traj(trajectories):
    plt.figure("trajectories")
    
    for i in range(len(trajectories)):
        trajectory = trajectories[i]
        idx = trajectory.shape[1]*100 + 11
        if i == 0:
            color = 'r--'
        elif i==1:
            color = 'g'
        else:
            color = 'b'
        for k in range(trajectory.shape[1]):
            plt.subplot(idx)
            idx += 1
            plt.plot(trajectory[:,k], color)
    plt.show() 


def train_dmp(trajectory):
    discrete_dmp_config['dof'] = 2
    dmp = DiscreteDMP(config=discrete_dmp_config)
    dmp.load_demo_trajectory(trajectory)
    dmp.train()

    return dmp

def test_dmp(dmp, speed=1., plot_trained=False):
    test_config = discrete_dmp_config
    test_config['dt'] = 0.001

    # play with the parameters
    start_offset = np.zeros(discrete_dmp_config['dof'])
    goal_offset = np.zeros(discrete_dmp_config['dof'])
    external_force = np.zeros(discrete_dmp_config['dof'])
    alpha_phaseStop = 50.

    test_config['y0'] = dmp._traj_data[0, 1:] + start_offset
    test_config['dy'] = np.zeros(discrete_dmp_config['dof'])
    test_config['goals'] = dmp._traj_data[-1, 1:] + goal_offset
    test_config['tau'] = 1./speed
    test_config['ac'] = alpha_phaseStop
    test_config['type'] = 1
    test_config['dof'] = 2

    if test_config['type'] == 3:
        test_config['extForce'] = external_force
    else:
        test_config['extForce'] = np.zeros(discrete_dmp_config['dof'])
    test_traj = dmp.generate_trajectory(config=test_config)

    if plot_trained:
        plot_traj([dmp._traj_data[:,1:], test_traj['pos'][:,1:]])

    test_traj = {
    'pos_traj': test_traj['pos'][:,1:],
    'vel_traj':test_traj['vel'][:,1:],
    'acc_traj':test_traj['acc'][:,1:]
    }
    
    return test_traj

if __name__ == '__main__':

    trajectory = np.array([[1,2],[1,3],[2,3],[2,4],[3,4],[3,5]])
    dmp = train_dmp(trajectory)
    test_traj = test_dmp(dmp, speed=2.,plot_trained=True)



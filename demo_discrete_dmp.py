import os
import numpy as np
import matplotlib.pyplot as plt
from discrete_dmp import DiscreteDMP
from config import discrete_dmp_config
from JustaPython.MouseUtils import MouseTracker


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

def plot_path(trajectory, true_points, custom_start = None, custom_goal = None):

    plt.plot(trajectory[:,0],trajectory[:,1], label = "New path", color = 'b')
    plt.scatter(true_points[:,0],true_points[:,1], label = "Original Path", color = 'y')

    if custom_start is not None:
        plt.scatter(custom_start[0], custom_start[1], label = "Custom start", color = 'r')

    if custom_goal is not None:
        plt.scatter(custom_goal[0], custom_goal[1], label = "Custom goal", color = 'g')

    plt.axes().set_aspect('equal', 'datalim')
    # handles, labels = plt.get_legend_handles_labels()
    plt.legend()

    plt.show()


def train_dmp(trajectory):
    discrete_dmp_config['dof'] = 2
    dmp = DiscreteDMP(config=discrete_dmp_config)
    dmp.load_demo_trajectory(trajectory)
    dmp.train()

    return dmp

def test_dmp(dmp, speed=1., plot_trained=False, custom_start = None, custom_goal = None):
    test_config = discrete_dmp_config
    test_config['dt'] = 0.001

    # play with the parameters
    if custom_start is None:
        new_start = dmp._traj_data[0, 1:] + np.zeros(discrete_dmp_config['dof'])
    else:
        new_start = custom_start

    if custom_goal is None:
        new_goal = dmp._traj_data[-1, 1:] + np.zeros(discrete_dmp_config['dof'])
    else:
        new_goal = custom_goal

    external_force = np.zeros(discrete_dmp_config['dof'])
    alpha_phaseStop = 50

    test_config['y0'] = new_start
    test_config['dy'] = np.zeros(discrete_dmp_config['dof'])
    test_config['goals'] = new_goal
    test_config['tau'] = 1./speed
    test_config['ac'] = alpha_phaseStop
    test_config['type'] = 3

    if test_config['type'] == 1:
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

    mt = MouseTracker(window_dim = [600, 400])

    # ----- record trajectory using mouse
    print "Draw trajectory ... "
    trajectory = mt.record_mousehold_path(record_interval = 0.01, close_on_mousebutton_up = True, verbose = False, inverted = True, keep_window_alive = True)

    # ----- custom start and end points for the dmp
    print "Click custom start and end points"
    strt_end = mt.get_mouse_click_coords(num_clicks = 2, inverted = True, keep_window_alive = True, verbose = False)

    if trajectory is not None:
        dmp = train_dmp(trajectory)

        # ----- the trajectory after modifying the start and goal, speed etc.
        test_traj = test_dmp(dmp, speed=1.,plot_trained=False, custom_start = strt_end[0,:], custom_goal = strt_end[1,:])

        # ----- plotting the 2d paths (actual and modified)
        plot_path(test_traj['pos_traj'], trajectory, custom_start = strt_end[0,:], custom_goal = strt_end[1,:])

    else:
        print "No data in trajectory!\n"

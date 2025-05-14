import os
import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from .process_h5 import *



def _process_bag(path):
    """ Process .mat files that is converted from .bag files """

    data_ = loadmat(r"{}".format(path))
    data_ = data_['data_ee_pose']
    L = data_.shape[1]

    p_raw     = []
    q_raw     = []
    t_raw     = []

    sample_step = 5
    vel_thresh  = 1e-3 
    
    for l in range(L):
        data_l = data_[0, l]['pose'][0,0]
        pos_traj  = data_l[:3, ::sample_step]
        quat_traj = data_l[3:7, ::sample_step]
        time_traj = data_l[-1, ::sample_step].reshape(1,-1)

        raw_diff_pos = np.diff(pos_traj)
        vel_mag = np.linalg.norm(raw_diff_pos, axis=0).flatten()
        first_non_zero_index = np.argmax(vel_mag > vel_thresh)
        last_non_zero_index = len(vel_mag) - 1 - np.argmax(vel_mag[::-1] > vel_thresh)

        if first_non_zero_index >= last_non_zero_index:
            raise Exception("Sorry, vel are all zero")

        pos_traj  = pos_traj[:, first_non_zero_index:last_non_zero_index]
        quat_traj = quat_traj[:, first_non_zero_index:last_non_zero_index]
        time_traj = time_traj[:, first_non_zero_index:last_non_zero_index]
        
        p_raw.append(pos_traj.T)
        q_raw.append([R.from_quat(quat_traj[:, i]) for i in range(quat_traj.shape[1]) ])
        t_raw.append(time_traj.reshape(time_traj.shape[1]))

    return p_raw, q_raw, t_raw




def _get_sequence(seq_file):
    """
    Returns a list of containing each line of `seq_file`
    as an element

    Args:
        seq_file (str): File with name of demonstration files
                        in each line

    Returns:
        [str]: List of demonstration files
    """
    seq = None
    with open(seq_file) as x:
        seq = [line.strip() for line in x]
    return seq




def load_clfd_dataset(task_id=1, num_traj=1, sub_sample=3):
    """
    Load data from clfd dataset

    Return:
    -------
        p_raw:  a LIST of L trajectories, each containing M observations of N dimension, or [M, N] ARRAY;
                M can vary and need not be same between trajectories

        q_raw:  a LIST of L trajectories, each containting a LIST of M (Scipy) Rotation objects;
                need to consistent with M from position
        
    Note:
    ----
        NO time stamp available in this dataset!

        [num_demos=9, trajectory_length=1000, data_dimension=7] 
        A data point consists of 7 elements: px,py,pz,qw,qx,qy,qz (3D position followed by quaternions in the scalar first format).
    """

    L = num_traj
    T = 10.0            # pick a time duration 

    file_path           = os.path.dirname(os.path.realpath(__file__))  
    dir_path            = os.path.dirname(file_path)
    data_path           = os.path.dirname(dir_path)

    seq_file    = os.path.join(data_path, "dataset", "pos_ori", "robottasks_pos_ori_sequence_4.txt")
    filenames   = _get_sequence(seq_file)
    datafile    = os.path.join(data_path, "dataset", "pos_ori", filenames[task_id])
    
    data        = np.load(datafile)[:, ::sub_sample, :]

    p_raw = []
    q_raw = []
    t_raw = []

    for l in range(L):
        M = data[l, :, :].shape[0]

        data_ori = np.zeros((M, 4))         # convert to scalar last format, consistent with Scipy convention
        w        = data[l, :, 3 ].copy()  
        xyz      = data[l, :, 4:].copy()
        data_ori[:, -1]  = w
        data_ori[:, 0:3] = xyz

        p_raw.append(data[l, :, :3])
        q_raw.append([R.from_quat(q) for q in data_ori.tolist()])
        t_raw.append(np.linspace(0, T, M, endpoint=False))   # hand engineer an equal-length time stamp


    return p_raw, q_raw, t_raw




def load_demo_dataset():
    """
    Load demo data recorded from demonstration


    """

    input_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "demo", "all.mat")
    
    return _process_bag(input_path)




def load_npy():

    traj = np.load("dataset/UMI/traj1.npy")

    q_raw = [R.from_matrix(traj[i, :3, :3]) for i in range(traj.shape[0])]

    p_raw = [traj[i, :3, -1] for i in range(traj.shape[0])]

    """provide dt"""
    # dt = 0.07

    """or provide T"""
    T = 5
    dt = T/traj.shape[0]

    t_raw = [dt*i for i in range(traj.shape[0])]

    return [np.vstack(p_raw)], [q_raw], [t_raw], dt





def load_UMI():

    traj = np.load("dataset/UMI/traj1.npy")

    q_raw = [R.from_matrix(traj[i, :3, :3]) for i in range(traj.shape[0])]

    p_raw = [traj[i, :3, -1] for i in range(traj.shape[0])]

    """provide dt"""
    # dt = 0.07

    """or provide T"""
    T = 5
    dt = T/traj.shape[0]

    t_raw = [dt*i for i in range(traj.shape[0])]

    return [np.vstack(p_raw)], [q_raw], [t_raw], dt



def load_kine_npy():
    
    p_list = []
    q_list = []
    t_list = []
    dt_list = []

    folder_path = 'dataset/kine/'
    files = os.listdir(folder_path)
    demo_files = [f for f in files if f.startswith('demo') and f.endswith('.npy')]
    traj_list = [np.load(os.path.join(folder_path, f), allow_pickle=True) for f in demo_files]

    for traj in traj_list:
        cutoff = 1000
        filtered_range = range(cutoff, len(traj)-cutoff, 20)

        p_list.append(np.vstack([traj[i]['position'] for i in filtered_range]))
        q_list.append([R.from_quat(traj[i]['orientation']) for i in filtered_range])

        timestamp = np.array([traj[i]['t'] for i in filtered_range])
        dt = np.mean(np.diff(timestamp))
        t_list.append([dt*i for i in range(len(traj))])
        dt_list.append(dt)

        # T = 20
        # dt = T/len(filtered_range)
        # t_raw = [dt*i for i in filtered_range]

        print(dt)

    return p_list, q_list, t_list, dt_list





def load_kine_npy_old():
    
    
    # traj = np.load("dataset/kine/demo_2024-08-07-11-43-13.npy", allow_pickle=True)
    traj = np.load("dataset/kine/demo_2024-08-13-14-54-28.npy", allow_pickle=True)

    cutoff = 1000
    filtered_range = range(cutoff, len(traj)-cutoff, 20)

    q_raw = [R.from_quat(traj[i]['orientation']) for i in filtered_range]

    p_raw = [traj[i]['position'] for i in filtered_range]

    timestamp = np.array([traj[i]['t'] for i in filtered_range])
    dt_list = np.diff(timestamp)
    dt = np.mean(dt_list)
    t_raw = [dt*i for i in range(len(traj))]

    # T = 20
    # dt = T/len(filtered_range)
    # t_raw = [dt*i for i in filtered_range]

    print(dt)

    return [np.vstack(p_raw)], [q_raw], [t_raw], dt



def load_h5(file_path, fixed_time = False):
    if os.path.exists(file_path):
        print(os.listdir(file_path))
        filenames = [int(f.split('.')[0]) for f in os.listdir(file_path) if f.split('.')[0].isdigit()]
        sorted_filenames = sorted(filenames)


        p_list = []
        q_list = []
        t_list = []
        dt_list = []

        for idx, file_name in enumerate(sorted_filenames):
            ee_pose, timestamps = read_data(file_path + '/' + str(file_name) + '.h5')
            velocity = calculate_velocity(ee_pose, timestamps)
            filtered_pose, filtered_timestamps = filter_low_velocity(ee_pose, timestamps, velocity, threshold=0.01)
            sample_pose, sample_timestamps = downsample_data(filtered_pose, filtered_timestamps, target_length=200)
            plot_ee_position(sample_pose)

            traj_len = len(sample_pose)

            q_raw = [R.from_quat(sample_pose[i, 3:]) for i in range(traj_len)]

            p_raw = [sample_pose[i, :3] for i in range(traj_len)]

            timestamp = np.array([sample_timestamps[i] for i in range(traj_len)])


            if fixed_time:
                T = 7.0
                dt = T/len(timestamp)
            else:
                dt = np.mean(np.diff(timestamp))

            t_raw = [dt*i for i in range(traj_len)]

            print(dt)
            
            p_list.append(np.vstack(p_raw))
            q_list.append(q_raw)
            t_list.append(t_raw)
            dt_list.append(dt)
        return p_list, q_list, t_list, dt_list
    else:
        print("h5 File does not exist")

def load_h5_UMI(file_path, fixed_time=False, show_plot=False):
    if os.path.exists(file_path):
        
        filenames = [int(f.split('.')[0]) for f in os.listdir(file_path) if f.split('.')[0].isdigit()]
        sorted_filenames = sorted(filenames)
        print(sorted_filenames)


        p_list = []
        q_list = []
        t_list = []
        dt_list = []

        for idx, file_name in enumerate(sorted_filenames):
            ee_pose, timestamps = read_data(file_path + '/' + str(file_name) + '.h5')

            
            ee_pose = ee_pose[20:]
            if show_plot:
                plot_ee_position(ee_pose, mat=True)

            traj_len = len(ee_pose)
            print("traj_len", traj_len)

            q_raw = [R.from_matrix(ee_pose[i, :3, :3]) for i in range(traj_len)]

            p_raw = [ee_pose[i, :3, -1] for i in range(traj_len)]

        
            if fixed_time:
                T = 4.0
                dt = T/traj_len
            else:
                timestamp = np.array([timestamps[i] for i in range(traj_len)])
                dt = np.mean(np.diff(timestamp))

            t_raw = [dt*i for i in range(traj_len)]

            print(dt)
            
            p_list.append(np.vstack(p_raw))
            q_list.append(q_raw)
            t_list.append(t_raw)
            dt_list.append(dt)
            


        return p_list, q_list, t_list, dt_list
    else:
        print("h5 File does not exist")

def load_single_h5_UMI(file_path, segment_id=0, fixed_time=False, show_plot=False):
    if os.path.exists(file_path):
        
        ee_pose, timestamps = read_data(file_path + '/' + str(segment_id + 1) + '.h5')

        ee_pose = ee_pose[5:]
        if show_plot:
            plot_ee_position(ee_pose, mat=True)

        traj_len = len(ee_pose)
        print("traj_len", traj_len)

        q_raw = [R.from_matrix(ee_pose[i, :3, :3]) for i in range(traj_len)]

        p_raw = [ee_pose[i, :3, -1] for i in range(traj_len)]

    
        if fixed_time:
            T = 4.0
            dt = T/traj_len
        else:
            timestamp = np.array([timestamps[i] for i in range(traj_len)])
            dt = np.mean(np.diff(timestamp))

        t_raw = [dt*i for i in range(traj_len)]
        
        return np.vstack(p_raw), q_raw, t_raw, dt
    else:
        print("h5 File does not exist")


def load_h5_keypose(file_path):
    if os.path.exists(file_path):
        
        filenames = [int(f.split('.')[0]) for f in os.listdir(file_path) if f.split('.')[0].isdigit()]
        sorted_filenames = sorted(filenames)
        print(sorted_filenames)

        obj_key_pose_in_obj_list = []
        for idx, file_name in enumerate(sorted_filenames):
            with h5py.File(file_path + '/' + str(file_name) + '.h5', 'r') as hf:
                obj_key_pose_in_obj = np.array(hf['obj_keypose_in_obj'])
                obj_key_pose_in_obj_list.append(obj_key_pose_in_obj)


        return obj_key_pose_in_obj_list
    else:
        print("h5 File does not exist")

def load_single_h5_keypose(file_path):
    if os.path.exists(file_path):

        obj_key_pose_in_obj = None
        with h5py.File(file_path + '/' + '1.h5', 'r') as hf:
            obj_key_pose_in_obj = np.array(hf['obj_keypose_in_obj'])

        return obj_key_pose_in_obj
    else:
        print("h5 File does not exist")

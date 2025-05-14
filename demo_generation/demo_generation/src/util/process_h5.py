import h5py
import numpy as np
import matplotlib.pyplot as plt
import os


def read_data(file_path):
    with h5py.File(file_path, 'r') as hf:
        ee_pose = np.array(hf['ee_state'])
        timestamps = np.array(hf['t'])
    return ee_pose, timestamps


def calculate_velocity(ee_pose, timestamps):
    delta_pose = np.diff(ee_pose, axis=0)
    delta_time = np.diff(timestamps)
    
    delta_time[delta_time == 0] = np.min(delta_time[delta_time > 0])
    
    velocity = np.linalg.norm(delta_pose, axis=1) / delta_time
    return velocity

def filter_low_velocity(ee_pose, timestamps, velocity, threshold=0.01):
    valid_indices = np.where(velocity > threshold)[0]
    valid_indices = np.concatenate(([0], valid_indices + 1))
    filtered_pose = ee_pose[valid_indices]
    filtered_timestamps = timestamps[valid_indices]
    
    return filtered_pose, filtered_timestamps

def downsample_data(ee_pose, timestamps, target_length=200):
    data_length = len(ee_pose)
    
    if data_length <= target_length:
        return ee_pose, timestamps

    indices = np.linspace(0, data_length - 1, target_length).astype(int)
    
    downsampled_pose = ee_pose[indices]
    downsampled_timestamps = timestamps[indices]
    
    return downsampled_pose, downsampled_timestamps

def plot_ee_position(ee_pose, mat=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if mat:
        x = ee_pose[:, 0, -1]
        y = ee_pose[:, 1, -1]
        z = ee_pose[:, 2, -1]


    else:
        x = ee_pose[:, 0]
        y = ee_pose[:, 1]
        z = ee_pose[:, 2]
    
    ax.plot(x, y, z, marker='o')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('End-Effector Position in 3D Space')

    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_zlim([z.min(), z.max()])

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    Xb = 0.5 * max_range * np.array([-1, 1])
    Yb = 0.5 * max_range * np.array([-1, 1])
    Zb = 0.5 * max_range * np.array([-1, 1])

    ax.set_xlim(x.mean() + Xb)
    ax.set_ylim(y.mean() + Yb)
    ax.set_zlim(z.mean() + Zb)
    
    plt.show()

def main():
    file_path = os.path.join(os.getcwd(), 'demo_data', '2024-08-18_18-01-36', 'demonstration_data.h5')
    
    if os.path.exists(file_path):
        ee_pose, timestamps = read_data(file_path)
        
        velocity = calculate_velocity(ee_pose, timestamps)
        
        filtered_pose, filtered_timestamps = filter_low_velocity(ee_pose, timestamps, velocity, threshold=0.01)
        
        downsampled_pose, downsampled_timestamps = downsample_data(filtered_pose, filtered_timestamps, target_length=200)
        
        print(filtered_pose.shape)
        print(downsampled_pose.shape)
        plot_ee_position(downsampled_pose)
    else:
        print(f"File {file_path} does not exist.")

if __name__ == '__main__':
    main()

from diffusion_policies.common.replay_buffer import ReplayBuffer
import numpy as np
import copy
import os
import zarr
from termcolor import cprint
from demo_generation.mask_util import restore_and_filter_pcd
import imageio
from scipy.spatial import cKDTree
from tqdm import tqdm
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import logging
from scipy.spatial.transform import Rotation as R

# Import EMP integration
from .emp_integration import EMPPlanner

class DemoGen:
    def __init__(self, cfg):
        self.data_root = cfg.data_root
        self.source_name = cfg.source_name
        
        self.task_n_object = cfg.task_n_object
        self.use_linear_interpolation = cfg.use_linear_interpolation
        self.interpolate_step_size = cfg.interpolate_step_size

        self.use_manual_parsing_frames = cfg.use_manual_parsing_frames
        self.parsing_frames = cfg.parsing_frames
        self.mask_names = cfg.mask_names

        self.gen_name = cfg.generation.range_name
        self.object_trans_range = cfg.trans_range[self.gen_name]["object"]
        self.target_trans_range = cfg.trans_range[self.gen_name]["target"]

        self.n_gen_per_source = cfg.generation.n_gen_per_source
        self.render_video = cfg.generation.render_video
        if self.render_video:
            cprint("[NOTE] Rendering video is enabled. It takes ~10s to render a single generated trajectory.", "yellow")
        self.gen_mode = cfg.generation.mode

        self.load_obstacle_config(cfg.generation)

        source_zarr = os.path.join(self.data_root, "datasets", "source", self.source_name + ".zarr")
        self._load_from_zarr(source_zarr)
        
        # Initialize EMP planner
        self.emp_planner = EMPPlanner()
        self.use_emp = cfg.get('use_emp', True)
        self.emp_initialized = False
        

    def _load_from_zarr(self, zarr_path):
        cprint(f"Loading data from {zarr_path}", "blue")
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud'])
        self.n_source_episodes = self.replay_buffer.n_episodes
        self.demo_name = zarr_path.split("/")[-1].split(".")[0]
    
    def initialize_emp(self, source_demo, segment_frames=None):
        try:
            if segment_frames is None:
                positions = np.asarray(source_demo["state"][:, :3])
                orientations = [R.from_quat([0, 0, 0, 1]) for _ in range(len(positions))]
                p_out = np.zeros_like(positions)
                q_out = orientations.copy()
                p_att = positions[-1]
                q_att = orientations[-1]
                success = success = self.emp_planner.load_emp_model(positions, orientations, p_out, q_out, p_att, q_att, dt=0.05, task_id=0)
                if not success:
                    print("[EMP INIT ERROR] Could not load model for task_id 0")
            else:
                for i, (start, end) in enumerate(zip(segment_frames[:-1], segment_frames[1:])):
                    start = int(start)
                    end = int(end)
                    positions = np.asarray(source_demo["state"][start:end, :3])
                    if positions.shape[0] < 5:
                        print(f"[EMP INIT WARNING] Too short segment {i} for EMP. Skipping.")
                        continue
                    orientations = [R.from_quat([0, 0, 0, 1]) for _ in range(len(positions))]
                    p_out = np.zeros_like(positions)
                    q_out = orientations.copy()
                    p_att = positions[-1]
                    q_att = orientations[-1]
                    success = success = self.emp_planner.load_emp_model(positions, orientations, p_out, q_out, p_att, q_att, dt=0.05, task_id=i)
                    if not success:
                        print(f"[EMP INIT ERROR] Could not load model for task_id {i}")
            self.emp_initialized = True
        except Exception as e:
            print(f"[EMP INIT CRITICAL] EMP initialization failed: {e}")

    def generate_trans_vectors(self, trans_range, n_demos, mode="random"):
        """
        Argument: trans_range: (2, 3)
            [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        Return: A list of translation vectors. (n_demos, 3)
        """
        x_min, x_max, y_min, y_max = trans_range[0][0], trans_range[1][0], trans_range[0][1], trans_range[1][1]
        if mode == "grid":
            n_side = int(np.sqrt(n_demos))
            if n_side ** 2 != n_demos or n_demos == 1:
                raise ValueError("In grid mode, n_demos must be a squared number larger than 1")
            x_values = [x_min + i / (n_side - 1) * (x_max - x_min) for i in range(n_side)]
            y_values = [y_min + i / (n_side - 1) * (y_max - y_min) for i in range(n_side)]
            xyz = list(set([(x, y, 0) for x in x_values for y in y_values]))
            return np.array(xyz)
        elif mode == "random":
            xyz = []
            for _ in range(n_demos):
                x = np.random.random() * (x_max - x_min) + x_min
                y = np.random.random() * (y_max - y_min) + y_min
                xyz.append([x, y, 0])
            return np.array(xyz)
        else:
            raise NotImplementedError
    
    def generate_demo(self):
        if self.task_n_object == 1:
            self.one_stage_augment(self.n_gen_per_source, self.render_video, self.gen_mode)
        elif self.task_n_object == 2:
            self.two_stage_augment(self.n_gen_per_source, self.render_video, self.gen_mode)
        else:
            raise NotImplementedError
    
    def one_stage_augment(self, n_demos, render_video=False, gen_mode='random'):
        # Prepare translation vectors
        trans_vectors = []
        if gen_mode == 'random':
            trans_vectors = self.generate_trans_vectors(self.object_trans_range, n_demos, mode="random")
        elif gen_mode == 'grid':
            def check_squared_number(arr):
                return np.isclose(np.sqrt(arr), np.round(np.sqrt(arr)))
            assert check_squared_number(n_demos), "n_demos must be a squared number"
            trans_vectors = self.generate_trans_vectors(self.object_trans_range, n_demos, mode="grid")

        generated_episodes = []

        for i in tqdm(range(self.n_source_episodes)):
            cprint(f"Generating demos for source demo {i}", "blue")
            source_demo = self.replay_buffer.get_episode(i)
            pcds = source_demo["point_cloud"]
            
            if self.use_manual_parsing_frames:
                skill_1_frame = self.parsing_frames["skill-1"]
            else:
                ee_poses = source_demo["state"][:, :3]
                skill_1_frame = self.parse_frames_one_stage(pcds, i, ee_poses)
            print(f"Skill-1: {skill_1_frame}")
            
            pcd_obj = self.get_objects_pcd_from_sam_mask(pcds[0], i, "object")
            obj_bbox = self.pcd_bbox(pcd_obj)
            
            # Initialize EMP if needed and using it
            if self.use_emp and not self.emp_initialized:
                self.initialize_emp(source_demo, [0, skill_1_frame, len(source_demo["state"])])

            for obj_trans_vec in tqdm(trans_vectors):
                ############## start of generating one episode ##############
                current_frame = 0

                traj_states = []
                traj_actions = []
                traj_pcds = []
                trans_sofar = np.zeros(3)

                ############# stage {motion-1} starts #############
                trans_togo = obj_trans_vec.copy()
                source_demo = self.replay_buffer.get_episode(i)
                start_pos = source_demo["state"][0][:3] - source_demo["action"][0][:3]  # home state
                end_pos = source_demo["state"][skill_1_frame-1][:3] + trans_togo
                
                # Use EMP to generate motion trajectory
                motion_traj = self.emp_planner.adapt_trajectory(
                    source_demo["state"][:skill_1_frame, :3],
                    start_pos,
                    end_pos,
                    task_id=0
                )

                # Inject external disturbance at a specific frame
                disturbance_frame = len(motion_traj) // 2
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                disturbance = 0.05 * direction  # 5 cm impulse
                motion_traj[disturbance_frame] += disturbance
                    
                # Extract actions from the trajectory
                steps = min(skill_1_frame, len(motion_traj))
                for j in range(steps):
                    if j == 0:
                        step_action = motion_traj[j] - start_pos
                    else:
                        step_action = motion_traj[j] - motion_traj[j-1]
                    
                    source_action = source_demo["action"][current_frame]
                    traj_actions.append(np.concatenate([step_action, source_action[3:]], axis=0))
                    trans_this_frame = step_action - source_action[:3]
                    trans_sofar[:3] += trans_this_frame
                    
                    # "state" and "point_cloud" consider the accumulated translation
                    state = source_demo["state"][current_frame].copy()
                    state[:3] = motion_traj[j]  # Use EMP trajectory point
                    traj_states.append(state)
                    
                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    pcd_obj, pcd_robot = self.pcd_divide(source_pcd, [obj_bbox])
                    pcd_obj = self.pcd_translate(pcd_obj, obj_trans_vec)
                    pcd_robot = self.pcd_translate(pcd_robot, trans_sofar)
                    traj_pcds.append(np.concatenate([pcd_robot, pcd_obj], axis=0))
                if hasattr(self, 'obstacle') and self.obstacle is not None:
                    obs_pcd = self.generate_obstacle_point_cloud()
                    traj_pcds[-1] = np.concatenate([traj_pcds[-1], obs_pcd], axis=0)
                    
                    current_frame += 1
                    if current_frame >= skill_1_frame:
                        break
                
                ############## stage {motion-1} ends #############
                num_frames = source_demo["state"].shape[0]
                ############# stage {skill-1} starts #############
                while current_frame < num_frames:
                    action = source_demo["action"][current_frame].copy()
                    traj_actions.append(action)

                    # "state" and "point_cloud" consider the accumulated translation
                    state = source_demo["state"][current_frame].copy()
                    state[:3] += trans_sofar
                    traj_states.append(state)
                    
                    pcd_obj_robot = source_demo["point_cloud"][current_frame].copy()
                    pcd_obj_robot = self.pcd_translate(pcd_obj_robot, trans_sofar)
                    traj_pcds.append(pcd_obj_robot)
                if hasattr(self, 'obstacle') and self.obstacle is not None:
                    obs_pcd = self.generate_obstacle_point_cloud()
                    traj_pcds[-1] = np.concatenate([traj_pcds[-1], obs_pcd], axis=0)

                    current_frame += 1
                    ############## stage {skill-1} ends #############

                generated_episode = {
                    "state": np.array(traj_states),
                    "action": np.array(traj_actions),
                    "point_cloud": np.array(traj_pcds)
                }
                generated_episodes.append(generated_episode)

                if render_video:
                    video_name = f"{i}_obj[{np.round(obj_trans_vec[0], 3)},{np.round(obj_trans_vec[1], 3)}].mp4"
                    video_path = os.path.join(self.data_root, "videos", self.source_name, self.gen_name, video_name)
                    self.point_cloud_to_video(generated_episode["point_cloud"], video_path, elev=20, azim=30)
                ############## end of generating one episode ##############
        
        # save the generated episodes
        save_path = os.path.join(self.data_root, "datasets", "generated", f"{self.source_name}_{self.gen_name}_{n_demos}.zarr")
        self.save_episodes(generated_episodes, save_path)
    
    def two_stage_augment(self, n_demos, render_video=False, gen_mode='random'):
        """
        An implementation of the DemoGen augmentation process for manipulation tasks involving two objects,
        enhanced with EMP for motion generation.
        """
        # Prepare translation vectors
        trans_vectors = []      # [n_demos, 6 (obj_xyz + targ_xyz)]
        if gen_mode == 'random':
            for _ in range(n_demos):
                obj_xyz = self.generate_trans_vectors(self.object_trans_range, 1, mode="random")[0]
                targ_xyz = self.generate_trans_vectors(self.target_trans_range, 1, mode="random")[0]
                trans_vectors.append(np.concatenate([obj_xyz, targ_xyz], axis=0))
        elif gen_mode == 'grid':
            def check_fourth_power(arr):
                fourth_roots = np.power(arr, 1/4)
                return np.isclose(fourth_roots, np.round(fourth_roots))
            assert check_fourth_power(n_demos), "n_demos must be a fourth power"
            sqrt_n_demos = int(np.sqrt(n_demos))
            obj_xyz = self.generate_trans_vectors(self.object_trans_range, sqrt_n_demos, mode="grid")
            targ_xyz = self.generate_trans_vectors(self.target_trans_range, sqrt_n_demos, mode="grid")
            for o_xyz in obj_xyz:
                for t_xyz in targ_xyz:
                    trans_vectors.append(np.concatenate([o_xyz, t_xyz], axis=0))
        else:
            raise NotImplementedError

        generated_episodes = []
        # For every source demo
        for i in range(self.n_source_episodes):
            cprint(f"Generating demos for source demo {i}", "blue")
            source_demo = self.replay_buffer.get_episode(i)
            pcds = source_demo["point_cloud"]
            
            if self.use_manual_parsing_frames:
                skill_1_frame = self.parsing_frames["skill-1"]
                motion_2_frame = self.parsing_frames["motion-2"]
                skill_2_frame = self.parsing_frames["skill-2"]
            else:
                ee_poses = source_demo["state"][:, :3]
                skill_1_frame, motion_2_frame, skill_2_frame = self.parse_frames_two_stage(pcds, i, ee_poses)
            
            print(f"Skill-1: {skill_1_frame}, Motion-2: {motion_2_frame}, Skill-2: {skill_2_frame}")
            
            pcd_obj = self.get_objects_pcd_from_sam_mask(pcds[0], i, "object")
            pcd_tar = self.get_objects_pcd_from_sam_mask(pcds[0], i, "target")
            obj_bbox = self.pcd_bbox(pcd_obj)
            tar_bbox = self.pcd_bbox(pcd_tar)
            
            # Initialize EMP if needed and using it
            if self.use_emp and not self.emp_initialized:
                # Initialize EMP with segmented trajectory
                self.initialize_emp(source_demo, [0, skill_1_frame, motion_2_frame, skill_2_frame, len(source_demo["state"])])

            # Generate demos according to translation vectors
            for trans_vec in tqdm(trans_vectors):
                obj_trans_vec = trans_vec[:3]
                tar_trans_vec = trans_vec[3:6]
                ############## start of generating one episode ##############
                current_frame = 0

                traj_states = []
                traj_actions = []
                traj_pcds = []
                trans_sofar = np.zeros(3)

                ############# stage {motion-1} starts #############
                trans_togo = obj_trans_vec.copy()
                source_demo = self.replay_buffer.get_episode(i)
                start_pos = source_demo["state"][0][:3] - source_demo["action"][0][:3] # home state
                end_pos = source_demo["state"][skill_1_frame-1][:3] + trans_togo
                
                # Use EMP to generate motion trajectory
                motion_traj = self.emp_planner.adapt_trajectory(
                    source_demo["state"][:skill_1_frame, :3],
                    start_pos,
                    end_pos,
                    task_id=0
                )

                # Inject external disturbance at a specific frame
                disturbance_frame = len(motion_traj) // 2
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                disturbance = 0.05 * direction  # 5 cm impulse
                motion_traj[disturbance_frame] += disturbance

                
                # Extract actions from the trajectory
                steps = min(skill_1_frame, len(motion_traj))
                for j in range(steps):
                    if j == 0:
                        step_action = motion_traj[j] - start_pos
                    else:
                        step_action = motion_traj[j] - motion_traj[j-1]
                    
                    source_action = source_demo["action"][current_frame]
                    traj_actions.append(np.concatenate([step_action, source_action[3:]], axis=0))
                    trans_this_frame = step_action - source_action[:3]
                    trans_sofar[:3] += trans_this_frame
                    
                    # "state" and "point_cloud" consider the accumulated translation
                    state = source_demo["state"][current_frame].copy()
                    state[:3] = motion_traj[j]  # Use EMP trajectory point
                    traj_states.append(state)
                    
                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    pcd_obj, pcd_tar, pcd_robot = self.pcd_divide(source_pcd, [obj_bbox, tar_bbox])
                    pcd_obj = self.pcd_translate(pcd_obj, obj_trans_vec)
                    pcd_tar = self.pcd_translate(pcd_tar, tar_trans_vec)
                    pcd_robot = self.pcd_translate(pcd_robot, trans_sofar)
                    traj_pcds.append(np.concatenate([pcd_robot, pcd_obj, pcd_tar], axis=0))
                if hasattr(self, 'obstacle') and self.obstacle is not None:
                    obs_pcd = self.generate_obstacle_point_cloud()
                    traj_pcds[-1] = np.concatenate([traj_pcds[-1], obs_pcd], axis=0)
                    
                    current_frame += 1
                    if current_frame >= skill_1_frame:
                        break
                
                ############## stage {motion-1} ends #############
                
                ############# stage {skill-1} starts #############
                while current_frame < motion_2_frame:
                    action = source_demo["action"][current_frame].copy()
                    traj_actions.append(action)

                    # "state" and "point_cloud" consider the accumulated translation
                    state = source_demo["state"][current_frame].copy()
                    state[:3] += trans_sofar
                    traj_states.append(state)
                    
                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    pcd_tar, pcd_obj_robot = self.pcd_divide(source_pcd, [tar_bbox])
                    pcd_tar = self.pcd_translate(pcd_tar, tar_trans_vec)
                    pcd_obj_robot = self.pcd_translate(pcd_obj_robot, trans_sofar)
                    traj_pcds.append(np.concatenate([pcd_obj_robot, pcd_tar], axis=0))
                if hasattr(self, 'obstacle') and self.obstacle is not None:
                    obs_pcd = self.generate_obstacle_point_cloud()
                    traj_pcds[-1] = np.concatenate([traj_pcds[-1], obs_pcd], axis=0)

                    current_frame += 1
                ############## stage {skill-1} ends #############
                
                ############# stage {motion-2} starts #############
                trans_togo = tar_trans_vec.copy()
                start_pos = traj_states[-1][:3]
                end_pos = source_demo["state"][skill_2_frame-1][:3] + trans_togo
                
                # Use EMP to generate motion trajectory
                motion_traj = self.emp_planner.adapt_trajectory(
                    source_demo["state"][motion_2_frame:skill_2_frame, :3],
                    start_pos,
                    end_pos,
                    task_id=2  # Third segment (index 2)
                )

                # Inject external disturbance at a specific frame
                disturbance_frame = len(motion_traj) // 2
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                disturbance = 0.05 * direction  # 5 cm impulse
                motion_traj[disturbance_frame] += disturbance

                
                # Extract actions from the trajectory
                steps = min(skill_2_frame - motion_2_frame, len(motion_traj))
                for j in range(steps):
                    if j == 0:
                        step_action = motion_traj[j] - start_pos
                    else:
                        step_action = motion_traj[j] - motion_traj[j-1]
                    
                    source_action = source_demo["action"][current_frame]
                    traj_actions.append(np.concatenate([step_action, source_action[3:]], axis=0))
                    trans_this_frame = step_action - source_action[:3]
                    trans_sofar[:3] += trans_this_frame
                    
                    # "state" and "point_cloud" consider the accumulated translation
                    state = source_demo["state"][current_frame].copy()
                    state[:3] = motion_traj[j]  # Use EMP trajectory point
                    traj_states.append(state)
                    
                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    pcd_tar, pcd_obj_robot = self.pcd_divide(source_pcd, [tar_bbox])
                    pcd_tar = self.pcd_translate(pcd_tar, tar_trans_vec)
                    pcd_obj_robot = self.pcd_translate(pcd_obj_robot, trans_sofar)
                    traj_pcds.append(np.concatenate([pcd_obj_robot, pcd_tar], axis=0))
                if hasattr(self, 'obstacle') and self.obstacle is not None:
                    obs_pcd = self.generate_obstacle_point_cloud()
                    traj_pcds[-1] = np.concatenate([traj_pcds[-1], obs_pcd], axis=0)
                    
                    current_frame += 1
                    if current_frame >= skill_2_frame:
                        break
                ############## stage {motion-2} ends #############
                
                ############# stage {skill-2} starts #############
                num_frames = source_demo["state"].shape[0]
                while current_frame < num_frames:
                    action = source_demo["action"][current_frame].copy()
                    traj_actions.append(action)

                    # "state" and "point_cloud" consider the accumulated translation
                    # "state" and "point_cloud" consider the accumulated translation
                    state = source_demo["state"][current_frame].copy()
                    state[:3] += trans_sofar
                    traj_states.append(state)
                    
                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    pcd_obj_robot = source_demo["point_cloud"][current_frame].copy()
                    pcd_obj_robot = self.pcd_translate(pcd_obj_robot, trans_sofar)
                    traj_pcds.append(pcd_obj_robot)
                if hasattr(self, 'obstacle') and self.obstacle is not None:
                    obs_pcd = self.generate_obstacle_point_cloud()
                    traj_pcds[-1] = np.concatenate([traj_pcds[-1], obs_pcd], axis=0)

                    current_frame += 1
                ############## stage {skill-2} ends #############

                generated_episode = {
                    "state": np.array(traj_states),
                    "action": np.array(traj_actions),
                    "point_cloud": np.array(traj_pcds)
                }
                generated_episodes.append(generated_episode)

                if render_video:
                    video_name = f"{i}_obj[{np.round(obj_trans_vec[0], 3)},{np.round(obj_trans_vec[1], 3)}]_tar[{np.round(tar_trans_vec[0], 3)},{np.round(tar_trans_vec[1], 3)}].mp4"
                    video_path = os.path.join(self.data_root, "videos", self.source_name, self.gen_name, video_name)
                    self.point_cloud_to_video(generated_episode["point_cloud"], video_path, elev=20, azim=30)
                ############## end of generating one episode ##############
        
        # save the generated episodes
        save_path = os.path.join(self.data_root, "datasets", "generated", f"{self.source_name}_{self.gen_name}_{n_demos}.zarr")
        self.save_episodes(generated_episodes, save_path)
    
    
    def load_obstacle_config(self, cfg):
        self.obstacle = cfg.get("obstacle", None)

    def generate_obstacle_point_cloud(self, density=500):
        if self.obstacle is None:
            return np.zeros((0, 3))

        shape = self.obstacle["shape"]
        size = np.array(self.obstacle["size"])
        pos = np.array(self.obstacle["position"])

        if shape == "square":
            x = np.random.uniform(-size[0]/2, size[0]/2, density)
            y = np.random.uniform(-size[1]/2, size[1]/2, density)
            z = np.random.uniform(-size[2]/2, size[2]/2, density)
            return np.stack([x, y, z], axis=1) + pos
        elif shape == "sphere":
            pts = np.random.randn(density, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            pts *= size[0] / 2
            return pts + pos
        else:
            raise ValueError(f"Unsupported obstacle shape: {shape}")


    def pcd_divide(self, pcd, bboxes):
        """
        Divide a point cloud into parts based on bounding box.
        
        Args:
            pcd: point cloud
            bboxes: list of bounding boxes
            
        Returns:
            list of point clouds, separated by bounding boxes, plus remaining points
        """
        pcds = []
        remain_pcd = pcd.copy()
        
        for bbox in bboxes:
            mask = ((remain_pcd[:, 0] >= bbox[0][0]) & (remain_pcd[:, 0] <= bbox[1][0]) & 
                   (remain_pcd[:, 1] >= bbox[0][1]) & (remain_pcd[:, 1] <= bbox[1][1]) & 
                   (remain_pcd[:, 2] >= bbox[0][2]) & (remain_pcd[:, 2] <= bbox[1][2]))
            pcds.append(remain_pcd[mask])
            remain_pcd = remain_pcd[~mask]
            
        pcds.append(remain_pcd)
        return pcds
    
    def pcd_translate(self, pcd, translation):
        """
        Translate point cloud by adding a translation vector.
        
        Args:
            pcd: point cloud to translate
            translation: translation vector [x, y, z]
            
        Returns:
            translated point cloud
        """
        if len(pcd) == 0:
            return pcd
        translated_pcd = pcd.copy()
        translated_pcd[:, :3] += translation
        return translated_pcd

    def pcd_bbox(self, pcd):
        """
        Compute axis-aligned bounding box of point cloud.
        
        Args:
            pcd: point cloud
            
        Returns:
            bbox: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        """
        min_vals = np.min(pcd[:, :3], axis=0)
        max_vals = np.max(pcd[:, :3], axis=0)
        return [min_vals, max_vals]
    
    def get_objects_pcd_from_sam_mask(self, pcd, episode_id, object_or_target="object"):
        """
        Get object point cloud from the segmentation mask.
        
        Args:
            pcd: point cloud
            episode_id: episode id
            object_type: "object" or "target"
            
        Returns:
            object point cloud
        """
        #mask_path = os.path.join(self.data_root, "mask", self.source_name, f"{episode_id}_{object_type}.jpg")
        mask = imageio.imread(os.path.join(self.data_root, f"sam_mask/{self.source_name}/{episode_id}/{self.mask_names[object_or_target]}.jpg"))
        mask = mask > 128
        filtered_pcd = restore_and_filter_pcd(pcd, mask)
        return filtered_pcd
    
    def parse_frames_one_stage(self, pcds, demo_idx, ee_poses, distance_mode="pcd2pcd", threshold_1=0.23):
        assert distance_mode in ["ee2pcd", "pcd2pcd"]
        for i in range(pcds.shape[0]):
            object_pcd = self.get_objects_pcd_from_sam_mask(pcds[i], demo_idx, "object")
            if distance_mode == "pcd2pcd":
                obj_bbox = self.pcd_bbox(object_pcd)
                source_pcd = pcds[i].copy()
                _, pcd_ee = self.pcd_divide(source_pcd, [obj_bbox])
                if self.chamfer_distance(pcd_ee, object_pcd) <= threshold_1:
                    print(f"Stage starts at frame {i}")
                    start_frame = i
                    break
            elif distance_mode == "ee2pcd":
                if self.average_distance_to_point_cloud(ee_poses[i], object_pcd) <= threshold_1:
                    print(f"Stage starts at frame {i}")
                    start_frame = i
                    break
        return start_frame
    
    def parse_frames_two_stage(self, pcds, demo_idx, ee_poses, distance_mode="ee2pcd", threshold_1=0.15, threshold_2=0.235, threshold_3=0.275,):
        """
        There are two ways to parse the frames of whole trajectory into object-centric segments: (1) Either by comparing the distance between 
            the end-effector and the object point cloud, (2) Or by manually specifying the frames when `self.use_manual_parsing_frames = True`.
        This function implements the first way. While it is an automatic process, you need to tune the distance thresholds to achieve a clean parse.
        Since DemoGen requires very few source demos, it is also feasible (actually recommended) to manually specify the frames for parsing.
        To manually decide the parsing frames, you can set the translation vectors to zero, run the DemoGen code, render the videos, and check the
            frame_idx on the left top of the video.
        """
        assert distance_mode in ["ee2pcd", "pcd2pcd"]
        stage = 1
        for i in range(pcds.shape[0]):
            object_pcd = self.get_objects_pcd_from_sam_mask(pcds[i], demo_idx, "object")
            target_pcd = self.get_objects_pcd_from_sam_mask(pcds[i], demo_idx, "target")
            if stage == 1:
                if distance_mode == "ee2pcd":            
                    if self.average_distance_to_point_cloud(ee_poses[i], object_pcd) <= threshold_1:
                        # visualizer.visualize_pointcloud(pcds[i])
                        stage = 2
                        skill_1_frame = i
                        object_origin_pcd = object_pcd
                elif distance_mode == "pcd2pcd":
                    obj_bbox = self.pcd_bbox(object_pcd)
                    tar_bbox = self.pcd_bbox(target_pcd)
                    source_pcd = pcds[i].copy()
                    _, _, ee_pcd = self.pcd_divide(source_pcd, [obj_bbox, tar_bbox])
                    if self.chamfer_distance(object_pcd, ee_pcd) <= threshold_1:
                        stage = 2
                        skill_1_frame = i
                        object_origin_pcd = object_pcd
                        
            elif stage == 2:
                if distance_mode == "ee2pcd":
                    if self.average_distance_to_point_cloud(ee_poses[i], object_origin_pcd) >= threshold_2:
                        stage = 3
                        motion_2_frame = i
                        # visualizer.visualize_pointcloud(pcds[i])
                elif distance_mode == "pcd2pcd":
                    tar_bbox = self.pcd_bbox(target_pcd)
                    source_pcd = pcds[i].copy()
                    _, ee_obj_pcd = self.pcd_divide(source_pcd, [tar_bbox])
                    if self.chamfer_distance(ee_obj_pcd, object_origin_pcd) >= threshold_2:
                        stage = 3
                        motion_2_frame = i
                        
            elif stage == 3:
                if distance_mode == "ee2pcd":
                    if self.average_distance_to_point_cloud(ee_poses[i], target_pcd) <= threshold_3:
                        skill_2_frame = i
                        # visualizer.visualize_pointcloud(pcds[i])
                        break
                elif distance_mode == "pcd2pcd":
                    tar_bbox = self.pcd_bbox(target_pcd)
                    source_pcd = pcds[i].copy()
                    _, ee_obj_pcd = self.pcd_divide(source_pcd, [tar_bbox])
                    if self.chamfer_distance(ee_obj_pcd, target_pcd) <= threshold_3:
                        skill_2_frame = i
                        break
                
        print(f"Stage 1: {skill_1_frame}, Pre-2: {motion_2_frame}, Stage 2: {skill_2_frame}")
        return skill_1_frame, motion_2_frame, skill_2_frame
    
    def save_episodes(self, episodes, save_path):
        """
        Save episodes to a zarr file.
        
        Args:
            episodes: list of episodes
            save_path: path to save the zarr file
        """
        cprint(f"Saving {len(episodes)} episodes to {save_path}", "blue")
        output_buffer = ReplayBuffer.create_empty(save_path)
        
        for episode in episodes:
            output_buffer.add_episode(episode)
            
        output_buffer.flush()
    
    def point_cloud_to_video(self, pcds, save_path, elev=30, azim=45):
        """
        Render point cloud sequence as video.
        
        Args:
            pcds: list of point clouds
            save_path: path to save the video
            elev: elevation angle for plot
            azim: azimuth angle for plot
        """
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
            
        images = []
        
        bounds = np.zeros((2, 3))
        for pcd in pcds:
            min_bound = np.min(pcd[:, :3], axis=0)
            max_bound = np.max(pcd[:, :3], axis=0)
            bounds[0] = np.minimum(bounds[0], min_bound) if not np.all(bounds[0] == 0) else min_bound
            bounds[1] = np.maximum(bounds[1], max_bound)
        
        margin = 0.05  # Add margin
        bounds[0] -= margin
        bounds[1] += margin
        
        for i, pcd in enumerate(pcds):
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Color the points (e.g., by height)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=pcd[:, 2], cmap='viridis', s=1)
            
            # Set consistent view
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(bounds[0][0], bounds[1][0])
            ax.set_ylim(bounds[0][1], bounds[1][1])
            ax.set_zlim(bounds[0][2], bounds[1][2])
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Frame {i}')
            
            # Convert to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(img)
            
            plt.close(fig)
        
        # Save as video
        imageio.mimsave(save_path, images, fps=10)
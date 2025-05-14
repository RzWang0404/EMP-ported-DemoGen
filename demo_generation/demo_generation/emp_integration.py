import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.linalg import null_space

from .src.util import load_tools, process_tools, quat_tools, plot_tools
from .src.se3_class import se3_class
# from src.se3_elastic.src.util import plot_tools
from .src.se3_elastic.elastic_pos_class import elastic_pos_class
from .src.se3_elastic.elastic_ori_class import elastic_ori_class


class EMPPlanner:
    """
    Class that integrates the EMP (Elastic Motion Planning) for trajectory adaptation.
    This provides a clean interface to use EMP models within DemoGen.
    """
    def __init__(self):
        self.emp_models = {}  # Store EMP models by task_id
        self.se3_obj_list = {}  # Store se3 objects by task_id
        self.use_orientation = False  # Default to position-only

    def load_emp_model(self, positions, orientations, p_out, q_out, p_att, q_att, dt=0.05, task_id=0):
        """
        Load and initialize an EMP model for the given trajectory segment.
        
        Args:
            positions: List of positions [n, 3]
            orientations: List of orientations as Rotation objects [n]
            p_out: Position outputs (velocity) [n, 3]
            q_out: Orientation outputs (angular velocity) [n]
            p_att: Position attractor
            q_att: Orientation attractor
            dt: Time step
            task_id: Task ID for this model
        """
        try:       
            # Process inputs for EMP
            p_in = np.array(positions)
            if len(p_in) < 5:
                raise ValueError("Too few trajectory points for EMP model.")
            if np.isnan(p_in).any():
                raise ValueError("NaNs detected in input positions.")
            if np.allclose(p_in, p_in[0]):
                raise ValueError("Degenerate trajectory: all points are identical.")

            p_in = np.array(positions)
            if len(p_in) < 5:
                raise ValueError("Too few trajectory points for EMP model.")
            if np.isnan(p_in).any():
                raise ValueError("NaNs detected in input positions.")
            if np.allclose(p_in, p_in[0]):
                raise ValueError("Degenerate trajectory: all points are identical.")
                
            q_in = orientations
            t_in = np.arange(len(positions)) * dt
            
            # Create EMP model
            se3_obj = se3_class('.', [p_in], [q_in], [p_out], [q_out], p_att, q_att, dt, 4, task_id)
            se3_obj.begin(write_json=False)
            
            # Store model
            self.se3_obj_list[task_id] = se3_obj
            print(f"EMP model loaded for task_id {task_id}")
            return True
            
        except Exception as e:
            print(f"[EMP ERROR] Failed to load model for task_id {task_id}: {e}")
            return False
        except Exception as e:
            print(f"Error initializing EMP model: {e}")
            return False
    
    def enable_orientation_tracking(self, enable=True):
        """
        Enable or disable orientation tracking.
        
        Args:
            enable: Whether to enable orientation tracking
        """
        self.use_orientation = enable
        print(f"Orientation tracking {'enabled' if enable else 'disabled'}")
    
    def adapt_trajectory(self, source_positions, start_pos, end_pos, task_id=0, step_size=0.05):
        """
        Adapt a trajectory from source to new start/end points using EMP.
        
        Args:
            source_positions: Source trajectory positions [n, 3]
            start_pos: New start position [3]
            end_pos: New end position [3]
            task_id: Task ID for the EMP model to use
            step_size: Integration step size
            
        Returns:
            Adapted trajectory positions [n, 3]
        """
        """
        if task_id not in self.se3_obj_list:
            print(f"No EMP model found for task_id {task_id}. Using linear interpolation.")
            # Fall back to linear interpolation
            t = np.linspace(0, 1, len(source_positions))
            return start_pos + t[:, np.newaxis] * (end_pos - start_pos)
        """
            
        #if task_id not in self.se3_obj_list:
            #raise RuntimeError(f"[EMP ERROR] No EMP model found for task_id {task_id}. Aborting.")
        
        try:
            se3_obj = self.se3_obj_list[int(task_id)]
            
            # Extract the GMM parameters
            linear_dict, _ = se3_obj._logOut(write_json=False)
            
            # Set up elastic position adaptation
            n_gaussian = int(linear_dict['K'])
            n_dim = int(linear_dict['M'])
            Prior = np.array(linear_dict['Prior'])
            Mu = np.array(linear_dict['Mu']).reshape((n_gaussian, n_dim))
            Sigma = np.array(linear_dict['Sigma']).reshape((n_gaussian, n_dim, n_dim))
            x_0 = np.array(linear_dict['x_0'])
            att = np.array(linear_dict['attractor'])
            
            # Create elastic position object
            elastic_pos_obj = elastic_pos_class(Prior, Mu, Sigma, x_0, att)
            
            # Calculate velocity constraints from source trajectory
            v_start = source_positions[1] - source_positions[0] if len(source_positions) > 1 else np.zeros_like(start_pos)
            v_end = source_positions[-1] - source_positions[-2] if len(source_positions) > 1 else np.zeros_like(end_pos)
            
            # Apply geometric constraints
            elastic_pos_obj._geo_constr(x_start=start_pos, x_end=end_pos, v_start=v_start, v_end=v_end)
            
            # Generate adapted trajectory
            data, _, gmm_struct, _, _, new_att = elastic_pos_obj.start_adapting()
            
            # Update the DS model with the new trajectory
            updated_positions = data[0]
            se3_obj.pos_ds.elasticUpdate(updated_positions, gmm_struct, new_att)
            
            # Simulate the trajectory using the updated model
            q_init = R.from_quat([0, 0, 0, 1])
            p_test, _, _, _, _, _ = se3_obj.sim(start_pos, q_init, step_size=step_size)
            
            self.tracked_trajectories.append(np.array(p_test))
            return np.array(p_test)

        
        except Exception as e:
            print(f"Error adapting trajectory with EMP: {e}")
            # Fall back to linear interpolation
            """
            t = np.linspace(0, 1, len(source_positions))
            return start_pos + t[:, np.newaxis] * (end_pos - start_pos)
            """
            
    def adapt_full_pose(self, source_positions, source_orientations, 
                        start_pos, end_pos, start_ori, end_ori, 
                        task_id=0, step_size=0.05):
        """
        Adapt a full pose trajectory (position + orientation) using EMP.
        
        Args:
            source_positions: Source trajectory positions [n, 3]
            source_orientations: Source trajectory orientations [n]
            start_pos: New start position [3]
            end_pos: New end position [3]
            start_ori: New start orientation (Rotation object)
            end_ori: New end orientation (Rotation object)
            task_id: Task ID for the EMP model to use
            step_size: Integration step size
            
        Returns:
            Tuple of (positions [n, 3], orientations [n])
        """    
        if task_id not in self.se3_obj_list:
            raise RuntimeError(f"[EMP ERROR] No EMP model found for task_id {task_id}. Aborting.")
            
        try:
            se3_obj = self.se3_obj_list[task_id]
            
            # Get the EMP parameters
            linear_dict, angular_dict = se3_obj._logOut(write_json=False)
            
            # Position adaptation
            n_gaussian_pos = linear_dict['K']
            n_dim_pos = linear_dict['M']
            Prior_pos = linear_dict['Prior']
            Mu_pos = np.array(linear_dict['Mu']).reshape((n_gaussian_pos, n_dim_pos))
            Sigma_pos = np.array(linear_dict['Sigma']).reshape((n_gaussian_pos, n_dim_pos, n_dim_pos))
            att_pos = np.array(linear_dict['attractor'])
            x0_pos = np.array(linear_dict['x_0'])
            
            # Initialize elastic position adapter
            elastic_pos_obj = elastic_pos_class(Prior_pos, Mu_pos, Sigma_pos, x0_pos, att_pos)
            elastic_pos_obj._geo_constr(
                x_start=start_pos, 
                x_end=end_pos, 
                v_start=source_positions[1]-source_positions[0], 
                v_end=source_positions[-1]-source_positions[-2]
            )
            
            # Adapt position trajectory
            pos_data, _, pos_gmm_struct, _, _, att_new_pos = elastic_pos_obj.start_adapting()
            adapted_positions = pos_data[0].T
            
            # Orientation adaptation
            n_gaussian_ori = angular_dict['K']
            n_dim_ori = angular_dict['M']
            Prior_ori = angular_dict['Prior']
            Mu_ori = np.array(angular_dict['Mu']).reshape((2 * n_gaussian_ori, n_dim_ori))
            Sigma_ori = np.array(se3_obj.ori_ds.gmm.Sigma_gt)
            att_ori = np.array(angular_dict['att_ori'])
            x0_ori = np.array(angular_dict['q_0'])
            
            # Handle orientation half-space issues
            q_end_adapted = end_ori
            q_start_adapted = start_ori
            
            # Initialize elastic orientation adapter
            elastic_ori_obj = elastic_ori_class(
                Prior_ori, Mu_ori, Sigma_ori, 
                q_start_adapted.as_quat(), q_end_adapted.as_quat()
            )
            
            # Set orientation constraints
            elastic_ori_obj._geo_constr(
                x_start=q_start_adapted, 
                x_end=q_end_adapted, 
                v_start=(source_orientations[1], source_orientations[0]),
                v_end=(source_orientations[-1], source_orientations[-2])
            )
            
            # Adapt orientation trajectory
            new_q_in, new_q_out, _, ori_gmm_struct, _, _, att_ori_new = elastic_ori_obj.start_adapting()
            
            # Update the EMP model with the adaptations
            se3_obj.pos_ds.elasticUpdate(adapted_positions, pos_gmm_struct, att_new_pos)
            se3_obj.ori_ds.elasticUpdate(new_q_in, new_q_out, ori_gmm_struct, att_ori_new)
            
            # Simulate the adapted trajectory
            p_test, q_test, _, _, _, _ = se3_obj.sim(start_pos, q_start_adapted, step_size=step_size)
            
            return np.array(p_test), q_test
            
        except Exception as e:
            print(f"Error adapting full pose trajectory with EMP: {e}")
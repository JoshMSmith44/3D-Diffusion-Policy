import numpy as np
import mujoco_py
import torch

def sample_claw_poses(n, scale=True):

    hand_model = mujoco_py.load_model_from_path("/home/josh/Courses/ESE6500/final_proj/3D-Diffusion-Policy/third_party/rrl-dependencies/mj_envs/mj_envs/hand_manipulation_suite/assets/DAPG_door.xml")
    def clip_and_scale(points, bounds):
        points = np.clip(points, bounds[:, 0], bounds[:, 1])
        ranges = bounds[:, 1] - bounds[:, 0]
        midpoints = (bounds[:, 1] + bounds[:, 0]) / 2.0
        normalized_points = 2 * (points - midpoints) / ranges
        return normalized_points

    finger_dev = 0.05
    mf_th_hinge = 0.7
    hinge_dev = 0.2

    offset = 6
    # Making a lookup table for joints by name. Ignoring the first (offset) joints. 
    u = {}
    actuator_names = [actuator for actuator in hand_model.actuator_names]
    for i in range(offset,hand_model.nu):
        name = actuator_names[i]
        u[name]=i-offset
        
    # Defining control range 
    ctrl_space = np.array(hand_model.actuator_ctrlrange)[offset:]
    ctrl_bounds = ctrl_space.copy()

    # Absolute Constraints
    #ctrl_bounds[u['A_WRJ1']] = [0,0] # Fix wrist yaw
    #ctrl_bounds[u['A_WRJ0'],1] = -0.5 # Tilt wrist back to emulate claw
    
    ctrl_bounds[u['A_MFJ1'],0] = 0.3 # Keep fingers slightly curled
    ctrl_bounds[u['A_MFJ3'],1] = -0.1 # point fingers at thumb
    ctrl_bounds[u['A_MFJ0'],1] = 0.2 # Prevent finger curling

    ctrl_bounds[u['A_LFJ4'],0] = 0.5 # Prevent finger curling

    ctrl_bounds[u['A_THJ0'],0] = -0.4 # Prevent thumb curling
    ctrl_bounds[u['A_THJ1'],1] = -0.1 # Prevent thumb curling
    ctrl_bounds[u['A_THJ3'],0] = 1.1 # Tuck thumb under
    ctrl_bounds[u['A_THJ4']] = [-0.2,0.7] # Claw closing range

    # Relative deviations (added to fixed joints later)

    ctrl_bounds[6-offset:10-offset] = [-finger_dev, finger_dev] # Keep fingers together
    ctrl_bounds[14-offset:23-offset] = [-finger_dev, finger_dev] # Keep fingers together
    ctrl_bounds[u['A_MFJ2']] = [-hinge_dev, hinge_dev] # Keep fingers in sync with thumb

    samples = np.random.uniform(low=ctrl_bounds[:,0], high=ctrl_bounds[:,1], size=(n,ctrl_space.shape[0]))

    samples[:,u['A_MFJ2']] += (samples[:,u['A_THJ4']] + mf_th_hinge)
    samples[:,u['A_FFJ0']] += samples[:,u['A_MFJ0']]; samples[:,u['A_RFJ0']] += samples[:,u['A_MFJ0']]
    samples[:,u['A_FFJ1']] += samples[:,u['A_MFJ1']]; samples[:,u['A_RFJ1']] += samples[:,u['A_MFJ1']]
    samples[:,u['A_FFJ2']] += samples[:,u['A_MFJ2']]; samples[:,u['A_RFJ2']] += samples[:,u['A_MFJ2']]
    samples[:,u['A_FFJ3']] += samples[:,u['A_MFJ3']]; samples[:,u['A_RFJ3']] += samples[:,u['A_MFJ3']]
    samples[:,u['A_LFJ0']] += samples[:,u['A_RFJ0']]
    samples[:,u['A_LFJ1']] += samples[:,u['A_RFJ1']]
    samples[:,u['A_LFJ2']] += samples[:,u['A_RFJ2']]
    samples[:,u['A_LFJ3']] += samples[:,u['A_RFJ3']]

    if scale:
        samples = clip_and_scale(samples,ctrl_space)

    return samples


class ConvexModel():
    def __init__(self):
        self.sample_points_unnormalized = sample_claw_poses(10000)
        self.offset = 6
    
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
        self.normalize_sample_points()

    def update_betas(self, betas):
        self.betas = betas
        self.a = 1-self.betas
        self.ah = np.cumprod(self.a)
    
    def normalize_sample_points(self):
        extended_sample_points = np.concatenate([np.zeros((self.sample_points_unnormalized.shape[0], self.offset)), self.sample_points_unnormalized], axis=1)
        src_pts = (self.normalizer['action'].normalize(extended_sample_points))[:, self.offset:]
        self.src_pts = np.array(src_pts.cpu())
    
    def get_model_output(self, trajectory, t):
        ah = self.ah[t]
        print("ah", ah)
        #calculate the optimal delta for each x0
        traj = np.array(trajectory.cpu(), dtype=np.double)
        diffs = (traj[:, :, self.offset:] - self.src_pts[:, None] * np.sqrt(ah))
        #deltas =  diffs / np.sqrt(1 - ah)

        t_var_inv = 1 / (1-ah) 
        euc_dist = -0.5 * (np.sum(diffs**2, axis=-1) * t_var_inv)
        euc_dist_max = np.max(euc_dist, axis=0, keepdims=True)
        euc_dist = euc_dist - euc_dist_max
        exps = np.exp(euc_dist)
        ps = exps / np.sum(exps, axis=0)


        mult = ps[:, :, None] * self.src_pts[:, None, :]#deltas


        avg_eth = np.sum(mult, axis=0)
        avg_eth = np.concatenate([traj[0, :, :self.offset], avg_eth], axis=1)
        avg_eth = torch.tensor(avg_eth[None, :, :], dtype=torch.float, device='cuda')
        return avg_eth
    

    
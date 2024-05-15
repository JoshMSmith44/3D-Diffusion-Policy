# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

import gym
# from abc import ABC
import numpy as np
from .rrl_encoder import Encoder, IdentityEncoder
from PIL import Image
import torch
from collections import deque
from mjrl.utils.gym_env import GymEnv
import matplotlib.pyplot as plt
from mujoco_py import MjRenderContextOffscreen
from scipy.spatial.transform import Rotation
import mujoco_py

_mj_envs = {'pen-v0', 'hammer-v0', 'door-v0', 'relocate-v0'}

def make_encoder(encoder, encoder_type, device, is_eval=True) :
    if not encoder :
        if encoder_type == 'resnet34' or encoder_type == 'resnet18' :
            encoder = Encoder(encoder_type)
        elif encoder_type == 'identity' :
            encoder = IdentityEncoder()
        else :
            print("Please enter valid encoder_type.")
            raise Exception
    if is_eval:
        encoder.eval()
    encoder.to(device)
    return encoder

def rotMatFromQuat(quat):
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    n = 1.0/np.sqrt(qx*qx+qy*qy+qz*qz+qw*qw)
    qw *= n
    qx *= n
    qy *= n
    qz *= n

    mat = np.array([[1.0 - 2.0*qy*qy - 2.0*qz*qz, 2.0*qx*qy - 2.0*qz*qw, 2.0*qx*qz + 2.0*qy*qw],
    [2.0*qx*qy + 2.0*qz*qw, 1.0 - 2.0*qx*qx - 2.0*qz*qz, 2.0*qy*qz - 2.0*qx*qw],
    [2.0*qx*qz - 2.0*qy*qw, 2.0*qy*qz + 2.0*qx*qw, 1.0 - 2.0*qx*qx - 2.0*qy*qy]])
    return mat


class BasicPairedAdroitEnv(gym.Env): # , ABC
    def __init__(self, env_name, cameras, latent_dim=512, hybrid_state=True, channels_first=False, 
    height=84, width=84, test_image=False, num_repeats=1, num_frames=1, encoder_type=None, device=None):

        self.viewer = None
        env = GymEnv("nocol-" + env_name)
        #env = GymEnv(env_name)
        self._env = env
        self._claw_env = GymEnv("claw-" + env_name)
        self._env.viewer = None
        self._claw_env.viewer = None

        if "nocol" in env.env.unwrapped.spec.id:
            env.env.unwrapped.spec.id = env.env.unwrapped.spec.id[env.env.unwrapped.spec.id.find("-") + 1 :]
        self.env_id = env.env.unwrapped.spec.id
        self.device = device

        self._num_repeats = num_repeats
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)

        self.encoder = None
        self.transforms = None
        self.encoder_type = encoder_type
        if encoder_type is not None:
            self.encoder = make_encoder(encoder=None, encoder_type=self.encoder_type, device=self.device, is_eval=True)
            self.transforms = self.encoder.get_transform()

        if test_image:
            print("======================adroit image test mode==============================")
            print("======================adroit image test mode==============================")
            print("======================adroit image test mode==============================")
            print("======================adroit image test mode==============================")
        self.test_image = test_image

        self.cameras = cameras
        self.latent_dim = latent_dim
        self.hybrid_state = hybrid_state
        self.channels_first = channels_first
        self.height = height
        self.width = width
        self.action_space = self._env.action_space
        self.env_kwargs = {'cameras' : cameras, 'latent_dim' : latent_dim, 'hybrid_state': hybrid_state,
                           'channels_first' : channels_first, 'height' : height, 'width' : width}

        shape = [3, self.width, self.height]
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )
        self.sim = env.env.sim
        self._env.spec.observation_dim = latent_dim

        if hybrid_state :
            if self.env_id in _mj_envs:
                self._env.spec.observation_dim += 24 # Assuming 24 states for adroit hand.

        self.spec = self._env.spec
        self.observation_dim = self.spec.observation_dim
        self.horizon = self._env.env.spec.max_episode_steps
        print("norm env action shape", self._env.action_space)
        print("claw env action shape", self._claw_env.action_space)
        print("")

        self.ik = True
        self.wait_until_still = True
        if self.wait_until_still:
            print("Wait until still")

    
    def do_ik(self):
        hand_sim = self._env.env.sim
        hand_model = hand_sim.model
        hand_data = hand_sim.data

        init_qpos = np.copy(hand_data.qpos)


        # Load the MuJoCo hand model
        claw_sim = self._claw_env.env.sim
        claw_model = claw_sim.model
        claw_data = claw_sim.data

        def jacobian():
            # Call mj_comPos (required for Jacobians).
            mujoco_py.functions.mj_comPos(hand_model, hand_data)

            # Get end-effector site Jacobian.
            #jac_pos_ff = np.empty((3, hand_model.nv))
            #jac_quat_ff = np.empty((3, hand_model.nv))
            #jac_pos_th = np.empty((3, hand_model.nv))
            #jac_quat_th = np.empty((3, hand_model.nv))
            #mujoco_py.mj_jacSite(hand_model, hand_data, jac_pos_ff, jac_quat_ff, hand_data.site('Tch_fftip').id)
            #mujoco_py.mj_jacSite(hand_model, hand_data, jac_pos_th, jac_quat_th, hand_data.site('Tch_thtip').id)

            jac_pos_ff = np.empty((3 * hand_model.nv))
            jac_pos_th = np.empty((3 * hand_model.nv))
            hand_data.get_site_jacp('Tch_mftip', jacp = jac_pos_ff)
            hand_data.get_site_jacp('Tch_thtip', jacp = jac_pos_th)
            jac_pos_ff = jac_pos_ff.reshape(3, hand_model.nv)
            jac_pos_th = jac_pos_th.reshape(3, hand_model.nv)

            # Stack jacobians in single 6x30 matrix
            return np.vstack((jac_pos_ff, jac_pos_th))
        
        def solve_ik(max_iterations=100, tolerance=1e-6):
            mujoco_py.functions.mj_kinematics(hand_model, hand_data)
            current_angles = hand_data.qpos
            current_positions = np.hstack((hand_data.get_site_xpos('Tch_mftip'), hand_data.get_site_xpos('Tch_thtip')))

            claw_left = claw_data.get_site_xpos('gripper_frame_2_ur5right')
            claw_right = claw_data.get_site_xpos('gripper_frame_1_ur5right')

            # Position residual.
            target_positions = np.hstack((claw_left, claw_right))

            for i in range(max_iterations):
                # Compute end effector positions
                hand_data.qpos[:] = current_angles
                mujoco_py.functions.mj_kinematics(hand_model, hand_data)
                current_positions = np.hstack((hand_data.get_site_xpos('Tch_mftip'), hand_data.get_site_xpos('Tch_thtip')))
                
                # Compute error between current and target positions
                error = target_positions - current_positions
                
                # Check convergence
                if np.linalg.norm(error) < tolerance:
                    print("Convergence achieved after", i+1, "iterations.")
                    return current_angles
                
                # Compute gradient using pseudo-inverse Jacobian
                jac = jacobian()
                jac_pinv = np.linalg.pinv(jac)
                position_gradient = np.dot(jac_pinv, error)        
                
                # Compute nullspace projector
                nullspace_projector = np.eye(hand_model.njnt) - np.dot(jac_pinv, jac)

                # Secondary task: Drive joints towards the centers of their limits
                #joint_limits_midpoints = 0.5 * (hand_model.jnt_range[:, 0] + hand_model.jnt_range[:, 1])
                joint_limits_midpoints = init_qpos
                joint_limits_error = joint_limits_midpoints - current_angles
                joint_limits_gradient = np.dot(nullspace_projector, joint_limits_error)
                
                # Update joint angles
                current_angles += position_gradient + joint_limits_gradient

            print("Max iterations reached without convergence.")
            return current_angles
        new_q = solve_ik()
        hand_data.qpos[:] = new_q
        #hand_data.qpos[28:] = 0
        mujoco_py.functions.mj_step(hand_model, hand_data)



  # return np.vstack((jac_pos_ff, jac_quat_ff, jac_pos_th, jac_quat_th))

    def get_obs(self,):
        # for our case, let's output the image, and then also the sensor features
        if self.env_id in _mj_envs :
            env_state = self._env.env.get_env_state()
            qp = env_state['qpos']

        if self.env_id == 'pen-v0':
            qp = qp[:-6]
        elif self.env_id == 'door-v0':
            qp = qp[4:-2]
        elif self.env_id == 'hammer-v0':
            qp = qp[2:-7]
        elif self.env_id == 'relocate-v0':
            qp = qp[6:-6]

        imgs = [] # number of image is number of camera

        if self.encoder is not None:
            for cam in self.cameras :
                img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=0)
                img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=0)
                # img = env.env.sim.render(width=84, height=84, mode='offscreen')
                img = img[::-1, :, : ] # Image given has to be flipped
                if self.channels_first :
                    img = img.transpose((2, 0, 1))
                #img = img.astype(np.uint8)
                img = Image.fromarray(img)
                img = self.transforms(img)
                imgs.append(img)

            inp_img = torch.stack(imgs).to(self.device) # [num_cam, C, H, W]
            z = self.encoder.get_features(inp_img).reshape(-1)
            # assert z.shape[0] == self.latent_dim, "Encoded feature length : {}, Expected : {}".format(z.shape[0], self.latent_dim)
            pixels = z
        else:
            if not self.test_image:
                for cam in self.cameras : # for each camera, render once
                    img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=0) # TODO device id will think later
                    img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=0) # TODO device id will think later
                    # img = img[::-1, :, : ] # Image given has to be flipped
                    if self.channels_first :
                        img = img.transpose((2, 0, 1)) # then it's 3 x width x height
                    # we should do channels first... (not sure why by default it's not, maybe they did some transpose when using the encoder?)
                    #img = img.astype(np.uint8)
                    # img = Image.fromarray(img) # TODO is this necessary?
                    imgs.append(img)
            else:
                img = (np.random.rand(1, 84, 84) * 255).astype(np.uint8)
                imgs.append(img)
            pixels = np.concatenate(imgs, axis=0)

        # TODO below are what we originally had... 
        # if not self.test_image:
        #     for cam in self.cameras : # for each camera, render once
        #         img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=0) # TODO device id will think later
        #         # img = img[::-1, :, : ] # Image given has to be flipped
        #         if self.channels_first :
        #             img = img.transpose((2, 0, 1)) # then it's 3 x width x height
        #         # we should do channels first... (not sure why by default it's not, maybe they did some transpose when using the encoder?)
        #         #img = img.astype(np.uint8)
        #         # img = Image.fromarray(img) # TODO is this necessary?
        #         imgs.append(img)
        # else:
        #     img = (np.random.rand(1, 84, 84) * 255).astype(np.uint8)
        #     imgs.append(img)
        # pixels = np.concatenate(imgs, axis=0)

        if not self.hybrid_state : # this defaults to True... so RRL uses hybrid state
            qp = None

        sensor_info = qp
        return pixels, sensor_info

    def get_env_infos(self):
        return self._env.get_env_infos()
    
    def set_seed(self, seed):
        return self._env.set_seed(seed)

    def get_stacked_pixels(self): #TODO fix it
        assert len(self._frames) == self._num_frames
        stacked_pixels = np.concatenate(list(self._frames), axis=0)
        return stacked_pixels
    
    def get_desired_claw_pose(self):
        #index_tip = self._env.env.sim.data.get_site_xpos("S_fftip")
        index_tip = self._env.env.sim.data.get_site_xpos("S_mftip")
        thumb_tip = self._env.env.sim.data.get_site_xpos("S_thtip")
        desired_claw_point = (index_tip + thumb_tip) / 2
        claw_init_joint_id = self._claw_env.env.sim.model.joint_name2id("ARTx")
        claw_init_body_id = self._claw_env.env.sim.model.body_name2id("forearm")
        #rot_mat = rotMatFromQuat(self._claw_env.env.sim.model.body_quat[claw_init_body_id])
        #print("rot mat", rot_mat)
        #jnt_world_diff = rot_mat @ self._claw_env.env.sim.model.jnt_pos[claw_init_joint_id]
        init_claw_pose = self._claw_env.env.sim.model.body_pos[claw_init_body_id]
        #print("init claw pose", init_claw_pose)
        claw_action_linear = desired_claw_point - init_claw_pose
        #print("desired pose", desired_claw_point)
        #print("init claw pose", init_claw_pose)
        x_vec = (index_tip - thumb_tip) / np.linalg.norm(index_tip - thumb_tip)
        metacarpal_pose = self._env.env.sim.data.get_site_xpos("Tch_ffmetacarpal")
        palm_to_midpoint_vec = (desired_claw_point - metacarpal_pose)/np.linalg.norm(desired_claw_point - metacarpal_pose)
        z_vec = np.cross(x_vec, palm_to_midpoint_vec)
        z_vec /= np.linalg.norm(z_vec)
        y_vec = np.cross(z_vec, x_vec)
        y_vec /= np.linalg.norm(x_vec)
        rot_mat = np.vstack([x_vec, y_vec, z_vec])
        des_rot = Rotation.from_matrix(rot_mat.T)
        eulers = des_rot.as_euler("XZY")
        claw_action = np.zeros(7)
        tip_dist = np.linalg.norm(index_tip - thumb_tip)
        if tip_dist > 0.085:
            claw_action_linear += x_vec * (tip_dist - 0.085)/2
        claw_action[:3] = claw_action_linear
        claw_action[3:6] = eulers
        claw_action[4] = eulers[2]
        claw_action[5] = eulers[1]
        claw_action[6] = 1 - (np.clip(tip_dist, 0, 0.085) / 0.085)

        return claw_action

    def reset(self):
        self._claw_env.reset()
        self._env.reset()
        if self.wait_until_still:
            self.prev_action = np.zeros(self.action_space.shape)
            self.prev_action[0] = -0.3
            self.step_until_still(self.prev_action)

        claw_action = self.get_desired_claw_pose()
        for i in range(20):
            self._claw_env.step(claw_action)
            #disp_img = self.get_paired_display()
            #plt.imshow(disp_img)
            #plt.show()
            thumb_tip = self._claw_env.env.sim.data.get_body_xpos("forearm")
        self.set_env_to_claw_env_state()

        pixels, sensor_info = self.get_obs()
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        stacked_pixels = self.get_stacked_pixels()
        return stacked_pixels, sensor_info
    
    def trim_action(self, action, max_diff = 0.05):
        diff = action - self.prev_action
        diff[diff > max_diff] = max_diff
        diff[diff < -max_diff] = -max_diff
        return self.prev_action + diff
    
    def step_until_still(self, action, max_iter = 100, vel_thresh = 0.03):
        obs, reward, done, env_info = self._env.step(action)
        count = 1
        while count < max_iter:
            self._env.env._elapsed_steps -= 1
            obs, reward, done, env_info = self._env.step(action)
            state = self._env.get_env_state()
            if np.max(np.abs(state['qvel'])) < vel_thresh:
                break
            count += 1
        return obs, reward, done, env_info
    
    def get_paired_display(self, cam_ind = 0):
        #_env_render_cont = MjRenderContextOffscreen(self._env.env.sim, device_id=0)
        #self._env.env.sim.add_render_context(_env_render_cont)
        self.cameras = ['fixed', 'vil_camera', 'top']
        hori_imgs = []
        for camera in self.cameras:
            hand_display = self._env.env.sim.render(512, 512, mode = 'offscreen', camera_name = camera, device_id = 0)
            hand_display = self._env.env.sim.render(512, 512, mode = 'offscreen', camera_name = camera, device_id = 0)
            #hand_display = self.main_view.render(512, 512, mode = 'offscreen', camera_name = self.cameras[cam_ind], device_id = 0)
            #_claw_env_render_cont = MjRenderContextOffscreen(self._claw_env.env.sim, device_id=0)
            #self._claw_env.env.sim.add_render_context(_claw_env_render_cont)
            claw_display = self._claw_env.env.sim.render(512, 512, mode = 'offscreen', camera_name = camera, device_id = 0)
            claw_display = self._claw_env.env.sim.render(512, 512, mode = 'offscreen', camera_name = camera, device_id = 0)
            img = np.concatenate([hand_display, claw_display], axis=1).astype(np.uint8)
            hori_imgs.append(img)
        vert_img = np.concatenate(hori_imgs, axis=0)
        return vert_img 

    def get_obs_for_first_state_but_without_reset(self):
        pixels, sensor_info = self.get_obs()
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        stacked_pixels = self.get_stacked_pixels()
        return stacked_pixels, sensor_info

    def step(self, action):
        reward_sum = 0.0
        discount_prod = 1.0 # TODO pen can terminate early 
        n_goal_achieved = 0
        #self._num_repeats = 1
        for i_action in range(self._num_repeats): 
            if self.wait_until_still:
                action = self.trim_action(action)
                self.prev_action = action
                obs, reward, done, env_info = self.step_until_still(action)
            else:
                obs, reward, done, env_info = self._env.step(action)
            if 'TimeLimit.truncated' in env_info:
                truncated = env_info['TimeLimit.truncated']
            else:
                truncated = False
            claw_action = self.get_desired_claw_pose()
            for j in range(20):
                _, claw_reward, _2, claw_env_info = self._claw_env.step(claw_action)
            env_info['TimeLimit.truncated'] = truncated
            self.set_env_to_claw_env_state()
            reward_sum += reward 
            if env_info['goal_achieved'] == True:
                n_goal_achieved += 1
            if done:
                break
        env_info['n_goal_achieved'] = n_goal_achieved
        # now get stacked frames
        pixels, sensor_info = self.get_obs()
        self._frames.append(pixels)
        stacked_pixels = self.get_stacked_pixels()
        return [stacked_pixels, sensor_info], reward_sum, done, env_info

    def set_env_state(self, state):
        return self._env.set_env_state(state)
    
    def get_env_state(self):
        return self._env.get_env_state()

    def set_claw_env_state(self, state):
        return self._claw_env.set_env_state(state)
    
    def get_claw_env_state(self):
        return self._claw_env.get_env_state()
    
    def set_env_to_claw_env_state(self):
        if self.ik:
            self.do_ik()
        claw_state = self.get_claw_env_state()
        env_state = self.get_env_state()
        if self.env_id == 'door-v0':
            #env_state['qpos'][:1] = claw_state['qpos'][:1]
            #env_state['qvel'][:1] = claw_state['qvel'][:1]
            env_state['qpos'][-2:] = claw_state['qpos'][-2:]
            env_state['door_body_pos'] = claw_state['door_body_pos']
        self._env.set_env_state(env_state)

    def evaluate_policy(self, policy,
    					num_episodes=5,
    					horizon=None,
    					gamma=1,
    					visual=False,
    					percentile=[],
    					get_full_dist=False,
    					mean_action=False,
    					init_env_state=None,
    					terminate_at_done=True,
    					seed=123):
        # TODO this needs to be rewritten

        self.set_seed(seed)
        horizon = self.horizon if horizon is None else horizon
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns = np.zeros(num_episodes)
        self.encoder.eval()

        for ep in range(num_episodes):
            o = self.reset()
            if init_env_state is not None:
                self.set_env_state(init_env_state)
            t, done = 0, False
            while t < horizon and (done == False or terminate_at_done == False):
                self.render() if visual is True else None
                o = self.get_obs(self._env.get_obs())
                a = policy.get_action(o)[1]['evaluation'] if mean_action is True else policy.get_action(o)[0]
                o, r, done, _ = self.step(a)
                ep_returns[ep] += (gamma ** t) * r
                t += 1

        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        base_stats = [mean_eval, std, min_eval, max_eval]

        percentile_stats = []
        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))

        full_dist = ep_returns if get_full_dist is True else None

        return [base_stats, percentile_stats, full_dist]

    def get_pixels_with_width_height(self, w, h, claw_env = False):
        imgs = [] # number of image is number of camera

        for cam in self.cameras : # for each camera, render once
            if claw_env:
                img = self._claw_env.env.sim.render(width=w, height=h, mode='offscreen', camera_name=cam, device_id=0) # TODO device id will think later
                img = self._claw_env.env.sim.render(width=w, height=h, mode='offscreen', camera_name=cam, device_id=0) # TODO device id will think later
            else:
                img = self._env.env.sim.render(width=w, height=h, mode='offscreen', camera_name=cam, device_id=0) # TODO device id will think later
                img = self._env.env.sim.render(width=w, height=h, mode='offscreen', camera_name=cam, device_id=0) # TODO device id will think later
            # img = img[::-1, :, : ] # Image given has to be flipped
            if self.channels_first :
                img = img.transpose((2, 0, 1)) # then it's 3 x width x height
            # we should do channels first... (not sure why by default it's not, maybe they did some transpose when using the encoder?)
            #img = img.astype(np.uint8)
            # img = Image.fromarray(img) # TODO is this necessary?
            imgs.append(img)

        pixels = np.concatenate(imgs, axis=0)
        return pixels

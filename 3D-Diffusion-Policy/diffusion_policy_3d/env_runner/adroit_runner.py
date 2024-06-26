import wandb
import numpy as np
import torch
import tqdm
from diffusion_policy_3d.env import AdroitEnv
from diffusion_policy_3d.gym_util.mjpc_diffusion_wrapper import MujocoPointcloudWrapperAdroit
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint
import cv2

import matplotlib.pyplot as plt

def export_video(frames, output_path, fps=30):
    num_frames, _, y_res, x_res = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (x_res, y_res))

    for frame in frames:
        # Convert the frame from numpy array to uint8
        frame = np.uint8(frame.transpose(1, 2, 0)[:, :, ::-1])
        # Write the frame to the video file
        out.write(frame)

    # Release the VideoWriter object
    out.release()



class AdroitRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_point_crop=True,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MujocoPointcloudWrapperAdroit(env=AdroitEnv(env_name=task_name, use_point_cloud=True),
                                                  env_name='adroit_'+task_name, use_point_crop=use_point_crop)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn()

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_goal_achieved = []
        all_success_rates = []
        all_rew_sums = []
        all_max_door = []
        all_max_knob = []
        


        self.eval_episodes = 100
        videos = []
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
                
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            num_goal_achieved = 0
            actual_step_count = 0
            rew_list = []
            max_door_angle = 0
            max_latch_angle = 0
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)
                    

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)
                # step env
                obs, reward, done, info = env.step(action)
                print("runner step")
                # all_goal_achieved.append(info['goal_achieved']
                rew_list.append(reward)
                print("info", info)
                num_goal_achieved += np.sum(info['goal_achieved'])
                door_angle = info['door_pos']
                latch_angle = info['latch_angle']
                if door_angle > max_door_angle:
                    max_door_angle = door_angle
                if latch_angle > max_latch_angle:
                    max_latch_angle = latch_angle
                done = np.all(done)
                actual_step_count += 1

            rew_sum = np.sum(np.array(rew_list))
            all_rew_sums.append(rew_sum)
            all_max_door.append(max_door_angle)
            all_max_knob.append(max_latch_angle)
            all_success_rates.append(info['goal_achieved'])
            all_goal_achieved.append(num_goal_achieved)
            videos.append(env.env.get_video())


        # log
        log_data = dict()
        

        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        log_data['mean max door'] = np.mean(all_max_door)
        log_data['mean max knob'] = np.mean(all_max_knob)
        log_data['mean rew sum'] = np.mean(all_rew_sums)

        videos = np.concatenate(videos, axis=0)
        #if len(videos.shape) == 5:
        #    videos = videos[:, 0]  # select first frame
        export_video(videos, './out.mp4', 10)
        #cv2.imshow("test", np.zeros((100, 100, 3), dtype=np.uint8))
        #cv2.waitKey(30)


        #videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        #print("after wandb video")
        #log_data[f'sim_video_eval'] = videos_wandb

        # clear out video buffer
        _ = env.reset()
        # clear memory
        videos = None
        del env

        return log_data

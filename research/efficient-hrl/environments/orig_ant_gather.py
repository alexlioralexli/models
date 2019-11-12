import sys
sys.path.insert(0, '/home/alexli/workspace/rllab-hrl')
from rllab.envs.mujoco.gather.gather_env import GatherEnv
from environments.ant import AntEnv
import math
import numpy as np
from rllab.envs.base import Step
class AntGatherEnv(GatherEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 3        # for hiro

    def __init__(
            self,
            n_apples=8,
            n_bombs=8,
            activity_range=6.,
            robot_object_spacing=2.,
            catch_range=1.,
            n_bins=10,
            sensor_range=6.,
            sensor_span=math.pi,
            coef_inner_rew=0.,
            dying_cost=-10,
            *args, **kwargs
    ):
        self.t = 0
        super().__init__(n_apples=n_apples,
            n_bombs=n_bombs,
            activity_range=10.0,
            robot_object_spacing=robot_object_spacing,
            catch_range=catch_range,
            n_bins=n_bins,
            sensor_range=sensor_range,
            sensor_span=sensor_span,
            coef_inner_rew=0.,
            dying_cost=dying_cost,
            *args, **kwargs)

        self.hiro = True
        self.frame_skip = 5


    def get_current_obs(self):
        # return sensor data along with data about itself
        self_obs = self.wrapped_env._get_obs()
        apple_readings, bomb_readings = self.get_readings()
        return np.concatenate([self_obs, apple_readings, bomb_readings, [self.t * 0.001]])


if __name__ == "__main__":
    env = AntGatherEnv()
    # import IPython; IPython.embed()
    # while True:
    #     env.reset()
    #     for _ in range(1000):
    #         env.render()
    #         _, reward, _, _ = env.step(env.action_space.sample())  # take a random action
    import ipdb; ipdb.set_trace()
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import sys
sys.path.insert(0, '/home/alexli/workspace/rllab-hrl')
from sandbox.snn4hrl.envs.mujoco.gather.ant_low_gear_gather_env import AntLowGearGatherEnv as AntGather
from sandbox.snn4hrl.envs.mujoco.gather.ant_normal_gear_gather_env import AntNormalGearGatherEnv
from sandbox.snn4hrl.envs.mujoco.gather.snake_gather_env import SnakeGatherEnv
from sandbox.snn4hrl.envs.mujoco.half_cheetah_adaptation import SparseHalfCheetahAdaptationEnv
from sandbox.snn4hrl.envs.mujoco.hopper_adaptation_withx import SparseHopperEnvWithX
from environments.ant_maze_env import AntMazeEnv
from environments.point_maze_env import PointMazeEnv
from environments.orig_ant_gather import AntGatherEnv

import tensorflow as tf
import gin.tf
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
import math

@gin.configurable
def create_maze_env(env_name=None, top_down_view=False):
  n_bins = 0
  manual_collision = False
  if env_name.startswith('AntGather'):
    keyword_args = dict(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=True)
    gym_env = AntGather(**keyword_args)
  if env_name.startswith('SnakeGather'):
    keyword_args = dict(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=True)
    gym_env = SnakeGatherEnv(**keyword_args)
  elif env_name.startswith('AntNormalGearGather'):
    keyword_args = dict(activity_range=10.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=True)
    gym_env = AntNormalGearGatherEnv(**keyword_args)
  elif env_name.startswith('HiroAntGather'):
    keyword_args = dict(activity_range=10.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=False)
    gym_env = AntGatherEnv(**keyword_args)
    import ipdb; ipdb.set_trace()
  elif env_name.startswith('SparseHalfCheetah'):
    gym_env = SparseHalfCheetahAdaptationEnv()
  elif env_name.startswith('SparseHopper'):
    gym_env = SparseHopperEnvWithX()
  else:
    if env_name.startswith('Ego'):
      n_bins = 8
      env_name = env_name[3:]
    if env_name.startswith('Ant'):
      cls = AntMazeEnv
      env_name = env_name[3:]
      maze_size_scaling = 8
    elif env_name.startswith('Point'):
      cls = PointMazeEnv
      manual_collision = True
      env_name = env_name[5:]
      maze_size_scaling = 4
    else:
      assert False, 'unknown env %s' % env_name

    maze_id = None
    observe_blocks = False
    put_spin_near_agent = False
    if env_name == 'Maze':
      maze_id = 'Maze'
    elif env_name == 'Push':
      maze_id = 'Push'
    elif env_name == 'Fall':
      maze_id = 'Fall'
    elif env_name == 'Block':
      maze_id = 'Block'
      put_spin_near_agent = True
      observe_blocks = True
    elif env_name == 'BlockMaze':
      maze_id = 'BlockMaze'
      put_spin_near_agent = True
      observe_blocks = True
    else:
      raise ValueError('Unknown maze environment %s' % env_name)

    gym_mujoco_kwargs = {
        'maze_id': maze_id,
        'n_bins': n_bins,
        'observe_blocks': observe_blocks,
        'put_spin_near_agent': put_spin_near_agent,
        'top_down_view': top_down_view,
        'manual_collision': manual_collision,
        'maze_size_scaling': maze_size_scaling
    }
    gym_env = cls(**gym_mujoco_kwargs)
  gym_env.reset()
  wrapped_env = gym_wrapper.GymWrapper(gym_env)
  return wrapped_env


class TFPyEnvironment(tf_py_environment.TFPyEnvironment):

  def __init__(self, *args, **kwargs):
    super(TFPyEnvironment, self).__init__(*args, **kwargs)

  def start_collect(self):
    pass

  def current_obs(self):
    time_step = self.current_time_step()
    return time_step.observation[0]  # For some reason, there is an extra dim.

  def step(self, actions):
    actions = tf.expand_dims(actions, 0)
    next_step = super(TFPyEnvironment, self).step(actions)
    return next_step.is_last()[0], next_step.reward[0], next_step.discount[0]

  def reset(self):
    return super(TFPyEnvironment, self).reset()

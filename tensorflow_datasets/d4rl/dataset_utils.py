# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors.
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

"""Utils to generate builders for D4RL datasets."""
from typing import Any, Dict

import dataclasses
import h5py
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.io import gfile
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """
D4RL is an open-source benchmark for offline reinforcement learning. It provides
standardized environments and datasets for training and benchmarking algorithms.
"""

_CITATION = """\
@misc{fu2020d4rl,
    title={D4RL: Datasets for Deep Data-Driven Reinforcement Learning},
    author={Justin Fu and Aviral Kumar and Ofir Nachum and George Tucker and Sergey Levine},
    year={2020},
    eprint={2004.07219},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
"""


@dataclasses.dataclass
class MujocoDatasetConfig(tfds.core.BuilderConfig):
  dataset_dir: str = 'gym_mujoco'
  file_suffix: str = 'medium'


# pytype: disable=wrong-keyword-args
MUJOCO_BUILDER_CONFIGS = [
    MujocoDatasetConfig(
        name='v0-expert',
        dataset_dir='gym_mujoco',
        file_suffix='expert'),
    MujocoDatasetConfig(
        name='v0-medium',
        dataset_dir='gym_mujoco',
        file_suffix='medium'),
    MujocoDatasetConfig(
        name='v0-medium-expert',
        dataset_dir='gym_mujoco',
        file_suffix='medium_expert'),
    MujocoDatasetConfig(
        name='v0-mixed',
        dataset_dir='gym_mujoco',
        file_suffix='mixed'),
    MujocoDatasetConfig(
        name='v0-random',
        dataset_dir='gym_mujoco',
        file_suffix='random'),
    MujocoDatasetConfig(
        name='v1-expert',
        dataset_dir='gym_mujoco_v1',
        file_suffix='expert-v1'),
    MujocoDatasetConfig(
        name='v1-medium',
        dataset_dir='gym_mujoco_v1',
        file_suffix='medium-v1'),
    MujocoDatasetConfig(
        name='v1-medium-expert',
        dataset_dir='gym_mujoco_v1',
        file_suffix='medium_expert-v1'),
    MujocoDatasetConfig(
        name='v1-medium-replay',
        dataset_dir='gym_mujoco_v1',
        file_suffix='medium_replay-v1'),
    MujocoDatasetConfig(
        name='v1-full-replay',
        dataset_dir='gym_mujoco_v1',
        file_suffix='full_replay-v1'),
    MujocoDatasetConfig(
        name='v1-random',
        dataset_dir='gym_mujoco_v1',
        file_suffix='random-v1'),
    MujocoDatasetConfig(
        name='v2-expert',
        dataset_dir='gym_mujoco_v2',
        file_suffix='expert-v2'),
    MujocoDatasetConfig(
        name='v2-full-replay',
        dataset_dir='gym_mujoco_v2',
        file_suffix='full_replay-v2'),
    MujocoDatasetConfig(
        name='v2-medium',
        dataset_dir='gym_mujoco_v2',
        file_suffix='medium-v2'),
    MujocoDatasetConfig(
        name='v2-medium-expert',
        dataset_dir='gym_mujoco_v2',
        file_suffix='medium_expert-v2'),
    MujocoDatasetConfig(
        name='v2-medium-replay',
        dataset_dir='gym_mujoco_v2',
        file_suffix='medium_replay-v2'),
    MujocoDatasetConfig(
        name='v2-random',
        dataset_dir='gym_mujoco_v2',
        file_suffix='random-v2'),
]


def _mujoco_replay_datasets():
  """Set of Mujoco datasets with replay."""
  return {
      'v1-medium-replay', 'v1-full-replay', 'v2-medium-replay', 'v2-full-replay'
  }


def _mujoco_full_metadata_datasets():
  """Set of Mujoco datasets that contain all of the metadata fields."""
  return {'v1-expert', 'v2-expert', 'v1-medium', 'v2-medium'}


def _mujoco_float_type(config_name: str):
  """Given a config name, returns the float type used in the dataset."""
  if config_name in _mujoco_replay_datasets():
    return tf.float64
  else:
    return tf.float32


def get_mujoco_features_dict(
    builder_config: MujocoDatasetConfig, obs_len: int, action_len: int,
    qpos_len: int, qvel_len: int) -> Dict[str, tfds.features.FeatureConnector]:
  """Builds the features dict of a Mujoco dataset.

  Args:
    builder_config: config of the Mujoco dataset.
    obs_len: first dimension of the obsercations.
    action_len: first dimension of the actions.
    qpos_len: first dimension of the infos/qpos field (ignored if the dataset
      does not include step metadata).
    qvel_len: first dimension of the infos/qvel field (ignored if the dataset
      does not include step metadata).

  Returns:
    Dictionary with the features of this dataset.
  """

  float_type = _mujoco_float_type(builder_config.name)

  steps_dict = {
      'observation': tfds.features.Tensor(shape=(obs_len,), dtype=float_type),
      'action': tfds.features.Tensor(shape=(action_len,), dtype=float_type),
      'reward': float_type,
      'is_terminal': tf.bool,
      'is_first': tf.bool,
      'discount': float_type,
  }
  # All except for v0 datasets contain step metadata
  if builder_config.dataset_dir != 'gym_mujoco':
    steps_dict['infos'] = {
        'action_log_probs': float_type,
        'qpos': tfds.features.Tensor(shape=(qpos_len,), dtype=float_type),
        'qvel': tfds.features.Tensor(shape=(qvel_len,), dtype=float_type),
    }

  episode_metadata = {}
  # Replay datasets contain only two fields of the metadata.
  if builder_config.name in _mujoco_replay_datasets():
    episode_metadata = {
        'algorithm': tf.string,
        'iteration': tf.int32,
    }
  if builder_config.name in _mujoco_full_metadata_datasets():
    episode_metadata = {
        'algorithm': tf.string,
        'iteration': tf.int32,
        'policy': {
            'fc0': {
                'bias':
                    tfds.features.Tensor(shape=(256,), dtype=float_type),
                'weight':
                    tfds.features.Tensor(
                        shape=(256, obs_len), dtype=float_type),
            },
            'fc1': {
                'bias':
                    tfds.features.Tensor(shape=(256,), dtype=float_type),
                'weight':
                    tfds.features.Tensor(shape=(256, 256), dtype=float_type),
            },
            'last_fc': {
                'bias':
                    tfds.features.Tensor(shape=(action_len,), dtype=float_type),
                'weight':
                    tfds.features.Tensor(
                        shape=(action_len, 256), dtype=float_type),
            },
            'last_fc_log_std': {
                'bias':
                    tfds.features.Tensor(shape=(action_len,), dtype=float_type),
                'weight':
                    tfds.features.Tensor(
                        shape=(action_len, 256), dtype=float_type),
            },
            'nonlinearity': tf.string,
            'output_distribution': tf.string,
        },
    }

  features_dict = {
      'steps': tfds.features.Dataset(steps_dict),
  }
  if episode_metadata:
    features_dict.update(episode_metadata)

  return features_dict


def description():
  return _DESCRIPTION


def citation():
  return _CITATION


def url():
  return 'https://sites.google.com/view/d4rl/home'


def generate_examples(file_path: str):
  """Provides a common generate_examples method for D4RL datasets."""
  with gfile.GFile(file_path, 'rb') as f:
    dataset_file = h5py.File(f, 'r')
    dataset_dict = {}
    for k in _get_dataset_keys(dataset_file):
      try:
        # first try loading as an array
        dataset_dict[k] = dataset_file[k][:]
      except ValueError as e:  # try loading as a scalar
        dataset_dict[k] = dataset_file[k][()]
    dataset_file.close()
  if 'timeouts' not in dataset_dict:
    raise ValueError('Only datasets with explicit timeouts are supported.')

  done = [
      terminal or timeout
      for (terminal,
           timeout) in zip(dataset_dict['terminals'], dataset_dict['timeouts'])
  ]
  # is_first corresponds to the done flag delayed by one step.
  dataset_dict['is_first'] = [True] + done[:-1]

  # Get step metadata
  infos_dict = {}
  if 'infos/qpos' in dataset_dict.keys():
    infos_dict = {
        'qpos': dataset_dict['infos/qpos'],
        'qvel': dataset_dict['infos/qvel'],
        'action_log_probs': dataset_dict['infos/action_log_probs']
    }
  # Flatten reward
  dataset_dict['rewards'] = np.squeeze(dataset_dict['rewards'])

  episode_metadata = _get_episode_metadata(dataset_dict)
  dataset_dict = {
      'observation': dataset_dict['observations'],
      'action': dataset_dict['actions'],
      'reward': dataset_dict['rewards'],
      'discount': np.ones_like(dataset_dict['rewards']),
      'is_terminal': dataset_dict['terminals'],
      'is_first': dataset_dict['is_first'],
  }
  if infos_dict:
    dataset_dict['infos'] = infos_dict
  num_steps = len(dataset_dict['is_first'])
  prev = 0
  counter = 0
  for pos in range(num_steps):
    if dataset_dict['is_first'][pos] and pos > prev:
      yield counter, _get_episode(dataset_dict, episode_metadata, prev, pos)
      prev = pos
      counter += 1
  if prev < num_steps:
    yield counter, _get_episode(dataset_dict, episode_metadata, prev, num_steps)


def _get_episode_metadata(dataset: Dict[str, Any]) -> Dict[str, Any]:
  """Generate a metadata dictionary using flattened metadata keys.

  Args:
    dataset: dictionary containing the dataset keys and values. Keys are
      flatened.

  Returns:
    Nested dictionary with the episode metadata.

  If the dataset contains:
  {
    'metadata/v1/v2': 1,
    'metadata/v3': 2,
  }
  Returns
  {
    'v1':{
      'v2': 1,
    }
    'v3': 2,
  }
  It assumes that the flattened metadata keys are well-formed.
  """
  episode_metadata = {}
  for k in dataset.keys():
    if 'metadata/' not in k:
      continue
    keys = k.split('/')[1:]
    nested_dict = episode_metadata
    leaf_value = dataset[k]
    for index, nested_key in enumerate(keys):
      if index == (len(keys) - 1):
        nested_dict[nested_key] = leaf_value
      else:
        if nested_key not in nested_dict:
          nested_dict[nested_key] = {}
        nested_dict = nested_dict[nested_key]

  return episode_metadata


def _get_episode(steps: Dict[str, Any], episode_metadata: Dict[str, Any],
                 begin: int, end: int) -> Dict[str, Any]:
  """Builds a full episode dict.

  Args:
      steps: a dict with all steps in a dataset
      episode_metadata: dict with the episode metadata
      begin: defines a starting position of an episode
      end: defines an ending position of an episode

  Returns:
     A dict with data specific to one episode, already broken into steps.
  """
  # It's an initial step if the episode is empty.
  episode = {}
  for k in ['is_first', 'observation', 'action', 'reward', 'discount']:
    episode[k] = steps[k][begin:end]
  episode['is_terminal'] = [False] * (end - begin)
  if 'infos' in steps.keys():
    episode['infos'] = {}
    for k in steps['infos'].keys():
      episode['infos'][k] = steps['infos'][k][begin:end]

  if steps['is_terminal'][end - 1]:
    # If the step is terminal, then we propagate the information to a next
    # state. This matches the definition in RLDS. See types.py.
    episode['is_first'] = np.concatenate((episode['is_first'], [False]))
    # Observation, action and reward are dummy.
    episode['observation'] = np.concatenate(
        (episode['observation'], [np.zeros_like(steps['observation'][0])]))
    episode['action'] = np.concatenate(
        (episode['action'], [np.zeros_like(steps['action'][0])]))
    episode['reward'] = np.concatenate(
        (episode['reward'], [np.zeros_like(steps['reward'][0])]))
    episode['discount'][-1] = 0.0
    episode['discount'] = np.array(
        np.concatenate((episode['discount'], [0.0])), dtype=np.float32)
    episode['is_terminal'] = np.concatenate((episode['is_terminal'], [True]))
    if 'infos' in steps.keys():
      for k in steps['infos'].keys():
        episode['infos'][k] = np.concatenate(
            (episode['infos'][k], [np.zeros_like(steps['infos'][k][0])]))
  full_episode = {'steps': episode}
  if episode_metadata:
    full_episode.update(episode_metadata)
  return full_episode


def _get_dataset_keys(h5file):
  """Gets the keys present in the D4RL dataset."""
  keys = []

  def visitor(name, item):
    if isinstance(item, h5py.Dataset):
      keys.append(name)

  h5file.visititems(visitor)
  return keys

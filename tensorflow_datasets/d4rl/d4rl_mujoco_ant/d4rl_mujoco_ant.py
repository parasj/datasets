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

"""D4Rl Ant dataset from Mujoco."""

from tensorflow_datasets.d4rl import dataset_utils
import tensorflow_datasets.public_api as tfds


class D4rlMujocoAnt(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for ant dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = dataset_utils.MUJOCO_BUILDER_CONFIGS

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    features_dict = dataset_utils.get_mujoco_features_dict(
        builder_config=self.builder_config,
        obs_len=111,
        action_len=8,
        qpos_len=15,
        qvel_len=14)
    return tfds.core.DatasetInfo(
        builder=self,
        description=dataset_utils.description(),
        features=tfds.features.FeaturesDict(features_dict),
        supervised_keys=None,  # disabled
        homepage=dataset_utils.url(),
        citation=dataset_utils.citation(),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    ds_dir = self.builder_config.dataset_dir
    ds_name = 'ant_'+self.builder_config.file_suffix + '.hdf5'
    path = dl_manager.download_and_extract({
        'file_path':
            'http://rail.eecs.berkeley.edu/datasets/offline_rl/' + ds_dir +
            '/' + ds_name
    })
    return {
        'train': self._generate_examples(path),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    file_path = path['file_path']
    return dataset_utils.generate_examples(file_path)

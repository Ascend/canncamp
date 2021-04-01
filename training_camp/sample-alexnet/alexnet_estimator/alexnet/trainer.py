# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import tensorflow as tf
import math
import time
import os
from . import train_helper
from .logger import rank0log

class Trainer(object):
    def __init__(self, session, config, data, model, logger):
        self.sess = session
        self.config = config
        self.data = data
        self.model = model
        self.logger = logger
        self.print_logger = self.logger.logger
        self.all_preds = []
        self.all_targets = []

        self.classifier, self.training_hook = self.get_npu_classifier()

    def get_npu_classifier(self):
        if self.config.chip == 'npu':
            from npu_bridge.estimator.npu.npu_config import NPURunConfig
            from npu_bridge.estimator.npu.npu_estimator import NPUEstimator

            ## refer to links
            # https://support.huaweicloud.com/tensorflowdevg-cann330alphaXtraining/atlasmprtg_13_0032.html
            # https://support.huaweicloud.com/tensorflowdevg-cann330alphaXtraining/atlastfadapi_07_0004.html
            if self.config.npu_profiling:
                from npu_bridge.estimator.npu.npu_config import ProfilingConfig
                work_dir = os.getcwd()
                profiling_dir = os.path.join(work_dir, "npu_profiling")
                if not os.path.exists(profiling_dir):
                    os.makedirs(profiling_dir)
                profiling_options = '{"output":"%s","task_trace": "on","aicpu": "on"}'  % (profiling_dir)
                profiling_config = ProfilingConfig(enable_profiling=True, profiling_options = profiling_options)

                run_config = NPURunConfig(
                    profiling_config=profiling_config,  ## 配置profiling参数
                    save_checkpoints_steps=112590,
                    session_config=self.sess.estimator_config,
                    model_dir=self.config.log_dir,
                    keep_checkpoint_max=self.config.max_checkpoint_to_save)
            else:
                run_config = NPURunConfig(
                    save_checkpoints_steps=112590,
                    session_config=self.sess.estimator_config,
                    model_dir=self.config.log_dir,
                    keep_checkpoint_max=self.config.max_checkpoint_to_save)

            classifier =NPUEstimator(
                model_fn= self.model.get_estimator_model_func,
                config= run_config
            )
        elif self.config.chip in ['gpu', 'cpu']:
            run_config = tf.estimator.RunConfig(
                save_checkpoints_steps=112590,
                session_config=self.sess.estimator_config,
                model_dir=self.config.log_dir,
                keep_checkpoint_max=self.config.max_checkpoint_to_save)
            classifier =tf.estimator.Estimator(
                model_fn= self.model.get_estimator_model_func,
                config= run_config
            )

        training_hooks = []
        training_hooks.append(self.logger)

        return classifier, training_hooks

    def train(self):
        print ('training steps: %d' % self.config.nstep)
        self.classifier.train( input_fn=lambda:self.data.get_train_input_fn(),
                               max_steps = self.config.nstep,
                               hooks = self.training_hook
                              )

    def evaluate(self):
        rank0log(self.print_logger, "Evaluating")
        rank0log(self.print_logger, "Validation dataset size: {}".format(self.config.num_evaluating_samples ))
        time.sleep(5)  # a little extra margin...
        try:
            ckpts = train_helper.sort_and_load_ckpts(self.config.checkpoint_dir)
            print("=========ckpt==========")
            print(ckpts)
            print("=========ckpt==========")
            for i, c in enumerate(ckpts):
                eval_result = self.classifier.evaluate(
                    input_fn=lambda: self.data.get_eval_input_fn(),
                    checkpoint_path=c['path'])
                #c['epoch'] = math.ceil(c['step']/ (self.config.num_training_samples/ (self.config.global_batch_size)))
                c['epoch'] = math.floor(c['step']/ (self.config.num_training_samples/ (self.config.global_batch_size)))
                c['top1'] = eval_result['val-top1acc']
                c['top5'] = eval_result['val-top5acc']
                c['loss'] = eval_result['loss']

            rank0log(self.print_logger, ' step  epoch  top1    top5     loss   checkpoint_time(UTC)')
            for i, c in enumerate(ckpts):
                if 'top1' not in c:
                    continue
                rank0log(self.print_logger,'{:5d}  {:5.1f}  {:5.3f}  {:6.2f}  {:6.2f}  {time}'
                         .format(c['step'],
                                 c['epoch'],
                                 c['top1']* 100,
                                 c['top5']* 100,
                                 c['loss'],
                                 time=time.strftime('%Y-%m-%d %H:%M:%S',
                                    time.localtime(c['mtime']))))
            rank0log(self.print_logger, "Finished evaluation")
        except KeyboardInterrupt:
            self.print_logger.error("Keyboard interrupt")

    def train_and_evaluate(self):
        epochs_between_evals = self.config.epochs_between_evals

        for i in range(self.config.max_epochs // epochs_between_evals):

            rank0log(self.print_logger, "Starting a training cycle")

            self.classifier.train(input_fn=lambda:self.data.get_train_input_fn(),
                            steps = self.config.nsteps_per_epoch*epochs_between_evals,
                            hooks = self.training_hook )

            rank0log(self.print_logger, "Starting to evaluate")
            rank0log(self.print_logger, "Validation dataset size: {}".format(self.config.num_evaluating_samples ))
            time.sleep(5)  # a little extra margin...

            ckpts = train_helper.sort_and_load_ckpts(self.config.log_dir)
            print("=========ckpt==========")
            print(ckpts)
            print("=========ckpt==========")
            c = ckpts[-1]
            eval_result = self.classifier.evaluate(
                input_fn=lambda: self.data.get_eval_input_fn(),
                checkpoint_path=c['path'])

            c['epoch'] = math.ceil(c['step'] / (self.config.num_training_samples / self.config.batch_size))
            c['top1'] = eval_result['val-top1acc']
            c['top5']= eval_result['val-top5acc']
            c['loss'] = eval_result['loss']

            rank0log(self.print_logger, ' step  epoch  top1    top5     loss   checkpoint_time(UTC)')

            rank0log(self.print_logger,'{:5d}  {:5.1f}  {:5.3f}  {:6.2f}  {:6.2f}  {time}'
                    .format(c['step'],
                            c['epoch'],
                            c['top1'] * 100,
                            c['top5'] * 100,
                            c['loss'],
                            time=time.strftime('%Y-%m-%d %H:%M:%S',
                                time.localtime(c['mtime']))))

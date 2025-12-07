# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Two-stage PPO training:
- Stage 1: Only EM reward (before switch_step)
- Stage 2: Efficiency reward (after switch_step)

Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import os
os.environ['NCCL_TIMEOUT'] = '1800'
from verl import DataProto
import torch
from verl.utils.reward_score import qa_em_efficiency
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np


def _select_rm_score_fn_stage1(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


def _select_rm_score_fn_stage2(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em_efficiency.compute_score_em
    else:
        raise NotImplementedError
        

def _select_em_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em_efficiency.compute_em_score
    else:
        raise NotImplementedError


class TwoStageRewardManager():
    """The two-stage reward manager.
    Stage 1: Only EM reward (before switch_step)
    Stage 2: Efficiency reward (after switch_step)
    """

    def __init__(self, tokenizer, num_examine, format_score=0., efficiency_score=0., 
                 switch_step=40, current_step=0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.efficiency_score = efficiency_score
        self.switch_step = switch_step  # Switch to stage 2 step number
        self.current_step = current_step  # Current training step
        self.last_stage = 1  # Track stage changes
        
        print(f"[TwoStageReward] Initialized with switch_step={switch_step}")
        print(f"[TwoStageReward] Stage 1 (steps 1-{switch_step}): EM reward only")
        print(f"[TwoStageReward] Stage 2 (steps {switch_step+1}+): Efficiency reward")


    def get_current_stage(self):
        """Get current training stage"""
        return 1 if self.current_step <= self.switch_step else 2

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # Get current step from meta_info
        if 'current_step' in data.meta_info:
            self.current_step = data.meta_info['current_step']

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        current_stage = self.get_current_stage()
        
        # Detect stage switching
        if current_stage != self.last_stage:
            print(f"[TwoStage] Stage switched from {self.last_stage} to {current_stage} at step {self.current_step}")
            self.last_stage = current_stage
        
        # Periodically print current stage info
        if self.current_step % 10 == 0:
            print(f"[TwoStage] Step {self.current_step}, Current Stage: {current_stage}")
        
        if current_stage == 2:
            search_times = data.meta_info['search_times']
            percentile_95 = np.percentile(search_times, 95)
            scaled_times = []
            for t in search_times:
                scaled_time = t
                # if percentile_95 > 0.5:
                scaled_time = scaled_time * 0.5 / percentile_95
                scaled_time = min(scaled_time, 0.5)
                scaled_times.append(scaled_time)
            
            avg_scale_time = sum(scaled_times) / len(scaled_times)
        else:
            scaled_times = [0] * len(data)
            avg_scale_time = 0

        already_print_data_sources = {}
        em_scores = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score and em_score functions
            data_source = data_item.non_tensor_batch['data_source']
            compute_em_fn = _select_em_score_fn(data_source)

            if current_stage == 1:
                # Stage 1: EM only
                compute_score_fn = _select_rm_score_fn_stage1(data_source)
                score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
                efficiency_score_value = 0
            else:
                # Stage 2: introduce efficiency reward
                compute_score_fn = _select_rm_score_fn_stage2(data_source)
                efficiency_score_value = avg_scale_time - scaled_times[i]
                score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, 
                                       format_score=self.format_score, efficiency_score=efficiency_score_value)
            
            # compute em score for logging
            em_score = compute_em_fn(solution_str=sequences_str, ground_truth=ground_truth)
            em_scores.append(em_score)

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                stage_info = f"[Stage {current_stage}]"
                if current_stage == 1:
                    print(f"{stage_info} Step {self.current_step}, EM Score: {score:.4f}")
                else:
                    print(f"{stage_info} Step {self.current_step}, Total Score: {score:.4f}, EM: {em_score:.4f}, Efficiency: {efficiency_score_value:.4f}")
                print(sequences_str)
        
        data.meta_info['em_scores'] = em_scores
        data.meta_info['current_stage'] = current_stage
        data.meta_info['current_step'] = self.current_step
        
        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN', 'NCCL_TIMEOUT': '1800'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    import torch
    import numpy as np
    import random
    
    def set_seed(seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"Global random seed set to: {seed}")
    
    seed = config.get('seed', '')
    if seed != '':
        set_seed(seed)
        
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    switch_step = getattr(config, 'switch_step', 1000)
    print(f"[TwoStage] Switch step set to: {switch_step}")

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = TwoStageRewardManager(tokenizer=tokenizer, num_examine=0, switch_step=switch_step)

    # Note that we always use function-based RM for validation
    val_reward_fn = TwoStageRewardManager(tokenizer=tokenizer, num_examine=1, switch_step=switch_step)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    
    trainer = RayPPOTrainer(config=config,
                           tokenizer=tokenizer,
                           role_worker_mapping=role_worker_mapping,
                           resource_pool_manager=resource_pool_manager,
                           ray_worker_group_cls=ray_worker_group_cls,
                           reward_fn=reward_fn,
                           val_reward_fn=val_reward_fn,
                           )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main() 
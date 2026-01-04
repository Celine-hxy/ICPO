# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import random
import uuid
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (compute_data_metrics,
                                           compute_throughout_metrics,
                                           compute_timing_metrics,
                                           process_validation_metrics)
from verl.trainer.ppo.ray_trainer import (RayPPOTrainer, ResourcePoolManager,
                                          Role, apply_kl_penalty,
                                          compute_advantage,
                                          compute_response_mask)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import (find_latest_ckpt_path,
                                                      should_save_ckpt_esi)
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import (get_response_mask, masked_mean,
                                         postprocess_data)
from verl.utils.tracking import ValidationGenerationsLogger

from ..utils.core_algos import EntController
from ..utils.demo_cache import DemoCache
from ..utils.metric_utils import compute_rollout_pass_rate
from ..utils.prompt_builder import ICLPromptBuilder

WorkerType = type[Worker]


class RayICPOTrainer(RayPPOTrainer):
    """Distributed  trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed  training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
        cold_start_demo_path: Optional[str] = None,
    ):
        """
        Initialize distributed  trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.cold_start_demo_path = self.config.actor_rollout_ref.demo_cache.cold_start_demo_path
        print(f"cold_start_demo_path: {self.cold_start_demo_path}")
        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        # define adaptive entropy controller
        adapt_ent_config = self.config.actor_rollout_ref.adaptive_entropy
        use_adapt_ent = adapt_ent_config.enabled
        self.ent_ctrl = EntController(init_ent_coef=adapt_ent_config.entropy_coeff,
                                                 max_ent_coef=adapt_ent_config.max_ent_coef,
                                                 min_ent_coef=adapt_ent_config.min_ent_coef,
                                                 delta_ent_coef=adapt_ent_config.delta_ent_coef,
                                                 target_ent=adapt_ent_config.target_ent,
                                                 use_adapt_ent=use_adapt_ent)
        self.adapt_ent_config = adapt_ent_config

        if config.critic.enable is not None:
            self.use_critic = bool(config.critic.enable)
        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            warnings.warn(
                "Disabled critic as algorithm.adv_estimator != gae. "
                "If it is not intended, please set critic.enable=True",
                stacklevel=2,
            )
            self.use_critic = False

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        
        # Initialize demo cache for storing high-quality samples
        cache_config = getattr(config.actor_rollout_ref, 'demo_cache', {})
        self.demo_cache = DemoCache(
            max_size=cache_config.get('max_size', 10000),
            min_pass_rate=cache_config.get('min_pass_rate', 0.95),
            cache_dir=cache_config.get('cache_dir', "./demo_cache"),
            similarity_threshold=cache_config.get('similarity_threshold', 0.8),
            max_cache_age=cache_config.get('max_cache_age', None)
        )

        # Cold start logic: if a preload path is specified, load the existing cache
        if self.cold_start_demo_path:
            if os.path.exists(self.cold_start_demo_path):
                self.demo_cache.load_cache(self.cold_start_demo_path)
                print(f"Loaded cold-start demo cache from {self.cold_start_demo_path} (size={len(self.demo_cache)})")
            else:
                warnings.warn(f"Cold-start demo cache file not found: {self.cold_start_demo_path}")
        # input("Pause...")
        
        # Mixed rollout configuration        
        self.n_rollouts = self.config.actor_rollout_ref.rollout.n
        self.n_off_policy = self.config.actor_rollout_ref.off_policy_rollout.rollout_num
        self.n_on_policy = self.n_rollouts - self.n_off_policy
        
        self.n_demo = cache_config.get('n_demo', 1)
        self.demo_match_strategy = cache_config.get('demo_match_strategy', 'random')
        
        # Validate rollout configuration
        if self.n_off_policy > self.n_rollouts:
            raise ValueError(f"n_off_policy ({self.n_off_policy}) cannot exceed n_rollouts ({self.n_rollouts})")


    def _print_batch_info(self, batch):
        # debug打印batch的各种值和形状
        print("=== [DEBUG] batch内容和形状 ===")
        if hasattr(batch, "batch"):
            for k, v in batch.batch.items():
                if hasattr(v, "shape"):
                    print(f"batch.batch['{k}'] shape: {v.shape}")
                else:
                    print(f"batch.batch['{k}']: {type(v)} (无shape)")
        if hasattr(batch, "non_tensor_batch"):
            for k, v in batch.non_tensor_batch.items():
                if hasattr(v, "shape"):
                    print(f"batch.non_tensor_batch['{k}'] shape: {v.shape}")
                elif isinstance(v, list):
                    print(f"batch.non_tensor_batch['{k}'] 列表长度: {len(v)}")
                else:
                    print(f"batch.non_tensor_batch['{k}']: {type(v)} (无shape)")
        if hasattr(batch, "meta_info"):
            print(f"batch.meta_info: {batch.meta_info}")
        print("=== [DEBUG] batch内容和形状 END ===")

    def _fetch_on_policy_batch(self, batch, on_indices):
        """
        Randomly select one on-policy sample, and return its prompt-related tensors (all truncated to max_prompt_length).
        Note: Each sample is 1D [L], do not use [:, :..].

        (TaskRunner pid=3042285) batch.batch['prompts'] shape: torch.Size([32, 4096])

        (TaskRunner pid=3042285) batch.batch['input_ids'] shape: torch.Size([32, 12288])
        (TaskRunner pid=3042285) batch.batch['position_ids'] shape: torch.Size([32, 12288])
        (TaskRunner pid=3042285) batch.batch['attention_mask'] shape: torch.Size([32, 12288])

        (TaskRunner pid=3042285) batch.batch['responses'] shape: torch.Size([32, 8192])
        (TaskRunner pid=3042285) batch.batch['response_mask'] shape: torch.Size([32, 8192])
        (TaskRunner pid=3042285) batch.batch['token_level_scores'] shape: torch.Size([32, 8192])
        (TaskRunner pid=3042285) batch.batch['token_level_rewards'] shape: torch.Size([32, 8192])
        # on_indices: [16, 20, 24, 28]
        """
        random_on_idx = random.choice(on_indices)

        max_pl = self.config.data.max_prompt_length
        on_prompt         = batch.batch["prompts"][random_on_idx][:max_pl].clone()
        on_input_ids      = batch.batch["input_ids"][random_on_idx][:max_pl].clone()
        on_position_ids   = batch.batch["position_ids"][random_on_idx][:max_pl].clone()
        on_attention_mask = batch.batch["attention_mask"][random_on_idx][:max_pl].clone()

        return on_prompt, on_input_ids, on_attention_mask, on_position_ids
    
    def _refill_on_batch_to_off_batch(self, batch, off_idx, on_prompt, on_input_ids, on_attention_mask, on_position_ids):
        """
        reprocess the batch for off-policy sample, replace off-policy samples' input_ids with on-policy samples' input_ids,
        and recompute attention_mask and position_ids
        on_prompt = torch.Size([4096])
        on_input_ids = torch.Size([4096])
        on_attention_mask = torch.Size([4096])
        on_position_ids = torch.Size([4096])
        """
        # === get seq for input_ids ===
        responses = batch.batch["responses"][off_idx]
        assert responses.size(-1) == self.config.data.max_response_length, \
            f"[Error] responses length is {responses.size(-1)}, which does not equal max_response_length={self.config.data.max_response_length}"
        assert torch.all(on_input_ids == on_prompt), f"on_input_ids: {on_input_ids[:5]}...{on_input_ids[-5:]} != on_prompt: {on_prompt[:5]}...{on_prompt[-5:]}"
        seq = torch.cat([on_prompt, responses], dim=-1)

        # === recompute attention_mask ===
        # set the attention_mask to 1 for the padding tokens
        response_attention_mask = batch.batch["response_mask"][off_idx]
        attention_mask = torch.cat([on_attention_mask, response_attention_mask], dim=-1)
        
        # === recompute position_ids ===
        # Use the last position of the prompt as the base, and increment the response segment by 1,2,3,...
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        last_pos = on_position_ids[-1]          # get the last position of the prompt (here is 3)
        resp_len = responses.size(-1)           # get the length of the response (here is 8192)
        delta_position_id = torch.arange(1, resp_len + 1, device=on_position_ids.device, dtype=on_position_ids.dtype)
        response_position_ids = last_pos + delta_position_id    # [4, 5, 6,...., 8192 + last_pos]
        position_ids = torch.cat([on_position_ids, response_position_ids], dim=-1)

        # === Fill back into batch ===
        batch.batch["prompts"][off_idx]         = on_prompt
        batch.batch["input_ids"][off_idx]       = seq
        batch.batch["attention_mask"][off_idx]  = attention_mask.to(torch.int64)   # Keep dtype consistent with upstream
        batch.batch["position_ids"][off_idx]    = position_ids.to(torch.int64)
        batch.non_tensor_batch["hinted"][off_idx] = 1

        return batch
    

    def _categorize_passrate(self, batch):
        """
        Traverse id2score and id2uid, count the number of zero/some/all pass and uids, and update the counts and uid lists passed in.
        """
        id2score = {"on": defaultdict(list), "off": defaultdict(list)}
        id2uid = {"on": defaultdict(list), "off": defaultdict(list)}
        bsz = len(batch)
        index = batch.non_tensor_batch["uid"]
        hinted = batch.non_tensor_batch["hinted"]
        scores = batch.batch["token_level_rewards"]  # (batch_size, token_length)
        for i in range(bsz):
            score = scores[i].sum(-1)
            key = "on" if hinted[i] == 0 else "off"
            id2score[key][index[i]].append(score)
            id2uid[key][index[i]].append(i)

        
        no_pass_uids, some_pass_uids, all_pass_uids = {"on": [], "off": []}, {"on": [], "off": []}, {"on": [], "off": []}
        zero_pass_count, all_pass_count, some_pass_count = {"on": 0, "off": 0}, {"on": 0, "off": 0}, {"on": 0, "off": 0}
        for key in ["on", "off"]:
            for idx in id2score[key]:
                random_uid = random.choice(id2uid[key][idx])  # randomly select one sample with the same uid
                if all(score == 0 for score in id2score[key][idx]):
                    zero_pass_count[key] += 1
                    no_pass_uids[key].append(random_uid)
                elif all(score > 0 for score in id2score[key][idx]):
                    all_pass_count[key] += 1
                    all_pass_uids[key].append(random_uid)
                else:
                    some_pass_count[key] += 1
                    some_pass_uids[key].append(random_uid)
        return {
            "no_pass_uids": no_pass_uids,
            "some_pass_uids": some_pass_uids,
            "all_pass_uids": all_pass_uids,
            "zero_pass_count": zero_pass_count,
            "all_pass_count": all_pass_count,
            "some_pass_count": some_pass_count
        }

    
    def _reorganize_batch(self, batch: DataProto):
        # group by uid
        # Retrieve the unique identifier (UUID) for each sample in the batch.
        # The 'uid' field is a numpy array of strings, where each element corresponds to a sample in the batch.
        # This enables us to group and track samples that originated from the same source,
        # which is particularly useful when samples are repeated or shuffled during rollout and advantage calculation.
        # The length of 'index' should be equal to the batch size (bsz).
        # Separately store uids and scores for on-policy and off-policy samples, using "hinted" as the distinguishing key
        
        id2uid = {"on": defaultdict(list), "off": defaultdict(list)}
        bsz = len(batch)
        index = batch.non_tensor_batch["uid"]
        hinted = batch.non_tensor_batch["hinted"]
        for i in range(bsz):
            key = "on" if hinted[i] == 0 else "off"
            id2uid[key][index[i]].append(i)

        # replace all off-policy samples' prompt with the same uid on-policy samples' prompt (input query)
        for off_uid in id2uid["off"]:
            off_indices = id2uid["off"][off_uid]
            # find on-policy samples with the same uid
            on_indices = id2uid["on"].get(off_uid, [])
            assert on_indices, "There must be at least one on-policy sample"
            # replace all off-policy samples' prompt
            for off_idx in off_indices:
                on_prompt, on_input_ids, on_attention_mask, on_position_ids = self._fetch_on_policy_batch(batch, on_indices)
                batch = self._refill_on_batch_to_off_batch(batch, off_idx, on_prompt, on_input_ids, on_attention_mask, on_position_ids)
                # self._print_batch_info(batch)

        return batch
    
    def _process_model_input(self, raw_prompts):
        model_inputs = self.tokenizer(raw_prompts, return_tensors="pt",
                                        add_special_tokens=False,
                                        padding=True,
                                        padding_side="left")
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="middle",
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        model_input_batch_dict = {}
        model_input_batch_dict["input_ids"] = input_ids                 # (batch_size, seq_len)
        model_input_batch_dict["attention_mask"] = attention_mask       # (batch_size, seq_len)
        model_input_batch_dict["position_ids"] = position_ids           # (batch_size, seq_len)
        return model_input_batch_dict

    def process_off_policy_batch(self, to_be_hinted_batch: DataProto):
        # add hint based on the answer
        raw_prompts = []
        
        # check if "raw_prompt" field exists
        if "raw_prompt" not in to_be_hinted_batch.non_tensor_batch:
            print("Error: 'raw_prompt' not found in non_tensor_batch")
            print(f"Available keys: {list(to_be_hinted_batch.non_tensor_batch.keys())}")
            raise KeyError("'raw_prompt' not found in non_tensor_batch")
        
        # for each sample to be hinted, add demo based on demo cache
        for i in range(len(to_be_hinted_batch)):
            system_prompt = ""
            raw_prompt_item = to_be_hinted_batch.non_tensor_batch["raw_prompt"][i]
            
            if len(raw_prompt_item) == 1:
                icl_prompt_builder = ICLPromptBuilder(w_inst=True, w_prefix=True)
                raw_question = raw_prompt_item[0]['content']
                raw_question = raw_question.split("Let's think step by step")[0].strip()
            
            elif raw_prompt_item[0]['role'] == "system":
                icl_prompt_builder = ICLPromptBuilder(w_inst=False, w_prefix=True)
                system_prompt = raw_prompt_item[0]['content']
                raw_question = raw_prompt_item[1]['content']
            
            else:
                raise ValueError(f"Unknown raw_prompt format: {raw_prompt_item}")

            # retrieve demo from demo cache
            demonstrations = self.demo_cache.get_n_demonstrations(
                target_prompt=raw_question,
                n_demo=self.n_demo,
                strategy=self.demo_match_strategy,
                num_candidates=max(7500, self.n_demo * 2),
                random_seed=np.random.randint(0, 999999)
            )

            final_prompt = icl_prompt_builder.construct_n_shot_prompt(demonstrations, system_prompt, raw_question, self.tokenizer)
            raw_prompts.append(final_prompt)
            print(f"[Final Prompt]\n{final_prompt[:]}")

        print(f"process_off_policy_batch: generated {len(raw_prompts)} prompts")
        if len(raw_prompts) == 0:
            raise ValueError("No prompts generated in process_off_policy_batch")

        hinted_batch_dict = self._process_model_input(raw_prompts)

        meta_info = to_be_hinted_batch.meta_info.copy()
        meta_info.pop('global_token_num', None)
        hinted_batch = DataProto.from_single_dict(data=hinted_batch_dict, meta_info=meta_info)
        
        # keep all non_tensor_batch fields in the original batch
        for key, value in to_be_hinted_batch.non_tensor_batch.items():
            if key not in hinted_batch.non_tensor_batch:
                hinted_batch.non_tensor_batch[key] = value
        
        hinted_batch_padded, pad_size = pad_dataproto_to_divisor(hinted_batch, self.actor_rollout_wg.world_size)
        
        hinted_batch_padded.meta_info["is_rollout"] = True
        hinted_batch_padded.meta_info["max_tokens"] = self.config.data.max_response_length
        
        # set "hinted" field to 1 (indicating off-policy)
        hinted_batch_padded.non_tensor_batch["hinted"] = np.ones(len(hinted_batch_padded), dtype=np.int32)
        
        print(f"process_off_policy_batch: final batch size = {len(hinted_batch_padded)}")

        
        return hinted_batch_padded



    def fit(self):
        """
        The training loop of .
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the  dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                
                batch_dict['hinted'] = np.zeros(len(batch_dict['input_ids']), dtype=np.int32)
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Assign a unique identifier (UUID) to each sample in the batch.
                # This is used to track and group samples throughout the training process,
                # especially when samples are repeated or shuffled for rollout and advantage calculation.
                # The UUIDs are stored in the non_tensor_batch["uid"] field as a numpy array of strings.
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "answer" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("answer")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")
                if "hinted" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("hinted")
                if "uid" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("uid")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )


                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps

                # repeat to align with repeated responses in rollout
                batch_on = batch.repeat(repeat_times=self.n_on_policy, interleave=True)
                gen_batch_on = gen_batch.repeat(repeat_times=self.n_on_policy, interleave=True)
                gen_batch_on.non_tensor_batch['hinted'] = np.zeros(len(gen_batch_on), dtype=np.int32)
                gen_batch_on = DataProto(
                    batch=gen_batch_on.batch,
                    non_tensor_batch=deepcopy(gen_batch_on.non_tensor_batch),
                    meta_info=deepcopy(gen_batch_on.meta_info)
                )

                if self.n_off_policy > 0:
                    batch_off = batch.repeat(repeat_times=self.n_off_policy, interleave=True)
                    gen_batch_off = gen_batch.repeat(repeat_times=self.n_off_policy, interleave=True)
                    print(f"Before processing: len(gen_batch_off) = {len(gen_batch_off)}")
                    print(f"gen_batch_off.batch keys: {gen_batch_off.batch.keys() if gen_batch_off.batch is not None else 'None'}")
                    
                    gen_batch_off = self.process_off_policy_batch(gen_batch_off)
                    gen_batch_off = DataProto(
                        batch=gen_batch_off.batch,
                        non_tensor_batch=deepcopy(gen_batch_off.non_tensor_batch),
                        meta_info=deepcopy(gen_batch_off.meta_info)
                    )
                    gen_batch = DataProto.concat([gen_batch_on, gen_batch_off])
                    batch_mixed = DataProto.concat([batch_on, batch_off])
                    # gen_batch = self._safe_concat(gen_batch_on, gen_batch_off)
                else:
                    gen_batch = gen_batch_on
                    batch_mixed = batch_on
                
                temperature = self.config.actor_rollout_ref.rollout.temperature
                gen_batch.meta_info["is_rollout"] = True
                gen_batch.meta_info["temperature"] = temperature

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    
                    # repeat to align with repeated responses in rollout
                    assert self.config.actor_rollout_ref.rollout.n == self.n_on_policy + self.n_off_policy, \
                        f"rollout.n ({self.config.actor_rollout_ref.rollout.n}) must equal n_on_policy({self.n_on_policy}) + n_off_policy({self.n_off_policy})"
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)

                    batch = batch_mixed.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    batch.meta_info["temperature"] = temperature
                    batch.meta_info["entropy_controller"] = self.ent_ctrl.__dict__

                    if self.config.actor_rollout_ref.hint.enabled:
                        # no kl in reward
                        pre_reward_tensor, pre_reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        batch.batch["token_level_scores"] = batch.batch["token_level_rewards"] = pre_reward_tensor
                        # pop pre-computed rewards
                        batch.batch.pop("token_level_rewards", None)
                        batch.batch.pop("token_level_scores", None)
                        
                        # reorganize on-policy and off-policy data, reward specifically
                        batch = self._reorganize_batch(batch)

                        post_reward_tensor, post_reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        batch.batch["token_level_scores"] = batch.batch["token_level_rewards"] = post_reward_tensor
                        # self._print_batch_info(batch)

                        batch.batch.pop("token_level_rewards", None)
                        batch.batch.pop("token_level_scores", None)


                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()
                            
                            # Save demo cache periodically
                            if hasattr(self, 'demo_cache') and self.demo_cache.cache_dir:
                                self.demo_cache.save_cache()
                                print(f"Info: Saved demo cache checkpoint with {len(self.demo_cache)} entries to {self.demo_cache.cache_dir}")

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                
                # Add demo cache and mixed rollout statistics to metrics
                if hasattr(self, 'demo_cache'):
                    cache_stats = self.demo_cache.get_cache_stats()
                    cache_metrics = {f"demo_cache/{k}": v for k, v in cache_stats.items()}
                    metrics.update(cache_metrics)
                    
                    # Add mixed rollout metrics
                    mixed_rollout_metrics = {
                        "demo/n_rollouts": self.n_rollouts,
                        "demo/n_off_policy": self.n_off_policy,
                        "demo/n_on_policy": self.n_rollouts - self.n_off_policy,
                        "demo/match_strategy": self.demo_match_strategy,
                        "demo/cache_size": len(self.demo_cache),
                        "demo/can_use_demo": len(self.demo_cache) >= self.n_demo and self.n_off_policy > 0,
                    }
                    
                    # Add off-policy specific metrics if available
                    if "is_off_policy" in batch.non_tensor_batch:
                        is_off_policy = batch.non_tensor_batch["is_off_policy"]
                        n_off_policy_samples = np.sum(is_off_policy)
                        n_on_policy_samples = len(is_off_policy) - n_off_policy_samples
                        
                        mixed_rollout_metrics.update({
                            "demo/actual_off_policy_count": n_off_policy_samples,
                            "demo/actual_on_policy_count": n_on_policy_samples,
                            "demo/off_policy_ratio": n_off_policy_samples / len(is_off_policy) if len(is_off_policy) > 0 else 0,
                        })
                        
                        # Calculate reward differences between on-policy and off-policy
                        if "token_level_scores" in batch.batch:
                            rewards = batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
                            on_policy_rewards = rewards[~is_off_policy]
                            off_policy_rewards = rewards[is_off_policy]
                            
                            if len(on_policy_rewards) > 0:
                                mixed_rollout_metrics["demo/on_policy_reward_mean"] = float(np.mean(on_policy_rewards))
                                mixed_rollout_metrics["demo/on_policy_reward_std"] = float(np.std(on_policy_rewards))
                            
                            if len(off_policy_rewards) > 0:
                                mixed_rollout_metrics["demo/off_policy_reward_mean"] = float(np.mean(off_policy_rewards))
                                mixed_rollout_metrics["demo/off_policy_reward_std"] = float(np.std(off_policy_rewards))
                                
                            if len(on_policy_rewards) > 0 and len(off_policy_rewards) > 0:
                                mixed_rollout_metrics["demo/reward_diff_off_minus_on"] = float(np.mean(off_policy_rewards) - np.mean(on_policy_rewards))
                    
                    metrics.update(mixed_rollout_metrics)
                
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    
                    # Save demo cache at the end of training
                    if hasattr(self, 'demo_cache') and self.demo_cache.cache_dir:
                        self.demo_cache.save_cache()
                        print(f"Info: Saved final demo cache with {len(self.demo_cache)} entries to {self.demo_cache.cache_dir}")
                    
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

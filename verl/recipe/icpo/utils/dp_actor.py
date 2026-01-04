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
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import (agg_loss, get_policy_loss_fn,
                                         kl_penalty)
from verl.utils.device import (get_device_id, get_device_name,
                               is_cuda_available, is_npu_available)
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import (prepare_dynamic_batch,
                                         restore_dynamic_batch)
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import (gather_outputs_and_unpad, ulysses_pad,
                                ulysses_pad_and_slice_inputs)
from verl.workers.actor import DataParallelPPOActor
from verl.workers.config import ActorConfig

from ..utils.core_algos import compute_icpo_policy_loss

if is_cuda_available:
    from flash_attn.bert_padding import (index_first_axis, pad_input,
                                         rearrange, unpad_input)
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import (
        index_first_axis, pad_input, rearrange, unpad_input)


__all__ = ["ICPODataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ICPODataParallelPPOActor(DataParallelPPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def get_entropy_metric(self, data, ent_ctrl_dict):
        entropy_coeff = ent_ctrl_dict["value"]
        entropy_loss_enabled = ent_ctrl_dict["entropy_loss_enabled"]
        advantages = data.batch['advantages']
        response_length = advantages.size(1)
        old_log_probs = data.batch["old_log_probs"]
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        pg_grad_scale = torch.sqrt(verl_F.masked_mean(advantages * advantages, response_mask)).item()
        entropy_grad_scale = torch.sqrt(verl_F.masked_mean(old_log_probs * old_log_probs, response_mask)).item()
        entropy_grad_scale_with_coeff = entropy_grad_scale * entropy_coeff * ent_ctrl_dict["entropy_loss_enabled"]
        entropy = -1 * verl_F.masked_mean(old_log_probs, response_mask).item()
        metrics = {
            "actor/pg_grad_scale" : pg_grad_scale,
            "actor/entropy_grad_scale": entropy_grad_scale,
            "actor/entropy_grad_scale_with_coeff": entropy_grad_scale_with_coeff,
            "actor/pg_grad_ratio" : pg_grad_scale / (pg_grad_scale + entropy_grad_scale_with_coeff) if pg_grad_scale + entropy_grad_scale_with_coeff > 1e-9 else 0,
            "actor/entropy_grad_ratio" : entropy_grad_scale_with_coeff / (pg_grad_scale + entropy_grad_scale_with_coeff) if pg_grad_scale + entropy_grad_scale_with_coeff > 1e-9 else 0,
            'actor/entropy_loss': entropy,
            'actor/entropy_loss_with_coeff': entropy * entropy_coeff * entropy_loss_enabled,
            'actor/entropy_coeff': entropy_coeff,
            'actor/entropy_coeff_realized': entropy_coeff * entropy_loss_enabled,
        }

        return metrics 

    def get_temperature_metric(self, data, temp_ctrl_dict):
        if temp_ctrl_dict["type"] == "pid":
            error = temp_ctrl_dict["last_error"]
            integral = temp_ctrl_dict["integral"]
            derivative = temp_ctrl_dict["last_derivative"]
            metrics = {"actor/temp_ctrl_error": error, "actor/temp_ctrl_integral": integral, "actor/temp_ctrl_derivative": derivative}
            return metrics
        else:
            return {}

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        ent_ctrl_dict = data.meta_info["entropy_controller"]
        temp_ctrl_dict = data.meta_info["temperature_controller"] if "temperature_controller" in data.meta_info else {"use_adapt_temp": False}
        # if temp_ctrl_dict["use_adapt_temp"]:
        #     temperature = temp_ctrl_dict["value"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        non_tensor_select_keys.append("hinted")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        if ent_ctrl_dict["use_adapt_ent"]:
            metrics.update(self.get_entropy_metric(data, ent_ctrl_dict))
        if temp_ctrl_dict["use_adapt_temp"]:
            metrics.update(self.get_temperature_metric(data, temp_ctrl_dict))
        metrics.update({"actor/temperature": temperature})
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    hinted = model_inputs.pop("hinted") if "hinted" in model_inputs else None

                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    entropy_coeff = ent_ctrl_dict["value"]
                    entropy_loss_enabled = ent_ctrl_dict["entropy_loss_enabled"]
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=entropy_loss_enabled
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if loss_mode == "icpo":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_icpo_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            hinted=hinted,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )
                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )

                    entropy_loss = 0.0
                    if entropy_loss_enabled:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                            'actor/entropy_ratio': (entropy_loss * entropy_coeff / policy_loss).detach().item(),
                            'actor/entropy_gt_pg': (entropy_loss * entropy_coeff > pg_loss).float().detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
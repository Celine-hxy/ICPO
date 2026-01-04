from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import (AdvantageEstimator, agg_loss,
                                         register_adv_est)


# From https://github.com/SkyworkAI/Skywork-OR1/blob/main/verl/trainer/ppo/core_algos.py
class EntController:
    def __init__(self, init_ent_coef, max_ent_coef, min_ent_coef, delta_ent_coef, target_ent, use_adapt_ent):
        self.type = "linear"
        self.value = init_ent_coef
        self.max_value = max_ent_coef
        self.min_value = min_ent_coef
        self.delta_ent_coef = delta_ent_coef
        self.target_ent = target_ent
        self.use_adapt_ent = use_adapt_ent
        self.entropy_loss_enabled = 1

    def update(self, current_ent):
        if not self.use_adapt_ent:
            return
            
        if current_ent < self.target_ent: 
            self.value += self.delta_ent_coef
        else:
            self.value -= self.delta_ent_coef

        self.value = float(np.clip(self.value, self.min_value, self.max_value))
        self.entropy_loss_enabled = int(current_ent < self.target_ent)


# PID controller for entropy loss
class EntPIDController:
    def __init__(self,
        init_ent_coef,
        max_ent_coef,
        min_ent_coef,
        target_ent,
        use_adapt_ent,
        kp,
        ki,
        kd,
    ):
        self.type = "pid"
        self.value = init_ent_coef
        self.max_value = max_ent_coef
        self.min_value = min_ent_coef
        self.target_ent = target_ent
        self.use_adapt_ent = use_adapt_ent
        self.entropy_loss_enabled = 1
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.last_error = 0
        self.last_derivative = 0

    def update(self, current_ent):
        if not self.use_adapt_ent:
            return
            
        error = self.target_ent - current_ent
        self.integral += error
        derivative = error - self.last_error
        self.last_error = error
        self.last_derivative = derivative
        delta = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.value += delta

        self.value = float(np.clip(self.value, self.min_value, self.max_value))
        self.entropy_loss_enabled = int(current_ent < self.target_ent)


class TempController:
    def __init__(self, init_temp, max_temp, min_temp, delta_temp, target_ent, warmup_steps, use_adapt_temp):
        self.type = "linear"
        self.value = init_temp
        self.max_value = max_temp
        self.min_value = min_temp
        self.delta_temp = delta_temp
        self.target_ent = target_ent
        self.warmup_steps = warmup_steps
        self.global_steps = 0
        self.use_adapt_temp = use_adapt_temp

    def update(self, current_ent):
        if not self.use_adapt_temp:
            return

        self.global_steps += 1
        if self.global_steps < self.warmup_steps:
            return 

        if current_ent < self.target_ent:
            self.value += self.delta_temp
        else:
            self.value -= self.delta_temp

        self.value = float(np.clip(self.value, self.min_value, self.max_value))


class TempPIDController:
    def __init__(self,
        init_temp,
        max_temp,
        min_temp,
        target_ent,
        warmup_steps,
        use_adapt_temp,
        kp,
        ki,
        kd,
    ):
        self.type = "pid"
        self.value = init_temp
        self.max_value = max_temp
        self.min_value = min_temp
        self.target_ent = target_ent
        self.warmup_steps = warmup_steps
        self.global_steps = 0
        self.use_adapt_temp = use_adapt_temp
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.last_error = 0
        self.last_derivative = 0

    def update(self, current_ent):
        if not self.use_adapt_temp:
            return

        self.global_steps += 1
        if self.global_steps < self.warmup_steps:
            return 

        error = self.target_ent - current_ent
        self.integral += error
        derivative = error - self.last_error
        self.last_error = error
        self.last_derivative = derivative
        delta = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.value += delta

        self.value = float(np.clip(self.value, self.min_value, self.max_value))


def compute_tpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    clipped_mask: torch.Tensor,
    index: np.ndarray,
    global_scale: float = 2.0,
    global_exponent: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:

    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2positive_sum = defaultdict(float)
    id2negative_sum = defaultdict(float)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            if scores[i] - 0.5 > 0:
                id2positive_sum[index[i]] += (scores[i] - 0.5) * 2.0
            else:
                id2negative_sum[index[i]] -= (scores[i] - 0.5) * 2.0

        for i in range(bsz):
            # if id2positive_sum[index[i]] and id2negative_sum[index[i]]:
            #     raw_score = (scores[i] - 0.5) * 2.0 # from (0, 1) to (-1, 1)
            #     scale = id2positive_sum[index[i]] if raw_score > 0 else id2negative_sum[index[i]]
            #     sign = 1.0 if raw_score > 0 else -1.0
            #     scores[i] = sign * (torch.abs(raw_score / scale) ** global_exponent) * global_scale
            # else:
            #     scores[i] = 0.0
            raw_score = (scores[i] - 0.5) * 2.0 # from (0, 1) to (-1, 1)
            scale = id2positive_sum[index[i]] if raw_score > 0 else id2negative_sum[index[i]]
            sign = 1.0 if raw_score > 0 else -1.0
            scores[i] = sign * (torch.abs(raw_score / scale) ** global_exponent) * global_scale

        scores = scores.unsqueeze(-1) * response_mask * (1 - clipped_mask)

    return scores, scores


def compute_icpo_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    hinted: np.ndarray,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        hinted (np.ndarray):
            Demo rollout flag, shape (batch_size,). Now represents samples that used demonstration.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    response_length = response_mask.shape[-1]   # (batch_size, response_length) -> response_length
    hinted = torch.tensor(hinted, device=log_prob.device)  # (batch_size,)
    hinted = hinted.unsqueeze(-1)   # (batch_size, 1)
    hinted = hinted.expand(-1, response_length)  # (batch_size, response_length)

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, (~hinted) * response_mask)

    hinted_negative_approx_kl = log_prob
    hinted_negative_approx_kl = torch.clamp(hinted_negative_approx_kl, min=-20.0, max=20.0)
    hinted_ratio = torch.exp(hinted_negative_approx_kl)
    # if config.policy_loss.enable_shaping:
    #     hinted_ratio = hinted_ratio / (hinted_ratio + 0.1)
    hinted_ratio = hinted_ratio / (hinted_ratio + 0.1)
    hinted_ppo_kl = verl_F.masked_mean(-hinted_negative_approx_kl, hinted * response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), (~hinted) * response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), (~hinted) * response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    hinted_pg_losses = -advantages * hinted_ratio

    hinted = hinted.float()
    pg_losses = pg_losses * (1 - hinted) + hinted_pg_losses * hinted
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
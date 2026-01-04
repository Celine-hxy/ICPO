"""
Demo Cache for  training.

This module implements a cache system that stores high-quality samples (pass rate = 1)
from the rollout process and provides them as demonstrations for training.
"""

import json
import os
import pickle
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from dataclasses import dataclass, field
from omegaconf import DictConfig
import time

from verl import DataProto
import random



@dataclass
class CacheEntry:
    """A single entry in the demo cache."""
    uid: str
    question: str  # The original question
    answer: str    # The model's response/answer
    reward: float
    pass_rate: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            import time
            self.timestamp = time.time()


class DemoCache:
    """
    Cache system for storing and retrieving high-quality demonstrations.
    
    This cache stores samples with pass_rate = 1 from rollout processes
    and provides them as demonstrations for training.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        min_pass_rate: float = 0.95,
        cache_dir: Optional[str] = None,
        similarity_threshold: float = 0.8,
        max_cache_age: Optional[float] = None,  # in seconds
    ):
        """
        Initialize the demo cache.
        
        Args:
            max_size: Maximum number of entries in the cache
            min_pass_rate: Minimum pass rate to consider a sample as high-quality
            cache_dir: Directory to persist cache (optional)
            similarity_threshold: Threshold for similarity-based retrieval
            max_cache_age: Maximum age of cache entries in seconds
        """
        self.max_size = max_size
        self.min_pass_rate = min_pass_rate
        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold
        self.max_cache_age = max_cache_age
        
        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}
        self.prompt_to_uid: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            "total_added": 0,
            "total_retrieved": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
        }
    
    def add_sample(
        self,
        uid: str,
        question: str,
        answer: str,
        reward: float,
        pass_rate: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a sample to the cache if it meets quality criteria.
        
        Args:
            uid: Unique identifier for the sample
            question: Input question
            answer: Model response/answer
            reward: Reward score
            pass_rate: Pass rate (0-1)
            metadata: Additional metadata
            
        Returns:
            bool: True if added, False otherwise
        """
        # Check if sample meets quality criteria
        if pass_rate < self.min_pass_rate:
            return False
        
        # Create cache entry
        entry = CacheEntry(
            uid=uid,
            question=question,
            answer=answer,
            reward=reward,
            pass_rate=pass_rate,
            timestamp=time.time(),  # 关键修复
            metadata=metadata or {}
        )
        
        # Add to cache
        if uid in self.cache:
            # Update existing entry
            self.cache[uid] = entry
        else:
            # Check if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            # Add new entry
            self.cache[uid] = entry
            self.prompt_to_uid[question].append(uid)
        
        self.stats["total_added"] += 1
        return True
    
    def get_n_demonstrations(
        self,
        target_prompt: str,
        n_demo: int = 3,
        strategy: str = "similarity",
        num_candidates: int = 10,
        random_seed: Optional[int] = None
    ) -> List[Tuple[str, str]]:
        """
        Retrieve multiple demonstrations for a given prompt.
        
        Args:
            target_prompt: The prompt to find demonstrations for
            n_demo: Number of demonstrations to retrieve
            strategy: Retrieval strategy ("similarity", "random", "best_reward")
            num_candidates: Number of candidates to consider
            random_seed: Random seed for reproducibility
            
        Returns:
            List[tuple[str, str]]: List of (question, answer) pairs
        """
        if not self.cache:
            self.stats["cache_misses"] += 1
            return []
        
        candidates = self._get_candidates(target_prompt, strategy, num_candidates, random_seed)
        
        if not candidates:
            self.stats["cache_misses"] += 1
            return []
        
        # Select top n candidates based on strategy
        if strategy == "best_reward":
            selected_candidates = sorted(candidates, key=lambda x: x.reward, reverse=True)[:n_demo]
        elif strategy == "similarity":
            selected_candidates = sorted(candidates, key=lambda x: x.metadata.get("similarity", 0), reverse=True)[:n_demo]
        else:  # random
            if random_seed is not None:
                np.random.seed(random_seed)
            if len(candidates) <= n_demo:
                selected_candidates = candidates
            else:
                indices = np.random.choice(len(candidates), n_demo, replace=False)
                selected_candidates = [candidates[i] for i in indices]
        
        self.stats["cache_hits"] += 1
        self.stats["total_retrieved"] += len(selected_candidates)
        
        return [(candidate.question, candidate.answer) for candidate in selected_candidates]
    
    def _get_candidates(
        self,
        target_prompt: str,
        strategy: str,
        num_candidates: int,
        random_seed: Optional[int] = None
    ) -> List[CacheEntry]:
        """Get candidate entries based on strategy."""
        candidates = []
        
        if strategy == "similarity":
            # Find similar questions based on content similarity
            for uid, entry in self.cache.items():
                similarity = self._compute_similarity(target_prompt, entry.question)
                if similarity >= self.similarity_threshold:
                    entry.metadata["similarity"] = similarity
                    candidates.append(entry)
            
            # Sort by similarity
            candidates.sort(key=lambda x: x.metadata.get("similarity", 0), reverse=True)
            
        elif strategy == "random":
            # Random selection
            if random_seed is not None:
                np.random.seed(random_seed)
            
            all_entries = list(self.cache.values())
            if len(all_entries) <= num_candidates:
                candidates = all_entries
            else:
                indices = np.random.choice(len(all_entries), num_candidates, replace=False)
                candidates = [all_entries[i] for i in indices]
                
        elif strategy == "best_reward":
            # Select based on reward
            all_entries = list(self.cache.values())
            candidates = sorted(all_entries, key=lambda x: x.reward, reverse=True)[:num_candidates]
        
        return candidates[:num_candidates]
    
    def _compute_similarity(self, question1: str, question2: str) -> float:
        """Compute similarity between two questions."""
        # Simple word overlap similarity (can be improved with embeddings)
        words1 = set(question1.lower().split())
        words2 = set(question2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _evict_oldest(self):
        """Evict the oldest entry from the cache."""
        if not self.cache:
            return
        
        # Find oldest entry
        oldest_uid = min(self.cache.keys(), key=lambda x: self.cache[x].timestamp)
        oldest_entry = self.cache[oldest_uid]
        
        # Remove from cache
        del self.cache[oldest_uid]
        
        # Remove from question mapping
        if oldest_entry.question in self.prompt_to_uid:
            self.prompt_to_uid[oldest_entry.question].remove(oldest_uid)
            if not self.prompt_to_uid[oldest_entry.question]:
                del self.prompt_to_uid[oldest_entry.question]
        
        self.stats["evictions"] += 1
    
    def update_from_batch(self, batch: DataProto, tokenizer) -> int:
        """
        Update cache from a training batch, only saving the highest reward trajectory per question.
        
        Args:
            batch: DataProto batch containing rollout results
            tokenizer: Tokenizer for decoding responses
            
        Returns:
            int: Number of new entries added
        """
        added_count = 0

        id2score = defaultdict(list)
        id2uid = defaultdict(list)
        bsz = len(batch)
        index = batch.non_tensor_batch["uid"]
        scores = batch.batch["token_level_rewards"]  # A tensor of shape (batch_size, token_length), containing the token-level rewards for each sample in the batch.
        for i in range(bsz):
            score = scores[i].sum(-1)               # For each sample in the batch, sum the token-level rewards to obtain the total reward for that sample.
            id2score[index[i]].append(score)        # Group the total reward by the sample's UUID.
            id2uid[index[i]].append(i)

        # find samples which all rollout passed (all scores are >0)
        hint_ids = []
        hinted_uids = []
        zero_pass_count, all_pass_count, some_pass_count = 0, 0, 0
        for idx in id2score:
            hint_ids.append(idx)
            hinted_uids.append(random.choice(id2uid[idx]))  # randomly select a sample from the same uid
            if all(score == 0 for score in id2score[idx]):
                zero_pass_count += 1
            elif all(score > 0 for score in id2score[idx]):
                # 精简写法：all_pass的样本随机选一条rollout结果存入demo_cache，适配batch数据格式
                chosen_idx = random.choice(id2uid[idx])
                # 获取原始问题
                raw_question = batch.non_tensor_batch["raw_prompt"][chosen_idx][0]['content']
                # 获取response和reward
                responses = batch.batch.get("responses", [])
                response_tokens = responses[chosen_idx]
                answer = tokenizer.decode(response_tokens, skip_special_tokens=True)
                self.add_sample(
                    uid=idx,
                    prompt=raw_question,
                    response=answer,
                    reward=id2score[idx][id2uid[idx].index(chosen_idx)],
                    pass_rate=sum(id2score[idx]) / len(id2score[idx]) if len(id2score[idx]) > 0 else 0,
                    metadata={"source": "update_from_batch"}
                )
                added_count += 1
                all_pass_count += 1
            else:
                some_pass_count += 1
        
        print(f"Debug: Successfully added {added_count} samples to cache")
        # 打印1个新加入cache的样本作为示例
        if added_count > 0:
            print("Example: Newly added sample to cache:")
            print(f"Question: {raw_question}")
            print(f"Answer: {answer}")
        return added_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.copy()
        stats.update({
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": stats["cache_hits"] / max(stats["total_retrieved"], 1),
            "avg_reward": np.mean([entry.reward for entry in self.cache.values()]) if self.cache else 0.0,
            "avg_pass_rate": np.mean([entry.pass_rate for entry in self.cache.values()]) if self.cache else 0.0,
        })
        return stats
    
    def save_cache(self, filepath: Optional[str] = None):
        """Save cache to disk."""
        if filepath is None:
            if self.cache_dir is None:
                raise ValueError("No cache directory specified")
            filepath = os.path.join(self.cache_dir, "demo_cache.pkl")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        cache_data = {
            "entries": self.cache,
            "stats": self.stats,
            "max_size": self.max_size,
            "min_pass_rate": self.min_pass_rate,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def load_cache(self, filepath: Optional[str] = None):
        """Load demonstration cache from disk. The format should be consistent with gen_demo_dummy.py."""
        print(f"Loading demo cache from {filepath} ...")
        try:
            with open(filepath, "rb") as f:
                cache_data = pickle.load(f)
            # Compatible with the structure from gen_demo_dummy.py
            entries = cache_data.get("entries", {})
            # If entries is a dict and its values are dicts (i.e., asdict structure), convert to CacheEntry
            self.cache.clear()
            for uid, entry in entries.items():
                if isinstance(entry, dict):
                    # Compatible with asdict structure
                    self.cache[uid] = CacheEntry(**entry)
                else:
                    self.cache[uid] = entry
            self.stats = cache_data.get("stats", self.stats)
            self.max_size = cache_data.get("max_size", self.max_size)
            self.min_pass_rate = cache_data.get("min_pass_rate", self.min_pass_rate)
            # Rebuild prompt_to_uid mapping
            self.prompt_to_uid.clear()
            for uid, entry in self.cache.items():
                self.prompt_to_uid[entry.question].append(uid)
            print(f"Demo cache loaded successfully, total {len(self.cache)} entries.")
        except Exception as e:
            print(f"Failed to load cache from {filepath}: {e}")
    
    def clear_cache(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.prompt_to_uid.clear()
        self.stats = {
            "total_added": 0,
            "total_retrieved": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
        }
    
    def __len__(self) -> int:
        return len(self.cache)
    
    def __contains__(self, uid: str) -> bool:
        return uid in self.cache

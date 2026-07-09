"""
双池专家系统 — 固定池 + 随机池

固定池: 每次审查都参与（Hickey/Carmack/Wardley/GeneKim/自审）
随机池: 按领域匹配，每次DPO步骤随机抽取1-2个
"""

import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter
from openai import OpenAI

from config import FIXED_POOL, RANDOM_POOL, API_CONFIG, DPO_CONFIG


@dataclass
class ExpertVote:
    expert_id: str
    expert_name: str
    lens: str
    preference: str          # "A" | "B" | "tie"
    confidence: float
    raw_response: str = ""


@dataclass
class PoolResult:
    """一次双池审查的完整结果"""
    prompt: str
    response_a: str
    response_b: str
    fixed_votes: List[ExpertVote] = field(default_factory=list)
    random_votes: List[ExpertVote] = field(default_factory=list)

    @property
    def all_votes(self) -> List[ExpertVote]:
        return self.fixed_votes + self.random_votes

    @property
    def weighted_preference(self) -> Tuple[str, float]:
        """加权汇总 → (偏好, 置信度)"""
        fixed_weights = {k: v["weight"] for k, v in FIXED_POOL.items()}
        random_weights = {k: 1.0 for k in RANDOM_POOL}

        score_a, score_b, total = 0.0, 0.0, 0.0
        for vote in self.all_votes:
            w = fixed_weights.get(vote.expert_id) or random_weights.get(vote.expert_id, 1.0)
            if vote.preference == "A":
                score_a += w * vote.confidence
            elif vote.preference == "B":
                score_b += w * vote.confidence
            total += w

        if total == 0:
            return ("tie", 0.0)
        score_a /= total
        score_b /= total

        if abs(score_a - score_b) < 0.15:
            return ("tie", max(score_a, score_b))
        if score_a > score_b:
            return ("A", score_a / (score_a + score_b))
        return ("B", score_b / (score_a + score_b))


class ExpertPool:
    """双池专家管理系统"""

    def __init__(self, client: Optional[OpenAI] = None):
        self.client = client
        self.fixed_pool = FIXED_POOL
        self.random_pool = RANDOM_POOL

    def sample_random_experts(self, n: int = 1, domain_hint: str = "") -> Dict:
        """随机池 — 按领域匹配采样"""
        candidates = self.random_pool.copy()
        if domain_hint:
            scored = []
            for eid, expert in candidates.items():
                score = sum(1 for d in expert.get("domains", [])
                           if d.lower() in domain_hint.lower())
                scored.append((eid, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            weights = [s[1] + 0.5 for s in scored]
            selected_ids = random.choices(
                [s[0] for s in scored], weights=weights,
                k=min(n, len(candidates)))
        else:
            selected_ids = random.sample(list(candidates.keys()), min(n, len(candidates)))
        return {eid: candidates[eid] for eid in selected_ids}

    def judge_pair(
        self, prompt: str, response_a: str, response_b: str,
        domain_hint: str = "",
    ) -> PoolResult:
        """执行一次完整双池审查"""
        result = PoolResult(prompt=prompt, response_a=response_a, response_b=response_b)

        # 随机化A/B顺序，消除位置偏差
        swap = random.random() < 0.5
        opt_a, opt_b = (response_b, response_a) if swap else (response_a, response_b)
        label_a, label_b = ("B", "A") if swap else ("A", "B")

        # 固定池 — 全部参与
        for eid, expert in self.fixed_pool.items():
            vote = self._call_expert(eid, expert, prompt, opt_a, opt_b, label_a, label_b)
            if vote:
                result.fixed_votes.append(vote)

        # 随机池 — 采样1-2个
        random_experts = self.sample_random_experts(n=random.randint(1, 2), domain_hint=domain_hint)
        for eid, expert in random_experts.items():
            vote = self._call_expert(eid, expert, prompt, opt_a, opt_b, label_a, label_b)
            if vote:
                result.random_votes.append(vote)

        return result

    def _call_expert(
        self, eid: str, expert: Dict,
        prompt: str, opt_a: str, opt_b: str,
        label_a: str, label_b: str,
    ) -> Optional[ExpertVote]:
        """调用单个专家API进行审查"""
        if self.client is None:
            return self._heuristic_vote(eid, expert, opt_a, opt_b, label_a, label_b)

        try:
            response = self.client.chat.completions.create(
                model=API_CONFIG["model"],
                messages=[
                    {"role": "system", "content": expert["prompt"]},
                    {"role": "user", "content": (
                        f"问题:\n{prompt}\n\n"
                        f"A:\n{opt_a}\n\n"
                        f"B:\n{opt_b}\n\n"
                        f"哪个回答更好？只输出 A 或 B。"
                    )},
                ],
                max_tokens=API_CONFIG["max_tokens"],
                temperature=API_CONFIG["temperature"],
            )
            raw = response.choices[0].message.content.strip().upper()
        except Exception as e:
            print(f"  [WARN] {eid} API: {e}")
            return self._heuristic_vote(eid, expert, opt_a, opt_b, label_a, label_b)

        if "A" in raw and "B" not in raw:
            pref = label_a
        elif "B" in raw and "A" not in raw:
            pref = label_b
        else:
            pref = "tie"

        return ExpertVote(
            expert_id=eid, expert_name=expert["name"],
            lens=expert["lens"], preference=pref,
            confidence=0.8 if pref != "tie" else 0.3, raw_response=raw,
        )

    def _heuristic_vote(self, eid, expert, opt_a, opt_b, label_a, label_b) -> ExpertVote:
        """API不可用时的启发式fallback"""
        len_a, len_b = len(opt_a), len(opt_b)
        if len_a > len_b * 1.5 or len_b > len_a * 1.5:
            pref = "tie"
        elif len_a >= len_b:
            pref = label_a
        else:
            pref = label_b
        return ExpertVote(
            expert_id=eid, expert_name=expert["name"],
            lens=expert["lens"], preference=pref,
            confidence=0.3, raw_response="[heuristic]",
        )

    def should_include(self, result: PoolResult) -> bool:
        """判断是否纳入DPO训练: 专家一致度 ≥ min_expert_agreement"""
        if not result.all_votes:
            return False
        votes = [v.preference for v in result.all_votes if v.preference != "tie"]
        if not votes:
            return False
        agreement = Counter(votes).most_common(1)[0][1] / len(votes)
        return agreement >= DPO_CONFIG["min_expert_agreement"]

    def summary(self, result: PoolResult) -> str:
        """审查摘要"""
        lines = [f"\n{'='*60}", "双池审查结果", f"{'='*60}"]
        lines.append(f"\nQ: {result.prompt[:120]}")
        lines.append(f"A ({len(result.response_a)}字): {result.response_a[:80]}...")
        lines.append(f"B ({len(result.response_b)}字): {result.response_b[:80]}...")
        lines.append(f"\n--- 固定池 ({len(result.fixed_votes)}位) ---")
        for v in result.fixed_votes:
            lines.append(f"  [{v.preference}] {v.expert_name} ({v.lens})")
        if result.random_votes:
            lines.append(f"\n--- 随机池 ({len(result.random_votes)}位) ---")
            for v in result.random_votes:
                lines.append(f"  [{v.preference}] {v.expert_name} ({v.lens})")
        pref, conf = result.weighted_preference
        lines.append(f"\n加权: {pref} (conf={conf:.2f})")
        return "\n".join(lines)

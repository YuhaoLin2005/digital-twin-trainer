"""
交付门禁 — 训练/DPO后的硬性质量检查
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Callable


@dataclass
class GateResult:
    passed: bool = False
    checks: Dict[str, bool] = field(default_factory=dict)
    details: Dict[str, str] = field(default_factory=dict)
    timestamp: str = ""


class DeliveryGate:
    """训练产出质量门禁"""

    def __init__(self, name: str = "default"):
        self.name = name
        self.checks: Dict[str, Callable] = {}
        self.history: List[GateResult] = []

    def add_check(self, name: str, fn: Callable):
        self.checks[name] = fn

    def run(self) -> GateResult:
        result = GateResult(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
        for name, fn in self.checks.items():
            try:
                passed, detail = fn()
                result.checks[name] = passed
                result.details[name] = detail
            except Exception as e:
                result.checks[name] = False
                result.details[name] = f"ERROR: {e}"
        result.passed = all(result.checks.values())
        self.history.append(result)
        return result

    def report(self, result: GateResult = None) -> str:
        if result is None and self.history:
            result = self.history[-1]
        if result is None:
            return "No gate results yet."
        lines = [f"\n{'='*60}", f"DELIVERY GATE: {self.name}", f"{'='*60}"]
        for name, passed in result.checks.items():
            icon = "[PASS]" if passed else "[FAIL]"
            detail = result.details.get(name, "")
            lines.append(f"  {icon} {name}")
            if not passed:
                lines.append(f"       → {detail}")
        lines.append(f"\n  VERDICT: {'PASS' if result.passed else 'FAIL'}")
        return "\n".join(lines)


def make_loss_check(loss_history: List[Dict]) -> tuple:
    if len(loss_history) < 2:
        return False, "Not enough data"
    first, last = loss_history[0]["loss"], loss_history[-1]["loss"]
    return last < first * 0.95, f"Loss {first:.4f} → {last:.4f}"


def make_checkpoint_check(path: str) -> tuple:
    exists = os.path.exists(path)
    return exists, f"Found: {path}" if exists else f"Missing: {path}"


def make_winrate_check(win_rate: float, min_rate: float = 0.5) -> tuple:
    ok = win_rate >= min_rate
    return ok, f"Win rate: {win_rate:.2f} (min: {min_rate})"

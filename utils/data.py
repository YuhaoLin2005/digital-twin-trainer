"""
数据处理 — 脱敏 + .claude配置 → 训练指令对
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
from config import PERSONA_SOURCES, SELF_MODEL, GROWTH_LOG_DIR, DATA_DIR


# ============================================================
# 1. 脱敏
# ============================================================

PII_PATTERNS = [
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
    (r'sk-[a-zA-Z0-9]{20,}', '[API_KEY]'),
    (r'ghp_[a-zA-Z0-9]{36}', '[GITHUB_TOKEN]'),
    (r'\b\d{3}[-.]?\d{4}[-.]?\d{4}\b', '[PHONE]'),
    (r'Bearer\s+[a-zA-Z0-9_\-\.]+', 'Bearer [TOKEN]'),
    (r'api_key\s*=\s*["\'][^"\']+["\']', 'api_key = "[REDACTED]"'),
    (r'token\s*=\s*["\'][^"\']+["\']', 'token = "[REDACTED]"'),
]


def sanitize_text(text: str) -> Tuple[str, int]:
    """脱敏文本 → (脱敏后文本, 替换次数)"""
    count = 0
    for pattern, replacement in PII_PATTERNS:
        new_text, n = re.subn(pattern, replacement, text)
        if n > 0:
            count += n
            text = new_text
    return text, count


def sanitize_file(filepath: Path) -> Tuple[str, int]:
    """脱敏单个文件"""
    if not filepath.exists():
        return "", 0
    content = filepath.read_text(encoding="utf-8", errors="replace")
    return sanitize_text(content)


# ============================================================
# 2. 配置提取 → 原则列表
# ============================================================

def extract_principles(text: str, source_name: str) -> List[Dict]:
    """从配置文本中提取可训练的原则

    检测规则类型:
      - behavior: 必须/MUST/强制 → 行为规则
      - principle: 核心/原则 → 设计原则
      - process: 启动/收尾/加载 → 流程规则
      - guideline: 建议/推荐 → 软指南
    """
    principles = []
    lines = text.split("\n")

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) < 10:
            continue

        rule_type = None
        priority = "medium"

        if any(kw in stripped for kw in ["必须", "MUST", "⛔", "HARD RULE", "强制"]):
            rule_type = "behavior"
            priority = "high"
        elif any(kw in stripped for kw in ["核心", "原则", "Principle"]):
            rule_type = "principle"
            priority = "high"
        elif any(kw in stripped for kw in ["启动", "收尾", "加载", "启动序列"]):
            rule_type = "process"
            priority = "high"
        elif any(kw in stripped for kw in ["建议", "推荐", "可以"]):
            rule_type = "guideline"
            priority = "medium"

        if rule_type:
            ctx_before = lines[i - 1].strip() if i > 0 else ""
            ctx_after = lines[i + 1].strip() if i < len(lines) - 1 else ""
            principles.append({
                "text": stripped,
                "type": rule_type,
                "priority": priority,
                "source": source_name,
                "context": f"{ctx_before} {ctx_after}"[:200],
            })

    return principles


# ============================================================
# 3. 原则 → 指令-回答对
# ============================================================

QUESTION_TEMPLATES = {
    "behavior": [
        "遇到这种情况你会怎么做？{context}",
        "你的处理方式是什么？{context}",
    ],
    "principle": [
        "你的核心原则是什么？",
        "在做决策时，你遵循什么方法论？",
    ],
    "process": [
        "开始工作时，你的第一步是什么？",
        "完成工作后，你做哪些检查？",
    ],
    "guideline": [
        "在这个场景下，你有什么建议？{context}",
        "你认为最佳实践是什么？",
    ],
}


def principle_to_qa(principle: Dict, augment: bool = True) -> List[Dict]:
    """原则 → 指令-回答对"""
    pairs = []
    templates = QUESTION_TEMPLATES.get(principle["type"], QUESTION_TEMPLATES["guideline"])

    for tmpl in templates[:2]:
        q = tmpl.format(context=principle.get("context", ""))
        pairs.append({
            "instruction": q,
            "output": principle["text"],
            "source": principle["source"],
            "priority": principle["priority"],
        })

    if augment and principle["priority"] == "high":
        keywords = [w for w in principle["text"].split() if len(w) >= 3][:5]
        if keywords:
            pairs.append({
                "instruction": f"{' '.join(keywords[:3])}——能展开说说吗？",
                "output": principle["text"],
                "source": principle["source"],
                "priority": principle["priority"],
            })

    return pairs


# growth-log 质量关键词 — 匹配有方法论价值的章节
GROWTH_KEYWORDS = [
    'why', 'how to apply', 'what worked', 'what broke', 'patterns',
    '核心发现', '教训', '洞察', '决定', '方法论', '翻车',
    'lesson', 'finding', 'insight', '关键', '沉淀', '模式',
    '事件序列', '诊断', '修复', '根因', '收敛',
]


def _extract_inline_qa(text: str, source: str) -> List[Dict]:
    """从文本中提取内联 Why/How 模式 → Q&A 对

    匹配模式:
      **Why:** <解释>
      **How to apply:** <应用>
    将 Why 和 How 配对成一条训练样本
    """
    pairs = []
    # 找所有 Why/How 配对
    why_pattern = re.compile(r'\*\*Why:\*\*\s*(.+?)(?=\n\n|\n\*\*|\Z)', re.DOTALL)
    how_pattern = re.compile(r'\*\*How to apply:\*\*\s*(.+?)(?=\n\n|\n\*\*|\Z)', re.DOTALL)

    whys = why_pattern.findall(text)
    hows = how_pattern.findall(text)

    for i, why_text in enumerate(whys):
        why_clean = why_text.strip()
        if len(why_clean) < 30:
            continue
        # 配对的 How（如果有）
        how_clean = hows[i].strip() if i < len(hows) else ""
        output = why_clean
        if how_clean:
            output += f"\n应用: {how_clean}"

        pairs.append({
            "instruction": "为什么这样做？说说背后的原因。",
            "output": output,
            "source": source,
            "priority": "high",
        })

    return pairs


def _split_numbered_items(text: str) -> List[str]:
    """将编号列表拆成独立条目

    "1. **同一问题追三层**..." → ["同一问题追三层: ...", ...]
    """
    # 匹配 `1. **title** content` 或 `1. content` 模式
    items = re.split(r'\n(?=\d+\.\s+\*\*)', text)
    if len(items) <= 1:
        items = re.split(r'\n(?=\d+\.\s)', text)
    return [item.strip() for item in items if len(item.strip()) > 40]


def load_persona_data(max_samples: int = 300) -> List[Dict]:
    """完整数据流水线: 加载 → 脱敏 → 提取 → 转换"""
    all_pairs = []
    total_sanitized = 0

    # 1) 核心配置文件
    for filepath, category in PERSONA_SOURCES:
        text, count = sanitize_file(filepath)
        total_sanitized += count
        if text:
            for p in extract_principles(text, filepath.name):
                all_pairs.extend(principle_to_qa(p, augment=True))

    # 2) self-model
    text, count = sanitize_file(SELF_MODEL)
    total_sanitized += count
    if text:
        for p in extract_principles(text, "self-model.md"):
            all_pairs.extend(principle_to_qa(p, augment=True))

    # 3) growth-logs — 全量加载，拆 ### 子节 + 内联Why/How
    if GROWTH_LOG_DIR.exists():
        log_files = sorted(
            GROWTH_LOG_DIR.glob("*.md"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )[:15]  # 从5篇扩到15篇
        for log_file in log_files:
            text, count = sanitize_file(log_file)
            total_sanitized += count
            if not text:
                continue

            # A) 内联 Why/How 提取
            all_pairs.extend(_extract_inline_qa(text, log_file.name))

            # B) 按 ## 和 ### 两级标题拆分
            for section in re.split(r'\n(?:##|###)\s+', text):
                section_lower = section.lower()
                if not any(kw in section_lower for kw in GROWTH_KEYWORDS):
                    continue

                lines = section.strip().split("\n")
                title = lines[0].strip() if lines else ""
                body_lines = lines[1:] if len(lines) > 1 else []

                # 如果章节包含编号列表，每个编号条目单独成样本
                body_text = "\n".join(body_lines)
                numbered_items = _split_numbered_items(body_text)

                if len(numbered_items) >= 2:
                    # 多个编号条目 → 每个单独成对
                    for item in numbered_items:
                        item_lines = item.strip().split("\n")
                        item_title = item_lines[0].strip() if item_lines else ""
                        item_body = "\n".join(item_lines[:8])  # 限制长度
                        if len(item_body) > 40:
                            all_pairs.append({
                                "instruction": f"从你的经验来看，{title}——{item_title}",
                                "output": item_body,
                                "source": log_file.name,
                                "priority": "high",
                            })
                else:
                    # 单个段落 → 整段作为回答
                    body = "\n".join(body_lines[:10])
                    if len(body) > 50:
                        all_pairs.append({
                            "instruction": f"说说你的经验：{title}",
                            "output": body.strip(),
                            "source": log_file.name,
                            "priority": "high",
                        })

    # 4) 补充 memory 目录关键文件
    MEMORY_DIR = GROWTH_LOG_DIR.parent  # .../memory/
    EXTRA_MEMORY_FILES = [
        "ratings-tracker.md",
        "user_profile.md",
        "dual-layer-mechanical-gate.md",
        "named-persona-adversarial-review.md",
    ]
    for fname in EXTRA_MEMORY_FILES:
        fpath = MEMORY_DIR / fname
        if not fpath.exists():
            continue
        text, count = sanitize_file(fpath)
        total_sanitized += count
        if text:
            for p in extract_principles(text, fname):
                all_pairs.extend(principle_to_qa(p, augment=True))
            # 也提取内联 Why/How
            all_pairs.extend(_extract_inline_qa(text, fname))

    print(f"  脱敏: {total_sanitized} 处替换")
    print(f"  提取: {len(all_pairs)} 条指令-回答对")

    # 去重 + 限数
    seen = set()
    unique = []
    for pair in all_pairs:
        key = pair["instruction"]
        if key not in seen:
            seen.add(key)
            unique.append(pair)

    if len(unique) > max_samples:
        high = [p for p in unique if p.get("priority") == "high"]
        rest = [p for p in unique if p.get("priority") != "high"]
        unique = high + rest[:max_samples - len(high)]

    print(f"  去重+采样: {len(unique)} 条（上限 {max_samples}）")
    return unique


def save_training_data(pairs: List[Dict], output_path: Path = None) -> Path:
    """保存 JSONL 训练数据"""
    if output_path is None:
        output_path = DATA_DIR / "persona_training_data.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"  已保存: {output_path} ({len(pairs)} 条)")
    return output_path

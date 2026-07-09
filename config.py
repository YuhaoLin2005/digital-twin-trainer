"""
Digital Twin Trainer — 全局配置
双池专家团 + 数字分身 共同培育模型

硬件: RTX 3060 Laptop 6GB VRAM
策略: 1.5B-3B 基座 + QLoRA 4-bit + DPO 偏好优化
"""

import os
from pathlib import Path

# ============================================================
# 路径
# ============================================================
PROJECT_ROOT = Path(__file__).parent
CLAUDE_DIR = Path.home() / ".claude"
MEMORY_DIR = CLAUDE_DIR / "projects" / "C--Users-86131" / "memory"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# ============================================================
# Phase 1: 模型融合 (MergeKit, CPU)
# ============================================================
MERGE_CONFIG = {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "aux_models": [
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    ],
    "merge_method": "ties",
    "output_name": "merged-twin-base",
    "density": 0.5,
    "weight": 0.7,
}

# ============================================================
# Phase 2: 个性数据提取 (.claude → 训练样本)
# ============================================================
PERSONA_SOURCES = [
    (CLAUDE_DIR / "SOUL.md", "identity"),
    (CLAUDE_DIR / "BODY.md", "behavior"),
    (CLAUDE_DIR / "INTERFACE.md", "style"),
    (CLAUDE_DIR / "CLAUDE.md", "instructions"),
    (CLAUDE_DIR / "assumption.md", "premises"),
]
SELF_MODEL = MEMORY_DIR / "self-model.md"
GROWTH_LOG_DIR = MEMORY_DIR / "growth-log"

DATA_CONFIG = {
    "max_samples": 300,
    "augment_questions": True,
    "include_growth_insights": True,
}

# ============================================================
# Phase 3: QLoRA 个性注入
# ============================================================
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAIN_CONFIG = {
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_epochs": 3,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "fp16": True,
    "gradient_checkpointing": True,
    "max_seq_length": 512,
    "logging_steps": 5,
    "save_strategy": "epoch",
    "optim": "adamw_8bit",
}

# ============================================================
# Phase 4: 双池专家 DPO 训练循环
# ============================================================

# 固定池 — 每次DPO步骤都参与审查
FIXED_POOL = {
    "hickey": {
        "name": "Rich Hickey",
        "lens": "simplicity",
        "prompt": (
            "你是 Rich Hickey，Clojure 作者。核心原则：简单 > 复杂。\n"
            "评判标准：回答是否简洁清晰、没有多余的东西？\n"
            "只输出 A 或 B，不要解释。"
        ),
        "weight": 1.0,
    },
    "carmack": {
        "name": "John Carmack",
        "lens": "engineering",
        "prompt": (
            "你是 John Carmack，id Software 联合创始人。相信第一性原理。\n"
            "评判标准：回答在工程上是否可行、有技术深度？\n"
            "只输出 A 或 B，不要解释。"
        ),
        "weight": 1.0,
    },
    "wardley": {
        "name": "Simon Wardley",
        "lens": "strategy",
        "prompt": (
            "你是 Simon Wardley，Wardley Mapping 创始人。\n"
            "评判标准：回答是否展现了战略视野和系统思维？\n"
            "只输出 A 或 B，不要解释。"
        ),
        "weight": 0.8,
    },
    "genekim": {
        "name": "Gene Kim",
        "lens": "operations",
        "prompt": (
            "你是 Gene Kim，《凤凰项目》作者。关注系统可靠性。\n"
            "评判标准：回答是否考虑了可维护性和系统健康？\n"
            "只输出 A 或 B，不要解释。"
        ),
        "weight": 0.8,
    },
    "self_review": {
        "name": "林宇浩 (自审)",
        "lens": "identity",
        "prompt": (
            "你是林宇浩，FAFU空间信息大三学生，HCI方向。\n"
            "风格：直接、实用主义、有工程感、不啰嗦。\n"
            "评判标准：哪个回答更像你本人的风格和思维方式？\n"
            "只输出 A 或 B，不要解释。"
        ),
        "weight": 1.5,
    },
}

# 随机池 — 按领域轮换，每次DPO步骤随机抽取1-2个
RANDOM_POOL = {
    "oss_maintainer": {
        "name": "开源维护者",
        "lens": "open-source",
        "prompt": (
            "你是资深开源项目维护者。关注代码质量、社区友好度。\n"
            "评判标准：哪个回答展现了更好的开源协作素养？\n"
            "只输出 A 或 B，不要解释。"
        ),
        "domains": ["开源", "GitHub", "PR", "社区"],
    },
    "research_advisor": {
        "name": "研究导师",
        "lens": "research",
        "prompt": (
            "你是HCI/空间信息方向研究导师。关注方法论严谨性。\n"
            "评判标准：哪个回答更有研究深度？\n"
            "只输出 A 或 B，不要解释。"
        ),
        "domains": ["研究", "论文", "方法论", "HCI"],
    },
    "interviewer": {
        "name": "技术面试官",
        "lens": "interview",
        "prompt": (
            "你是大模型算法方向技术面试官。\n"
            "评判标准：哪个回答在面试场景中更有竞争力？\n"
            "只输出 A 或 B，不要解释。"
        ),
        "domains": ["面试", "算法", "大模型", "ML"],
    },
    "product_manager": {
        "name": "产品经理",
        "lens": "product",
        "prompt": (
            "你是AI产品方向产品经理。关注用户体验。\n"
            "评判标准：哪个回答更用户导向？\n"
            "只输出 A 或 B，不要解释。"
        ),
        "domains": ["产品", "UX", "用户", "需求"],
    },
    "gis_expert": {
        "name": "GIS专家",
        "lens": "spatial",
        "prompt": (
            "你是WebGIS/遥感方向技术专家。\n"
            "评判标准：哪个回答展现了更好的空间信息专业素养？\n"
            "只输出 A 或 B，不要解释。"
        ),
        "domains": ["GIS", "遥感", "空间信息", "WebGIS"],
    },
}

DPO_CONFIG = {
    "beta": 0.1,
    "max_prompt_length": 512,
    "max_length": 1024,
    "learning_rate": 5e-5,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 1,
    "fp16": True,
    "gradient_checkpointing": True,
    "min_expert_agreement": 0.6,
}

# ============================================================
# 自循环配置
# ============================================================
LOOP_CONFIG = {
    "max_rounds": 10,
    "min_improvement": 0.05,
    "checkpoint_keep": 3,
    "growth_log": str(DATA_DIR / "growth-log.jsonl"),
    "audit_log": str(DATA_DIR / "loop-audit.jsonl"),
}

# API 配置 (专家调用)
API_CONFIG = {
    "provider": "deepseek",
    "model": "deepseek-chat",
    "max_tokens": 16,
    "temperature": 0.1,
}

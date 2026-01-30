import os

# 配置常量：模型部署方案
# 格式：列表的列表，每个子列表代表一个模型实例占用的 GPU ID 集合
# 可以通过命令行参数 --gpu-config "[[0],[1]]" 进行覆盖
# MODEL_DEPLOYMENT_CONFIG = [[4], [5], [6, 7]]
MODEL_DEPLOYMENT_CONFIG = [[0], [1], [2, 3]]



import sys
import json
import re
import copy
import random
import threading
from dataclasses import dataclass
from pathlib import Path
import torch
import asyncio
import time
from collections import defaultdict, Counter
from typing import List, Optional, Dict, Any, Union, Literal, Mapping
from contextlib import asynccontextmanager
from loguru import logger

from fastapi import FastAPI, HTTPException, Body, Depends
from pydantic import BaseModel, Field, ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer

# ATTN_MODEL_PATH = "/models/Qwen3-4B-Instruct-2507"
# ATTN_MODEL_PATH = "/models/llama-3.2-3b-instruct"
# ATTN_MODEL_PATH = "/models/gemma-3-4b-it"
# # 长文本阈值：当请求的 token 数超过此值时，优先使用多卡实例
# LARGE_REQUEST_THRESHOLD = 100000
# # 最大模型处理 Token 数：超过此值时，使用线性 fallback 策略，不进行模型前向计算
# MAX_TOKEN_FOR_MODEL = 300000

# ATTN_MODEL_PATH = "/models/qwen3-1.7b"
# LARGE_REQUEST_THRESHOLD = 100000
# MAX_TOKEN_FOR_MODEL = 300000

ATTN_MODEL_PATH = "/models/qwen3-8b"
LARGE_REQUEST_THRESHOLD = 100000
MAX_TOKEN_FOR_MODEL = 170000

ATTN_RATIO = 0.20
ATTN_TAIL = 2
ATTN_HEAD = 0

# Ensure log directory exists (used for service.log and request artifacts)
Path("log").mkdir(parents=True, exist_ok=True)

# Configure logger to write to file, overwriting each time
logger.add(
    "log/service.log",
    mode="w",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)


# --- Pydantic Models & Dataclasses ---


# 单条消息结构
class Message(BaseModel):
    role: str
    content: str

    # Allow extra fields like tool_calls, etc.
    model_config = ConfigDict(extra="allow")


# --- Visualization Model ---
class VisualizationData(BaseModel):
    html: str
    token_scores: List[float]


# --- Request Model ---
class CompressRequest(BaseModel):
    messages: List[Dict[str, Any]]
    attn_ratio: float = ATTN_RATIO
    attn_head: int = ATTN_HEAD
    attn_tail: int = ATTN_TAIL

    chunking_method: Literal[
        "token", "line", "message", "block", "message_block", "message_line"
    ]

    # Block splitting method: "double_newline" or "ppl" (PPL-based semantic splitting)
    block_split_method: Literal["double_newline", "ppl"] = "double_newline"
    # PPL spike detection threshold coefficient (smaller = more sensitive)
    ppl_spike_threshold_k: float = 0.2
    # PPL spike detection method: "std", "robust_std", "iqr", "mad"
    ppl_spike_method: Literal["std", "robust_std", "iqr", "mad"] = "std"
    # Minimum number of lines per block for PPL splitting
    ppl_min_block_lines: int = 1

    # Block scoring method: "mean" for simple average, "top_pct" for top-k percentage mean
    block_score_method: Literal["mean", "top_pct"] = "mean"
    # Top percentage for top_pct method (default 10%)
    block_score_top_pct: float = 0.1

    # Hierarchical filtering parameters (for message_block and message_line)
    # Ratio of messages to keep in the first stage (message-level filtering)
    hierarchical_message_ratio: float = 0.5

    # New fields to control compression of different parts
    compress_tool_call: bool = False
    compress_tool_response: bool = True
    compress_assistant_content: bool = False

    # Optional: explicit step indices if the client wants to control tail protection precisely
    step_indices: Optional[List[int]] = None
    # Whether to return visualization data
    return_visualization: bool = False
    # Randomize scores for ablation study
    randomize: bool = False

    # Candidate selection method:
    # - "greedy": 贪心算法，按 score 降序依次选择（原有实现）
    # - "knapsack": 0/1 背包动态规划，在 token 预算内最大化总 score
    selection_method: Literal["greedy", "knapsack"] = "greedy"

    # Attention layer selection:
    # - int: specific layer index
    # - List[int]: list of layer indices
    # - "low", "middle", "high": bottom/middle/top 1/3 layers
    # - "mean": all layers (default)
    # - None: all layers (default)
    attn_layers: Optional[
        Union[int, List[int], Literal["low", "middle", "high", "mean"]]
    ] = None

    model_config = ConfigDict(extra="forbid")


@dataclass
class Span:
    """一次可压缩/可标注的文本区间（Span）。

    这是服务内部使用的数据结构：
    - `extract_spans` 阶段仅负责生成字符级范围（char_start/char_end）。
    - tokenize 后会将字符范围映射为 token 范围（token_start/token_end）。
    - 后续压缩逻辑根据 span_type/step_idx 等字段决定是否压缩、以及写回到哪一类字段。

    字段说明:
        span_type: 区间类型，取值如："assistant_content" / "tool_call" / "tool_response" / "other"。
        msg_idx: 该 span 属于第几条 message（与 request.messages 的索引对齐）。
        char_start/char_end: 在 `full_text` 中的绝对字符范围（半开区间）。
        step_idx: 对应的 step 索引（用于 tail 保护），可能为 None。
        tc_idx: 若 span_type 为 tool_call，表示该 message 内第几个 tool_call（用于写回）。
        token_start/token_end: tokenize 后的 token 范围（半开区间），初始为 None。
    """

    span_type: str
    msg_idx: int
    char_start: int
    char_end: int
    step_idx: Optional[int] = None
    tc_idx: Optional[int] = None
    token_start: Optional[int] = None
    token_end: Optional[int] = None


@dataclass
class TokenScores:
    """注意力聚合得到的 token 分数。

    字段说明:
        token_scores: 可能平滑后的 token 分数（用于 token chunking 候选打分）。
        raw_token_scores: 对齐后的原始分数（用于可视化、以及非 token chunking）。
        token_log_probs: 每个 token 的 log probability（可选），用于 PPL block 切分。
            shape: (seq_len - 1,)，对应 input_ids[1:] 的 log prob。
    """

    token_scores: torch.Tensor
    raw_token_scores: torch.Tensor
    token_log_probs: Optional[torch.Tensor] = None


@dataclass
class ProtectionConfig:
    """保护配置（Head/Tail 保护）。

    字段说明:
        msg_step_indices: 每条 message 对应的 step 索引。
        tail_keep_start: step_idx >= tail_keep_start 的区域视为 tail（跳过压缩）。
        head_keep_end: step_idx < head_keep_end 的区域视为 head（跳过压缩）。
    """

    msg_step_indices: List[int]
    tail_keep_start: int
    head_keep_end: int


@dataclass
class TokenizationResult:
    """tokenize 结果与 offset_mapping。"""

    encoded: Any
    offset_mapping: list[list[int]]


@dataclass
class Candidate:
    """一个压缩候选块。

    字段说明:
        score: 候选块的分数（用于排序/随机选择）。
        token_indices: 候选块对应的 token 下标列表。
    """

    score: float
    token_indices: list[int]


@dataclass
class TokenCountStats:
    """压缩前后的 token 统计。"""

    old_tokens: int
    new_tokens: int


# --- Response Model ---
class CompressResponse(BaseModel):
    compressed_messages: List[Dict[str, Any]]
    stats: Dict[str, Any]
    visualization: Optional[VisualizationData] = None


# --- Helper Classes & Functions ---


class AttnResource:
    """注意力模型/分词器资源的进程内单例（支持多模型实例部署）。

    目标：
        - 管理多个模型实例以支持负载均衡和不同配置（单卡/多卡）。
        - 避免重复加载 tokenizer。
        - 线程安全的单例获取。
    """

    _instance: Optional["AttnResource"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "AttnResource":
        """获取 `AttnResource` 的全局单例（按需初始化）。"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(ATTN_MODEL_PATH)
        return cls._instance

    @classmethod
    def cleanup_instance_cuda_cache(cls) -> None:
        """若单例已创建，则清理 CUDA cache（用于服务关闭/释放显存碎片）。"""
        if cls._instance is None:
            return
        cls._instance.cleanup_cuda_cache()

    def __init__(self, model_path: str):
        logger.info(f"Loading tokenizer from {model_path}...")
        self.model_path: str = model_path
        # Tokenizer 是 CPU 运行的，全局共享一个
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
        self.instances = []
        try:
            num_gpus = torch.cuda.device_count()
            logger.info(f"Detected {num_gpus} GPUs available.")
        except Exception:
            num_gpus = 0
            logger.warning("Failed to detect GPUs, assuming 0.")

        for i, gpu_ids in enumerate(MODEL_DEPLOYMENT_CONFIG):
            logger.info(f"Initializing instance {i} on GPUs {gpu_ids}...")
            
            # 构造 max_memory 限制 accelerate 只使用分配给当前实例的 GPU
            # 对于不属于该实例的 GPU，设置显存限额为 0
            curr_max_memory = {}
            for gid in range(num_gpus):
                if gid in gpu_ids:
                    try:
                        # 获取显存大小，留少量余量或全用
                        mem_bytes = torch.cuda.get_device_properties(gid).total_memory
                        curr_max_memory[gid] = mem_bytes
                    except Exception:
                        curr_max_memory[gid] = "80GiB" # Fallback
                else:
                    curr_max_memory[gid] = 0
            
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    max_memory=curr_max_memory,
                    dtype="auto",
                    attn_implementation="eager",
                )
                model.eval()

                # 记录该实例的设备分布
                dm = getattr(model, "hf_device_map", None)
                if dm is not None:
                    c = Counter()
                    for k, v in dm.items():
                        if ".layers." in k:
                            c[v] += 1
                    logger.info(f"Instance {i} layers per device: {dict(c)}")
                
                self.instances.append({
                    "model": model,
                    "lock": threading.Lock(),
                    "gpus": gpu_ids,
                    "id": i
                })
                
                # Verify that the model loaded correctly on GPU
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                logger.info(f"Model instance {i} loaded successfully.")

            except Exception as e:
                logger.error(f"Failed to load instance {i} on GPUs {gpu_ids}: {e}")
                raise e

        logger.info(f"All {len(self.instances)} model instances loaded successfully.")

    def get_model_instance(self, num_tokens: int = 0) -> tuple[Any, threading.Lock]:
        """根据请求长度选择模型实例。
        
        逻辑：
        1. 如果 num_tokens >= LARGE_REQUEST_THRESHOLD 且存在多卡部署的实例，则优先从多卡实例中选择。
        2. 否则，从所有可用实例中随机选择。
        """
        # import pdb; pdb.set_trace()
        
        multi_gpu_candidates = [inst for inst in self.instances if len(inst["gpus"]) > 1]
        candidates = self.instances
        if num_tokens >= LARGE_REQUEST_THRESHOLD and multi_gpu_candidates:
            # 大文本优先分配给多卡实例
            candidates = multi_gpu_candidates
        
        if not candidates:
            candidates = self.instances
            
        choice = random.choice(candidates)
        return choice["model"], choice["lock"]

    def cleanup_cuda_cache(self):
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.warning("Failed to cleanup CUDA cache.")


def write_json_overwrite(path: str, obj: Any) -> None:
    """以覆盖方式写入 JSON 文件。

    用途：
    - 写入请求/响应落盘（如 log/original_messages.json、log/compressed_messages.json）。

    行为：
    - 自动创建父目录。
    - 失败时仅记录 warning，不影响主流程。
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write {path}: {e}")


def compute_protection_config(
    request: CompressRequest,
    original_messages: List[Dict[str, Any]],
) -> ProtectionConfig:
    """推导每条消息的 step_indices，并计算 head_keep_end / tail_keep_start。

    这里抽取了 endpoint 里与“保护策略”相关的逻辑：
    1) `msg_step_indices` 的推导：
       - 若请求提供 `request.step_indices` 且长度与 messages 一致，则直接使用；
       - 若长度不一致，则 warning 并退化为顺序索引；
       - 若未提供，则默认使用顺序索引（将每条消息视为一个 step）。
    2) `tail_keep_start` 的计算：
       - `total_steps = max(step_indices) + 1`（若为空则用 message 数兜底）；
       - `tail_keep_start = total_steps - max(0, min(attn_tail, total_steps))`。
    3) `head_keep_end` 的计算：
       - 直接取 `request.attn_head`。

    返回:
        ProtectionConfig
    """
    if request.step_indices:
        msg_step_indices = request.step_indices
        if len(msg_step_indices) != len(original_messages):
            logger.warning(
                "Warning: step_indices length mismatch. Falling back to sequential."
            )
            msg_step_indices = list(range(len(original_messages)))
    else:
        msg_step_indices = list(range(len(original_messages)))

    total_steps = (
        max(msg_step_indices) + 1 if msg_step_indices else len(original_messages)
    )
    tail_keep_start = total_steps - max(0, min(request.attn_tail, total_steps))
    head_keep_end = request.attn_head

    return ProtectionConfig(
        msg_step_indices=msg_step_indices,
        tail_keep_start=tail_keep_start,
        head_keep_end=head_keep_end,
    )


def extract_json_value_span(s: str, *, key: str) -> Optional[tuple[int, int]]:
    """在一个 JSON 字符串中，提取某个 key 对应 value 的字符切片范围。

    该函数用于从 Qwen 的 `<tool_call>` JSON 片段中，精准定位 `arguments`（或其它 key）
    对应的 value 区间，避免把 tool_call 的其它字段（如 name）也当成可压缩文本。

    约定:
        - 返回的是半开区间 (start, end)，可用于 `s[start:end]`。
        - 支持 value 为对象 `{...}`、数组 `[...]`、字符串 `"..."` 或简单字面量。
        - 这是一个“轻量扫描器”，不是完整 JSON parser；遇到不规范输入可能返回 None。

    参数:
        s: JSON 字符串（通常是 `<tool_call>` 标签里的内容）。
        key: 需要提取的键名。

    返回:
        (start, end) 或 None。
    """
    m = re.search(rf'"{re.escape(key)}"\s*:\s*', s)
    if not m:
        return None

    i = m.end()
    while i < len(s) and s[i].isspace():
        i += 1
    if i >= len(s):
        return None

    first = s[i]

    def _scan_bracket(open_ch: str, close_ch: str) -> Optional[tuple[int, int]]:
        # 扫描匹配的括号范围，期间需要正确跳过字符串内部的括号字符
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(s)):
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return (i, j + 1)
        return None

    if first == "{":
        return _scan_bracket("{", "}")
    if first == "[":
        return _scan_bracket("[", "]")

    if first == '"':
        esc = False
        for j in range(i + 1, len(s)):
            ch = s[j]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                return (i, j + 1)
        return None

    j = i
    while j < len(s) and s[j] not in ",\n\r}":
        j += 1
    return (i, j) if j > i else None


def extract_spans(
    full_text: str,
    *,
    expected_msg_len: int,
    msg_step_indices: List[int],
    messages: Optional[List[Dict[str, Any]]] = None,
) -> List[Span]:
    """从 `apply_chat_template` 生成的完整文本中抽取可压缩区间（spans）。

    本函数负责：
        1) 按 chat 模板拆分每条消息（role + content）；
        2) 自动识别模型格式（Qwen/Llama）；
        3) 识别 tool_call/tool_response 并生成对应的 Span。

    支持格式：
        - Qwen: <|im_start|>role\ncontent<|im_end|>
          - tool_call: <tool_call>\n...\n</tool_call>
          - tool_response: <tool_response>\n...\n</tool_response>
        - Llama: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
          - tool_call: content 为 JSON 且包含 "name" 和 "parameters"
          - tool_response: role 为 "ipython" 或 "tool"

    参数:
        full_text: tokenizer.apply_chat_template(...) 的输出。
        expected_msg_len: 期望的消息条数（通常等于 request.messages 长度）。
        msg_step_indices: 每条消息对应的 step 索引列表，用于 tail 保护。

    返回:
        spans 列表（`Span` 形式）。
    """
    # 自动检测模板格式
    is_llama = "<|start_header_id|>" in full_text
    is_gemma = "<start_of_turn>" in full_text
    
    all_spans: List[Span] = []

    if is_llama:
        # Llama 格式处理
        # Pattern: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
        pattern = re.compile(
            r"<\|start_header_id\|>([^\n]+)<\|end_header_id\|>\n\n(.*?)(?:<\|eot_id\|>|$)",
            re.S,
        )
        
        for msg_index, m in enumerate(pattern.finditer(full_text)):
            if msg_index >= expected_msg_len:
                break
            
            role = m.group(1).strip()
            msg_content = m.group(2)
            msg_start = m.start(2)
            step_idx = (
                msg_step_indices[msg_index] if msg_index < len(msg_step_indices) else None
            )

            if role == "assistant":
                # Llama tool call 也是 assistant role，但 content 是 JSON
                # 尝试提取 parameters
                # 注意：模板只支持单 tool_call，所以如果看起来像 tool call，就整个视为 tool_call
                arg_span = extract_json_value_span(msg_content, key="parameters")
                
                # 简单的启发式检查：是否包含 "name" 和 "parameters" 且是大括号包裹
                is_tool_call = False
                if arg_span and '"name"' in msg_content:
                     is_tool_call = True

                if is_tool_call and arg_span is not None:
                    # 这是一个 tool call
                    arg_start, arg_end = arg_span
                    all_spans.append(
                        Span(
                            span_type="tool_call",
                            msg_idx=msg_index,
                            tc_idx=0, # Llama 模板目前限制单 tool call
                            char_start=msg_start + arg_start,
                            char_end=msg_start + arg_end,
                            step_idx=step_idx,
                        )
                    )
                else:
                    # 普通 assistant 内容
                    all_spans.append(
                        Span(
                            span_type="assistant_content",
                            msg_idx=msg_index,
                            char_start=msg_start,
                            char_end=msg_start + len(msg_content),
                            step_idx=step_idx,
                        )
                    )
            
            elif role in ("ipython", "tool"):
                # tool response
                all_spans.append(
                    Span(
                        span_type="tool_response",
                        msg_idx=msg_index,
                        char_start=msg_start,
                        char_end=msg_start + len(msg_content),
                        step_idx=step_idx,
                    )
                )
            
            else:
                # user, system, etc.
                all_spans.append(
                    Span(
                        span_type="other",
                        msg_idx=msg_index,
                        char_start=msg_start,
                        char_end=msg_start + len(msg_content),
                        step_idx=step_idx,
                    )
                )

    elif is_gemma:
        # Gemma 格式处理
        # Pattern: <start_of_turn>role\ncontent<end_of_turn>
        pattern = re.compile(
            r"<start_of_turn>(.*?)\n(.*?)(?:<end_of_turn>|$)",
            re.S,
        )

        # 检测是否因为 System message 被合并而需要偏移索引
        # Gemma template 通常会将 system message 合并到第一条 user message 中
        # 导致 full_text 中的 turn 数量比 messages 少 1（如果有 system）
        # 且第一个 turn (user) 对应的是 messages[1] (user)
        msg_offset = 0
        if messages and len(messages) > 0 and messages[0].get("role") == "system":
            msg_offset = 1

        for match_index, m in enumerate(pattern.finditer(full_text)):
            msg_index = match_index + msg_offset

            if msg_index >= expected_msg_len:
                break
            
            role = m.group(1).strip()
            msg_content = m.group(2)
            msg_start = m.start(2)
            step_idx = (
                msg_step_indices[msg_index]
                if msg_index < len(msg_step_indices)
                else None
            )

            if role == "assistan":
                # Gemma 使用 model 代表 assistant
                all_spans.append(
                    Span(
                        span_type="assistant_content",
                        msg_idx=msg_index,
                        char_start=msg_start,
                        char_end=msg_start + len(msg_content),
                        step_idx=step_idx,
                    )
                )
            elif role in ("user", "tool"):
                all_spans.append(
                    Span(
                        span_type="tool_response",
                        msg_idx=msg_index,
                        char_start=msg_start,
                        char_end=msg_start + len(msg_content),
                        step_idx=step_idx,
                    )
                )
            else:
                # user, system, etc.
                all_spans.append(
                    Span(
                        span_type="other",
                        msg_idx=msg_index,
                        char_start=msg_start,
                        char_end=msg_start + len(msg_content),
                        step_idx=step_idx,
                    )
                )

    else:
        # Qwen 格式处理 (原有逻辑)
        pattern = re.compile(
            r"<\|im_start\|>([^\n]+)\n(.*?)(?:<\|im_end\|>|$)",
            re.S,
        )
        tool_response_pattern = re.compile(
            r"^\s*<tool_response>\n?(.*?)\n?</tool_response>\s*$",
            re.S,
        )
        tool_call_pattern = re.compile(
            r"<tool_call>\n(.*?)\n</tool_call>",
            re.S,
        )

        for msg_index, m in enumerate(pattern.finditer(full_text)):
            # 为了与 request.messages 对齐，只处理前 expected_msg_len 条模板消息
            if msg_index >= expected_msg_len:
                break
            role = m.group(1).strip()
            msg_content = m.group(2)
            msg_start = m.start(2)
            step_idx = (
                msg_step_indices[msg_index] if msg_index < len(msg_step_indices) else None
            )

            if role == "assistant":
                last_end = 0
                tc_count = 0
                for tc_match in tool_call_pattern.finditer(msg_content):
                    # tool_call 之前的 assistant 普通文本
                    if tc_match.start() > last_end:
                        all_spans.append(
                            Span(
                                span_type="assistant_content",
                                msg_idx=msg_index,
                                char_start=msg_start + last_end,
                                char_end=msg_start + tc_match.start(),
                                step_idx=step_idx,
                            )
                        )
                    tc_json_str = tc_match.group(1)
                    # tool_call 的 JSON 文本，尝试只抽取 arguments value
                    arg_span = extract_json_value_span(tc_json_str, key="arguments")
                    if arg_span is not None:
                        arg_start, arg_end = arg_span
                        all_spans.append(
                            Span(
                                span_type="tool_call",
                                msg_idx=msg_index,
                                tc_idx=tc_count,
                                char_start=msg_start + tc_match.start(1) + arg_start,
                                char_end=msg_start + tc_match.start(1) + arg_end,
                                step_idx=step_idx,
                            )
                        )
                    else:
                        # 若无法定位 arguments，则退化为整个 tool_call JSON（仍限定在 tool_call 内部）
                        all_spans.append(
                            Span(
                                span_type="tool_call",
                                msg_idx=msg_index,
                                tc_idx=tc_count,
                                char_start=msg_start + tc_match.start(1),
                                char_end=msg_start + tc_match.end(1),
                                step_idx=step_idx,
                            )
                        )

                    last_end = tc_match.end()
                    tc_count += 1

                if last_end < len(msg_content):
                    # assistant 最后的剩余文本
                    all_spans.append(
                        Span(
                            span_type="assistant_content",
                            msg_idx=msg_index,
                            char_start=msg_start + last_end,
                            char_end=msg_start + len(msg_content),
                            step_idx=step_idx,
                        )
                    )

            elif role in ("user", "tool"):
                # 兼容：tool role 或 user role 中以 <tool_response> 包裹的内容
                tm = tool_response_pattern.match(msg_content)
                if tm:
                    all_spans.append(
                        Span(
                            span_type="tool_response",
                            msg_idx=msg_index,
                            char_start=msg_start + tm.start(1),
                            char_end=msg_start + tm.end(1),
                            step_idx=step_idx,
                        )
                    )
                else:
                    all_spans.append(
                        Span(
                            span_type="other",
                            msg_idx=msg_index,
                            char_start=msg_start,
                            char_end=msg_start + len(msg_content),
                            step_idx=step_idx,
                        )
                    )
            else:
                all_spans.append(
                    Span(
                        span_type="other",
                        msg_idx=msg_index,
                        char_start=msg_start,
                        char_end=msg_start + len(msg_content),
                        step_idx=step_idx,
                    )
                )

    return all_spans


def tokenize_and_map_spans_to_token_ranges(
    *,
    tokenizer: Any,
    full_text: str,
    spans: List[Span],
    device: Any,
) -> TokenizationResult:
    """对 `full_text` 做 tokenize，并将 span 的字符范围映射到 token 范围。

    这个函数把 endpoint 内部两段强相关逻辑封装起来：
    1) 调用 tokenizer 生成 `encoded`（包含 input_ids / attention_mask / offset_mapping）；
    2) 使用 offset_mapping（token -> [char_start, char_end)）把每个 span 的
       `char_start/char_end` 映射为 `token_start/token_end`（半开区间）。

    约定与注意：
        - 会原地写回 `spans[i].token_start/token_end`，以便后续压缩流程直接使用。
        - 映射过程使用一个单调递增的 `token_idx` 游标来加速，因此假设 `spans` 基本按
            `char_start` 递增（当前 `extract_spans` 的生成顺序满足该假设）。

    参数:
        tokenizer: HuggingFace tokenizer。
        full_text: 待分词的完整文本（apply_chat_template 的输出）。
        spans: `extract_spans` 生成的 Span 列表。
        device: 将 `encoded` 移动到的设备（通常是 model.device）。

    返回:
        (encoded, offset_mapping)
        - encoded: tokenizer 的输出（已 `.to(device)`，且已 pop 掉 offset_mapping）。
        - offset_mapping: `List[[start, end]]`，长度等于 token 数。
    """
    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        return_attention_mask=True,
        return_offsets_mapping=True,
        add_special_tokens=True,
    ).to(device)

    offset_mapping: list[list[int]] = encoded.pop("offset_mapping")[0].tolist()
    token_idx = 0
    num_tokens = len(offset_mapping)

    for span in spans:
        span_start = span.char_start
        span_end = span.char_end
        token_start, token_end = None, None

        # 先把游标推进到可能覆盖 span_start 的 token
        while token_idx < num_tokens and offset_mapping[token_idx][1] <= span_start:
            token_idx += 1

        # token_start：span_start 落在哪个 token 的字符范围内
        if token_idx < num_tokens:
            start, end = offset_mapping[token_idx]
            if start <= span_start < end:
                token_start = token_idx

        # token_end：找到第一个 start >= span_end 的 token（半开区间）
        curr = token_idx
        while curr < num_tokens:
            start, end = offset_mapping[curr]
            if start >= span_end:
                token_end = curr
                break
            if start < span_end <= end:
                token_end = curr + 1
                break
            curr += 1

        span.token_start = token_start
        span.token_end = token_end

    return TokenizationResult(encoded=encoded, offset_mapping=offset_mapping)


@dataclass
class ForwardResult:
    """前向传播结果，包含 attentions 和 log_probs。

    字段说明:
        attentions: 最后一个 token 的注意力分布。
        log_probs: 每个 token 的 log probability，shape (seq_len - 1,)。
            对应 input_ids[1:] 的 log prob。
    """

    attentions: Optional[List[torch.Tensor]]
    log_probs: Optional[torch.Tensor] = None


# 两段式前向传播以获取最后一个token的注意力（同时返回 log_probs）
def forward_last_token_attentions_two_pass(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    return_log_probs: bool = False,
) -> ForwardResult:
    """
    Returns attentions for the last token and optionally the full log_probs.

    参数:
        model: HuggingFace 模型。
        input_ids: 输入 token IDs，shape (1, seq_len)。
        attention_mask: 注意力掩码。
        return_log_probs: 是否计算并返回 log probabilities（用于 PPL block 切分）。

    返回:
        ForwardResult，包含 attentions 和 log_probs。
    """
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        return ForwardResult(attentions=None, log_probs=None)

    seq_len = int(input_ids.size(1))
    if seq_len <= 0:
        return ForwardResult(attentions=None, log_probs=None)

    # 辅助函数：计算 log probs
    def compute_log_probs(
        logits: torch.Tensor,
        labels: torch.Tensor,
        is_shifted: bool = False,
        device: str = "cpu",
        batch_size: int = 1000
    ) -> torch.Tensor:
        """
        计算 log probabilities。
        
        Args:
            logits: 预测的 logits。
                - 若 is_shifted=False (默认): 视为完整 logits (1, seq, V)，需要内部做 shift (logits[:-1])。
                - 若 is_shifted=True: 视为已对齐的 logits (seq-1, V)，直接与 labels 计算。
            labels: 目标 token IDs。
                - 若 is_shifted=False: 完整 input_ids (1, seq)，内部取 labels[1:]。
                - 若 is_shifted=True: 已对齐的 labels (seq-1,)。
            is_shifted: 指示输入是否已经完成了 shift 操作。
            device: 计算设备，例如 "cpu" 或 "cuda"。默认为 "cpu"。
            batch_size: 当使用 GPU (device!="cpu") 计算时，每次计算的 token 数量。分批计算以节省显存。
        """
        # 1. 统一处理 shifting 和 view
        if not is_shifted:
            # 标准模式：logits[:-1] 预测 labels[1:]
            shift_logits = logits[0, :-1, :]
            shift_labels = labels[0, 1:]
        else:
            # 已对齐模式
            shift_logits = logits
            shift_labels = labels

        if shift_labels.numel() == 0:
            return torch.tensor([], device="cpu")

        # 展平以便统一处理 (Total_Tokens, Vocab_Size) 和 (Total_Tokens)
        shift_logits = shift_logits.contiguous().view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # 2. 根据设备选择计算路径
        if device == "cpu":
            # CPU 模式：原有逻辑，全部移动到 CPU 一次性计算
            # 移动到 CPU 计算以节省显存
            shift_logits = shift_logits.cpu()
            shift_labels = shift_labels.cpu()

            neg_log_probs = torch.nn.functional.cross_entropy(
                shift_logits,
                shift_labels,
                reduction="none",
            )
            return -neg_log_probs
        
        else:
            # GPU 分批计算模式
            log_probs_list = []
            total_tokens = shift_labels.size(0)
            
            for i in range(0, total_tokens, batch_size):
                end = min(i + batch_size, total_tokens)
                
                # cross_entropy 返回的是 -log_prob
                chunk_neg_log_probs = torch.nn.functional.cross_entropy(
                    shift_logits[i:end],
                    shift_labels[i:end],
                    reduction="none"
                )
                
                # 计算完成后立即转回 CPU 并释放显存引用，方便后续合并
                log_probs_list.append(-chunk_neg_log_probs.cpu())
                
                del chunk_neg_log_probs
            
            # 合并结果
            return torch.cat(log_probs_list)

    if seq_len == 1:
        try:
            model.set_attn_implementation("eager")
        except Exception as e:
            logger.warning(f"attn single-token switch fallback (eager): {e}")
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
        
        log_probs = None
        if return_log_probs:
            log_probs = compute_log_probs(out.logits, input_ids)
            
        return ForwardResult(attentions=out.attentions, log_probs=log_probs)

    prefill_ids = input_ids[:, :-1]
    last_ids = input_ids[:, -1:]
    prefill_mask = attention_mask[:, :-1] if attention_mask is not None else None

    try:
        model.set_attn_implementation("sdpa")
    except Exception as e:
        logger.warning(f"attn two-pass prefill switch fallback (sdpa): {e}")
    prefill_out = model(
        input_ids=prefill_ids,
        attention_mask=prefill_mask,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )

    try:
        model.set_attn_implementation("eager")
    except Exception as e:
        logger.warning(f"attn two-pass last-token switch fallback (eager): {e}")
    out_last = model(
        input_ids=last_ids,
        attention_mask=attention_mask,
        past_key_values=prefill_out.past_key_values,
        use_cache=False,
        output_attentions=True,
        return_dict=True,
    )

    log_probs = None
    if return_log_probs:
        # PPL 计算只需要 prefill 的 logits (预测 input_ids[1:])
        # prefill_logits shape: (1, seq_len-1, vocab_size)
        # 对应需要预测的 labels 是 input_ids[0, 1:] (即整个 input 的后续部分)
        
        # 注意：compute_log_probs 内部会自动做 shift: logits[:-1] vs labels[1:]
        # 但我们这里的 prefill_logits 已经是 "input_ids[:-1] 对应的输出" 了。
        # 如果直接传给 compute_log_probs，它会再次切片掉最后一个 logit，导致丢失 input_ids[-1] 的预测。
        # 所以我们这里需要手动构造一个 "看起来像完整 logits" 的 tensor 或者修改 helper。
        # 为了复用 helper 且不修改 helper 的语义（helper 假定输入是 full logits），
        # 我们可以稍微 hack 一下或者直接调用 F.cross_entropy。
        
        # 直接调用 F.cross_entropy 最清晰：
        target_logits = prefill_out.logits[0]  # (seq_len-1, V)
        target_labels = input_ids[0, 1:].to(target_logits.device) # (seq_len-1,)
        log_probs = compute_log_probs(target_logits, target_labels, is_shifted=True, device='gpu')
    # import pdb; pdb.set_trace()
    return ForwardResult(attentions=out_last.attentions, log_probs=log_probs)


def compute_token_scores(
    model: Any,
    encoded: Any,
    chunking_method: str,
    compute_token_log_probs: bool = False,
    attn_layers: Optional[Union[int, List[int], str]] = None,
) -> TokenScores:
    """计算每个 token 的注意力分数，并返回平滑后的分数与原始分数。

    本函数抽取了 endpoint 内与"注意力分数计算"强相关的步骤：
    1) 调用 `forward_last_token_attentions_two_pass` 获取最后一个 token 的 attentions（及 logits）；
    2) 根据 attn_layers 筛选特定层（如有）；
    3) stack/聚合得到 (seq_len,) 的 `token_scores`；
    4) 将 `token_scores` 对齐到 `encoded["input_ids"]` 的 seq_len（pad/截断）；
    5) 仅当 `chunking_method == "token"` 时做平滑（conv1d moving average）。
    6) 可选：计算每个 token 的 log probability（用于 PPL block 切分）。

    返回值约定：
    - `raw_token_scores` 是对齐后的"未平滑"分数；用于可视化、以及非 token chunking。
    - `token_scores` 是可能被平滑后的分数；用于 token 级候选的打分。
    - `token_log_probs` 是每个 token 的 log probability（可选），用于 PPL block 切分。

    参数:
        model: 用于前向计算注意力的模型。
        encoded: tokenizer 输出（至少包含 input_ids，可能包含 attention_mask）。
        chunking_method: chunking 策略；仅当为 "token" 时启用平滑。
        compute_token_log_probs: 是否计算 token log probs（用于 PPL block 切分）。
        attn_layers: 指定参与计算的 attention 层。可以是单个索引、索引列表、"low"/"middle"/"high" 或 "mean"。

    返回:
        TokenScores，包含 token_scores, raw_token_scores, token_log_probs。

    异常:
        ValueError: 当模型未返回 attentions（为空/None）时抛出，便于上层按"skipped"处理。
    """
    with torch.inference_mode():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

        forward_result = forward_last_token_attentions_two_pass(
            model=model,
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            return_log_probs=compute_token_log_probs,
        )
        torch.cuda.synchronize()

    attentions = forward_result.attentions
    token_log_probs = forward_result.log_probs

    if not attentions:
        raise ValueError("no attentions returned")

    # Filter layers based on attn_layers
    num_layers = len(attentions)
    selected_indices = list(range(num_layers))
    if attn_layers is not None:
        if isinstance(attn_layers, int):
            idx = attn_layers if attn_layers >= 0 else num_layers + attn_layers
            if 0 <= idx < num_layers:
                selected_indices = [idx]
        elif isinstance(attn_layers, list):
            valid_indices = []
            for idx in attn_layers:
                real_idx = idx if idx >= 0 else num_layers + idx
                if 0 <= real_idx < num_layers:
                    valid_indices.append(real_idx)
            if valid_indices:
                selected_indices = valid_indices
        elif isinstance(attn_layers, str):
            one_third = max(1, num_layers // 3)
            if attn_layers == "low":
                selected_indices = list(range(0, one_third))
            elif attn_layers == "middle":
                selected_indices = list(range(one_third, 2 * one_third))
            elif attn_layers == "high":
                selected_indices = list(range(2 * one_third, num_layers))
            elif attn_layers == "mean":
                selected_indices = list(range(num_layers))
    # If selection ends up empty (e.g. invalid indices), fallback to all layers (or keep previous logic?)
    # Here we fallback to all layers to be safe, or we could warn.
    if not selected_indices:
        selected_indices = list(range(num_layers))
        
    logger.info(f"Selected layers: {selected_indices}")

    selected_attentions = [attentions[i] for i in selected_indices]
    attn_tensor = torch.stack(selected_attentions)
    
    last_q_idx = -1 if attn_tensor.shape[-2] > 1 else 0
    last_token_attn = attn_tensor[:, 0, :, last_q_idx, :]
    token_scores = last_token_attn.mean(dim=(0, 1))

    seq_len = int(encoded["input_ids"].shape[1])
    if int(token_scores.numel()) != seq_len:
        if int(token_scores.numel()) > seq_len:
            token_scores = token_scores[:seq_len]
        else:
            token_scores = torch.nn.functional.pad(
                token_scores, (0, seq_len - int(token_scores.numel()))
            )

    # Keep raw scores for visualization and non-token chunking
    raw_token_scores = token_scores.clone()

    # Smoothing (Only for token level chunking)
    # token_scores = smooth_token_scores_for_chunking(
    #     token_scores=token_scores,
    #     chunking_method=chunking_method,
    # )

    return TokenScores(
        token_scores=token_scores,
        raw_token_scores=raw_token_scores,
        token_log_probs=token_log_probs,
    )


def smooth_token_scores_for_chunking(
    *, token_scores: torch.Tensor, chunking_method: str
) -> torch.Tensor:
    """对 token_scores 做平滑（仅 token chunking）。

    与原先 compute_token_scores 内的 conv1d moving average 逻辑一致。
    """
    if chunking_method != "token":
        return token_scores

    orig_len = int(token_scores.numel())
    window_size = 10
    kernel = (
        torch.ones(
            1,
            1,
            window_size,
            device=token_scores.device,
            dtype=token_scores.dtype,
        )
        / window_size
    )
    token_scores = token_scores.view(1, 1, -1)
    token_scores = torch.nn.functional.conv1d(
        token_scores, kernel, padding=window_size // 2
    )
    token_scores = token_scores.view(-1)[:orig_len]
    return token_scores


def should_skip_model_forward(request: CompressRequest) -> bool:
    """判断本次请求是否应跳过模型前向计算注意力。

    需求：
    - randomize=True 时不需要调用模型；
    - attn_ratio 为 0 或 1 时不需要调用模型。
    这里对 0/1 做了轻微放宽：<=0 或 >=1 都视为同类边界情况。
    """
    # PPL 切分依赖模型前向计算得到的 token_log_probs，因此不能跳过
    if request.block_split_method == "ppl":
        return False
    
    if request.randomize:
        return True

    if request.attn_ratio <= 0.0:
        return True
    if request.attn_ratio >= 1.0:
        return True
    return False


def build_fallback_token_scores(
    *,
    encoded: Any,
    chunking_method: str,
    randomize: bool,
    linear: bool = False,
) -> TokenScores:
    """不调用模型时构造 TokenScores。

    - randomize=True：用随机分数（便于可视化/ablation），再按需平滑。
    - linear=True: 线性递增分数（最早最低，越后越高）。
    - 否则：用全 0 分数（不会触发模型前向；排序时相当于稳定的默认顺序）。
    """
    device = encoded["input_ids"].device
    seq_len = int(encoded["input_ids"].shape[1])
    if randomize:
        raw = torch.rand((seq_len,), device=device, dtype=torch.float32)
    elif linear:
        # 0.0 to 1.0
        raw = torch.linspace(0, 1, steps=seq_len, device=device, dtype=torch.float32)
    else:
        raw = torch.zeros((seq_len,), device=device, dtype=torch.float32)

    token_scores = smooth_token_scores_for_chunking(
        token_scores=raw.clone(),
        chunking_method=chunking_method,
    )
    return TokenScores(token_scores=token_scores, raw_token_scores=raw)


def is_span_type_compressible(
    span_type: Optional[str], request: CompressRequest
) -> bool:
    """判断某个 span 类型在当前请求配置下是否允许被压缩。

    这里的“可压缩”仅指：该类型是否被请求参数打开（例如 `compress_tool_response=True`）。
    不包含 tail 保护、token 范围合法性等其它过滤条件。

    参数:
        span_type: span 的类型字符串，来源于 span 字典的 `type` 字段。
            可能为 None（例如不完整数据），此时返回 False。
        request: `CompressRequest`，包含用户选择压缩哪些区域的开关。

    返回:
        True 表示该 span 类型在本次请求中允许进入压缩候选流程；否则 False。
    """
    if span_type == "tool_response":
        return request.compress_tool_response
    if span_type == "tool_call":
        return request.compress_tool_call
    if span_type == "assistant_content":
        return request.compress_assistant_content
    return False


def is_step_protected(
    step_idx: Optional[int], *, head_keep_end: int, tail_keep_start: int
) -> bool:
    """判断某个 step 是否落在 head/tail 保护范围内。

    当前逻辑：
    1) 当 `step_idx < head_keep_end` 时，属于 head 保护。
    2) 当 `step_idx >= tail_keep_start` 时，属于 tail 保护。
    这些区域为了保证上下文的完整性，会跳过压缩。

    参数:
        step_idx: 当前 span / message 对应的步骤索引，可能为 None。
        head_keep_end: head 保护区间的结束 step（不包含）。
        tail_keep_start: tail 保护区间的起始 step（包含）。

    返回:
        True 表示处于保护范围（应跳过压缩）；False 表示不在保护范围。
    """
    if step_idx is None:
        return False
    return step_idx < head_keep_end or step_idx >= tail_keep_start


def _split_span_into_blocks_double_newline(
    span_text: str,
    span_char_start: int,
) -> list[tuple[int, int]]:
    """按双换行（\\n{2,}）将 span 文本切分为多个 block。

    参数:
        span_text: span 的原始文本。
        span_char_start: span 在 full_text 中的起始字符位置。

    返回:
        block 范围列表，每个元素为 (char_start, char_end)，相对于 full_text 的绝对位置。
    """
    block_ranges: list[tuple[int, int]] = []
    
    current_pos = 0
    block_start = span_char_start
    # 记录当前 block 最后一个"非空行"的内容结束位置（绝对坐标）
    last_content_end = span_char_start
    has_content = False

    # 使用 splitlines(keepends=True) 处理换行，逻辑与 PPL 切分保持一致
    for line in span_text.splitlines(keepends=True):
        line_len = len(line)
        
        # 计算该行内容的长度（去除末尾换行符）
        content_len = line_len
        if line.endswith("\r\n"):
            content_len -= 2
        elif line.endswith("\n") or line.endswith("\r"):
            content_len -= 1
        
        # 判断是否是空行（即内容长度为0，只含换行符）
        if content_len == 0:
            # 遇到空行，视为分隔符（或分隔符的一部分）
            if has_content:
                # 结算上一个 block
                # 注意：根据原正则 (\r?\n){2,} 的行为，Block 不包含触发分割的末尾换行符
                block_ranges.append((block_start, last_content_end))
                has_content = False
            
            # 更新 block_start 跳过当前空行
            current_pos += line_len
            block_start = span_char_start + current_pos
            last_content_end = block_start
        else:
            # 非空行
            has_content = True
            # 更新 last_content_end 为当前行内容的结束位置
            # 注意：这里仅包含当前行的内容，不含其换行符。
            # 但如果这是 block 中间行，其实下一行会接续上来，不会在这里截断。
            # 这个 last_content_end 仅在下一行是空行触发结算时生效。
            last_content_end = span_char_start + current_pos + content_len
            current_pos += line_len

    # 添加最后一个 block（如果有剩余内容）
    if has_content:
        # 对于最后一个 block，包含一直到文本结束的所有字符（包括最后的换行符）
        block_ranges.append((block_start, span_char_start + len(span_text)))

    return block_ranges


def _split_span_into_blocks_ppl(
    span_text: str,
    span_char_start: int,
    *,
    token_log_probs: torch.Tensor,
    offset_mapping: list[tuple[int, int]],
    spike_threshold_k: float = 0.2,
    spike_method: Literal["std", "robust_std", "iqr", "mad"] = "std",
    min_block_lines: int = 1,
) -> list[tuple[int, int]]:
    """基于 PPL（困惑度）突变检测将 span 文本切分为多个 block。

    核心思想：
        1. 按行分割 span 文本
        2. 使用预计算的 token_log_probs（来自 compute_token_scores）
        3. 根据每行的 token 边界统计每行的 PPL
        4. 检测 PPL 突变点（spike），在突变点处切分 block

    参数:
        span_text: span 的原始文本。
        span_char_start: span 在 full_text 中的起始字符位置。
        token_log_probs: 预计算的每个 token 的 log probability，shape (seq_len - 1,)。
            对应 input_ids[1:] 的 log prob。
        offset_mapping: token 到字符的映射，[(char_start, char_end), ...]，基于 full_text。
        spike_threshold_k: 突变检测的阈值系数（越小越敏感）。
        spike_method: 阈值计算方法，支持 "std", "robust_std", "iqr", "mad"。
        min_block_lines: 每个 block 的最小行数。

    返回:
        block 范围列表，每个元素为 (char_start, char_end)，相对于 full_text 的绝对位置。
    """
    import math
    import numpy as np

    span_char_end = span_char_start + len(span_text)

    # 按行分割（同时计算原始字符位置）
    # 使用 splitlines(keepends=True) 可以自动处理各种换行符（\n, \r\n, \r 等）
    # 并保留它们以便准确计算字符偏移
    line_char_ranges: list[tuple[int, int]] = []
    lines: list[str] = []

    current_pos = 0
    # splitlines 对于空字符串返回 []，这会导致 len(lines)=0，
    # 下面的判断 if len(lines) <= 1 会直接返回整个 span，符合预期
    for line_with_end in span_text.splitlines(keepends=True):
        # 计算该行内容的长度（去除末尾换行符）
        content_len = len(line_with_end)
        if line_with_end.endswith("\r\n"):
            content_len -= 2
        elif line_with_end.endswith("\n") or line_with_end.endswith("\r"):
            content_len -= 1
        
        # 记录内容（不含换行符，用于 PPL 计算）
        lines.append(line_with_end[:content_len])
        
        # 记录内容的绝对字符范围 [start, end)
        start = span_char_start + current_pos
        end = start + content_len
        line_char_ranges.append((start, end))
        
        # 推进游标（包含换行符的完整长度）
        current_pos += len(line_with_end)

    if len(lines) <= 1:
        # 只有一行或空文本，直接返回整个 span
        return [(span_char_start, span_char_end)]

    # ========== 检查预计算的 token_log_probs ==========
    if token_log_probs is None:
        logger.warning("PPL block split: token_log_probs is None, returning whole span")
        return [(span_char_start, span_char_end)]

    if len(token_log_probs) == 0:
        return [(span_char_start, span_char_end)]

    # ========== 计算每行的 PPL ==========
    # line_char_ranges 已经计算好了

    # 将字符范围映射到 token 范围
    def find_token_range_for_char_range(
        char_start: int, char_end: int, offset_mapping: list, start_search_idx: int = 0
    ) -> tuple[int, int]:
        """找到覆盖 [char_start, char_end) 的 token 范围 [tok_start, tok_end)"""
        tok_start = None
        tok_end = None
        
        # 优化：从 start_search_idx 开始搜索
        # offset_mapping 是按顺序排列的，所以不需要每次从头遍历
        for tok_idx in range(start_search_idx, len(offset_mapping)):
            c_start, c_end = offset_mapping[tok_idx]

            # 跳过特殊 token（offset 为 (0, 0)）
            if c_start == c_end == 0 and tok_idx > 0:
                continue

            # 提前结束条件：当前 token 起始位置已超过目标范围结束位置
            # 假设 offset_mapping 是按字符顺序排列的
            if c_start >= char_end:
                break

            # 检查是否与目标字符范围有交集
            if c_end > char_start and c_start < char_end:
                if tok_start is None:
                    tok_start = tok_idx
                tok_end = tok_idx + 1
            
        if tok_start is None:
            return (0, 0)
        return (tok_start, tok_end)

    line_ppls: list[float] = []
    current_search_idx = 0  # 记录上一次搜索结束的位置，作为下一次搜索的起点
    # import pdb; pdb.set_trace()
    for line_idx, (char_start, char_end) in enumerate(line_char_ranges):
        if char_start >= char_end:
            # 空行
            line_ppls.append(0.0)
            continue

        tok_start, tok_end = find_token_range_for_char_range(
            char_start, char_end, offset_mapping, start_search_idx=current_search_idx
        )

        if tok_start >= tok_end:
            line_ppls.append(0.0)
            continue

        # 更新搜索起点：下一行肯定从当前行的 tok_start 开始（或之后）
        current_search_idx = tok_start

        # token_log_probs 是 shifted 的，对应 input_ids[1:]
        # tok_start 对应 input_ids[tok_start]，其 loss 在 token_log_probs[tok_start - 1]
        # 所以该行的 loss 范围是 [tok_start - 1, tok_end - 1)
        loss_start = max(0, tok_start - 1)
        loss_end = min(len(token_log_probs), tok_end - 1)

        if loss_start >= loss_end:
            line_ppls.append(0.0)
            continue

        line_log_probs = token_log_probs[loss_start:loss_end]
        avg_log_prob = line_log_probs.mean().item()
        ppl = math.exp(-avg_log_prob)
        # 转换为 log2 尺度（与 longcodezip 保持一致）
        ppl = math.log2(ppl) if ppl > 0 else 0.0
        line_ppls.append(ppl)

    # ========== 计算自适应阈值 ==========
    valid_ppls = [p for p in line_ppls if not math.isinf(p) and not math.isnan(p)]
    if len(valid_ppls) < 3:
        # 数据太少，无法计算阈值，返回整个 span
        return [(span_char_start, span_char_end)]

    # 计算相邻 diff 分布（用于动态阈值）
    valid_diffs = []
    for i in range(1, len(line_ppls)):
        curr = line_ppls[i]
        prev = line_ppls[i - 1]
        if (
            not math.isinf(curr)
            and not math.isnan(curr)
            and not math.isinf(prev)
            and not math.isnan(prev)
        ):
            valid_diffs.append(abs(curr - prev))

    if not valid_diffs:
        # 无法计算 diff 分布，返回整个 span
        return [(span_char_start, span_char_end)]

    valid_diffs_arr = np.array(valid_diffs)

    # # 计算直方图用于日志展示
    # # 优化：针对长尾分布，限制直方图显示范围到 P99 附近，避免极少数大值导致直方图大部分为空
    # p99 = np.percentile(valid_diffs_arr, 95)
    # max_val = np.max(valid_diffs_arr)
    
    # # 动态设定显示上限：如果 P99 显著小于 Max (例如 10 倍差异)，则截断显示
    # # 这样可以更清晰地看到主要数据的分布情况
    # if p99 > 0 and max_val > 10 * p99:
    #     upper_limit = p99 * 2.0  # 显示范围扩展到 P99 的 2 倍
    # else:
    #     upper_limit = max_val
    
    # if upper_limit == 0:
    #     upper_limit = 1.0

    # # 统计直方图
    # hist_counts, hist_edges = np.histogram(valid_diffs_arr, bins=20, range=(0, upper_limit))
    # outliers_count = np.sum(valid_diffs_arr > upper_limit)

    # hist_lines = []
    # max_count = hist_counts.max() if hist_counts.size > 0 else 1
    
    # for i, count in enumerate(hist_counts):
    #     # 归一化条形长度，最长 30 个字符
    #     bar_len = int(30 * count / max_count)
    #     bar = "#" * bar_len
    #     line = f"  {hist_edges[i]:8.4f} - {hist_edges[i+1]:8.4f} | {count:4d} | {bar}"
    #     hist_lines.append(line)
    
    # if outliers_count > 0:
    #     hist_lines.append(f"  > {upper_limit:8.4f}          | {outliers_count:4d} | (Outliers)")
    
    # hist_str = "\n".join(hist_lines)

    # # Log diff distribution
    # logger.info(
    #     f"PPL Diff Stats:\n"
    #     f"  n={len(valid_diffs_arr)}, mean={np.mean(valid_diffs_arr):.4f}, std={np.std(valid_diffs_arr):.4f}\n"
    #     f"  min={np.min(valid_diffs_arr):.4f}, max={np.max(valid_diffs_arr):.4f}, p50={np.percentile(valid_diffs_arr, 50):.4f}, p90={np.percentile(valid_diffs_arr, 90):.4f}, p99={np.percentile(valid_diffs_arr, 99):.4f}\n"
    #     f"Histogram:\n{hist_str}"
    # )

    # 根据方法计算阈值
    if spike_method == "std":
        mean_val = np.mean(valid_diffs_arr)
        std_val = np.std(valid_diffs_arr)
        threshold = mean_val + spike_threshold_k * std_val
    elif spike_method == "robust_std":
        median_val = np.median(valid_diffs_arr)
        mad = np.median(np.abs(valid_diffs_arr - median_val))
        robust_std = mad * 1.4826
        threshold = median_val + spike_threshold_k * robust_std
    elif spike_method == "iqr":
        q25 = np.percentile(valid_diffs_arr, 25)
        q75 = np.percentile(valid_diffs_arr, 75)
        iqr = q75 - q25
        threshold = q75 + spike_threshold_k * iqr
    elif spike_method == "mad":
        median_val = np.median(valid_diffs_arr)
        mad = np.median(np.abs(valid_diffs_arr - median_val))
        threshold = median_val + spike_threshold_k * mad
    else:
        # 默认使用 std
        mean_val = np.mean(valid_diffs_arr)
        std_val = np.std(valid_diffs_arr)
        threshold = mean_val + spike_threshold_k * std_val

    # ========== 检测 PPL 突变点 ==========
    spike_indices: list[int] = []
    for i in range(1, len(line_ppls) - 1):
        current = line_ppls[i]
        left = line_ppls[i - 1]
        right = line_ppls[i + 1]

        # 跳过无效值
        if math.isinf(current) or math.isnan(current):
            continue
        if math.isinf(left) or math.isnan(left):
            left = current
        if math.isinf(right) or math.isnan(right):
            right = current

        left_diff = current - left
        right_diff = current - right

        # 条件：当前 PPL 显著高于两侧邻居
        if (left_diff >= threshold or right_diff >= threshold) and (
            left_diff >= 0 and right_diff >= 0
        ):
            spike_indices.append(i)

    # ========== 在突变点处切分 block ==========
    # 切分点：在 spike 行之后切分
    split_points = [0] + [idx for idx in spike_indices] + [len(lines)]
    # logger.info(f"PPLs: {len(line_ppls)}, Valid PPLs: {len(valid_ppls)}")
    # logger.info(f"Split points: {len(lines)} lines -> {len(split_points)} split points")

    # 合并过小的 block
    merged_split_points = [split_points[0]]
    for i in range(1, len(split_points)):
        if split_points[i] - merged_split_points[-1] >= min_block_lines:
            merged_split_points.append(split_points[i])
    if merged_split_points[-1] != len(lines):
        merged_split_points.append(len(lines))

    # 生成 block 范围（字符级别）
    block_ranges: list[tuple[int, int]] = []
    for i in range(len(merged_split_points) - 1):
        start_line = merged_split_points[i]
        end_line = merged_split_points[i + 1]

        if start_line >= end_line:
            continue

        # 计算字符范围
        block_char_start = line_char_ranges[start_line][0]
        block_char_end = line_char_ranges[end_line - 1][1]

        # block_char_start/end 已经是绝对位置
        abs_start = block_char_start
        abs_end = block_char_end

        if abs_start < abs_end:
            block_ranges.append((abs_start, abs_end))
    # import pdb; pdb.set_trace()
    return block_ranges


def split_span_into_blocks(
    *,
    full_text: str,
    span: Span,
    method: str = "double_newline",
    token_log_probs: Optional[torch.Tensor] = None,
    offset_mapping: Optional[list[tuple[int, int]]] = None,
    ppl_spike_threshold_k: float = 0.2,
    ppl_spike_method: Literal["std", "robust_std", "iqr", "mad"] = "std",
    ppl_min_block_lines: int = 1,
) -> list[tuple[int, int]]:
    """将 span 切分为多个 block。

    这是 block 切分的统一入口函数，支持多种切分方法。

    参数:
        full_text: 完整的文本（apply_chat_template 的输出）。
        span: 要切分的 Span 对象。
        method: 切分方法，目前支持：
            - "double_newline": 按双换行（\\n{2,}）切分段落（默认）。
            - "ppl": 基于 PPL 突变检测切分语义块。
        token_log_probs: 预计算的每个 token 的 log probability（仅 ppl 方法需要）。
        offset_mapping: token 到字符的映射（仅 ppl 方法需要）。
        ppl_spike_threshold_k: PPL 突变检测的阈值系数（越小越敏感）。
        ppl_spike_method: PPL 阈值计算方法。
        ppl_min_block_lines: 每个 block 的最小行数。

    返回:
        block 范围列表，每个元素为 (char_start, char_end)，相对于 full_text 的绝对位置。
        如果切分结果为空，则返回整个 span 作为一个 block。
    """
    s_start = span.char_start
    s_end = span.char_end
    s_text = full_text[s_start:s_end]

    if method == "double_newline":
        block_ranges = _split_span_into_blocks_double_newline(s_text, s_start)
    elif method == "ppl":
        if token_log_probs is None or offset_mapping is None:
            logger.warning(
                "PPL block split requires token_log_probs/offset_mapping, "
                "falling back to double_newline"
            )
            block_ranges = _split_span_into_blocks_double_newline(s_text, s_start)
            if not block_ranges:
                block_ranges = [(s_start, s_end)]
            return block_ranges

        block_ranges = _split_span_into_blocks_ppl(
            s_text,
            s_start,
            token_log_probs=token_log_probs,
            offset_mapping=offset_mapping,
            spike_threshold_k=ppl_spike_threshold_k,
            spike_method=ppl_spike_method,
            min_block_lines=ppl_min_block_lines,
        )
    else:
        # 未知方法，将整个 span 作为一个 block
        logger.warning(f"Unknown block split method: {method}, using whole span")
        block_ranges = [(s_start, s_end)]

    # 如果没有切分出任何 block，则将整个 span 作为一个 block
    if not block_ranges:
        block_ranges = [(s_start, s_end)]

    return block_ranges


def _compute_block_score(
    token_scores: torch.Tensor,
    token_indices: list[int],
    *,
    method: Literal["mean", "top_pct"] = "top_pct",
    top_pct: float = 0.1,
) -> float:
    """计算 block 的聚合分数。

    支持两种聚合方式：
    - "mean"：简单均值，适合 block 大小较均匀的场景。
    - "top_pct"：Top-pct Mean，取 block 内 top p% 的 token 分数再求均值。
      这种方法可以减少"长 block 被均值稀释"和"短 block 偶然偏高"的问题。

    参数:
        token_scores: 所有 token 的分数张量。
        token_indices: 当前 block 包含的 token 下标列表。
        method: 聚合方式，"mean" 或 "top_pct"。
        top_pct: 当 method="top_pct" 时，取 top 多少比例的 token（默认 0.1 即 10%）。

    返回:
        block 的聚合分数。
    """
    if not token_indices:
        return 0.0

    block_scores = token_scores[token_indices]

    if method == "mean":
        return block_scores.mean().item()

    # top_pct method
    k = max(1, int(len(token_indices) * top_pct))

    # 如果 k >= block 长度，直接返回均值
    if k >= len(token_indices):
        return block_scores.mean().item()

    top_k_scores, _ = torch.topk(block_scores, k)
    return top_k_scores.mean().item()


@dataclass
class MessageCandidate:
    """一个 message 级别的压缩候选。

    用于分层筛选（message_block / message_line）的第一阶段。

    字段说明:
        msg_idx: message 索引（与 request.messages 对齐）。
        score: message 的聚合分数。
        token_count: message 内可压缩区域的 token 总数。
        spans: 该 message 内的所有可压缩 spans。
    """

    msg_idx: int
    score: float
    token_count: int
    spans: list[Span]


def build_message_candidates(
    *,
    spans: List[Span],
    token_scores: torch.Tensor,
    request: CompressRequest,
    head_keep_end: int,
    tail_keep_start: int,
) -> list[MessageCandidate]:
    """构建 message 级别的候选列表。

    遍历所有可压缩的 spans，按 msg_idx 分组，计算每个 message 的聚合分数。

    参数:
        spans: 所有 spans（已完成 token_range 映射）。
        token_scores: 每个 token 的注意力分数。
        request: 原始请求（包含压缩开关）。
        head_keep_end: head 保护的结束 step。
        tail_keep_start: tail 保护的起始 step。

    返回:
        MessageCandidate 列表，每个元素代表一个可压缩的 message。
    """
    # 按 msg_idx 分组 spans
    msg_spans: dict[int, list[Span]] = defaultdict(list)

    for span in spans:
        span_type = span.span_type
        if not is_span_type_compressible(span_type, request):
            continue

        step_idx = span.step_idx
        if is_step_protected(
            step_idx, head_keep_end=head_keep_end, tail_keep_start=tail_keep_start
        ):
            continue

        ts = span.token_start
        te = span.token_end
        if ts is None or te is None or ts >= te:
            continue

        msg_spans[span.msg_idx].append(span)

    # 为每个 message 计算聚合分数
    candidates: list[MessageCandidate] = []
    for msg_idx, spans_in_msg in msg_spans.items():
        all_token_indices: list[int] = []
        for span in spans_in_msg:
            ts = span.token_start
            te = span.token_end
            if ts is not None and te is not None:
                all_token_indices.extend(range(ts, te))

        if not all_token_indices:
            continue

        # 使用与 block 相同的打分方法
        score = _compute_block_score(
            token_scores,
            all_token_indices,
            method=request.block_score_method,
            top_pct=request.block_score_top_pct,
        )

        candidates.append(
            MessageCandidate(
                msg_idx=msg_idx,
                score=score,
                token_count=len(all_token_indices),
                spans=spans_in_msg,
            )
        )

    return candidates


def select_top_messages(
    *,
    message_candidates: list[MessageCandidate],
    message_ratio: float,
    randomize: bool,
) -> set[int]:
    """选择保留的 message 索引集合。

    按分数排序（或随机），选择 top message_ratio 比例的 messages。

    参数:
        message_candidates: MessageCandidate 列表。
        message_ratio: 保留的 message 比例。
        randomize: 是否随机选择。

    返回:
        被选中保留的 msg_idx 集合。
    """
    if not message_candidates:
        return set()

    if randomize:
        random.shuffle(message_candidates)
    else:
        # 按分数降序排序
        message_candidates = sorted(
            message_candidates, key=lambda m: m.score, reverse=True
        )

    # 按 token 数计算保留目标
    total_tokens = sum(m.token_count for m in message_candidates)
    target_tokens = max(1, int(total_tokens * message_ratio))

    selected_msg_indices: set[int] = set()
    current_tokens = 0

    for mc in message_candidates:
        if current_tokens >= target_tokens:
            break
        selected_msg_indices.add(mc.msg_idx)
        current_tokens += mc.token_count

    return selected_msg_indices


def build_hierarchical_candidates(
    *,
    full_text: str,
    spans: List[Span],
    offset_mapping: list[list[int]],
    token_scores: torch.Tensor,
    request: CompressRequest,
    head_keep_end: int,
    tail_keep_start: int,
    token_log_probs: Optional[torch.Tensor] = None,
) -> list[Candidate]:
    """构建分层筛选的候选块（用于 message_block 和 message_line）。

    分层筛选逻辑：
    1. 第一阶段：按 message 级别筛选，选出 top hierarchical_message_ratio 的 messages
    2. 第二阶段：在保留的 messages 中，按 block 或 line 构建候选

    参数:
        full_text: apply_chat_template 生成的完整文本。
        spans: 所有 spans（已完成 token_range 映射）。
        offset_mapping: token 的字符范围列表。
        token_scores: 每个 token 的注意力分数。
        request: 原始请求。
        head_keep_end: head 保护的结束 step。
        tail_keep_start: tail 保护的起始 step。
        token_log_probs: 预计算的每个 token 的 log probability（用于 PPL block 切分）。
    返回:
        candidates 列表（仅包含被选中 messages 内的 block/line 候选）。
    """
    # 第一阶段：构建 message 级候选并选择
    message_candidates = build_message_candidates(
        spans=spans,
        token_scores=token_scores,
        request=request,
        head_keep_end=head_keep_end,
        tail_keep_start=tail_keep_start,
    )

    selected_msg_indices = select_top_messages(
        message_candidates=message_candidates,
        message_ratio=request.hierarchical_message_ratio,
        randomize=request.randomize,
    )

    logger.info(
        f"Hierarchical filtering: selected {len(selected_msg_indices)} messages "
        f"out of {len(message_candidates)} candidates "
        f"(ratio={request.hierarchical_message_ratio})"
    )

    # 第二阶段：在保留的 messages 中构建 block/line 候选
    candidates: list[Candidate] = []

    # 确定第二阶段的 chunking 方法
    if request.chunking_method == "message_block":
        secondary_method = "block"
    elif request.chunking_method == "message_line":
        secondary_method = "line"
    else:
        # Fallback，不应该发生
        secondary_method = "block"

    for span in spans:
        span_type = span.span_type
        if not is_span_type_compressible(span_type, request):
            continue

        step_idx = span.step_idx
        if is_step_protected(
            step_idx, head_keep_end=head_keep_end, tail_keep_start=tail_keep_start
        ):
            continue

        ts = span.token_start
        te = span.token_end
        if ts is None or te is None or ts >= te:
            continue

        span_token_indices = list(range(ts, te))
        if not span_token_indices:
            continue
        
        # 检查该 span 的 message 是否被选中
        if span.msg_idx not in selected_msg_indices:
            # 未被选中的 message，整个 span 作为一个候选，但分数设为 -inf
            # 这样在 select_mask 时会被排到最后，基本不会被保留
            candidates.append(
                Candidate(score=float("-inf"), token_indices=span_token_indices)
            )
            continue

        # tool_call.arguments：按"整个 arguments"做 all-or-nothing 压缩
        if span_type == "tool_call":
            score = token_scores[span_token_indices].mean().item()
            candidates.append(Candidate(score=score, token_indices=span_token_indices))
            continue

        # 根据第二阶段方法构建候选
        if secondary_method == "line":
            _build_line_candidates_for_span(
                full_text=full_text,
                span=span,
                offset_mapping=offset_mapping,
                token_scores=token_scores,
                candidates=candidates,
            )
        elif secondary_method == "block":
            _build_block_candidates_for_span(
                full_text=full_text,
                span=span,
                offset_mapping=offset_mapping,
                token_scores=token_scores,
                request=request,
                candidates=candidates,
                token_log_probs=token_log_probs,
            )

    return candidates


def _build_line_candidates_for_span(
    *,
    full_text: str,
    span: Span,
    offset_mapping: list[list[int]],
    token_scores: torch.Tensor,
    candidates: list[Candidate],
) -> None:
    """为单个 span 构建 line 级候选（原地添加到 candidates）。"""
    ts = span.token_start
    te = span.token_end
    if ts is None or te is None:
        return

    s_start = span.char_start
    s_end = span.char_end
    s_text = full_text[s_start:s_end]

    line_ranges: list[tuple[int, int]] = []
    line_start = 0
    for i, char in enumerate(s_text):
        if char == "\n":
            line_ranges.append((s_start + line_start, s_start + i + 1))
            line_start = i + 1
    if line_start < len(s_text):
        line_ranges.append((s_start + line_start, s_start + len(s_text)))
    if not line_ranges:
        line_ranges = [(s_start, s_end)]

    span_token_indices = list(range(ts, te))
    current_line_idx = 0
    current_line_tokens: list[int] = []

    for t_idx in span_token_indices:
        t_start, _t_end = offset_mapping[t_idx]

        while current_line_idx < len(line_ranges):
            _l_start, l_end = line_ranges[current_line_idx]
            if t_start >= l_end:
                if current_line_tokens:
                    score = token_scores[current_line_tokens].mean().item()
                    candidates.append(
                        Candidate(score=score, token_indices=current_line_tokens)
                    )
                    current_line_tokens = []
                current_line_idx += 1
                continue
            current_line_tokens.append(t_idx)
            break

    if current_line_tokens:
        score = token_scores[current_line_tokens].mean().item()
        candidates.append(Candidate(score=score, token_indices=current_line_tokens))


def _build_block_candidates_for_span(
    *,
    full_text: str,
    span: Span,
    offset_mapping: list[list[int]],
    token_scores: torch.Tensor,
    request: CompressRequest,
    candidates: list[Candidate],
    token_log_probs: Optional[torch.Tensor] = None,
) -> None:
    """为单个 span 构建 block 级候选（原地添加到 candidates）。"""
    ts = span.token_start
    te = span.token_end
    if ts is None or te is None:
        return

    # 使用统一的 block 切分函数
    block_ranges = split_span_into_blocks(
        full_text=full_text,
        span=span,
        method=request.block_split_method,
        token_log_probs=token_log_probs,
        offset_mapping=offset_mapping,
        ppl_spike_threshold_k=request.ppl_spike_threshold_k,
        ppl_spike_method=request.ppl_spike_method,
        ppl_min_block_lines=request.ppl_min_block_lines,
    )

    span_token_indices = list(range(ts, te))
    # 为每个 block 分配 token 并计算分数
    current_block_idx = 0
    current_block_tokens: list[int] = []

    for t_idx in span_token_indices:
        t_start, _t_end = offset_mapping[t_idx]

        while current_block_idx < len(block_ranges):
            _b_start, b_end = block_ranges[current_block_idx]
            if t_start >= b_end:
                # 当前 token 不在这个 block 内，先提交当前 block
                if current_block_tokens:
                    score = _compute_block_score(
                        token_scores,
                        current_block_tokens,
                        method=request.block_score_method,
                        top_pct=request.block_score_top_pct,
                    )
                    candidates.append(
                        Candidate(score=score, token_indices=current_block_tokens)
                    )
                    current_block_tokens = []
                current_block_idx += 1
                continue
            # token 属于当前 block
            current_block_tokens.append(t_idx)
            break

    if current_block_tokens:
        score = _compute_block_score(
            token_scores,
            current_block_tokens,
            method=request.block_score_method,
            top_pct=request.block_score_top_pct,
        )
        candidates.append(Candidate(score=score, token_indices=current_block_tokens))


def build_candidates(
    *,
    full_text: str,
    spans: List[Span],
    offset_mapping: list[list[int]],
    token_scores: torch.Tensor,
    request: CompressRequest,
    head_keep_end: int,
    tail_keep_start: int,
    token_log_probs: Optional[torch.Tensor] = None,
) -> list[Candidate]:
    """构建压缩候选块 candidates。

    candidates 的元素形态为 `(score, token_indices)`：
    - score: 一个 float，用于在后续排序/随机选择。
    - token_indices: 一个 token index 列表，表示该候选块对应的 token 区间。

    该函数会：
    - 仅从允许压缩的 span（由 `is_span_type_compressible` 决定）中生成候选；
    - 跳过 head/tail 保护范围内的 span（由 `is_step_protected` 决定）；
    - 按 `chunking_method` 生成 token/message/line 级候选。

    参数:
        full_text: apply_chat_template 生成的完整文本（用于 line chunking 的按行切分）。
        spans: `extract_spans` 返回并完成 token_range 映射后的 spans。
        offset_mapping: token 的字符范围列表（token -> [char_start, char_end)）。
        token_scores: 每个 token 的注意力分数（可能平滑）。
        request: 原始请求（包含 chunking_method 与压缩开关）。
        head_keep_end: head 保护的结束 step。
        tail_keep_start: tail 保护的起始 step。
        token_log_probs: 预计算的每个 token 的 log probability（用于 PPL block 切分）。

    返回:
        candidates 列表。
    """
    candidates: list[Candidate] = []

    for span in spans:
        span_type = span.span_type
        if not is_span_type_compressible(span_type, request):
            continue

        step_idx = span.step_idx
        # 如果这一步骤在 head/tail 保护范围内，则跳过压缩
        if is_step_protected(
            step_idx, head_keep_end=head_keep_end, tail_keep_start=tail_keep_start
        ):
            continue

        ts = span.token_start
        te = span.token_end
        if ts is None or te is None or ts >= te:
            continue

        span_token_indices = list(range(ts, te))
        if not span_token_indices:
            continue

        # tool_call.arguments：按“整个 arguments”做 all-or-nothing 压缩
        # 无论 chunking_method 如何，都只生成一个候选（整个 span）。
        if span_type == "tool_call":
            score = token_scores[span_token_indices].mean().item()
            candidates.append(Candidate(score=score, token_indices=span_token_indices))
            continue

        if request.chunking_method == "token":
            for idx in span_token_indices:
                candidates.append(
                    Candidate(score=token_scores[idx].item(), token_indices=[idx])
                )

        elif request.chunking_method == "message":
            score = token_scores[span_token_indices].mean().item()
            candidates.append(Candidate(score=score, token_indices=span_token_indices))

        elif request.chunking_method == "line":
            _build_line_candidates_for_span(
                full_text=full_text,
                span=span,
                offset_mapping=offset_mapping,
                token_scores=token_scores,
                candidates=candidates,
            )

        elif request.chunking_method == "block":
            _build_block_candidates_for_span(
                full_text=full_text,
                span=span,
                offset_mapping=offset_mapping,
                token_scores=token_scores,
                request=request,
                candidates=candidates,
                token_log_probs=token_log_probs,
            )

    return candidates


def _knapsack_select(
    candidates: list[Candidate],
    capacity: int,
) -> list[int]:
    """使用 0/1 背包动态规划选择候选块，在 token 预算内最大化总 score。

    参数:
        candidates: 候选块列表，每个候选块包含 score 和 token_indices。
        capacity: token 预算（背包容量）。

    返回:
        被选中的候选块索引列表。
    """
    n = len(candidates)
    if n == 0 or capacity <= 0:
        return []

    # 预计算每个候选块的 weight（token 数量）
    weights = [len(c.token_indices) for c in candidates]
    # 将 score 转换为整数以便 DP（乘以大系数保留精度）
    # 使用原始 float score 进行 DP（Python 支持浮点数比较）
    values = [c.score for c in candidates]

    # dp[w] 表示容量为 w 时能获得的最大 score
    dp: list[float] = [0.0] * (capacity + 1)
    
    # 记录每个状态的选择（用于回溯）
    # picks[i][w] = 1 表示第 i 个物品在容量 w 时被选中
    # 使用 bytearray 以节省内存 (N * Capacity bytes)
    try:
        picks = [bytearray(capacity + 1) for _ in range(n)]
    except MemoryError:
        logger.warning("Knapsack out of memory, falling back to greedy.")
        return _greedy_select(candidates, capacity, False)

    for i in range(n):
        w_i = weights[i]
        v_i = values[i]
        # 逆序遍历避免重复选择同一物品
        for w in range(capacity, w_i - 1, -1):
            if dp[w - w_i] + v_i > dp[w]:
                dp[w] = dp[w - w_i] + v_i
                picks[i][w] = 1

    # 回溯找出被选中的候选块
    selected: list[int] = []
    
    # 找到最优容量（可能不需要用满）
    best_w = 0
    max_val = -1.0
    for w in range(capacity + 1):
        if dp[w] > max_val:
            max_val = dp[w]
            best_w = w
            
    curr_w = best_w
    for i in range(n - 1, -1, -1):
        if picks[i][curr_w]:
            selected.append(i)
            curr_w -= weights[i]

    selected.sort()

    logger.info(
        f"Knapsack selection: capacity={capacity}, selected {len(selected)} candidates, "
        f"total_score={sum(values[i] for i in selected):.4f}, "
        f"total_tokens={sum(weights[i] for i in selected)}"
    )

    return selected


def _greedy_select(
    candidates: list[Candidate],
    target_tokens: int,
    randomize: bool,
) -> list[int]:
    """使用贪心算法选择候选块。

    参数:
        candidates: 候选块列表。
        target_tokens: 目标保留 token 数量。
        randomize: 是否随机选择。

    返回:
        被选中的候选块索引列表。
    """
    n = len(candidates)
    if n == 0 or target_tokens <= 0:
        return []

    indices = list(range(n))
    if randomize:
        random.shuffle(indices)
        logger.info("Randomize enabled: selecting candidates in random order")
    else:
        # 按 score 降序排序索引
        indices.sort(key=lambda i: candidates[i].score, reverse=True)

    selected: list[int] = []
    current_tokens: int = 0
    for i in indices:
        current_tokens += len(candidates[i].token_indices)
        if current_tokens > target_tokens:
            break
        selected.append(i)

    return selected


def select_mask(
    *,
    candidates: list[Candidate],
    attn_ratio: float,
    randomize: bool,
    seq_len: int,
    device: Any,
    selection_method: Literal["greedy", "knapsack"] = "greedy",
    total_tokens_override: Optional[int] = None,
) -> torch.Tensor:
    """根据 candidates 选择要保留的 token，返回 mask。

    支持两种筛选策略：
    - greedy: 贪心算法，按 score 降序依次选择（原有实现）；当 randomize=True 时打乱顺序随机选择。
    - knapsack: 0/1 背包动态规划，在 token 预算内最大化总 score。

    参数:
        candidates: `(score, token_indices)` 列表。
        attn_ratio: 保留比例。
        randomize: 是否随机选择候选（仅对 greedy 方法有效）。
        seq_len: 序列长度（mask 长度）。
        device: mask 所在 device。
        selection_method: 筛选策略，"greedy" 或 "knapsack"。
        total_tokens_override: 可选。显式指定用于计算 target_tokens 的分母（总 token 数）。
            若不提供，则默认使用 sum(len(c.token_indices))。
            这在分层筛选（Candidates 不包含全文）时非常有用，可以保持 attn_ratio 的全局语义。

    返回:
        shape 为 (seq_len,) 的 bool Tensor。
    """
    mask: torch.Tensor = torch.zeros((seq_len,), dtype=torch.bool, device=device)
    
    if total_tokens_override is not None:
        base_tokens = total_tokens_override
    else:
        base_tokens = sum(len(c.token_indices) for c in candidates)
        
    target_tokens: int = max(1, int(base_tokens * attn_ratio))

    if selection_method == "knapsack":
        # 0/1 背包：在 token 预算内最大化总 score
        selected_indices = _knapsack_select(candidates, target_tokens)
        logger.info(f"Using knapsack selection with target_tokens={target_tokens}")
    else:
        # 贪心算法（原有实现）
        selected_indices = _greedy_select(candidates, target_tokens, randomize)
        logger.info(f"Using greedy selection with target_tokens={target_tokens}")
    # import pdb; pdb.set_trace()
    # 根据选中的候选块索引构建 mask
    for idx in selected_indices:
        mask[candidates[idx].token_indices] = True

    return mask


def group_spans_by_msg(spans: List[Span]) -> dict[int, list[Span]]:
    """按 message 索引对 spans 分组。

    参数:
        spans: Span 列表。

    返回:
        {msg_idx: [Span, ...]}。
    """
    msg_to_spans: dict[int, list[Span]] = defaultdict(list)
    for span in spans:
        msg_to_spans[span.msg_idx].append(span)
    return msg_to_spans


# 计算字符串的token数量
def count_token(s: str) -> int:
    resource = AttnResource.instance()
    tokenizer = resource.tokenizer
    tokens = tokenizer.tokenize(s)
    return len(tokens)


def apply_mask_to_messages_and_count_tokens(
    *,
    original_messages: List[Dict[str, Any]],
    compressed_messages: List[Dict[str, Any]],
    spans_by_msg: dict[int, list[Span]],
    mask: torch.Tensor,
    tokenizer: Any,
    input_ids_cpu: torch.Tensor,
    request: CompressRequest,
    head_keep_end: int,
    tail_keep_start: int,
) -> TokenCountStats:
    """将 mask 应用到压缩候选区域，并返回压缩前/后的 token 数。

    说明：
    - 该函数会原地修改 `compressed_messages`（其通常是 original_messages 的 deep copy）。
    - token 统计逻辑与原 endpoint 保持一致：
      - 对 content 计数
      - 对 tool_calls[].function.arguments 计数
    - 写回逻辑与原 endpoint 保持一致：
      - assistant_content/tool_response 写回到 message["content"]
      - tool_call 写回到 message["tool_calls"][tc_idx]["function"]["arguments"]
    """
    old_tokens_count: int = 0
    new_tokens_count: int = 0

    for idx, orig_msg in enumerate(original_messages):
        new_msg: Dict[str, Any] = compressed_messages[idx]
        spans: list[Span] = spans_by_msg.get(idx, [])

        # Calculate original tokens for stats
        msg_old_tokens: int = count_token(orig_msg.get("content") or "")
        for tc in orig_msg.get("tool_calls") or []:
            msg_old_tokens += count_token(tc.get("function", {}).get("arguments") or "")
        old_tokens_count += msg_old_tokens

        # Update fields based on spans
        for span in spans:
            ts = span.token_start
            te = span.token_end
            if ts is None or te is None or ts >= te:
                continue

            span_type = span.span_type
            step_idx = span.step_idx
            is_candidate = is_span_type_compressible(span_type, request)
            if is_candidate and not is_step_protected(
                step_idx, head_keep_end=head_keep_end, tail_keep_start=tail_keep_start
            ):
                keep_indices: list[int] = [i for i in range(ts, te) if mask[i].item()]

                if span_type == "tool_call":
                    # 对整个 arguments：要么全保留，要么全删除（写成 {}）
                    tc_idx = span.tc_idx or 0
                    if len(keep_indices) == (te - ts):
                        # 原原本本保留原始 arguments（避免因 decode/格式化造成差异）
                        orig_tc_list = orig_msg.get("tool_calls") or []
                        if tc_idx < len(orig_tc_list):
                            orig_args = (
                                orig_tc_list[tc_idx]
                                .get("function", {})
                                .get("arguments")
                            )
                            new_content = orig_args if orig_args is not None else "{}"
                        else:
                            new_content = "{}"
                    else:
                        new_content = "{}"
                else:
                    keep_token_ids: torch.Tensor = input_ids_cpu[keep_indices]
                    new_content = tokenizer.decode(
                        keep_token_ids, skip_special_tokens=True
                    )

                if span_type != "tool_call" and len(keep_indices) < (te - ts):
                    if not keep_indices:
                        new_content = "(System reminder: long content deleted for better efficiency)"
                    else:
                        new_content = "(System reminder: compressed for better efficiency)\n" + new_content

                if span_type == "assistant_content" or span_type == "tool_response":
                    new_msg["content"] = new_content
                elif span_type == "tool_call":
                    tc_idx = span.tc_idx or 0
                    if "tool_calls" in new_msg and tc_idx < len(new_msg["tool_calls"]):
                        new_msg["tool_calls"][tc_idx]["function"][
                            "arguments"
                        ] = new_content

        # Calculate new tokens for stats
        msg_new_tokens: int = count_token(new_msg.get("content") or "")
        for tc in new_msg.get("tool_calls") or []:
            msg_new_tokens += count_token(tc.get("function", {}).get("arguments") or "")
        new_tokens_count += msg_new_tokens
    # import pdb; pdb.set_trace()
    return TokenCountStats(old_tokens=old_tokens_count, new_tokens=new_tokens_count)


def normalize_scores_for_viz(raw_token_scores: torch.Tensor) -> torch.Tensor:
    """把 raw_token_scores 归一化到 [0, 1]（log-scale）。

    可视化中 token 的注意力分数通常呈长尾分布；直接线性归一化会导致大多数 token 的
    颜色差异不明显。因此这里对 `raw_token_scores` 做 `log(x + eps)` 后再做 min-max。

    参数:
        raw_token_scores: shape 为 (seq_len,) 的非负分数张量。

    返回:
        shape 为 (seq_len,) 的张量，范围约在 [0, 1]。
    """
    eps = 1e-10
    log_scores = torch.log(raw_token_scores + eps)
    min_log = log_scores.min().item()
    max_log = log_scores.max().item()

    if max_log > min_log:
        return (log_scores - min_log) / (max_log - min_log)
    return torch.zeros_like(raw_token_scores)


def compute_dropped_mask(
    *,
    spans: List[Span],
    mask: torch.Tensor,
    request: CompressRequest,
    head_keep_end: int,
    tail_keep_start: int,
    raw_token_scores: torch.Tensor,
) -> torch.Tensor:
    """为可视化计算 dropped_mask。

    dropped_mask 的语义：
    - 仅在“可压缩区域 + 非保护区域”内，mask 为 False 的 token 视为 dropped。
    - 其它区域 dropped=False。
    """
    dropped_mask = torch.zeros_like(raw_token_scores, dtype=torch.bool)
    for span in spans:
        span_type = span.span_type
        if not is_span_type_compressible(span_type, request):
            continue

        step_idx = span.step_idx
        if is_step_protected(
            step_idx, head_keep_end=head_keep_end, tail_keep_start=tail_keep_start
        ):
            continue

        ts = span.token_start
        te = span.token_end
        if ts is None or te is None or ts >= te:
            continue

        dropped_mask[ts:te] = ~mask[ts:te]
    return dropped_mask


def render_visualization_html(
    *,
    tokenizer: Any,
    full_text: str,
    offset_mapping: list[list[int]],
    input_ids_list: list[int],
    norm_scores_list: list[float],
    raw_scores_list: list[float],
    mask_list: list[bool],
    spans: List[Span],
    request: CompressRequest,
    head_keep_end: int,
    tail_keep_start: int,
) -> str:
    """将 token 分数渲染成 HTML（背景色热力图 + dropped 删除线）。

    注意：删除线在“字符级区间”内渲染。
    原因：某些 tokenizer 会把 `}}` 等合并进同一个 token。
    如果仅按 token 粒度决定 dropped，会把 span 外的字符也一并标为 dropped。
    这里用 offset_mapping + full_text，把每个 token 再切成若干字符子段，只对
    (可压缩 span 的字符范围 ∩ 当前 token 字符范围) 的部分应用删除线。
    """

    def _escape_html(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not intervals:
            return []
        intervals = sorted(intervals)
        merged: list[tuple[int, int]] = []
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    # Build compressible character intervals (non-protected)
    compressible_intervals: list[tuple[int, int]] = []
    for span in spans:
        if not is_span_type_compressible(span.span_type, request):
            continue
        if is_step_protected(
            span.step_idx, head_keep_end=head_keep_end, tail_keep_start=tail_keep_start
        ):
            continue
        if span.char_end > span.char_start:
            compressible_intervals.append((span.char_start, span.char_end))

    compressible_intervals = _merge_intervals(compressible_intervals)

    def _token_segments(
        *, token_char_start: int, token_char_end: int, dropped: bool
    ) -> list[tuple[int, int, bool]]:
        """Split token [start,end) into segments; only overlap with compressible spans can be dropped."""
        if token_char_end <= token_char_start or not compressible_intervals:
            return [(token_char_start, token_char_end, False)]

        segs: list[tuple[int, int, bool]] = []
        cursor = token_char_start
        for s, e in compressible_intervals:
            if e <= cursor:
                continue
            if s >= token_char_end:
                break

            if cursor < s:
                segs.append((cursor, min(s, token_char_end), False))

            inter_start = max(cursor, s, token_char_start)
            inter_end = min(token_char_end, e)
            if inter_end > inter_start:
                segs.append((inter_start, inter_end, dropped))
                cursor = inter_end

            if cursor >= token_char_end:
                break

        if cursor < token_char_end:
            segs.append((cursor, token_char_end, False))

        return [seg for seg in segs if seg[1] > seg[0]]

    html_parts: list[str] = []
    for idx, (tid, norm_score, raw_score, keep_flag) in enumerate(
        zip(input_ids_list, norm_scores_list, raw_scores_list, mask_list)
    ):
        # token may have no char mapping (special tokens)
        if idx < len(offset_mapping):
            t_start, t_end = offset_mapping[idx]
        else:
            t_start, t_end = (0, 0)

        if t_end > t_start and t_end <= len(full_text):
            token_text_raw = full_text[t_start:t_end]
        else:
            token_text_raw = tokenizer.decode([tid])

        bg_color = f"rgba(255, 0, 0, {norm_score:.2f})"
        # dropped only makes sense inside compressible intervals
        dropped = not bool(keep_flag)

        # Split into char-level segments for accurate dropped visualization
        if t_end > t_start and t_end <= len(full_text):
            segments = _token_segments(
                token_char_start=t_start, token_char_end=t_end, dropped=dropped
            )
            for s, e, seg_dropped in segments:
                seg_text = _escape_html(full_text[s:e])
                style = f"background-color: {bg_color};"
                if seg_dropped:
                    style += " text-decoration: line-through; color: rgba(0, 0, 0, 0.4);"
                html_parts.append(
                    f'<span style="{style}" title="Score: {raw_score:.6f} (Norm: {norm_score:.2f})">{seg_text}</span>'
                )
        else:
            # Fallback: no offset mapping; render as a whole token
            token_text = _escape_html(token_text_raw)
            style = f"background-color: {bg_color};"
            html_parts.append(
                f'<span style="{style}" title="Score: {raw_score:.6f} (Norm: {norm_score:.2f})">{token_text}</span>'
            )

    visualization_html = (
        f'<div style="font-family: monospace; white-space: pre-wrap; border: 1px solid #ccc; padding: 10px;">'
        f'<div style="margin-bottom: 10px; color: #666; font-size: 0.8em;">'
        f"Visualization uses Log-Scale normalization. Hover over tokens to see raw scores. "
        f"Dropped tokens are shown with strikethrough and faded text."
        f"</div>"
        f'{"".join(html_parts)}</div>'
    )
    return visualization_html


# --- FastAPI App ---

# request_sem: asyncio.Semaphore = asyncio.Semaphore(1)


# # 限制并发请求数为2
# async def limit_concurrency():
#     async with request_sem:
#         start_time = time.time()
#         try:
#             yield
#         finally:
#             elapsed_time = time.time() - start_time
#             logger.info(f"Request processed in {elapsed_time:.4f} seconds")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    AttnResource.instance()
    yield
    # Cleanup on shutdown
    AttnResource.cleanup_instance_cuda_cache()


app = FastAPI(lifespan=lifespan)


@app.post(
    "/compress",
    response_model=CompressResponse,
)
def compress_messages_endpoint(request: CompressRequest) -> CompressResponse:
    """
    接收压缩请求，利用一个较小的模型计算注意力分数，对消息列表中的特定内容（如工具调用结果）进行压缩。
    主要步骤包括：
    1. 解析请求中的消息列表。
    2. 使用预加载的模型对消息进行分词和前向传播，获取注意力分数。
    3. 根据注意力分数和设定的压缩比例，筛选出关键的 Token。
    4. 重构消息内容，仅保留高关注度的部分，从而减少 Token 数量。
    5. 返回压缩后的消息列表及压缩统计信息。
    """
    original_messages: List[Dict[str, Any]] = request.messages
    request_params = request.model_dump(exclude={"messages"})
    logger.info(f"Received compress request: {len(original_messages)} messages, params: {request_params}")
    start_time = time.time()

    # Persist the raw request messages (overwrite each request)
    write_json_overwrite("log/original_messages.json", original_messages)

    protection_cfg: ProtectionConfig = compute_protection_config(
        request=request,
        original_messages=original_messages,
    )
    msg_step_indices: List[int] = protection_cfg.msg_step_indices
    head_keep_end: int = protection_cfg.head_keep_end
    tail_keep_start: int = protection_cfg.tail_keep_start

    resource: AttnResource = AttnResource.instance()
    tokenizer: Any = resource.tokenizer
    # model: Any = resource.model  # 延迟获取模型实例

    # Apply chat template
    try:
        full_text: str = tokenizer.apply_chat_template(
            original_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception as e:
        # We already persisted original_messages above.
        raise HTTPException(
            status_code=400, detail=f"Error applying chat template: {str(e)}"
        )

    expected_msg_len: int = len(original_messages)
    all_spans: List[Span] = extract_spans(
        full_text,
        expected_msg_len=expected_msg_len,
        msg_step_indices=msg_step_indices,
        messages=original_messages,
    )

    # Tokenize + 将 span 的字符范围映射为 token 范围
    # 先放在 CPU 上，等选定模型实例后再移动
    tok_res: TokenizationResult = tokenize_and_map_spans_to_token_ranges(
        tokenizer=tokenizer,
        full_text=full_text,
        spans=all_spans,
        device="cpu",
    )
    encoded: Any = tok_res.encoded
    offset_mapping: list[list[int]] = tok_res.offset_mapping
    # 保存一份 CPU 上的 input_ids 供后续使用（因为 encoded 可能会被移到 GPU）
    input_ids_cpu: torch.Tensor = encoded["input_ids"][0].clone()

    num_tokens: int = len(offset_mapping)
    logger.info(f"Total tokens: {num_tokens}")

    # Check for max token limit
    use_linear_fallback = False
    if (
        MAX_TOKEN_FOR_MODEL is not None
        and num_tokens > MAX_TOKEN_FOR_MODEL
    ):
        logger.info(
            f"Input tokens {num_tokens} exceeds limit {MAX_TOKEN_FOR_MODEL}. "
            "Using linear fallback and forcing message_line chunking."
        )
        use_linear_fallback = True
        request.chunking_method = "message_line"

    # Compute token scores
    # - randomize=True 或 attn_ratio 为 0/1：不需要调用模型（跳过注意力前向）
    if should_skip_model_forward(request) or use_linear_fallback:
        logger.info(
            f"Skip model forward: randomize={request.randomize}, attn_ratio={request.attn_ratio}, linear_fallback={use_linear_fallback}"
        )
        score_res: TokenScores = build_fallback_token_scores(
            encoded=encoded,
            chunking_method=request.chunking_method,
            randomize=request.randomize,
            linear=use_linear_fallback,
        )
    else:
        # 根据 token 数量选择模型实例
        model, instance_lock = resource.get_model_instance(num_tokens)

        # attention aggregation + align + optional smoothing
        try:
            with instance_lock:
                # 将输入移动到模型所在设备（对于多卡模型，移到 device_map 的第一块卡即可）
                # Use the device of the first layer to ensure correct placement for split models
                if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                     target_device = model.model.embed_tokens.weight.device
                else:
                     target_device = model.device
                
                try:
                    encoded["input_ids"] = encoded["input_ids"].to(target_device)
                    if "attention_mask" in encoded:
                        encoded["attention_mask"] = encoded["attention_mask"].to(target_device)
                except RuntimeError as e:
                    logger.error(f"Failed to move tensors to device {target_device}: {e}")
                    # Log memory status
                    if torch.cuda.is_available():
                        logger.error(f"CUDA Memory Summary: {torch.cuda.memory_summary()}")
                    raise e
            
                score_res = compute_token_scores(
                    model=model,
                    encoded=encoded,
                    chunking_method=request.chunking_method,
                    compute_token_log_probs=request.block_split_method == "ppl",
                    attn_layers=request.attn_layers,
                )
                score_res.token_scores = score_res.token_scores.detach().cpu()
                score_res.raw_token_scores = score_res.raw_token_scores.detach().cpu()
                if score_res.token_log_probs is not None:
                    score_res.token_log_probs = score_res.token_log_probs.detach().cpu()
                
                # 显式清理缓存可能影响并发性能，但在显存紧张时是必要的
                # resource.cleanup_cuda_cache() 
        except ValueError as e:
            if str(e) == "no attentions returned":
                resource.cleanup_cuda_cache()
                resp = CompressResponse(
                    compressed_messages=original_messages,
                    stats={"status": "skipped", "reason": "no attentions returned"},
                )
                write_json_overwrite(
                    "log/compressed_messages.json", resp.compressed_messages
                )
                return resp
            raise

    token_scores = score_res.token_scores
    raw_token_scores = score_res.raw_token_scores
    token_log_probs = score_res.token_log_probs  # 预计算的 token log probs，用于 PPL block 切分

    # Compression Logic
    # 根据 chunking_method 选择候选构建策略
    if request.chunking_method in ("message_block", "message_line"):
        # 分层筛选：先筛选 message，再在保留的 message 中筛选 block/line
        candidates: list[Candidate] = build_hierarchical_candidates(
            full_text=full_text,
            spans=all_spans,
            offset_mapping=offset_mapping,
            token_scores=token_scores,
            request=request,
            head_keep_end=head_keep_end,
            tail_keep_start=tail_keep_start,
            token_log_probs=token_log_probs,
        )
    else:
        # 普通候选构建
        candidates: list[Candidate] = build_candidates(
            full_text=full_text,
            spans=all_spans,
            offset_mapping=offset_mapping,
            token_scores=token_scores,
            request=request,
            head_keep_end=head_keep_end,
            tail_keep_start=tail_keep_start,
            token_log_probs=token_log_probs,
        )

    if candidates:
        avg_tokens = sum(len(c.token_indices) for c in candidates) / len(candidates)
        logger.info(
            f"Splitting stats: Split {sum(len(c.token_indices) for c in candidates)} tokens into {len(candidates)} candidates, avg {avg_tokens:.1f} tokens/candidate"
        )

    if not candidates:
        logger.info("No candidates found for compression.")
        resp = CompressResponse(
            compressed_messages=original_messages,
            stats={"status": "skipped", "reason": "no candidates to compress"},
        )
        write_json_overwrite("log/compressed_messages.json", resp.compressed_messages)
        logger.info(f"Request processed in {time.time() - start_time:.4f} seconds")
        return resp

    mask: torch.Tensor = select_mask(
        candidates=candidates,
        attn_ratio=request.attn_ratio,
        randomize=request.randomize,
        seq_len=int(token_scores.numel()),
        device=token_scores.device,
        selection_method=request.selection_method,
        total_tokens_override=None,
    )

    compressed_messages: List[Dict[str, Any]] = copy.deepcopy(original_messages)

    spans_by_msg: dict[int, list[Span]] = group_spans_by_msg(all_spans)
    token_stats: TokenCountStats = apply_mask_to_messages_and_count_tokens(
        original_messages=original_messages,
        compressed_messages=compressed_messages,
        spans_by_msg=spans_by_msg,
        mask=mask,
        tokenizer=tokenizer,
        input_ids_cpu=input_ids_cpu,
        request=request,
        head_keep_end=head_keep_end,
        tail_keep_start=tail_keep_start,
    )
    old_tokens_count: int = token_stats.old_tokens
    new_tokens_count: int = token_stats.new_tokens

    # --- Visualization Generation ---
    visualization_data: Optional[VisualizationData] = None
    if request.return_visualization:
        norm_scores: torch.Tensor = normalize_scores_for_viz(raw_token_scores)

        input_ids_list: list[int] = input_ids_cpu.tolist()
        norm_scores_list: list[float] = norm_scores.tolist()
        raw_scores_list: list[float] = raw_token_scores.tolist()
        mask_list: list[bool] = mask.tolist()

        visualization_html: str = render_visualization_html(
            tokenizer=tokenizer,
            full_text=full_text,
            offset_mapping=offset_mapping,
            input_ids_list=input_ids_list,
            norm_scores_list=norm_scores_list,
            raw_scores_list=raw_scores_list,
            mask_list=mask_list,
            spans=all_spans,
            request=request,
            head_keep_end=head_keep_end,
            tail_keep_start=tail_keep_start,
        )
        visualization_data = VisualizationData(
            html=visualization_html, token_scores=raw_scores_list
        )

    logger.info(
        f"Compression completed: old tokens={old_tokens_count}, new tokens={new_tokens_count}, ratio={(new_tokens_count / old_tokens_count) if old_tokens_count > 0 else 1.0:.4f}, chunking_method={request.chunking_method}"
    )
    resp = CompressResponse(
        compressed_messages=compressed_messages,
        stats={
            "status": "success",
            "old_tokens": old_tokens_count,
            "new_tokens": new_tokens_count,
            "ratio": (
                new_tokens_count / old_tokens_count if old_tokens_count > 0 else 1.0
            ),
        },
        visualization=visualization_data,
    )

    # Persist the compressed messages (overwrite each request)
    write_json_overwrite("log/compressed_messages.json", resp.compressed_messages)
    logger.info(f"Request processed in {time.time() - start_time:.4f} seconds")
    return resp


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Attention Compress Service")
    parser.add_argument("--port", type=int, default=46405, help="Port to run the service on")
    parser.add_argument(
        "--gpu-config",
        type=str,
        help="GPU deployment config in JSON format, e.g. '[[0], [1]]' or '[[0,1], [2,3]]'",
    )
    parser.add_argument(
        "--max-token-for-model",
        type=int,
        help="Max input tokens to use the model. If exceeded, use linear fallback.",
    )
    args = parser.parse_args()

    if args.max_token_for_model is not None:
        MAX_TOKEN_FOR_MODEL = args.max_token_for_model
        logger.info(f"Overridden MAX_TOKEN_FOR_MODEL via args: {MAX_TOKEN_FOR_MODEL}")

    if args.gpu_config:
        try:
            # Parse JSON config
            config = json.loads(args.gpu_config)
            if not isinstance(config, list):
                raise ValueError("Config must be a list")

            # Update global configuration
            MODEL_DEPLOYMENT_CONFIG.clear()
            MODEL_DEPLOYMENT_CONFIG.extend(config)
            logger.info(
                f"Overridden MODEL_DEPLOYMENT_CONFIG via args: {MODEL_DEPLOYMENT_CONFIG}"
            )
        except Exception as e:
            logger.error(f"Failed to parse --gpu-config: {e}")
            exit(1)

    uvicorn.run(app, host="0.0.0.0", port=args.port)

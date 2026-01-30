from __future__ import annotations

from utils.llm_polytool import get_llm_response

import os
import time
import typing
import tiktoken
import lz4.frame
import json
import random
import sys
import httpx
from collections import defaultdict

analysis_args = json.loads(os.environ['TRAJ_ANALYSIS'].strip())
print('-- analysis args:', analysis_args)

# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
_llm_lingua = None
def get_lingua():
    global _llm_lingua
    if not _llm_lingua:
        from llmlingua import PromptCompressor
        _llm_lingua = PromptCompressor(
            # model_name='microsoft/llmlingua-2-xlm-roberta-large-meetingbank',
            use_llmlingua2=True,
            device_map='cpu',
        )
    return _llm_lingua

def count_comp(b):
    return len(lz4.frame.compress(b))

_token_encoding = tiktoken.encoding_for_model('gpt-4o')
def count_token(s):
    #return len(s)
    return len(_token_encoding.encode(s))

if typing.TYPE_CHECKING:
    from .expert import MessageManager

SYS_PROMPT = """
You will analyze and compress a given step in a trajectory of an AI agent solving a software bug.

In the trajectory, each step is marked in <step id="..."></step>.
The agent will think in <think>, call external tools as marked in <call tool="..."></call>. Its result is marked in <result></result> within the <step> tag.

Your job is to compress the text within the given id to avoid harming efficiency, typically shortening it to 20%-50% of the original length.
Meanwhile, keep the compressed text useful such that you are able to continue the trajectory as close as the original path.

- You should ONLY remove redundant texts, which are either irrelevant to future steps or duplicated by other texts in the trajectory.
- Replace the text to remove to "..." and a short takeaway, e.g. "... (same as the content below)".
- You should keep the original structure unchanged, e.g., XML tags, Python indentation and line numbers.
- Again, keep useful details in the original content unchanged, e.g., XML tags, Python indentation and line numbers.

Typical examples:
- If the step opens a huge file but only one part is necessary for future steps, replace other parts to "... (unrelated function XXX, YYY)".
- If the step runs a verbose test script and everything goes fine, replace the verbose part to "... (expected output)".
- If the step uses str_replace_editor to modify a file and the content can be inferred by the content after it, replace the tool call argument to "... (see results below)".

You should only process the text within the <step> tag with the given id. STOP OUTPUT IMMEDIATELY AFTER </step>.
""".strip()

#以下是论文中的prompt
LLM_SUMMARY_PROMPT = """You are maintaining a context-aware state summary for an interactive agent working on software engineering tasks (specifically, bug fixes within the SWE-bench framework). You will be given a list of turns corresponding to actions taken by the agent and their resulting observations, and the most recent previous summary if one exists. Track:

USER_CONTEXT: (Preserve essential user requirements, goals, and clarifications in concise form)

COMPLETED: (Sub-tasks completed so far, with brief results)
PENDING: (Key sub-tasks that still need to be done)
CURRENT_STATE: (Current variables, data structures, or relevant environmental state not covered elsewhere)
CODE_STATE: (File paths, function signatures, data structures followed by their current state)
TESTS: (Failing cases, error messages, outputs)
CHANGES: (Code edits, variable updates)
DEPS: (Dependencies, imports, external calls)

PRIORITIZE:

1. Capture key user requirements and goals (from the initial issue)
2. Distinguish between completed and pending sub-tasks clearly
3. Keep all sections concise and relevant to fixing the issue
4. Focus on information that quantifies the agent's progress towards a solution

Example format:

USER_CONTEXT: Fix a ZeroDivisionError in utils.calculate_ratio when denominator is zero. The function should return 0 in this case.
COMPLETED:
1. Reproduced ZeroDivisionError with calculate_ratio(5, 0).
2. Identified missing check for denominator == 0 in utils.py.
3. Added test case test_calculate_ratio_zero_denominator to tests/test_utils.py.
4. Modified utils.calculate_ratio to return 0 if denominator is 0.
5. Verified test_calculate_ratio_zero_denominator now passes.
6. All tests in tests/test_utils.py are passing. 
PENDING: 
1. Prepare final patch file for submission. 
CODE_STATE: 
1. app/utils.py: calculate_ratio function MODIFIED.
2. tests/test_utils.py: test_calculate_ratio_zero_denominator ADDED. 
TESTS:
1. tests/test_utils.py::test_calculate_ratio_zero_denominator: PASSED (previously FAILED).
2. All tests in tests/test_utils.py: PASSED. 
CHANGES:
1. app/utils.py: calculate_ratio(): Added conditional if denominator == 0: return 0.
2. tests/test_utils.py: Added test_calculate_ratio_zero_denominator to assert behavior with zero denominator. 
DEPS: None modified.
"""

# 以下是gemini cli的llm summary prompt
LLM_SUMMARY_PROMPT_GEMINI_CLI = r"""
You are the component that summarizes internal chat history into a given structure.

When the conversation history grows too large, you will be invoked to distill the entire history into a concise, structured XML snapshot. This snapshot is CRITICAL, as it will become the agent's *only* memory of the past. The agent will resume its work based solely on this snapshot. All crucial details, plans, errors, and user directives MUST be preserved.

First, you will think through the entire history in a private <scratchpad>. Review the user's overall goal, the agent's actions, tool outputs, file modifications, and any unresolved questions. Identify every piece of information that is essential for future actions.

After your reasoning is complete, generate the final <state_snapshot> XML object. Be incredibly dense with information. Omit any irrelevant conversational filler.

The structure MUST be as follows:

<state_snapshot>
    <overall_goal>
        <!-- A single, concise sentence describing the user's high-level objective. -->
        <!-- Example: "Refactor the authentication service to use a new JWT library." -->
    </overall_goal>

    <key_knowledge>
        <!-- Crucial facts, conventions, and constraints the agent must remember based on the conversation history and interaction with the user. Use bullet points. -->
        <!-- Example:
         - Build Command: \`npm run build\`
         - Testing: Tests are run with \`npm test\`. Test files must end in \`.test.ts\`.
         - API Endpoint: The primary API endpoint is \`https://api.example.com/v2\`.
         
        -->
    </key_knowledge>

    <file_system_state>
        <!-- List files that have been created, read, modified, or deleted. Note their status and critical learnings. -->
        <!-- Example:
         - CWD: \`/home/user/project/src\`
         - READ: \`package.json\` - Confirmed 'axios' is a dependency.
         - MODIFIED: \`services/auth.ts\` - Replaced 'jsonwebtoken' with 'jose'.
         - CREATED: \`tests/new-feature.test.ts\` - Initial test structure for the new feature.
        -->
    </file_system_state>

    <recent_actions>
        <!-- A summary of the last few significant agent actions and their outcomes. Focus on facts. -->
        <!-- Example:
         - Ran \`grep 'old_function'\` which returned 3 results in 2 files.
         - Ran \`npm run test\`, which failed due to a snapshot mismatch in \`UserProfile.test.ts\`.
         - Ran \`ls -F static/\` and discovered image assets are stored as \`.webp\`.
        -->
    </recent_actions>

    <current_plan>
        <!-- The agent's step-by-step plan. Mark completed steps. -->
        <!-- Example:
         1. [DONE] Identify all files using the deprecated 'UserAPI'.
         2. [IN PROGRESS] Refactor \`src/components/UserProfile.tsx\` to use the new 'ProfileAPI'.
         3. [TODO] Refactor the remaining files.
         4. [TODO] Update tests to reflect the API change.
        -->
    </current_plan>
</state_snapshot>
""".strip()

##### BEGIN ARGS

MODE = analysis_args['mode'].strip()

FIX_MODEL = analysis_args.get('fix_model', 'claude4-sonnet').strip()

#MODEL = analysis_args.get('model', 'gemini-2.5-flash').strip()
MODEL = analysis_args.get('model', 'gpt-5-mini-2025-08-07').strip()
LINGUA_RATIO = float(analysis_args.get('lingua_ratio', .25))

BYPASS_FILTER = 'gpt-5-' in MODEL

N_CTX_BEFORE = int(analysis_args.get('ctx_before', 1))
N_CTX_AFTER = int(analysis_args.get('ctx_after', 2))

SHOW_CTX = bool(int(analysis_args.get('show_ctx', 1)))
USE_LZ4 = bool(int(analysis_args.get('use_lz4', 0)))

THRESHOLD_TOKENS = int(analysis_args.get('threshold', 500))

SUMMARY_MODEL = analysis_args.get('summary_model', MODEL).strip()
SUMMARY_N = int(analysis_args.get('summary_n', 21))
SUMMARY_TAIL = int(analysis_args.get('summary_tail', 10))

OBSERVATION_WINDOW = int(analysis_args.get('obs_window', 2))
_default_placeholder = 'Environment Observation Ommited'
OBSERVATION_PLACEHOLDER = analysis_args.get('obs_placeholder', _default_placeholder).strip()
OBS_MASKING_THRESHOLD_TOKENS = int(analysis_args.get('obs_masking_threshold_tokens', 300))

ATTN_TAIL = int(analysis_args.get('attn_tail', 2)) #保留末尾若干项不变
ATTN_ROLLING_M = int(analysis_args.get('attn_rolling_m', 0)) # 滚动压缩步长
ATTN_MASK_HISTORY = bool(int(analysis_args.get('attn_mask_history', 0))) # 是否mask滚动窗口之前的msg
ATTN_REFRESH_FIXED = bool(int(analysis_args.get('attn_refresh_fixed', 0))) # 是否在滚动时刷新固定区
ATTN_RATIO = float(analysis_args.get('attn_ratio', 0.20))
ATTN_SERVICE_URL = analysis_args.get('attn_service_url', 'http://100.69.30.24:46405/compress').strip()
ATTN_SERVICE_TIMEOUT = float(analysis_args.get('attn_service_timeout', 600))
ATTN_CHUNKING_METHOD = analysis_args.get('attn_chunking_method', 'token').strip() # "token", "line", "message"
ATTN_COMPRESS_TOOL_RESPONSE = bool(int(analysis_args.get('attn_compress_tool_response', 1)))
ATTN_COMPRESS_TOOL_CALL = bool(int(analysis_args.get('attn_compress_tool_call', 0)))
ATTN_COMPRESS_ASSISTANT_CONTENT = bool(int(analysis_args.get('attn_compress_assistant_content', 0)))
ATTN_RANDOMIZE = bool(int(analysis_args.get('attn_randomize', 0)))
BLOCK_SPLIT_METHOD = analysis_args.get('attn_block_split_method', 'double_newline').strip() # "double_newline", "ppl"
BLOCK_SCORE_METHOD = analysis_args.get('attn_block_score_method', 'mean').strip() # "mean", "top_pct"
SELECTION_METHOD = analysis_args.get('attn_selection_method', 'greedy').strip() # "greedy", "knapsack"
PPL_SPIKE_THRESHOLD_K = float(analysis_args.get('attn_ppl_spike_threshold_k', 1.2))
PPL_SPIKE_METHOD = analysis_args.get('attn_ppl_spike_method', 'iqr').strip() # "std", "robust_std", "iqr", "mad"
ATTN_LAYERS = analysis_args.get('attn_layers', None) # None, int, list[int], "low", "middle", "high"
if ATTN_LAYERS and isinstance(ATTN_LAYERS, str):
    ATTN_LAYERS = ATTN_LAYERS.strip()


def _format_observation_placeholder(masked_count: int | None = None) -> str:
    template = OBSERVATION_PLACEHOLDER or _default_placeholder
    try:
        return template.format(window=OBSERVATION_WINDOW, masked=masked_count)
    except Exception:
        return template

def _call_attn_service(payload: dict[str, typing.Any]) -> dict[str, typing.Any]:
    try:
        response = httpx.post(
            ATTN_SERVICE_URL,
            json=payload,
            timeout=ATTN_SERVICE_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        raise RuntimeError(f'attn service http {exc.response.status_code}: {detail}') from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f'attn service request failed: {exc}') from exc

    try:
        return response.json()
    except ValueError as exc:
        body = response.text
        raise RuntimeError(f'attn service invalid json: {body[:200]}') from exc

##### END ARGS

def perform_analysis_step_llmlingua(mgr: MessageManager):
    idx = mgr.count_turn() - 1 - N_CTX_AFTER
    traj_content = mgr.extract_step_into_traj(idx)

    new_content = get_lingua().compress_prompt(traj_content, rate=LINGUA_RATIO, force_tokens=['\n', '?'])

    mgr.metrics['erase_tot_count'] += 1
    mgr.metrics['erase_in_tokens'] += count_token(traj_content)
    mgr.metrics['erase_out_tokens'] += count_token(new_content['compressed_prompt'])

    mgr.perform_erase_step(idx, new_content['compressed_prompt'], traj_content)

def perform_analysis_step_random_drop(mgr: MessageManager):
    idx = mgr.count_turn() - 1 - N_CTX_AFTER
    traj_content = mgr.extract_step_into_traj(idx)

    def drop_tokens(ts: list[int], n: float) -> tuple[str, int]:
        deletable_indices = []
        for ind, t in enumerate(ts):
            try:
                _token_encoding.decode_single_token_bytes(t).decode()
            except UnicodeDecodeError:
                pass
            else:
                deletable_indices.append(ind)

        deleted_indices = random.sample(deletable_indices, min(int(n), len(deletable_indices)))
        remaining_tokens = [t for ind, t in enumerate(ts) if ind not in deleted_indices]

        print('tot', len(ts), 'deletable', len(deletable_indices), 'deleted', len(deleted_indices), 'remaining', len(remaining_tokens))

        return _token_encoding.decode(remaining_tokens), len(remaining_tokens)

    old_tokens = _token_encoding.encode(traj_content)
    new_content, len_new_tokens = drop_tokens(old_tokens, len(old_tokens) * (1-LINGUA_RATIO))

    mgr.metrics['erase_tot_count'] += 1
    mgr.metrics['erase_in_tokens'] += len(old_tokens)
    mgr.metrics['erase_out_tokens'] += len_new_tokens

    mgr.perform_erase_step(idx, new_content, traj_content)


def perform_analysis_step_delete(mgr: MessageManager):
    idx = mgr.count_turn() - 1 - N_CTX_AFTER
    traj_content = mgr.extract_step_into_traj(idx)

    mgr.metrics['erase_tot_count'] += 1
    mgr.metrics['erase_in_tokens'] += count_token(traj_content)
    mgr.metrics['erase_out_tokens'] += 0

    mgr.perform_erase_step(idx, None, traj_content)


def perform_analysis_step_ours(mgr: MessageManager):
    idx = mgr.count_turn() - 1 - N_CTX_AFTER
    traj_content = [
        mgr.extract_step_into_traj(i, BYPASS_FILTER) # -1 is user prompt
        for i in range(idx - N_CTX_BEFORE, idx + N_CTX_AFTER + 1)
    ]

    if not SHOW_CTX:
        for i in range(len(traj_content)):
            if i != N_CTX_BEFORE:
                traj_content[i] = ''

    sys_prompt = SYS_PROMPT

    str_traj_content = '\n'.join(traj_content)

    if BYPASS_FILTER:
        sys_prompt = sys_prompt.replace('think', 'talk')
        sys_prompt = sys_prompt.replace('agent', 'engineer')

    user_prompt = f'{str_traj_content}\n\nNow, compress the step {idx}.'

    if mgr.USE_CACHING:
        sys_prompt = [{
            'type': 'text',
            'text': sys_prompt,
            'cache_control': {'type': 'ephemeral'},
        }]

    msgs = [
        {
            'role': 'system',
            'content': sys_prompt,
        },
        {
            'role': 'user',
            'content': user_prompt,
        },
        {
            'role': 'assistant',
            'content': f'Sure. Here is the compressed content of step {idx}: <step id="{idx}">',
        }
    ]

    # kwargs = dict(temperature = 0.0, n = 1, stop='</step>')
    # if 'qwen3-235b-a22b-instruct-2507' in MODEL: # overcome gateway timeout
    #     kwargs['stream'] = True
    kwargs = dict(temperature = 0.7, n = 1)

    expert_answer_list, finish_reason_list, usage = get_llm_response(MODEL, msgs, [], kwargs)
    if usage["completion_tokens"] is None:
        print('wtf analysis result', usage, expert_answer_list, finish_reason_list, finish_reason_list)
        return

    mgr.metrics['analysis_cost_tokens'] += usage['total_tokens']
    mgr.metrics['analysis_prompt_tokens'] += usage["prompt_tokens"]
    mgr.metrics['analysis_completion_tokens'] += usage["completion_tokens"]

    expert_answer = expert_answer_list[0]

    #print(expert_answer['content'])

    content, splitter, leftover_content = expert_answer['content'].partition('</step>')

    if not splitter:
        if 'stop' not in finish_reason_list:
            print('!!! invalid eraser', finish_reason_list, expert_answer)
            return

    if '<step' in content[:200]:
        content = content.partition('<step')[2]
        if '>' in content[:20]:
            content = content.partition('>')[2]

    old_tokens = count_token(traj_content[N_CTX_BEFORE])
    new_tokens = count_token(content)

    if old_tokens - new_tokens >= 400 or new_tokens < .8 * old_tokens:
        print('!!! erased', old_tokens, '->', new_tokens)

        mgr.metrics['erase_tot_count'] += 1
        mgr.metrics['erase_in_tokens'] += old_tokens
        mgr.metrics['erase_out_tokens'] += new_tokens

        mgr.perform_erase_step(idx, content, traj_content[N_CTX_BEFORE])
    else:
        print('  no erase', old_tokens, '->', new_tokens)

def perform_analysis_step_obs_masking(mgr: MessageManager):
    if OBSERVATION_WINDOW <= 0:
        print('obsmask disabled, window <= 0')
        return 0

    idx = mgr.count_turn()
    target_idx = idx - OBSERVATION_WINDOW

    if target_idx < 0:
        print('obsmask skipped, not enough steps yet')
        return 0
    #判断一下target_idx这一步的observation token数是否超过阈值，若不超过则不删
    old_tokens = count_token(mgr.extract_step_into_traj(target_idx, BYPASS_FILTER))
    if old_tokens < OBS_MASKING_THRESHOLD_TOKENS:
        print('obsmask skipped, observation tokens below threshold at step', target_idx)
        return 0

    placeholder = _format_observation_placeholder()
    masked_contents = mgr.perform_obs_masking_step(target_idx, placeholder)

    if not masked_contents:
        print('obsmask skipped, nothing to mask at step', target_idx)
        return 0

    new_tokens = count_token(placeholder)
    for content in masked_contents:
        old_tokens = count_token(content)
        mgr.metrics['erase_tot_count'] += 1
        mgr.metrics['erase_in_tokens'] += old_tokens
        mgr.metrics['erase_out_tokens'] += new_tokens

    print('obsmask masked', len(masked_contents), 'observations at step', target_idx)
    return 1

def perform_analysis_step_llm_summary(mgr: MessageManager) -> bool:
    if SUMMARY_N <= 0:
        print('llm_summary disabled, summary_n <= 0')
        return False

    total_steps = mgr.count_turn() #这个步数是压缩过的步数
    start_idx = 0
    end_idx = total_steps - SUMMARY_TAIL - 1

    if end_idx - start_idx + 1 < SUMMARY_N:
        print('llm_summary skipped (not enough steps)', end_idx - start_idx + 1)
        return False

    prior_summary_text = None
    if getattr(mgr, 'summary_has_result', False):
        for step in mgr.steps:
            if step and step[0].get('agent_keep_content'):
                prior_summary_text = step[0].get('content', '')
                break

    base_context = prior_summary_text or mgr.user_message.get('content', '')
    has_prior_summary_ctx = prior_summary_text is not None
    content_start_idx = start_idx + 1 if has_prior_summary_ctx else start_idx

    if end_idx < content_start_idx:
        print('llm_summary skipped (no new steps)')
        return False

    full_span_content = [
        mgr.extract_step_into_traj(i, BYPASS_FILTER)
        for i in range(start_idx, end_idx + 1)
    ]

    traj_content = [
        mgr.extract_step_into_traj(i, BYPASS_FILTER)
        for i in range(content_start_idx, end_idx + 1)
    ]

    orig_concat = '\n'.join(full_span_content)
    llm_input_concat = '\n'.join(traj_content)
    sys_prompt = LLM_SUMMARY_PROMPT
    if mgr.USE_CACHING:
        sys_prompt = [{
            'type': 'text',
            'text': LLM_SUMMARY_PROMPT,
            'cache_control': {'type': 'ephemeral'},
        }]

    user_prompt = ""
    if prior_summary_text:
        user_prompt += f"<PREVIOUS SUMMARY>:\n{prior_summary_text}\n </PREVIOUS SUMMARY>\n\n"
    user_prompt += llm_input_concat

    kwargs = dict(temperature=0.7, n=1)
    expert_answer_list, finish_reason_list, usage = get_llm_response(
        SUMMARY_MODEL,
        [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        [],
        kwargs
    )
    if usage["completion_tokens"] is None:
        print('llm_summary invalid usage', usage)
        return False

    expert_answer = expert_answer_list[0]
    summary_text = (expert_answer.get('content') or '').strip()
    if not summary_text:
        print('llm_summary empty response')
        return False

    old_tokens = sum(count_token(c) for c in full_span_content)
    seen_tokens = sum(count_token(c) for c in traj_content) + count_token(base_context)
    new_tokens = count_token(summary_text)

    mgr.metrics['analysis_cost_tokens'] += usage['total_tokens']
    mgr.metrics['analysis_prompt_tokens'] += usage["prompt_tokens"]
    mgr.metrics['analysis_completion_tokens'] += usage["completion_tokens"]
    mgr.metrics['seen_tokens'] += seen_tokens

    mgr.metrics['erase_tot_count'] += 1
    mgr.metrics['erase_in_tokens'] += old_tokens
    mgr.metrics['erase_out_tokens'] += new_tokens

    mgr.perform_summary_step(start_idx, end_idx, summary_text, orig_concat)
    print('llm_summary replaced steps', start_idx, '-', end_idx, 'with', new_tokens, 'tokens')
    return True

def _update_rolling_head(mgr: MessageManager) -> int:
    """Updates attn_fixed_head based on the rolling strategy."""
    if not hasattr(mgr, 'attn_fixed_head'):
        mgr.attn_fixed_head = 0
    
    if ATTN_ROLLING_M <= 0:
        return mgr.attn_fixed_head

    total_steps = len(mgr.steps)
    compress_end = total_steps - ATTN_TAIL
    
    if ATTN_MASK_HISTORY:
        # Mode 2 (Rolling Mask): Sliding Window Strategy
        # The window size is kept constant at ATTN_ROLLING_M.
        # Head moves at every step to maintain the window size.
        # [Masked History (0...Head)] + [Active Window (Head...Head+M)] + [Tail]
        mgr.attn_fixed_head = max(0, compress_end - ATTN_ROLLING_M)
            
    else:
        # Mode 3 (Rolling Fixed): Step-wise Strategy
        # The window grows until it reaches ATTN_ROLLING_M, then we advance Head.
        # This reduces the frequency of "fixing" (and potentially refreshing) the history.
        current_window_len = compress_end - mgr.attn_fixed_head
        
        if current_window_len >= ATTN_ROLLING_M:
            mgr.attn_fixed_head += ATTN_ROLLING_M
            print(f'attn compress rolling (fixed): advanced head to {mgr.attn_fixed_head} (accumulated {current_window_len} steps)')
        
    return mgr.attn_fixed_head

def _mask_history_steps(mgr: MessageManager, head_idx: int):
    """Masks observations in steps before head_idx."""
    if head_idx <= 0:
        return
    count_masked = 0
    
    # We try to mask every step before head
    
    # Check tokens
    try:
        content_str = mgr.extract_step_into_traj(head_idx-1, BYPASS_FILTER)
    except Exception:
        return
        
    old_tokens = count_token(content_str)
    # Implicitly checks if already masked because placeholder is short (< THRESHOLD)
    if old_tokens < OBS_MASKING_THRESHOLD_TOKENS:
        return
    
    placeholder = _format_observation_placeholder()
    
    # Perform mask
    masked_contents = mgr.perform_obs_masking_step(head_idx-1, placeholder)
    
    if masked_contents:
        new_tokens = count_token(placeholder)
        for content in masked_contents:
            mgr.metrics['erase_tot_count'] += 1
            mgr.metrics['erase_in_tokens'] += count_token(content)
            mgr.metrics['erase_out_tokens'] += new_tokens
        count_masked += 1

    if count_masked > 0:
        print(f'attn compress rolling: masked {count_masked} steps before head {head_idx}')

def _merge_local_and_service_messages(mgr: MessageManager, service_msgs: list, service_head: int):
    """Refactored logic for merging preserved local messages and service messages."""
    num_preserved_msgs = 0
    if service_head > 0:
        for step in mgr.steps[:service_head]:
            num_preserved_msgs += len(step)

    if num_preserved_msgs > 0:
        current_step_msgs = []
        for step in mgr.steps:
            for msg in step:
                current_step_msgs.append(msg)
        
        if len(current_step_msgs) >= num_preserved_msgs and len(service_msgs) >= num_preserved_msgs:
            # 拼接: Head(Preserved Local) + Body/Tail(Service)
            service_msgs = current_step_msgs[:num_preserved_msgs] + service_msgs[num_preserved_msgs:]
            mgr.replace_steps_with_compressed_messages(service_msgs)
        else:
            print(f'attn compress warning: msg len mismatch during merge local={len(current_step_msgs)} service={len(service_msgs)} preserved={num_preserved_msgs}')
            mgr.replace_steps_with_compressed_messages(service_msgs)
    else:
        mgr.replace_steps_with_compressed_messages(service_msgs)
    
    return service_msgs

def perform_analysis_step_attention_compress(mgr: MessageManager):
    # 初始化 head
    if not hasattr(mgr, 'attn_fixed_head'):
        mgr.attn_fixed_head = 0
    old_fixed_head = mgr.attn_fixed_head

    # 区分三种模式：
    # 1. 动态全压缩 (Dynamic Full Compression): ATTN_ROLLING_M <= 0
    # 2. 滚动 Mask (Rolling Mask): ATTN_ROLLING_M > 0 and ATTN_MASK_HISTORY = True
    # 3. 滚动压缩 (Rolling Fixed): ATTN_ROLLING_M > 0 and ATTN_MASK_HISTORY = False
    
    service_head = 0

    if ATTN_ROLLING_M <= 0:
        # Mode 1: Dynamic Full Compression
        # 每次都全量压缩，Head 始终为 0
        mgr.attn_fixed_head = 0
        service_head = 0
    else:
        # Rolling Logic
        mgr.attn_fixed_head = _update_rolling_head(mgr)
        
        if ATTN_MASK_HISTORY:
            # Mode 2: Rolling Mask
            # Mask 掉 Head 之前的消息
            _mask_history_steps(mgr, mgr.attn_fixed_head)
            
            # Service 只需要处理 Head 之后的消息
            service_head = mgr.attn_fixed_head
            
        else:
            # Mode 3: Rolling Fixed
            # 如果开启了 Refresh 且发生了滚动，则重置 Service Head 为 0 以刷新固定区
            if ATTN_REFRESH_FIXED and mgr.attn_fixed_head > old_fixed_head:
                service_head = 0
            else:
                service_head = mgr.attn_fixed_head

    # 获取原始消息列表 (Masking applied if Mode 2)
    # original_messages最前面两项分别是初始system prompt和初始user message，之后无需压缩。
    original_messages = mgr.extract_format_messages(fixed_head=service_head)
    if not original_messages:
        print('attn compress skipped (empty messages)')
        return

    # 记录每条message对应的step idx，便于尾部保留
    msg_step_indices = [-1, -1]  # system + user
    for step_idx, step in enumerate(mgr.steps):
        msg_step_indices.extend([step_idx] * len(step))
    if len(msg_step_indices) != len(original_messages):
        print('attn compress warning: step index alignment mismatch', len(msg_step_indices), len(original_messages))
        raise RuntimeError('step index alignment mismatch')

    payload = {
        'messages': original_messages,
        'attn_ratio': ATTN_RATIO,
        "hierarchical_message_ratio": min(ATTN_RATIO*2, 1.0),
        'attn_head': service_head,
        'attn_tail': ATTN_TAIL,
        'chunking_method': ATTN_CHUNKING_METHOD,
        'compress_tool_response': ATTN_COMPRESS_TOOL_RESPONSE,
        'compress_tool_call': ATTN_COMPRESS_TOOL_CALL,
        'compress_assistant_content': ATTN_COMPRESS_ASSISTANT_CONTENT,
        'step_indices': msg_step_indices,
        'randomize': ATTN_RANDOMIZE,
        'block_score_method': BLOCK_SCORE_METHOD, # "mean", "top_pct"
        'block_split_method': BLOCK_SPLIT_METHOD, # "double_newline", "ppl"
        'selection_method': SELECTION_METHOD, # "greedy", "knapsack"
        'ppl_spike_threshold_k': PPL_SPIKE_THRESHOLD_K,
        'ppl_spike_method': PPL_SPIKE_METHOD,
        'attn_layers': ATTN_LAYERS, # None, int, list[int], "low", "middle", "high"
    }

    try:
        response = _call_attn_service(payload)
    except Exception as exc:
        print('attn compress skipped (service error):', exc)
        return

    compressed_messages = response.get('compressed_messages')
    stats = response.get('stats', {})
    if not compressed_messages:
        print('attn compress skipped (empty response)')
        return

    if len(compressed_messages) != len(original_messages):
        print('attn compress warning: response len mismatch', len(compressed_messages), len(original_messages))
        if len(compressed_messages) < len(original_messages):
            return
        compressed_messages = compressed_messages[:len(original_messages)]

    status = stats.get('status')
    if status and status != 'success':
        reason = stats.get('reason', '')
        if reason:
            print(f'attn compress skipped ({status}): {reason}')
        else:
            print(f'attn compress skipped ({status})')
        return

    old_tokens = sum(
        count_token(msg.get('content', ''))
        for msg in original_messages[2:]
        if msg.get('role') == 'tool'
    )
    if old_tokens <= 0:
        print('attn compress skipped (no tool tokens)')
        return

    service_msgs = compressed_messages[2:] # 忽略掉最开始的两个固定消息(sys, user)

    # Apply results based on mode
    if ATTN_ROLLING_M <= 0:
        # Mode 1: Dynamic Full -> Replace All
        mgr.replace_steps_with_compressed_messages(service_msgs)
    
    elif ATTN_MASK_HISTORY:
        # Mode 2: Rolling Mask -> Merge Local (Masked) Head + Service Tail
        # service_head equals attn_fixed_head here
        service_msgs = _merge_local_and_service_messages(mgr, service_msgs, service_head)
        
    else:
        # Mode 3: Rolling Fixed
        # If Refreshed (service_head=0), Replace All. Else Merge.
        if service_head <= 0:
            mgr.replace_steps_with_compressed_messages(service_msgs)
        else:
            service_msgs = _merge_local_and_service_messages(mgr, service_msgs, service_head)
    
    new_tokens = sum(
        count_token(msg.get('content', ''))
        for msg in service_msgs
        if msg.get('role') == 'tool'
    )

    mgr.metrics['erase_tot_count'] += 1
    mgr.metrics['erase_in_tokens'] += old_tokens
    mgr.metrics['erase_out_tokens'] += new_tokens

    # 输出压缩前后的token数对比
    print('attn compress tool results: tokens', old_tokens, '->', new_tokens)

def should_perform_analysis(mgr: MessageManager):
    turn = mgr.count_turn()
    if turn < N_CTX_BEFORE + N_CTX_AFTER:
        return False

    traj_content = [
        mgr.extract_step_into_traj(i)
        for i in range(turn - N_CTX_BEFORE - N_CTX_AFTER - 1, turn)
    ]

    tokens = count_token(traj_content[N_CTX_BEFORE])
    mgr.metrics['seen_tokens'] += tokens

    if tokens < THRESHOLD_TOKENS:
        print('!!! no analysis (too short)', tokens)
        return False

    if not USE_LZ4:
        print('!!! DO analysis (length ok)', tokens)
        return True

    else:
        x1 = count_comp(''.join(traj_content[-(N_CTX_AFTER):]).encode('utf-8'))
        x2 = count_comp(''.join(traj_content[-(N_CTX_AFTER+1):]).encode('utf-8'))

        save_rate = 1 - max(0, x2-x1) / len(traj_content[N_CTX_BEFORE].encode('utf-8'))
        save_tokens = tokens * save_rate

        if save_tokens >= THRESHOLD_TOKENS:
            print('!!! DO analysis', tokens, save_rate, save_tokens)
            return True
        else:
            print('!!! no analysis', tokens, save_rate, save_tokens)
            return False

def maybe_perform_analysis_step(mgr: MessageManager):
    start_time = time.time()
    try:
        # judge if analysis is necessary
        if MODE == 'skip':
            print('skipped traj analysis')
            return

        # 新增的两种压缩方式，不走should_perform_analysis的判断。
        if MODE == 'llm_summary':
            if perform_analysis_step_llm_summary(mgr):
                mgr.metrics['analysis_count'] += 1
            return
        elif MODE == 'obs_mask':
            if perform_analysis_step_obs_masking(mgr):
                mgr.metrics['analysis_count'] += 1
            return
        elif MODE == 'attn':
            perform_analysis_step_attention_compress(mgr)
            mgr.metrics['analysis_count'] += 1
            return

        if not should_perform_analysis(mgr):
            return

        # okay perform analysis now
        print('===== begin traj analysis =====')
        mgr.metrics['analysis_count'] += 1

        if MODE == 'lingua':
            perform_analysis_step_llmlingua(mgr)
        elif MODE == 'random':
            perform_analysis_step_random_drop(mgr)
        elif MODE == 'delete':
            perform_analysis_step_delete(mgr)
        elif MODE == 'ours':
            perform_analysis_step_ours(mgr)
        else:
            raise ZeroDivisionError(f'wtf mode: {analysis_args}')
    finally:
        mgr.metrics['analysis_time'] += time.time() - start_time


if __name__ == '__main__':
    # 简易自测：构造假消息，依赖 attn 压缩服务可用且可访问。
    # 注意：运行前确保 TRAJ_ANALYSIS 环境变量已设置，否则本文件 import 时会抛错。
    # 压缩逻辑在 perform_analysis_step_attention_compress 内通过服务完成。
    # 兼容两种运行方式：
    # 1) python -m agents.traj_analyzer  -> 有包上下文，直接相对导入
    # 2) python agents/traj_analyzer.py -> 无包上下文，需要手动设置 __package__ 和 sys.path
    if __package__ in (None, ""):
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        __package__ = "agents"
        # 避免在作为脚本运行时被相对导入再次加载一遍（导致打印两次、重复加载模型）
        sys.modules.setdefault('agents.traj_analyzer', sys.modules[__name__])

    from .expert import MessageManager
    
    test_metrics = defaultdict(int)
    test_metrics['analysis_args'] = analysis_args
    mgr = MessageManager(
        project_path='.',
        issue='demo',
        sandbox=None,
        metrics=test_metrics,
        turn_reminder=False,
        max_turn=5,
        tools=[],
    )
    
    import textwrap

    # 给一个更接近真实轨迹的自测：tool 输出里包含少量关键信息（add 函数）和大量冗余信息（无关代码/注释/日志）。
    mgr.user_message = {
        'role': 'user',
        'content': (
            'Find the exact function that adds two numbers in calc.py, and output the complete function code.\n'
            '- Output ONLY the function definition (def ... + body).\n'
            '- Do NOT include any extra explanation.\n'
        ),
    }

    noisy_tail = ''.join(
        f'# other information {i:03d}: '
        'lorem ipsum dolor sit amet, consectetur adipiscing elit; '
        'sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n'
        for i in range(1, 2500)
    )

    calc_py_dump = (
        textwrap.dedent(
            '''\
            $ nl -ba calc.py | sed -n '1,220p'
                 1  """calc.py - toy calculator utilities.
                 2
                 3  This file is intentionally noisy for testing context compression.
                 4  The important part is the *exact* add() function below.
                 5  """
                 6
                 7  from __future__ import annotations
                 8
                 9  from dataclasses import dataclass
                10  from typing import Any
                11
                12  # --- lots of unrelated helpers ---
                13  def mul(a: int, b: int) -> int:
                14      return a * b
                15
                16  def sub(a: int, b: int) -> int:
                17      return a - b
                18
                19  def add(a: int, b: int) -> int:
                20      """Add two numbers and return the sum.
                21
                22      (KEY) This is the exact function requested.
                23      """
                24      return a + b
                25
                26  def div(a: int, b: int) -> float:
                27      if b == 0:
                28          raise ZeroDivisionError('b must not be zero')
                29      return a / b
                30
                31  # --- more irrelevant content omitted ---
            '''
        ).rstrip('\n')
        + '\n\n'
        + noisy_tail
    )

    mgr.push_step(
        {
            'role': 'assistant',
            'content': 'Opening calc.py to locate the exact add() function.',
            'tool_calls': [
                {
                    'id': 'call0',
                    'type': 'function',
                    'function': {
                        'name': 'bash',
                        'arguments': json.dumps({'command': "nl -ba calc.py | sed -n '1,220p'"}),
                    },
                }
            ],
        },
        [
            {
                'role': 'tool',
                'content': calc_py_dump,
                'tool_call_id': 'call0',
                'agent_caller': ('bash', {'command': "nl -ba calc.py | sed -n '1,220p'"}),
            }
        ],
    )

    extract_dump = (
        textwrap.dedent(
            '''\
            $ python -c "<script omitted>"
            [INFO] Parsed calc.py successfully.
            [INFO] Found candidate: add(a, b) at lines 19-24.
            === EXTRACTED FUNCTION BEGIN ===
            def add(a: int, b: int) -> int:
                """Add two numbers and return the sum.

                (KEY) This is the exact function requested.
                """
                return a + b
            === EXTRACTED FUNCTION END ===
            '''
        ).rstrip('\n')
        + '\n\n'
        + noisy_tail
    )

    mgr.push_step(
        {
            'role': 'assistant',
            'content': 'Extracting the add() function precisely to ensure we output the full definition.',
            'tool_calls': [
                {
                    'id': 'call1',
                    'type': 'function',
                    'function': {
                        'name': 'bash',
                        'arguments': json.dumps({'command': 'python -c "<script omitted>"'}),
                    },
                }
            ],
        },
        [
            {
                'role': 'tool',
                'content': extract_dump,
                'tool_call_id': 'call1',
                'agent_caller': ('bash', {'command': 'python -c "<script omitted>"'}),
            }
        ],
    )
    
    # 再增加两步很简短的，来测试不被压缩的情况
    mgr.push_step(
        {
            'role': 'assistant',
            'content': 'The add() function has been extracted successfully.',
        },
        [],
    )
    
    mgr.push_step(
        {
            'role': 'assistant',
            'content': 'Outputting the final result.',
        },
        [
            {
                'role': 'tool',
                'content': 'def add(a: int, b: int) -> int:\n    """Add two numbers and return the sum.\n\n    (KEY) This is the exact function requested.\n    """\n    return a + b\n',
                'tool_call_id': None,
                'agent_caller': None,
            }
            ],
    )

    # 压缩前后对比
    tool_before = [msg['content'] for step in mgr.steps for msg in step if msg['role'] == 'tool']
    # print('Tool messages before:', tool_before)

    # 调用压缩
    perform_analysis_step_attention_compress(mgr)

    tool_after = [msg['content'] for step in mgr.steps for msg in step if msg['role'] == 'tool']
    # print('Tool messages after:', tool_after)
    # steps = mgr.steps
    # for i, step in enumerate(steps):
    #     print(f'--- Step {i+1} ---')
    #     for msg in step:
    #         role = msg.get('role', '')
    #         content = msg.get('content', '')
    #         print(f'[{role}]:\n{content}\n')

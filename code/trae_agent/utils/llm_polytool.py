import random
import os
import itertools
import threading
import collections
import time
import json
import openai
import logging
import hashlib
from typing import Optional
from google import genai
from google.genai import types

logging.getLogger('httpx').setLevel(logging.ERROR)

### BEGIN CACHE

class HashKey:
    def __init__(self, info):
        self.cache_key = json.dumps(info, sort_keys=True)
        self.cache_hash = int(hashlib.sha256(self.cache_key.encode()).hexdigest()[:8], 16)

class NullCache:
    def __init__(self):
        pass

    def get(self, k: HashKey) -> Optional[object]:
        return None

    def put(self, k: HashKey, v: object):
        pass

llm_cache_chat = NullCache()

### END CACHE

def send_request_azure(endpoint, api_key):
    def s(model, messages, tools, kwargs):
        max_tokens = 8192  # range: [1, 8192]
        data = {
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens,
            'tools': tools,
            **kwargs,
        }

        if not tools: # Invalid 'tools': empty array. Expected an array with minimum length 1, but got an empty array instead.
            del data["tools"]

        if 'gpt-5-' in model:
            # Unsupported value: 'temperature' does not support 0 with this model. Only the default (1) value is supported.
            if 'temperature' in data:
                del data['temperature']

            # Unsupported parameter: 'stop' is not supported with this model.
            if 'stop' in data:
                del data['stop']

            data['reasoning_effort'] = 'low'

        hk = HashKey(data)

        res = llm_cache_chat.get(hk)
        if res:
            # print('cache hit')
            return res

        client = openai.AzureOpenAI(
            azure_endpoint=endpoint,
            api_version="2024-03-01-preview",
            api_key=api_key,
        )

        max_retries = 5
        retries = 0
        while retries < max_retries:
            try:
                completion = client.chat.completions.create(**data)
                if completion is None:
                    raise Exception("completion is None")

                if data.get('stream', False):
                    assert not data.get('tools', [])

                    resp_json = {
                        'choices': [{
                            'message': {
                                'role': 'assistant',
                                'content': '',
                                'refusal': None,
                                'annotations': None,
                                'audio': None,
                                'function_call': None,
                                'tool_calls': None,
                                'reasoning_content': '',
                            },
                            'finish_reason': None,
                            'index': 0,
                            'logprobs': None,
                        }],
                        'usage': {},
                    }

                    for ev in completion:
                        ev = ev.model_dump()
                        c = ev['choices']
                        if c:
                            assert len(c) == 1
                            c = c[0]
                            if c['finish_reason']:
                                resp_json['choices'][0]['finish_reason'] = c['finish_reason']
                            if c['delta'] and c['delta']['content']:
                                resp_json['choices'][0]['message']['content'] += c['delta']['content']
                        if ev.get('usage', None):
                            resp_json['usage'] = ev['usage']

                else:
                    resp_json = completion.model_dump()

                llm_cache_chat.put(hk, resp_json)
                return resp_json
            except (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError, openai.APIConnectionError, openai.LengthFinishReasonError, openai.ContentFilterFinishReasonError) as e:
                print(f"An error occurred: {type(e)} {e}")
                if retries < max_retries:
                    time.sleep(2 ** retries)
                retries += 1
            except Exception as e: # (openai.APIStatusError, openai.BadRequestError)
                print(f"A fatal error occurred: {type(e)} {e}")
                raise e

        print(f"Maximum retries ({max_retries}) exceeded.")
        return None

    return s

def send_request_openai(base_url, api_keys):
    def s(model, messages, tools, kwargs):
        max_tokens = 8192  # range: [1, 8192]
        data = {
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens,
            'tools': tools,
            **kwargs,
        }

        if not tools: # Invalid 'tools': empty array. Expected an array with minimum length 1, but got an empty array instead.
            del data["tools"]

        if 'gpt-5-' in model:
            # Unsupported value: 'temperature' does not support 0 with this model. Only the default (1) value is supported.
            if 'temperature' in data:
                del data['temperature']

            # Unsupported parameter: 'stop' is not supported with this model.
            if 'stop' in data:
                del data['stop']

            data['reasoning_effort'] = 'low'
        if 'gemini-' in model:
            if 'n' in data:
                del data['n']
            if 'temperature' in data:
                del data['temperature']
            for msg in data['messages']:
                for k in list(msg.keys()):
                    if msg[k] is None:
                        del msg[k]

        hk = HashKey(data)

        res = llm_cache_chat.get(hk)
        if res:
            # print('cache hit')
            return res

        max_retries = 5
        retries = 0
        while retries < max_retries:
            try:
                client = openai.OpenAI(
                    base_url=base_url,
                    api_key=random.choice(api_keys),
                )
                
                completion = client.chat.completions.create(**data)
                if completion is None:
                    raise Exception("completion is None")

                # if completion.choices[0].message.content == "":
                # raise Exception("completion.choices[0].message.content is empty")
                if completion.choices[0].message.content is None:
                    completion.choices[0].message.content = ""
                resp_json = completion.model_dump()

                llm_cache_chat.put(hk, resp_json)
                return resp_json
            except Exception as e:
                print(f"An error occurred: {e}")
                print(f"Retrying {retries} times...")
                # 打印出发生异常时，输入的完整信息
                print("Input data that caused the exception:")
                print(data)
                if retries < max_retries:
                    time.sleep(min(2 ** retries, 180))
                retries += 1

        print(f"Maximum retries ({max_retries}) exceeded.")
        return None

    return s

def send_request_gemini(base_url, api_keys, official_api_keys=[]):
    class LoadBalancer:
        def __init__(self, official_keys):
            self.official_keys = official_keys
            self.lock = threading.Lock()
            self.key_idx = 0
            self.history = collections.deque(maxlen=10000)
            self.threshold = 200000
            self.update_counter = 0

        def get_next_official_key(self):
            with self.lock:
                if not self.official_keys:
                    return None
                k = self.official_keys[self.key_idx]
                self.key_idx = (self.key_idx + 1) % len(self.official_keys)
                return k

        def should_use_official(self, current_len):
            with self.lock:
                self.history.append(current_len)
                self.update_counter += 1
                if self.update_counter >= 100 and len(self.history) >= 100:
                    sorted_hist = sorted(self.history)
                    idx = int(len(sorted_hist) * 0.80)
                    if idx < len(sorted_hist):
                        self.threshold = sorted_hist[idx]
                    self.update_counter = 0
                effective_threshold = self.threshold if len(self.history) >= 100 else 200000
                return current_len > effective_threshold

    lb = LoadBalancer(official_api_keys)

    def s(model, messages, tools, kwargs):
        max_tokens = 8192
        data = {
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens,
            'tools': tools,
            **kwargs,
        }

        hk = HashKey(data)
        res = llm_cache_chat.get(hk)
        if res:
            return res

        total_len = len(json.dumps(messages))
        use_official_strategy = lb.should_use_official(total_len) and bool(official_api_keys)

        max_retries = 7
        retries = 0
        while retries < max_retries:
            try:
                use_official_key = use_official_strategy and (retries < 4)
                
                if use_official_key:
                    current_api_key = lb.get_next_official_key()
                    client_kwargs = {'api_key': current_api_key}
                else:
                    current_api_key = random.choice(api_keys)
                    client_kwargs = {'api_key': current_api_key}
                    if base_url:
                        client_kwargs['http_options'] = {'base_url': base_url}
                
                client = genai.Client(**client_kwargs)
                
                gemini_contents = []
                system_instruction = None
                id_to_name = {}

                for msg in messages:
                    role = msg.get('role')
                    content = msg.get('content')
                    
                    if role == 'system':
                        if isinstance(content, list):
                            text_parts = []
                            for part in content:
                                if isinstance(part, dict) and part.get('type') == 'text':
                                    text_parts.append(part.get('text', ''))
                                elif isinstance(part, str):
                                    text_parts.append(part)
                            system_instruction = "".join(text_parts)
                        else:
                            system_instruction = content
                    elif role == 'user':
                        gemini_contents.append(types.Content(
                            role='user',
                            parts=[types.Part.from_text(text=content)]
                        ))
                    elif role == 'assistant':
                        parts = []
                        if content:
                            parts.append(types.Part.from_text(text=content))
                        
                        tool_calls = msg.get('tool_calls')
                        if tool_calls:
                            for tc in tool_calls:
                                name = tc['function']['name']
                                args = json.loads(tc['function']['arguments'])
                                id_to_name[tc['id']] = name
                                parts.append(types.Part.from_function_call(name=name, args=args))
                        
                        gemini_contents.append(types.Content(role='model', parts=parts))
                    elif role == 'tool':
                        tool_call_id = msg.get('tool_call_id')
                        name = id_to_name.get(tool_call_id)
                        if name:
                            # Ensure response is a dict
                            response_payload = {'result': content}
                            gemini_contents.append(types.Content(
                                role='user',
                                parts=[types.Part.from_function_response(name=name, response=response_payload)]
                            ))

                gemini_tools = None
                if tools:
                    function_declarations = []
                    for t in tools:
                        if t.get('type') == 'function':
                            function_declarations.append(t['function'])
                    if function_declarations:
                        gemini_tools = [types.Tool(function_declarations=function_declarations)]

                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=gemini_tools,
                    temperature=kwargs.get('temperature'),
                    max_output_tokens=kwargs.get('max_tokens', 8192),
                    top_p=kwargs.get('top_p'),
                )

                response = client.models.generate_content(
                    model=model,
                    contents=gemini_contents,
                    config=config
                )

                if not response.candidates:
                     raise Exception("No candidates returned")
                
                candidate = response.candidates[0]
                message_content = ""
                message_tool_calls = []

                for part in candidate.content.parts:
                    if part.text:
                        message_content += part.text
                    if part.function_call:
                        call_id = f"call_{int(time.time()*1000)}_{random.randint(1000,9999)}"
                        message_tool_calls.append({
                            'id': call_id,
                            'type': 'function',
                            'function': {
                                'name': part.function_call.name,
                                'arguments': json.dumps(part.function_call.args)
                            }
                        })
                if message_content is None:
                    message_content = ""
                    
                resp_msg = {
                    'role': 'assistant',
                    'content': message_content,
                }
                if message_tool_calls:
                    resp_msg['tool_calls'] = message_tool_calls

                usage = {}
                if response.usage_metadata:
                    usage = {
                        'prompt_tokens': response.usage_metadata.prompt_token_count,
                        'completion_tokens': response.usage_metadata.candidates_token_count,
                        'total_tokens': response.usage_metadata.total_token_count
                    }

                resp_json = {
                    'choices': [{
                        'message': resp_msg,
                        'finish_reason': 'tool_calls' if message_tool_calls else 'stop',
                        'index': 0
                    }],
                    'usage': usage
                }
                
                llm_cache_chat.put(hk, resp_json)
                return resp_json

            except Exception as e:
                print(f"An error occurred: {e}")
                print(f"Retrying {retries} times...")
                # print("Input data that caused the exception:")
                # print(data)
                if retries < max_retries:
                    time.sleep(min(2 ** retries, 180))
                retries += 1
        
        print(f"Maximum retries ({max_retries}) exceeded.")
        return None

    return s

GEMINI_API_KEYS = os.getenv('GEMINI_API_KEYS').split(',') if os.getenv('GEMINI_API_KEYS') else []
random.shuffle(GEMINI_API_KEYS)

UPSTREAMS_PER_MODEL = {
    'gemini-3-flash-preview': send_request_gemini('https://base_url', ['api_key'], official_api_keys=GEMINI_API_KEYS),
    'gpt-5-mini-2025-08-07': send_request_openai('https://base_url', ['api_key']),
    'claude4-sonnet': send_request_openai('https://base_url', ['api_key']),
    'claude35-haiku': send_request_openai('https://base_url', ['api_key']),
    'deepseek-chat': send_request_openai('https://base_url', ['api_key']),
    'qwen3-235b-a22b-instruct-2507': send_request_openai('https://base_url', ['sk-ooo']),
    'qwen3-next-80b-a3b-instruct': send_request_openai('https://base_url', ['sk-ooo']),
    'qwen3-coder-30B-a3B-instruct': send_request_openai('https://base_url', ['sk-ooo']),
}

def get_llm_response(model: str, messages, tools, kwargs):
    upstream = UPSTREAMS_PER_MODEL[model]
    # time.sleep(10)
    decoded_answer = []
    finish_reason = []
    assistant_response = upstream(model, messages, tools, kwargs)
    if not assistant_response:
        raise RuntimeError('no response from api')
    # print(assistant_response)
    for choice in assistant_response["choices"]:
        decoded_answer.append(choice["message"])
        finish_reason.append(choice["finish_reason"])
    return decoded_answer, finish_reason, assistant_response["usage"]

if __name__ == "__main__":
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": """
    Custom editing tool for viewing, creating and editing files
    * State is persistent across command calls and discussions with the user
    * If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
    * The `create` command cannot be used if the specified `path` already exists as a file !!! If you know that the `path` already exists, please remove it first and then perform the `create` operation!
    * If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
    * The `undo_edit` command will revert the last edit made to the file at `path`

    Notes for using the `str_replace` command:
    * The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
    * If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
    * The `new_str` parameter should contain the edited lines that should replace the `old_str`
        """,
                "parameters": {
                    "properties": {
                        "command": {
                            "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
                            "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                            "type": "string",
                        },
                        "file_text": {
                            "description": "Required parameter of `create` command, with the content of the file to be created.",
                            "type": "string",
                        },
                        "insert_line": {
                            "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
                            "type": "integer",
                        },
                        "new_str": {
                            "description": "Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.",
                            "type": "string",
                        },
                        "old_str": {
                            "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
                            "type": "string",
                        },
                        "path": {
                            "description": "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.",
                            "type": "string",
                        },
                        "view_range": {
                            "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
                            "items": {"type": "integer"},
                            "type": "array",
                        },
                    },
                    "required": ["command", "path"],
                    "type": "object",
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": """
    Run commands in a bash shell
    * When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
    * You have access to a mirror of common linux and python packages via apt and pip.
    * State is persistent across command calls and discussions with the user.
    * To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
    * Please avoid commands that may produce a very large amount of output.
    * Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.
        """,
                "parameters": {
                    "properties": {
                        "command": {
                            "description": "The bash command to run. Required unless the tool is being restarted.",
                            "type": "string",
                        },
                    },
                    "type": "object",
                }
            }
        },
        {

            "type": "function",
            "function": {
                "name": "task_done",
                "description": """
                Report the completion of the task. Note that you cannot call this tool before any verfication is done. You can write reproduce / test script to verify your solution.
                """,
                "parameters": {
                    "properties": {},
                    "type": "object",
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "think",
                "description": "Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed. For example, if you explore the repo and discover the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective. Alternatively, if you receive some test results, call this tool to brainstorm ways to fix the failing tests.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "Your thoughts."
                        }
                    },
                    "required": ["thought"],
                },
            },
        },
    ]
    print(get_llm_response(
        "gemini-2.5-flash",
        [
                {"role": "system", "content": "You respond to what the user says."},
                {"role": "user", "content": "hello"},
        ],
        TOOLS,
        dict(temperature = 0.0, n = 1, stream=False),
    ))

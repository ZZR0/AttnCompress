# Artifact of AttnCompess: Dynamic Attention-Guided Trajectory Compression for Software Engineering Agents

Structure:

- `code/`: The tool implementation of AgentDiet on Trae Agent
  - `code/attn_compress/attn_compress_service.py`: The attention-based trajectory compression service.
  - `code/trae_agent/agents/traj_analyzer.py`: The reflection module introduced in the paper
  - `code/main.sh`: Parameters for main experiment settings
  - `code/subjects/`: The list of subjects used for the experiment
- `result/`: Scripts to render tables and figures of experiment results in the paper
  - `result/trajs.7z`: Raw trajectories collected in the experiment

## ➡️ Reproduce the full experiment

> **WARNING:**
> 
> The estimated cost to run the full experiment includes ~$2k LLM fees, ~500GB of disk space, and ~2 months of time (depending on your rate limit).
> 
> Therefore, it is highly recommended to either: 
> 
> - run the experiment only on a small subset, or
> - inspect the collected trajectories instead.
> 
> Instructions for such alternatives are provided in other sections below.

### Install dependencies

The artifact has the below requirements:

- An x86-64 Linux machine with Python 3.11+ (we used Debian 12)
- Docker
- Evaluation harness of SWE-bench Verified (refer to https://github.com/SWE-bench/SWE-bench)
- Evaluation harness of Multi-SWE-bench Flash (refer to https://github.com/multi-swe-bench/multi-swe-bench)
- Other Python packages (`cd code && pip install -r requirements.txt`)

You also need to prepare necessary data files in the below locations:

- `~/miniconda3/envs/py312/` (install with [Miniconda](https://www.anaconda.com/download))
- `~/Multi-SWE-bench-flash/multi_swe_bench_flash.jsonl` (download from [HuggingFace](https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench-flash/blob/main/multi_swe_bench_flash.jsonl))

### Apply for LLM API keys

You need API keys to call various LLMs. You can apply a key from [OpenRouter](https://openrouter.ai/).

Then, modify the `UPSTREAMS_PER_MODEL` variable in `code/trae_agent/utils/llm_polytool.py` to fill in your API keys. For example:

```python
UPSTREAMS_PER_MODEL = {
    'qwen3-coder-30B-a3B-instruct': send_request_openai('https://your_base_url', 'your_api_key'),
    # ...
}
```

For the full experiment, you need access to the below LLMs:

- `gemini-3-flash` (Gemini 3 Flash)
- `qwen3-235b-a22b-instruct-2507` (Qwen3)
- `qwen3-coder-30B-a3B-instruct` (Qwen3)

### Start the compression service

If you intend to run experiments that use the `attncompress` mode (as specified in `code/trae_agent/main.sh`), you must start the compression service first:

```bash
python code/attn_compress/attn_compress_service.py --gpu-config "[[0], [1], [2, 3]]"
```

Note: You may need to adjust the `--gpu-config` according to your hardware. This service must be running in the background for the agent to perform trajectory compression.

### Run the experiment

Use `cd code/trae_agent && ./main.sh` to run the full experiment. You will see outputs like this:

```bash
username@machine:/path/to/artifact/code/trae_agent$ ./main.sh 
=== name=design_space/baseline benchmark=swebench-verified-appr100 {"mode": "skip"}
-- analysis args: {'mode': 'skip'}
tot tasks: 100
processing:  django__django-16667
processing:  django__django-11477
processing:  django__django-15987
processing:  django__django-16315
processing:  pylint-dev__pylint-7080
processing:  sphinx-doc__sphinx-11510
processing:  astropy__astropy-14598
processing:  pylint-dev__pylint-6903
processing:  sphinx-doc__sphinx-9698
== finished (val = fail / gen = task_done): django__django-16667 @ Mon Sep  8 15:24:34 2025
processing:  sphinx-doc__sphinx-7910
== finished (val = fail / gen = turn_capped): sphinx-doc__sphinx-11510 @ Mon Sep  8 15:27:20 2025
processing:  django__django-11451
......
```

Since the experiment will run for a long time, it is highly recommended to run it in a `screen` or `tmux`.

The trajectories, logs, and generated patches will be saved in `code/out/`.

## ➡️ Reproduce on a subset

The file `code/trae_agent/main.sh` includes main experiment settings to run.

Each line in the file follows the format: `out_name|benchmark|arg_json`. You can delete or comment out (with `#`) lines that you want to skip.

For example, if you only want to reproduce the experiment on Multi-SWE-bench Flash, delete everything except for the last four lines (`multi/...`).

## ➡️ Inspect collected trajectories

We are aware that the experiment may be costly to run, so we collected trajectories for future researchers to interpret the results without re-running the experiment.

Collected trajectories are located in `result/trajs_table1.7z` and `result/trajs_table23t.7z`. You can extract the zipped file with the command `7z x trajs_table1.7z` and `7z x trajs_table23t.7z`.

The python script at `result/analysis.py` reads the trajectories and renders the experiment results. Refer to that script for how to use the trajectories.
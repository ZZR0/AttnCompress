# BEGIN harness hack

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import logging
logging.getLogger('datasets').setLevel(logging.ERROR)
logging.getLogger('datasets.load').setLevel(logging.ERROR)
logging.getLogger('datasets.packaged_modules.cache.cache').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", message="Using an unlimited age cache is not recommended")

# END harness hack

import tempfile
import json
import time
import secrets
import traceback
import contextlib
import multiprocessing
from multiprocessing import Lock
from pathlib import Path

_SWEBENCH_INSTANCE_TIMEOUT = int(os.environ.get("SWEBENCH_INSTANCE_TIMEOUT", "1200"))
_SWEBENCH_VALIDATE_TIMEOUT = int(
    os.environ.get(
        "SWEBENCH_VALIDATE_TIMEOUT",
        str(_SWEBENCH_INSTANCE_TIMEOUT + 300),
    )
)
_MULTISWEBENCH_INSTANCE_TIMEOUT = int(
    os.environ.get("MULTISWEBENCH_INSTANCE_TIMEOUT", "1800")
)
_MULTISWEBENCH_VALIDATE_TIMEOUT = int(
    os.environ.get(
        "MULTISWEBENCH_VALIDATE_TIMEOUT",
        str(_MULTISWEBENCH_INSTANCE_TIMEOUT + 300),
    )
)


def _subprocess_runner(result_queue, target, args, kwargs):
    try:
        result = target(*args, **kwargs)
    except Exception:
        result_queue.put(("err", traceback.format_exc()))
    else:
        result_queue.put(("ok", result))

def _get_mp_context():
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context("spawn")


def _run_with_timeout(target, args, kwargs, timeout, label):
    if not timeout or timeout <= 0:
        return target(*args, **kwargs)
    ctx = _get_mp_context()
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_subprocess_runner,
        args=(result_queue, target, args, kwargs),
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        raise TimeoutError(f"{label} timed out after {timeout} seconds")
    if result_queue.empty():
        raise RuntimeError(f"{label} exited without a result")
    status, payload = result_queue.get()
    if status == "ok":
        return payload
    raise RuntimeError(payload)


def _run_swebench_eval(params):
    from swebench.harness.run_evaluation import main as run_eval_main
    return run_eval_main(**params)


def _run_multiswebench_eval(params):
    from multi_swe_bench.harness.run_evaluation import CliArgs
    CliArgs.from_dict(params).run()
    return True


def validate_swebench(patches: dict[str, str], lock: Lock) -> dict[str, bool]:
    # with lock:
        # time.sleep(2) # avoid possible race

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f'{tmpdir}/patch.jsonl', 'w') as f:
            for k, v in patches.items():
                line = json.dumps({
                    'instance_id': k,
                    'model_name_or_path': 'play',
                    'model_patch': v,
                })
                f.write(line + '\n')

        with contextlib.chdir(tmpdir):
            e = None
            res_path = None

            for _retry in range(7):
                try:
                    eval_params = {
                        'dataset_name': 'SWE-bench/SWE-bench_Verified',
                        'split': 'test',
                        'instance_ids': list(patches.keys()),
                        'predictions_path': f'{tmpdir}/patch.jsonl',
                        'max_workers': 1,
                        'force_rebuild': False,
                        'cache_level': 'instance',
                        'clean': False,
                        'open_file_limit': 4096,
                        'run_id': secrets.token_urlsafe(8),
                        'timeout': _SWEBENCH_INSTANCE_TIMEOUT,
                        'namespace': 'swebench',
                        'rewrite_reports': False,
                        'modal': False,
                        'instance_image_tag': 'latest',
                        'report_dir': tmpdir,
                    }
                    eval_timeout = max(
                        _SWEBENCH_VALIDATE_TIMEOUT,
                        _SWEBENCH_INSTANCE_TIMEOUT * max(1, len(patches)) + 300,
                    )
                    res_path = _run_with_timeout(
                        _run_swebench_eval,
                        args=(eval_params,),
                        kwargs={},
                        timeout=eval_timeout,
                        label="swebench evaluation",
                    )
                except Exception as ee:
                    traceback.print_exc()
                    e = ee
                    print(f"Attempt {_retry + 1} failed")
                else:
                    break

            if res_path:
                with res_path.open() as f:
                    res = json.load(f)

                return {k: (k in res['resolved_ids']) for k in patches.keys()}

            else:
                raise e


MSB_FLASH_FILES = Path(os.path.join(os.path.dirname(__file__), '../../../multi-swe-bench/data/Multi-SWE-bench-flash')) #Path('~/Multi-SWE-bench-flash').expanduser()


def validate_multiswebench(patches: dict[str, str], lock: Lock) -> dict[str, bool]:
    import docker

    #with lock:
        #time.sleep(2) # avoid possible race

    id_mapping = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f'{tmpdir}/patch.jsonl', 'w') as f:
            for k, v in patches.items():
                org_repo, _, number = k.rpartition('-')
                org, _, repo = org_repo.partition('__')
                line = json.dumps({
                    "org": org,
                    "repo": repo,
                    "number": number,
                    "fix_patch": v,
                })
                f.write(line + '\n')
                id_mapping[k] = f'{org}/{repo}:pr-{number}'

        with contextlib.chdir(tmpdir):
            e = None
            res_path = None

            Path('workdir').mkdir()
            Path('log').mkdir()
            Path('output').mkdir()

            with lock:
                # Ensure nix_swe container is runningAdd commentMore actions
                try:
                    client = docker.from_env()
                    try:
                        container = client.containers.get("nix_swe")
                    except docker.errors.NotFound:
                        client.containers.run("mswebench/nix_swe:v1.0", "true", name="nix_swe")
                except Exception as e:
                    print(f"Error starting nix_swe container: {e}")
                    raise e

            for _retry in range(2):
                try:
                    eval_params = {
                        "mode": "evaluation",
                        "workdir": f'{tmpdir}/workdir',
                        "patch_files": [f'{tmpdir}/patch.jsonl'],
                        "dataset_files": [str(MSB_FLASH_FILES / 'multi_swe_bench_flash.jsonl')],
                        "force_build": False,
                        "output_dir": f'{tmpdir}/output',
                        "specifics": [id_mapping[k] for k in patches.keys()],
                        "skips": [],
                        "repo_dir": str(MSB_FLASH_FILES / 'repos'), # not needed after tweaks
                        "need_clone": False,
                        "global_env": [],
                        "clear_env": True,
                        "stop_on_error": True,
                        "max_workers": 1,
                        "max_workers_build_image": 1,
                        "max_workers_run_instance": 1,
                        "fix_patch_run_cmd": "",
                        "log_dir": f'{tmpdir}/log',
                        "log_level": "WARNING",
                        "log_to_console": True,
                        "human_mode": True,
                    }
                    eval_timeout = max(
                        _MULTISWEBENCH_VALIDATE_TIMEOUT,
                        _MULTISWEBENCH_INSTANCE_TIMEOUT * max(1, len(patches)) + 300,
                    )
                    _run_with_timeout(
                        _run_multiswebench_eval,
                        args=(eval_params,),
                        kwargs={},
                        timeout=eval_timeout,
                        label="multiswebench evaluation",
                    )
                    time.sleep(.5)
                    res_path = Path(f'{tmpdir}/output/final_report.json')
                    assert res_path.is_file()
                except Exception as ee:
                    traceback.print_exc()
                    e = ee
                else:
                    break

            if res_path:
                with res_path.open() as f:
                    res = json.load(f)

                assert not res.get('error_ids', []), f"error_ids: {res.get('error_ids', [])}"

                return {k: (id_mapping[k] in res['resolved_ids']) for k in patches.keys()}

            else:
                raise e

validators = {
    'swebench': validate_swebench,
    'multiswebench': validate_multiswebench,
}

def test_swebench():
    patch = r"diff --git a/astropy/wcs/wcsapi/wrappers/sliced_wcs.py b/astropy/wcs/wcsapi/wrappers/sliced_wcs.py\nindex d7605b078..cf96ade36 100644\n--- a/astropy/wcs/wcsapi/wrappers/sliced_wcs.py\n+++ b/astropy/wcs/wcsapi/wrappers/sliced_wcs.py\n@@ -246,12 +246,16 @@ class SlicedLowLevelWCS(BaseWCSWrapper):\n         world_arrays = tuple(map(np.asanyarray, world_arrays))\n         world_arrays_new = []\n         iworld_curr = -1\n+        idropped = -1\n         for iworld in range(self._wcs.world_n_dim):\n             if iworld in self._world_keep:\n                 iworld_curr += 1\n                 world_arrays_new.append(world_arrays[iworld_curr])\n             else:\n-                world_arrays_new.append(1.)\n+                idropped += 1\n+                # Use the actual world coordinate value for the dropped dimension\n+                dropped_value = self.dropped_world_dimensions['value'][idropped]\n+                world_arrays_new.append(dropped_value)\n \n         world_arrays_new = np.broadcast_arrays(*world_arrays_new)\n         pixel_arrays = list(self._wcs.world_to_pixel_values(*world_arrays_new))\n"
    res = validators['swebench']({
        'astropy__astropy-13579': patch,
    }, Lock())
    print(res)

def test_multiswebench():
    patch = r"diff --git a/CHANGELOG.md b/CHANGELOG.md\nindex ffd43278c..39b5e14d7 100644\n--- a/CHANGELOG.md\n+++ b/CHANGELOG.md\n@@ -12,6 +12,8 @@\n   This can be useful to speed up searches in cases where you know that there are only N results.\n   Using this option is also (slightly) faster than piping to `head -n <count>` where `fd` can only\n   exit when it finds the search results `<count> + 1`.\n+- Add new `--min-depth <depth>` and `--exact-depth <depth>` options in addition to the existing option\n+  to limit the maximum depth. See #404.\n - Add the alias `-1` for `--max-results=1`, see #561. (@SimplyDanny).\n - Support additional ANSI font styles in `LS_COLORS`: faint, slow blink, rapid blink, dimmed, hidden and strikethrough.\n \ndiff --git a/doc/fd.1 b/doc/fd.1\nindex d1334e339..6575c2a4f 100644\n--- a/doc/fd.1\n+++ b/doc/fd.1\n@@ -110,6 +110,12 @@ Limit directory traversal to at most\n .I d\n levels of depth. By default, there is no limit on the search depth.\n .TP\n+.BI \"\\-\\-min\\-depth \" d\n+Only show search results starting at the given depth. See also: '--max-depth' and '--exact-depth'.\n+.TP\n+.BI \"\\-\\-exact\\-depth \" d\n+Only show search results at the exact given depth. This is an alias for '--min-depth <depth> --max-depth <depth>'.\n+.TP\n .BI \"\\-t, \\-\\-type \" filetype\n Filter search by type:\n .RS\ndiff --git a/src/app.rs b/src/app.rs\nindex 47a71dde1..3b33d0ee9 100644\n--- a/src/app.rs\n+++ b/src/app.rs\n@@ -168,10 +168,11 @@ pub fn build_app() -> App<'static, 'static> {\n                 ),\n         )\n         .arg(\n-            Arg::with_name(\"depth\")\n+            Arg::with_name(\"max-depth\")\n                 .long(\"max-depth\")\n                 .short(\"d\")\n                 .takes_value(true)\n+                .value_name(\"depth\")\n                 .help(\"Set maximum search depth (default: none)\")\n                 .long_help(\n                     \"Limit the directory traversal to a given depth. By default, there is no \\\n@@ -185,6 +186,29 @@ pub fn build_app() -> App<'static, 'static> {\n                 .hidden(true)\n                 .takes_value(true)\n         )\n+        .arg(\n+            Arg::with_name(\"min-depth\")\n+                .long(\"min-depth\")\n+                .takes_value(true)\n+                .value_name(\"depth\")\n+                .hidden_short_help(true)\n+                .long_help(\n+                    \"Only show search results starting at the given depth. \\\n+                     See also: '--max-depth' and '--exact-depth'\",\n+                ),\n+        )\n+        .arg(\n+            Arg::with_name(\"exact-depth\")\n+                .long(\"exact-depth\")\n+                .takes_value(true)\n+                .value_name(\"depth\")\n+                .hidden_short_help(true)\n+                .conflicts_with_all(&[\"max-depth\", \"min-depth\"])\n+                .long_help(\n+                    \"Only show search results at the exact given depth. This is an alias for \\\n+                     '--min-depth <depth> --max-depth <depth>'.\",\n+                ),\n+        )\n         .arg(\n             Arg::with_name(\"file-type\")\n                 .long(\"type\")\ndiff --git a/src/main.rs b/src/main.rs\nindex 587277853..bf43e4d36 100644\n--- a/src/main.rs\n+++ b/src/main.rs\n@@ -226,8 +226,13 @@ fn run() -> Result<ExitCode> {\n         one_file_system: matches.is_present(\"one-file-system\"),\n         null_separator: matches.is_present(\"null_separator\"),\n         max_depth: matches\n-            .value_of(\"depth\")\n+            .value_of(\"max-depth\")\n             .or_else(|| matches.value_of(\"rg-depth\"))\n+            .or_else(|| matches.value_of(\"exact-depth\"))\n+            .and_then(|n| usize::from_str_radix(n, 10).ok()),\n+        min_depth: matches\n+            .value_of(\"min-depth\")\n+            .or_else(|| matches.value_of(\"exact-depth\"))\n             .and_then(|n| usize::from_str_radix(n, 10).ok()),\n         threads: std::cmp::max(\n             matches\n@@ -296,7 +301,13 @@ fn run() -> Result<ExitCode> {\n             .value_of(\"max-results\")\n             .and_then(|n| usize::from_str_radix(n, 10).ok())\n             .filter(|&n| n != 0)\n-            .or_else(|| if matches.is_present(\"max-one-result\") { Some(1) } else { None }),\n+            .or_else(|| {\n+                if matches.is_present(\"max-one-result\") {\n+                    Some(1)\n+                } else {\n+                    None\n+                }\n+            }),\n     };\n \n     let re = RegexBuilder::new(&pattern_regex)\ndiff --git a/src/options.rs b/src/options.rs\nindex e7fb44566..3f52516a6 100644\n--- a/src/options.rs\n+++ b/src/options.rs\n@@ -40,6 +40,9 @@ pub struct Options {\n     /// all files under subdirectories of the current directory, etc.\n     pub max_depth: Option<usize>,\n \n+    /// The minimum depth for reported entries, or `None`.\n+    pub min_depth: Option<usize>,\n+\n     /// The number of threads to use.\n     pub threads: usize,\n \ndiff --git a/src/walk.rs b/src/walk.rs\nindex 2181a41ac..34b862c21 100644\n--- a/src/walk.rs\n+++ b/src/walk.rs\n@@ -283,6 +283,13 @@ impl DirEntry {\n             DirEntry::BrokenSymlink(_) => None,\n         }\n     }\n+\n+    pub fn depth(&self) -> Option<usize> {\n+        match self {\n+            DirEntry::Normal(e) => Some(e.depth()),\n+            DirEntry::BrokenSymlink(_) => None,\n+        }\n+    }\n }\n \n fn spawn_senders(\n@@ -338,6 +345,12 @@ fn spawn_senders(\n                 }\n             };\n \n+            if let Some(min_depth) = config.min_depth {\n+                if entry.depth().map_or(true, |d| d < min_depth) {\n+                    return ignore::WalkState::Continue;\n+                }\n+            }\n+\n             // Check the name first, since it doesn't require metadata\n             let entry_path = entry.path();\n \n"
    patch = patch.replace('/', '/')
    res = validators['multiswebench']({
        'sharkdp__fd-569': patch,
    }, Lock())
    print(res)

if __name__ == '__main__':
    test_swebench()

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def _patch_requests():
    retry = Retry(
        total=10,
        connect=10,
        read=10,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)

    old_init = requests.Session.__init__
    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self.mount("https://", adapter)
        self.mount("http://", adapter)
        self.trust_env = True
        # 避免连接复用触发代理边缘问题
        self.headers.update({"Connection": "close"})
    requests.Session.__init__ = new_init

_patch_requests()


from utils.swebench_validate import validators, MSB_FLASH_FILES # import first because it set envs for logging
from agents.expert import Expert
from utils.sandbox import Sandbox
from utils.agent_util import save_trajectory, save_patches
import argparse
import json
import multiprocessing
import os
import time
import sys
import traceback
import random
import fcntl
from pathlib import Path
from datetime import datetime

# 避免tokenizers并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_autodebug_task_claude_tool(args, item, lock, single_inst_mode):
    print(f"processing: ", item["instance_id"])
    output_path = args.output_path
    output_file_path = os.path.join(output_path, f'task_{item["instance_id"]}.log')

    with open(output_file_path, 'a+', buffering=1) as log_file:
        try:
            fcntl.flock(log_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            print(f"Task {item['instance_id']} is already running (locked). Skipping.")
            return

        log_file.seek(0)
        log_file.truncate()

        if not single_inst_mode:
            sys.stdout = log_file
            sys.stderr = log_file

        from agents.traj_analyzer import analysis_args
        trajectory = {
            'result': {
                'gen': '',
                'val': '',
            },
            'metrics': {
                'analysis_args': analysis_args,

                'tot_step': 0,

                'cost_tokens': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,

                'analysis_cost_tokens': 0,
                'analysis_prompt_tokens': 0,
                'analysis_completion_tokens': 0,

                'analysis_count': 0,
                'analysis_time': 0,

                'erase_tot_count': 0,

                'seen_tokens': 0,
                'erase_in_tokens': 0,
                'erase_out_tokens': 0,
            },
            'input': None,
            'messages': [],
        }

        try:
            #print(f"*********************{index}/500*********************")
            p2p_retry = 1
            current_try = 0
            valid_patch = ''
            while(current_try < p2p_retry):
                print("current_try:", current_try)
                now = datetime.now()
                datetime_str = now.strftime('%Y%m%d%H%M%S')
                print("time: ", datetime_str)
                current_try += 1

                expert = None

                try:
                    namespace, image_name, tag = item['docker_info']
                    image_name = image_name.lower()

                    # 确保 Agent 阶段使用默认 Docker daemon（validator 可能设置过 DOCKER_HOST）
                    os.environ.pop('DOCKER_HOST', None)

                    sandbox = Sandbox(namespace, image_name, tag, item)
                    sandbox.start_container()
                    project_path = sandbox.get_project_path()
                    expert = Expert(sandbox)

                    print("Expert Start!!!")
                    start_time = time.time()
                    _, final_patch, _ = expert.run(project_path, item, trajectory)
                    trajectory['metrics']['run_time'] = time.time() - start_time

                    if final_patch.strip() == '':
                        if current_try < p2p_retry:
                            try:
                                sandbox.stop_container()
                                sandbox.destroy_all_checkpoints()
                            except Exception as e:
                                print(f"Error during cleanup: {e}")
                            continue

                    valid_patch = final_patch
                    sandbox.stop_container()
                    sandbox.destroy_all_checkpoints()
                    break

                except Exception as e:
                    print(f"Error occurred: {e}")
                    trajectory['result']['gen'] = f'err-{type(e)}'
                    if expert and expert.mgr:
                        trajectory['messages'] = [m for s in expert.mgr.steps for m in s]

                    traceback.print_exc()
                    if sandbox:
                        sandbox.stop_container()
                        sandbox.destroy_all_checkpoints()

            save_patches(instance_id=item["instance_id"], patches_path=args.patches_path, patches=valid_patch)

            if valid_patch.strip():
                print('========== validating patch', time.ctime())
                try:
                    val = validators[item['validator']]({item['instance_id']: valid_patch}, lock)
                    if val[item['instance_id']]:
                        trajectory['result']['val'] = 'pass'
                        print(f"========== PASS: This instance has been successfully resolved!!")
                    else:
                        trajectory['result']['val'] = 'fail'
                        print('========== FAIL: Validation failed !!')
                except Exception:
                    if single_inst_mode:
                        raise
                    else:
                        print(f'========== ERROR: Validation error !! (id = {item["instance_id"]})')
                        traceback.print_exc()
                        trajectory['result']['val'] = 'error'
            else:
                trajectory['result']['val'] = 'nopatch'
                print(f"========== NOPATCH: Not resolved by agent !!")

            save_trajectory(instance_id=item["instance_id"], traj_dir=args.log_path, trajectory=trajectory)

        finally:
            if not single_inst_mode:
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__

            print(f"== finished (val = {trajectory['result']['val']} / gen = {trajectory['result']['gen']}): {item['instance_id']} @ {time.ctime()}")


def worker(task_queue, lock, single_inst_mode, finished_count, total_tasks):
    while True:
        task = task_queue.get()
        if task is None:
            break
        args, item = task
        
        if not single_inst_mode:
            # 随机延迟 1-10 秒，避免并发过高导致 Docker 启动压力过大
            time.sleep(random.uniform(1, 10))

        run_autodebug_task_claude_tool(args, item, lock, single_inst_mode)
        
        with finished_count.get_lock():
            finished_count.value += 1
            current_finished = finished_count.value
        
        print(f"Task Progress: {current_finished}/{total_tasks} completed, {total_tasks - current_finished} remaining.")

def main(args, lock):
    TRAJ_ANALYSIS: dict = json.loads(os.environ.get('TRAJ_ANALYSIS', '{}'))
    num_processes = int(TRAJ_ANALYSIS.get('worker_num', 8)) if TRAJ_ANALYSIS else 8

    SPECIFIC_ID = os.environ.get('INSTANCE_ID')
    if SPECIFIC_ID:
        print('!! use specific id', SPECIFIC_ID)

    jsonList = []
    need_run_set = set()

    if args.benchmark.startswith('swebench-verified-'):
        with open('../data/swebench-verified.json', 'r') as file:
            jsonList = json.load(file)

        for d in jsonList:
            d['validator'] = 'swebench'
            d['docker_info'] = ("swebench", 'sweb.eval.x86_64.' + d["instance_id"].replace('__', '_1776_'), 'latest')
            d['turn_reminder'] = True
            d['max_turn'] = 100

        subset_fn = {
            'swebench-verified-appr100': '../subjects/approach_100.json',
            'swebench-verified-eval200': '../subjects/eval_200.json',
        }[args.benchmark]

        with open(subset_fn, 'r') as file:
            for k in json.load(file):
                need_run_set.add(k)

    elif args.benchmark == 'multiswebench-flash':
        with (MSB_FLASH_FILES / 'multi_swe_bench_flash.jsonl').open() as file:
            for l in file.read().splitlines():
                d = json.loads(l)
                issue = d['resolved_issues'][0]
                jsonList.append({
                    **d,
                    'base_commit': None,
                    'custom_cwd': f'/home/{d["repo"]}',
                    'problem_statement': f'{issue["title"]}\n\n{issue["body"]}',
                    'validator': 'multiswebench',
                    'docker_info': ('mswebench', f'{d["org"]}_m_{d["repo"]}', f'pr-{d["number"]}'),
                    'turn_reminder': False,
                    'max_turn': 100,
                })
                need_run_set.add(d['instance_id'])

    else:
        raise RuntimeError(f'unknown benchmark: {args.benchmark}')

    if SPECIFIC_ID:
        need_run_set = set(SPECIFIC_ID.split(','))

    tasks = []
    for t in jsonList:
        if t["instance_id"] in need_run_set.copy():
            done = False
            bugid = t["instance_id"]

            def _is_done(log_json):
                result = log_json.get('result', {}) or {}
                val = str(result.get('val', '')).lower()

                if val in {'pass', 'fail', 'nopatch'} and "err" not in str(result.get('gen', '')).lower():
                    return True
                return False

            for serial in range(9, 0, -1):
                log_p = Path(args.log_path) / f'{t["instance_id"]}_{serial}.json'
                if log_p.is_file():
                    with open(log_p, 'r') as f:
                        d = json.load(f)
                    # if "11490" in str(log_p):
                    #     print('check log:', log_p)
                    #     print(d["result"]["val"])
                    #     print(str(d)[:500])
                    if _is_done(d):
                        print('!!done', bugid)
                        done = True
                    break

            if not done:
                tasks.append((args, t))
            need_run_set.remove(bugid)

    print('tot tasks:', len(tasks))

    if need_run_set:
        print('!!! invalid tasks:', need_run_set)
        time.sleep(10)

    will_del = []
    for _, task in tasks:
        bugid = task['instance_id']
        for p in Path(args.output_path).glob(f'task_{bugid}*.log'):
            print('delete', p)
            will_del.append(p)
        for p in Path(args.patches_path).glob(f'{bugid}_*.patch'):
            print('delete', p)
            will_del.append(p)
        for p in Path(args.log_path).glob(f'{bugid}_*.json'):
            print('delete', p)
            will_del.append(p)

    # if will_del:
    #    print(f'will delete {len(will_del)} files in 10 seconds')
    #    time.sleep(10)
    #    for p in will_del:
    #        p.unlink()

    # random.seed(666)
    # random.shuffle(tasks)
    
    # 把tasks按照instance_id排序，保证每次运行顺序一致
    tasks.sort(key=lambda x: x[1]['instance_id'])

    total_tasks = len(tasks)
    finished_count = multiprocessing.Value('i', 0)

    if len(tasks) <= 1: # use single process and disable file logging if only one job
        num_processes = 0

    task_queue = multiprocessing.Queue()

    pool = []
    for _ in range(num_processes):
        p = multiprocessing.Process(target=worker, args=(task_queue, lock, False, finished_count, total_tasks))
        p.start()
        pool.append(p)

    for task in tasks:
        task_queue.put(task)
    
    for _ in range(num_processes):
        task_queue.put(None)

    if num_processes==0:
        task_queue.put(None)
        worker(task_queue, lock, True, finished_count, total_tasks)

    for p in pool:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to handle paths.")
    parser.add_argument("--benchmark", required=True, help="Benchmark name to use")
    parser.add_argument("--log_path", required=True, help="Path to the log directory")
    parser.add_argument("--patches_path", required=True, help="Path to the patches directory")
    parser.add_argument("--output_path", required=True, help="Path to the output directory")
    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.patches_path):
        os.makedirs(args.patches_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    # process_instances(args)
    from multiprocessing import Lock
    l = Lock()
    main(args, l)

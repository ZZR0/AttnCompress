import os
import json
import shutil
from pathlib import Path
from collections import Counter
import tabulate
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pd.set_option("display.max_rows", 200)
# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

mode_cost_map = {
    "gemini3flash": [0.5, 3, 0.05],
    "qwen3-235b": [2/7, 8/7, 2/7/10],
    "qwen3-30b": [1.5/7, 6/7, 1.5/7/10],
}
# 定义一个函数来从日志文件中提取指标    
def get_metrics_from_log(log_path: Path):
    """
    输入： log_path: Path - 指向日志文件的路径
    输出： 一个字典，key是metric名称，value是对应的值
    """
    with open(log_path, 'r') as f:
        d = json.load(f)
    
    metrics = d["metrics"]
    metrics["gen"] = d["result"]["gen"]
    metrics["val"] = d["result"]["val"]
    return metrics

# subjs_appr_100 = []
# with open('../code/subjects/approach_100.json') as f:
#     for name in json.load(f):
#         subjs_appr_100.append(name)

def display_results_in_folder(folder_path:Path, output_file: str):
    # 获取所有顶层配置文件夹
    base_config_dirs = [p for p in folder_path.iterdir() if p.is_dir()]
    base_config_dirs.sort()

    all_results = pd.DataFrame()
    tot = 0

    for base_config_dir in base_config_dirs:
        base_config_name = base_config_dir.name
                
        config_name = f"{base_config_name}"
        
        # 枚举目录下所有的日志文件
        log_files = list(base_config_dir.glob('*.json'))
        total_logs = len(log_files)

        all_metrics = []
        skipped_err = 0
        for log_file in tqdm(log_files, desc=f'Processing {config_name}'):
            metrics = get_metrics_from_log(log_file)
            # # 如果这个log运行结果是错误，则跳过
            # if metrics["gen"].startswith("err") or metrics["val"].startswith("err"):
            #     skipped_err += 1
            #     continue
            metrics['config'] = config_name
            all_metrics.append(metrics)
        # print(skipped_err)
        
        if not all_metrics:
            if total_logs > 0:
                print(f"[SKIP] {config_name}: total_logs={total_logs}, valid_logs=0, err_logs={skipped_err}")
            continue

        # 对该配置下的一些指标进行计算    
        config_df = pd.DataFrame(all_metrics)
        # tot: 总共的实例数
        if len(config_df) > tot:
            tot = len(config_df)
        # keep%: 进入处理环节的token，每个实例erase_out/erase_in的比值的平均值
        config_df['keep%'] = config_df['erase_out_tokens'] / config_df['erase_in_tokens']
        keep = config_df['keep%'].mean()
        # Input: 输入token数的平均值
        Input = config_df['prompt_tokens'].mean()
        # Output: 输出token数的平均值
        Output = config_df['completion_tokens'].mean()
        # Cache: 缓存token数的平均值
        Cache = config_df['cached_tokens'].mean() if 'cached_tokens' in config_df.columns else 0
        # pass% : 成功通过的实例数占比
        passed = (config_df['val'] == "pass").sum() / tot
        passed_count = (config_df['val'] == "pass").sum()
        # step: 平均步数
        step = config_df['tot_step'].mean()
        # passed_step: 成功通过的实例的平均步数
        passed_step = config_df[config_df['val'] == "pass"]['tot_step'].mean()
        # Add_Input: 每个实例为了压缩，增加的input token数的平均值
        Add_Input = config_df['analysis_prompt_tokens'].mean()
        # Add_Output: 每个实例为了压缩，增加的output token数的平均值
        Add_Output = config_df['analysis_completion_tokens'].mean()
        # turn_capped%： 被限制在最大步数的实例占比
        turn_capped = (config_df['gen'] == "turn_capped").sum() / tot
        model_name = config_name.split('_')[1]
        Cost = mode_cost_map[model_name][0] * (Input-Cache) / 1e6 \
                + mode_cost_map[model_name][1] * Output / 1e6 \
                + mode_cost_map[model_name][2] * Cache / 1e6
        Cost_ = mode_cost_map[model_name][0] * Add_Input / 1e6 \
                + mode_cost_map[model_name][1] * Add_Output / 1e6
        Cost_Total = Cost + Cost_
        
        all_results = pd.concat([all_results, pd.DataFrame({
            'config': [config_name],
            'tot': [tot],
            'passed_count': [passed_count],
            'pass%': [passed*100],
            'Input': [Input / 1e3],
            'Output': [Output / 1e3],
            'Cost_Total': [Cost_Total],
            'step': [step],
            'passed_step': [passed_step],
            'Cache': [Cache / 1e3],
            'Cost': [Cost],
            'Cost+': [Cost_],
            'keep%': [keep*100],
            'Add_Input': [Add_Input / 1e3],
            'Add_Output': [Add_Output / 1e3],
            'turn_capped%': [turn_capped],
        })], ignore_index=True)

    # 筛选config中包含关键词的展示
    keyword = ''
    individual_runs = all_results[all_results['config'].str.contains(keyword)].reset_index(drop=True)
    # 提取排序关键字
    sort_key = individual_runs['config'].str.split('_').str[-1].str.split('/').str[0]
    
    # 指定排序顺序 (可根据需要修改，不在列表中的元素会排在最后)
    # 注意：这里的名称需要与 sort_key 提取出的字符串匹配（例如 'ours' 而不是 'ori'，除非你的配置名里确实是 'ori'）
    custom_order = ['skip', 'random', 'lingua', 'llm-summary', 'obs-mask', 'ours', 'attncompess']
    individual_runs['sort_helper'] = pd.Categorical(sort_key, categories=custom_order, ordered=True)
    
    individual_runs = individual_runs.sort_values(by=['sort_helper', 'config'])
    individual_runs = individual_runs.drop(columns=['sort_helper'])

    print(f"Individual Config Results: {keyword}")
    print(individual_runs.round({
        'keep%': 2,
        'Input': 0, 
        'Output': 0, 
        'Cache': 0,
        'step': 1, 
        'passed_step': 1, 
        'pass%': 2,
        'Add_Input': 0,
        'Add_Output': 0,
        'Run_Time': 2,
        'Ana_Time': 2,
    }))

    # 将结果保存到 CSV 文件
    output_dir = Path("./result")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_file
    individual_runs.to_csv(output_file, index=False)
    print(f"\n结果已保存至: {output_file}")


def plot_radar_chart(df: pd.DataFrame, output_path: Path):
    name_map = {
        "python": "Python",
        "javascript": "JavaScript",
        "java": "Java",
        "c++": "C++",
        "c": "C",
        "go": "Go",
        "rust": "Rust",
        "typescript": "TypeScript",
        
        "attncompess": "AttnCompress",
        "obs_mask": "ObsMask",
        "lingua": "Lingua",
        "llm_summary": "LLM Summary",
        "skip": "Original",
        "random": "Random",
        "ours": "AgentDiet",
    }
    # 提取方法名
    def get_method_name(config_str):
        base = config_str.split('/')[0]
        if "play_qwen3-30b_" in base:
            base = base.replace("play_qwen3-30b_", "")
            return name_map.get(base, base)
        return name_map.get(base, base)
    
    df = df.copy()
    df['method'] = df['config'].apply(get_method_name)
    df['lang'] = df['lang'].apply(lambda x: name_map.get(x, x))
    
    # 筛选要显示的方法
    methods_to_show = [
        "Original",
        "AttnCompress",
        "AgentDiet", 
        "LLM Summary", 
        # "Lingua",
        # "ObsMask",
        # "Random"
    ]
    
    # 准备数据
    pivot_df = df.pivot(index='method', columns='lang', values='passed_count').fillna(0)
    
    # 过滤 pivot_df
    pivot_df = pivot_df[pivot_df.index.isin(methods_to_show)]
    
    # 语言标签，确保 Java 在最上面（即列表的第一个）
    labels = list(pivot_df.columns)
    if "Java" in labels:
        labels.remove("Java")
        labels.insert(0, "Java")
    
    # 重新排列 pivot_df 的列
    pivot_df = pivot_df[labels]
    
    num_vars = len(labels)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += [angles[0]]  # 闭合
    
    fig, ax = plt.subplots(figsize=(5, 3.5), subplot_kw=dict(polar=True))
    
    # 设置标签
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], labels)
    
    # 设置径向标签（y轴）
    max_val = pivot_df.max().max()
    grid_step = 4
    
    # 生成基础刻度
    ticks = list(range(0, int(max_val) + 1, grid_step))
    ticks = [t for t in ticks if t < max_val]  # 先排除掉等于或大于最大值的部分
    
    # 总是添加最大值作为边界
    ticks.append(max_val)
    
    # 如果倒数第二个刻度离最大值太近（小于步长的一半），则移除倒数第二个刻度
    # 例如：[0, 4, 8, 12, 16, 17] -> 移除 16 -> [0, 4, 8, 12, 17]
    if len(ticks) >= 2 and (ticks[-1] - ticks[-2] < grid_step * 0.5):
        ticks.pop(-2)
    
    ax.set_ylim(0, max_val)
    ax.set_yticks(ticks)
    
    # 生成标签
    tick_labels = [f"{int(t)}" if int(t) == t else f"{t:.1f}" for t in ticks]
    ax.set_yticklabels(tick_labels)
    
    # 绘制每个方法
    for method in pivot_df.index:
        values = pivot_df.loc[method].tolist()
        values += [values[0]]
        ax.plot(angles, values, linewidth=1.5, label=method)
        # ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='center right', bbox_to_anchor=(1.63, 0.6))
    plt.title("Passed Instances (#) by Method and Programing Language", x=0.7, y=1.1)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Radar chart saved to {output_path}")


def display_results_by_language(folder_path: Path, output_file: str, lang_json_path: Path):
    # 加载语言映射
    with open(lang_json_path, 'r') as f:
        lang_map = json.load(f)

    # 获取所有顶层配置文件夹
    base_config_dirs = [p for p in folder_path.iterdir() if p.is_dir()]
    base_config_dirs.sort()

    all_results = pd.DataFrame()

    for base_config_dir in base_config_dirs:
        base_config_name = base_config_dir.name
        
        log_files = list(base_config_dir.glob('*.json'))
        config_name = f"{base_config_name}"
        
        # 按照语言对指标进行分组
        lang_to_metrics = {}
        
        for log_file in tqdm(log_files, desc=f'Processing {config_name}', leave=False):
            metrics = get_metrics_from_log(log_file)
            # 如果这个log运行结果是错误，则跳过
            if metrics["gen"].startswith("err") or metrics["val"].startswith("err"):
                continue
            
            instance_id = log_file.stem.replace(".json", "")
            lang = lang_map.get(instance_id, "unknown")
            
            if lang not in lang_to_metrics:
                lang_to_metrics[lang] = []
            lang_to_metrics[lang].append(metrics)
        
        for lang, all_metrics in lang_to_metrics.items():
            if not all_metrics:
                continue

            # 对该配置和该语言下的一些指标进行计算    
            config_df = pd.DataFrame(all_metrics)
            # tot: 总共的实例数
            tot = len(config_df)
            # keep%: 进入处理环节的token，每个实例erase_out/erase_in的比值的平均值
            config_df['keep%'] = config_df['erase_out_tokens'] / config_df['erase_in_tokens']
            keep = config_df['keep%'].mean()
            # Input: 输入token数的平均值
            Input = config_df['prompt_tokens'].mean()
            # Output: 输出token数的平均值
            Output = config_df['completion_tokens'].mean()
            # Cache: 缓存token数的平均值
            Cache = config_df['cached_tokens'].mean() if 'cached_tokens' in config_df.columns else 0
            # pass% : 成功通过的实例数占比
            passed = (config_df['val'] == "pass").sum() / tot
            passed_count = (config_df['val'] == "pass").sum()
            # step: 平均步数
            step = config_df['tot_step'].mean()
            # passed_step: 成功通过的实例的平均步数
            passed_step = config_df[config_df['val'] == "pass"]['tot_step'].mean()
            # Add_Input: 每个实例为了压缩，增加的input token数的平均值
            Add_Input = config_df['analysis_prompt_tokens'].mean()
            # Add_Output: 每个实例为了压缩，增加的output token数的平均值
            Add_Output = config_df['analysis_completion_tokens'].mean()
            # turn_capped%： 被限制在最大步数的实例占比
            turn_capped = (config_df['gen'] == "turn_capped").sum() / tot
            # Run_Time: 平均运行时间
            Run_Time = config_df['run_time'].mean() if 'run_time' in config_df.columns else 0
            # Ana_Time: 平均分析时间
            Ana_Time = config_df['analysis_time'].mean() if 'analysis_time' in config_df.columns else 0
            
            model_name = config_name.split('_')[1]
            # 兼容不同模型的成本计算
            if model_name in mode_cost_map:
                Cost = mode_cost_map[model_name][0] * (Input-Cache) / 1e6 \
                        + mode_cost_map[model_name][1] * Output / 1e6 \
                        + mode_cost_map[model_name][2] * Cache / 1e6
                Cost_ = mode_cost_map[model_name][0] * Add_Input / 1e6 \
                        + mode_cost_map[model_name][1] * Add_Output / 1e6
                Cost_Total = Cost + Cost_
            else:
                Cost = Cost_ = Cost_Total = 0
            
            all_results = pd.concat([all_results, pd.DataFrame({
                'config': [config_name],
                'lang': [lang],
                'tot': [tot],
                'passed_count': [passed_count],
                'pass%': [passed*100],
                'Input': [Input],
                'Output': [Output],
                'Cache': [Cache],
                'Cost': [Cost],
                'Cost+': [Cost_],
                'Cost_Total': [Cost_Total],
                'step': [step],
                'passed_step': [passed_step],
                'keep%': [keep*100],
                'Add_Input': [Add_Input],
                'Add_Output': [Add_Output],
                'Run_Time': [Run_Time],
                'Ana_Time': [Ana_Time],
                'turn_capped%': [turn_capped],
            })], ignore_index=True)

    if all_results.empty:
        print("No results found.")
        return

    # 排序：先按 config，再按 lang
    individual_runs = all_results.sort_values(by=['config', 'lang']).reset_index(drop=True)

    print(f"Results grouped by Language:")
    print(individual_runs.round({
        'keep%': 2,
        'Input': 0, 
        'Output': 0, 
        'Cache': 0,
        'step': 1, 
        'passed_step': 1, 
        'pass%': 2,
        'Add_Input': 0,
        'Add_Output': 0,
        'Run_Time': 2,
        'Ana_Time': 2,
    }))

    # 将结果保存到 CSV 文件
    output_dir = Path("./result")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_file
    individual_runs.to_csv(output_file, index=False)
    print(f"\n结果已保存至: {output_file}")
    
    # 绘制雷达图
    try:
        plot_radar_chart(individual_runs, output_file.with_suffix('.pdf'))
    except Exception as e:
        print(f"Error plotting radar chart: {e}")


def display_time_results(output_file: str):
    # 获取所有顶层配置文件夹
    base_config_dirs = [
        Path(os.path.join(os.path.dirname(__file__), "trajs/time_appr100/play_qwen3-30b_skip")),
        Path(os.path.join(os.path.dirname(__file__), "trajs/time_appr100/play_qwen3-30b_attncompess")),
        Path(os.path.join(os.path.dirname(__file__), "trajs/time_appr100/play_qwen3-30b_random")),
        Path(os.path.join(os.path.dirname(__file__), "trajs/time_appr100/play_qwen3-30b_lingua")),
        Path(os.path.join(os.path.dirname(__file__), "trajs/time_appr100/play_qwen3-30b_llm_summary")),
        Path(os.path.join(os.path.dirname(__file__), "trajs/time_appr100/play_qwen3-30b_obs_mask")),
        Path(os.path.join(os.path.dirname(__file__), "trajs/time_appr100/play_qwen3-30b_ours")),
    ]

    all_results = pd.DataFrame()

    for base_config_dir in base_config_dirs:
        base_config_name = base_config_dir.name
        
        config_name = f"{base_config_name}"
        
        # 枚举目录下所有的日志文件
        log_files = list(base_config_dir.glob('*.json'))[:]
        total_logs = len(log_files)

        all_metrics = []
        skipped_err = 0
        for log_file in tqdm(log_files, desc=f'Processing {config_name}'):
            metrics = get_metrics_from_log(log_file)
            # # 如果这个log运行结果是错误，则跳过
            # if metrics["gen"].startswith("err") or metrics["val"].startswith("err"):
            #     skipped_err += 1
            #     continue
            metrics['config'] = config_name
            all_metrics.append(metrics)
        # print(skipped_err)
        
        if not all_metrics:
            if total_logs > 0:
                print(f"[SKIP] {config_name}: total_logs={total_logs}, valid_logs=0, err_logs={skipped_err}")
            continue

        # 对该配置下的一些指标进行计算    
        config_df = pd.DataFrame(all_metrics)
        # tot: 总共的实例数
        tot = len(config_df)
        # keep%: 进入处理环节的token，每个实例erase_out/erase_in的比值的平均值
        config_df['keep%'] = config_df['erase_out_tokens'] / config_df['erase_in_tokens']
        keep = config_df['keep%'].mean()
        # Input: 输入token数的平均值
        Input = config_df['prompt_tokens'].mean()
        # Output: 输出token数的平均值
        Output = config_df['completion_tokens'].mean()
        # Cache: 缓存token数的平均值
        Cache = config_df['cached_tokens'].mean() if 'cached_tokens' in config_df.columns else 0
        # pass% : 成功通过的实例数占比
        passed = (config_df['val'] == "pass").sum() / tot
        passed_count = (config_df['val'] == "pass").sum()
        # step: 平均步数
        step = config_df['tot_step'].mean()
        # passed_step: 成功通过的实例的平均步数
        passed_step = config_df[config_df['val'] == "pass"]['tot_step'].mean()
        # Add_Input: 每个实例为了压缩，增加的input token数的平均值
        Add_Input = config_df['analysis_prompt_tokens'].mean()
        # Add_Output: 每个实例为了压缩，增加的output token数的平均值
        Add_Output = config_df['analysis_completion_tokens'].mean()
        # turn_capped%： 被限制在最大步数的实例占比
        turn_capped = (config_df['gen'] == "turn_capped").sum() / tot
        # Run_Time: 平均运行时间
        Run_Time = config_df['run_time'].mean() if 'run_time' in config_df.columns else 0
        # Ana_Time: 平均分析时间
        # Ana_Time = config_df['analysis_time'].mean() / config_df['analysis_count'].mean() if 'analysis_time' in config_df.columns else 0
        Ana_Time = config_df['analysis_time'].mean() if 'analysis_time' in config_df.columns else 0
        Agent_Time = Run_Time - Ana_Time
        
        all_results = pd.concat([all_results, pd.DataFrame({
            'config': [config_name],
            'tot': [tot],
            'passed_count': [passed_count],
            'pass%': [passed*100],
            'keep%': [keep*100],
            'Input': [Input],
            'Output': [Output],
            'Cache': [Cache],
            'step': [step],
            'passed_step': [passed_step],
            'Add_Input': [Add_Input],
            'Add_Output': [Add_Output],
            'Run_Time': [Run_Time],
            'Ana_Time': [Ana_Time],
            'Agent_Time': [Agent_Time],
            'turn_capped%': [turn_capped],
        })], ignore_index=True)

    # 筛选config中包含关键词的展示
    keyword = ''
    individual_runs = all_results[all_results['config'].str.contains(keyword)].reset_index(drop=True)

    print(f"Individual Config Results: {keyword}")
    print(individual_runs.round({
        'keep%': 2,
        'Input': 0, 
        'Output': 0, 
        'Cache': 0,
        'step': 1, 
        'passed_step': 1, 
        'pass%': 2,
        'Add_Input': 0,
        'Add_Output': 0,
        'Run_Time': 2,
        'Ana_Time': 2,
    }))

    # 将结果保存到 CSV 文件
    output_dir = Path("./result")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / output_file
    individual_runs.to_csv(output_file_path, index=False)
    print(f"\n结果已保存至: {output_file_path}")

if __name__ == "__main__":

    display_results_in_folder(Path("./result/trajs/table1"), "analysis_results_table1.csv")
    display_results_in_folder(Path("./result/trajs/table2"), "analysis_results_table2.csv")
    display_results_in_folder(Path("./result/trajs/table3"), "analysis_results_table3.csv")
    
    display_results_by_language(
        Path("./result/trajs/table3"), 
        "analysis_results_table3_by_lang.csv", 
        Path("./code/subjects/mutltiswebench_lang.json")
    )
    
    display_time_results("analysis_results_time.csv")
    
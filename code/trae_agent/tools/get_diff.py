import argparse
import subprocess
import os
def git_diff_to_patch(project_path, base_commit=None):
    try:
        os.chdir(project_path)
        if not base_commit:
            stdout = subprocess.check_output(['git', '--no-pager', 'diff', '--ignore-submodules=all']).decode()
            print(stdout)
        else:
            stdout = subprocess.check_output(['git', '--no-pager', 'diff', '--ignore-submodules=all', base_commit, 'HEAD']).decode()
            print(stdout)
    except Exception as error:
        print("git diff error: ", error)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get diff in repo.')
    parser.add_argument('-p', '--project_path', help='path of the project', required=True)
    parser.add_argument('-c', '--base_commit', help='bash commit id', required=False)
    args = parser.parse_args()
    git_diff_to_patch(args.project_path, args.base_commit)
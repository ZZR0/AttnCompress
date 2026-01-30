import docker
from docker.errors import ImageNotFound
import pexpect
import time 
import subprocess
import os
import random

from docker.models.containers import Container

# os.environ['DOCKER_HOST'] = f'unix:///run/user/{os.getuid()}/docker.sock'

class Sandbox:
    def __init__(self, namespace, name, tag, instance):
        self.namespace = namespace
        self.name = name
        self.tag = tag
        self.client = docker.from_env(timeout=300)
        self.container: Container | None = None
        self.shell = None
        self.commit_id = instance["base_commit"]
        self.instance_id = instance["instance_id"]
        self.shell_ready_ts = 0
        self.registered_checkpoints = []

        self.custom_cwd = instance.get('custom_cwd', None)
        print('custom_cwd', self.custom_cwd)

    def get_project_path(self):
        project_path = self.container.exec_run("pwd").output.decode(errors='replace').strip()
        return project_path

    def apply_patch(self, patch, project_path):
        random_integer = str(random.randint(1, 10000))
        with open(f"/tmp/{random_integer}.diff", "w", encoding="utf-8") as file:
            file.write(patch)
        copy_db_cmd = f"docker cp /tmp/{random_integer}.diff {self.container.name}:{project_path}/patch.diff"
        subprocess.run(copy_db_cmd, check=True, shell=True)
        apply_command = f"git apply --ignore-space-change --ignore-whitespace {project_path}/patch.diff"
        output = self.container.exec_run(cmd = apply_command, workdir = project_path).output.decode(errors='replace').strip()
        print("git_apply: ", output)
        return output
        


    def get_file_content(self, file_path, start_line = None, end_line = None):
        file_name = os.path.basename(file_path)
        copy_db_cmd = f"docker cp {self.container.name}:{file_path} /tmp/{file_name}"
        subprocess.run(copy_db_cmd, check=True, shell=True)
        if not os.path.exists(f"/tmp/{file_name}"):
            print(f"Error Occurred: {copy_db_cmd} Failed!")
            return None
        
        snippet_lines = []
        with open(f"/tmp/{file_name}", 'r', encoding='utf-8') as f:
            code = f.read()
            lines = code.split('\n')
            if start_line < 0:
                start_line = 1
            if end_line >= len(lines):
                end_line = len(lines)

            snippet_lines = [f"【{i + start_line + 1}】{line}" for i, line in enumerate(lines[start_line:end_line + 1])]

        subprocess.run(f"rm /tmp/{file_name}", check=True, shell=True)
        return '\n'.join(snippet_lines)


    def start_container_build(self):
        image = f"{self.namespace}/{self.name}:{self.tag}"
        self.container = self.client.containers.run(image, detach=True, tty=True, stdin_open=True, privileged=True)
        print(f"Container {self.container.short_id} started with image {image}")

    def start_container(self):
        self.destroy_all_checkpoints() # cleanup leftover checkpoints

        image = f"{self.namespace}/{self.name}:{self.tag}"
        #host_path = '/tmp'
        try:
            self.client.images.get(image)
        except ImageNotFound:
            print(f"Image {image} not found locally. Attempting to pull...")
            try:
                self.client.images.pull(image)
            except Exception as e:
                print(f"Failed to pull image {image}: {e}")
                raise e

        self.container = self.client.containers.run(
            image,
            detach=True,
            tty=True,
            stdin_open=True,
            privileged=True,
            environment={
                # "http_proxy": "http://10.0.2.2:10810",
                # "https_proxy": "http://10.0.2.2:10810",
                "http_proxy": "http://172.17.0.1:10810",
                "https_proxy": "http://172.17.0.1:10810",
            },
            **({'working_dir': self.custom_cwd} if self.custom_cwd else {}),
            # volumes={'/hdd2/zzr/models/llmlingua-2-xlm-roberta-large-meetingbank/': {'bind': '/models/llmlingua-2-xlm-roberta-large-meetingbank/', 'mode': 'rw'}},
            #volumes={host_path: {'bind': container_path, 'mode': 'rw'}},
        )
        print(f"Container {self.container.short_id} started with image {image}")
        _ = self.container.exec_run(cmd="mkdir -p /home/swe-bench/conda_envs/")
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        project_directory = os.path.dirname(current_directory)
        cmd = f"chmod -R 777 {project_directory}/tools && docker cp {project_directory}/tools {self.container.name}:/home/swe-bench"
        subprocess.run(cmd, check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # install_res = self.container.exec_run(cmd="conda create -p /home/swe-bench/conda_envs/py312/ python=3.12")
        # print('install_res: ', install_res)
        copy_python_cmd = f"docker cp /home/swe-bench/miniconda3/envs/py312 {self.container.name}:/home/swe-bench/conda_envs/py312/"
        subprocess.run(copy_python_cmd, check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if self.commit_id:
            checkout_res = self.container.exec_run(f"git checkout {self.commit_id}")
            print('checkout: ',checkout_res)

        self.spawn_bg_processes()

    def spawn_bg_processes(self):
        self.start_shell()

    def get_diff_result(self, project_path: str, base_commit=None):
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                if not base_commit:
                    res = self.container.exec_run(f"/home/swe-bench/conda_envs/py312/bin/python3 /home/swe-bench/tools/get_diff.py -p {project_path}").output.decode(errors='replace')
                else:
                    res = self.container.exec_run(f"/home/swe-bench/conda_envs/py312/bin/python3 /home/swe-bench/tools/get_diff.py -p {project_path} -c {base_commit}").output.decode(errors='replace')
                    print(base_commit)
                    print("diff res: ", res)
                return res
            except Exception as e:
                print(f"Attempt {retries + 1}: An error occurred while executing the command - {e}")
                time.sleep(5)
                retries += 1
        print(f"Failed to execute the command after {max_retries} attempts.")
        return ''
    def start_shell(self):
        if self.container:
            if self.shell and self.shell.isalive():
                try:
                    self.shell.close(force=True)
                except Exception: # pexpect.exceptions.ExceptionPexpect: Could not terminate the child.
                    pass
            command = f'docker exec -it {self.container.id} /bin/bash'
            self.shell = pexpect.spawn(command, maxread=200000)
            self.shell.expect([r'\$ ', r'# '], timeout=10)
        else:
            raise Exception("Container not started. Call start_container() first.")
    def get_session(self):
        self.start_shell()
        class Session:
            def __init__(self, sandbox):
                self.sandbox = sandbox
            def execute(self, command, timeout=180):
                delay = self.sandbox.shell_ready_ts - time.time()
                if delay>0:
                    time.sleep(delay)

                try:
                    if command[-1] != '&':
                        self.sandbox.shell.sendline(command + " && sleep 0.5")
                    else:
                        self.sandbox.shell.sendline(command)
                    before = ''
                    try_limit = 5
                    current_try = 0
                    self.sandbox.shell.before = b''
                    self.sandbox.shell.after = b''
                    self.sandbox.shell.buffer = b''
                    time.sleep(.5)
                    self.sandbox.shell.expect([r'swe-bench@.*:.*\$ ', r'root@.*:.*# '], timeout)
                    output = self.sandbox.shell.before.decode('utf-8', errors='replace') + self.sandbox.shell.after.decode('utf-8', errors='replace') + self.sandbox.shell.buffer.decode('utf-8', errors='replace')

                    #output = output.rpartition('')
                    output_lines = output.split('\r\n')
                    if len(output_lines) > 1:
                        output_lines = output_lines[1:-1]
                    # result_message = '### Observation: ' + '\n'.join(output_lines)
                    result_message = '\n'.join(output_lines).replace("\x1b[?2004l\r", "")
                    # truncation_length = 5000
                    # if len(result_message) > truncation_length:
                    #     return result_message[:truncation_length] + "\n...[Truncation]"
                    return result_message
                except pexpect.TIMEOUT:
                    partial_output = ''
                    if isinstance(self.sandbox.shell.before, bytes):
                        partial_output += self.sandbox.shell.before.decode('utf-8', errors='replace')
                    if isinstance(self.sandbox.shell.after, bytes):
                        partial_output += self.sandbox.shell.after.decode('utf-8', errors='replace')
                    if isinstance(self.sandbox.shell.buffer, bytes):
                        partial_output += self.sandbox.shell.buffer.decode('utf-8', errors='replace')
                    partial_output_lines = partial_output.split('\n')
                    if len(partial_output_lines) > 1:
                        partial_output_lines = partial_output_lines[1:-1]
                        partial_output = '\n'.join(partial_output_lines)
                    return f"Command timed out after {timeout} seconds. Partial output:\n + {partial_output}"
            def close(self):
                if self.sandbox.shell:
                    try:
                        self.sandbox.shell.sendline('exit')
                        self.sandbox.shell.expect(pexpect.EOF)
                    except pexpect.TIMEOUT:
                        pass
                    self.sandbox.shell.close(force=True)
                    self.sandbox.shell = None
        return Session(self)
    def stop_container(self):
        if self.container:
            if self.shell and self.shell.isalive():
                self.shell.close(force=True)
                self.shell = None
            try:
                self.container.stop(timeout=10)
                self.container.remove()
                print(f"Container {self.container.short_id} stopped and removed")
                self.container = None
            except Exception as e:
                print(f"Error stopping/removing container: {e}")
    
    def copy_to_host(self, docker_path, host_path):
        copy_cmd = f"docker cp {self.container.name}:{docker_path} {host_path}"
        subprocess.run(copy_cmd, check=True, shell=True)

    def make_checkpoint(self):
        image = f"ckpt-{self.name}-{self.tag}"
        ckpt_tag = f'{int(time.time())}-{int(random.random()*1000000)}'
        self.container.commit(image, ckpt_tag, pause=False)

        self.registered_checkpoints.append(ckpt_tag)
        return ckpt_tag

    def restore_checkpoint(self, ckpt_tag):
        self.stop_container()
        image = f"ckpt-{self.name}-{self.tag}:{ckpt_tag}"
        self.container = self.client.containers.run(
            image,
            detach=True,
            tty=True,
            stdin_open=True,
            privileged=True,
        )
        self.spawn_bg_processes()

    def destroy_all_checkpoints(self):
        print(f'!! deleting {len(self.registered_checkpoints)} checkpoints')
        for ckpt_tag in self.registered_checkpoints:
            image = f"ckpt-{self.name}-{self.tag}:{ckpt_tag}"
            self.client.images.remove(image, force=True)

        self.registered_checkpoints.clear()

    def __del__(self):
        try:
            self.client.close()
        except:
            pass

if __name__ == "__main__":
    sandbox = Sandbox("mswebench", "ponylang_m_ponyc", "pr-2007", {'base_commit': None, 'instance_id': None})
    sandbox.start_container_build()
    session = sandbox.get_session()
    print('-----')
    output = session.execute("ls")
    print(output)
    print('-----')
    output = session.execute("sleep 70")
    print(output)
    print('-----')
    session = sandbox.get_session()
    output = session.execute("ls")
    print(output)
    # output = session.execute("cd astropy")
    # print(output)
    # output = session.execute("ls")
    # print(output)
    # output = session.execute("conda env list")
    # print(output)
    # output = session.execute("cd miniconda3")
    # output = session.execute("ls")
    # print(output)
    # output = session.execute("pytest --no-header -rA --tb=no -p no:cacheprovider")
    # print(output)
    session.close()
    # session2 = sandbox.get_session()
    # output = session2.execute("pwd")
    # print(output)
    # session2.close()
    sandbox.stop_container()

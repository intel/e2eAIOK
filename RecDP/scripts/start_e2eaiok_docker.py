
import argparse
import os, sys
import subprocess
from subprocess import PIPE, STDOUT
import shutil
import logging
from time import sleep

current_folder = os.getcwd()

def parse_args(args):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-b', '--backend',choices=['spark', 'hadoop-spark'],default='spark')
    parser.add_argument('-dp', '--dataset_path',type=str,default="../e2eaiok_dataset",help='large capacity folder for dataset storing')
    parser.add_argument('--spark_shuffle_dir',type=str,default="../spark_local_dir",help='large capacity folder for spark temporary storage')
    parser.add_argument('--proxy', type=str, default=None, help='proxy for pip and apt install')
    parser.add_argument('--log_path',type=str,default="./e2eaiok_docker_building.log",help='large capacity folder for dataset storing')
    parser.add_argument('-w', '--workers',nargs='+', default=[], help='worker host list')

    return parser.parse_args(args)

def decode_subprocess_output(pipe, logger):
    ret = []
    for line in iter(pipe.readline, b''):
        ret.append(line.decode("utf-8").strip())
        logger.info(line.decode("utf-8").strip())
    return ret

def log_subprocess_output(pipe, logger, idx = ""):
    for line in iter(pipe.readline, b''):
        logger.info(f"[{idx}]" + line.decode("utf-8").strip())

def get_install_cmd():
    import platform
    os = platform.linux_distribution()[0]
    if 'debian' in os:
        return "apt install -y "
    else:
        return "yum install -y "

def fix_cmdline(cmdline, workers, use_ssh = False):
    if not isinstance(cmdline, list):
        cmdline = cmdline.split()
    local = os.uname()[1]
    ret = []
    if len(workers) == 0:
        workers.append(local)
    for n in workers:
        if n == local:
            if use_ssh:
                ret.append(["ssh", "localhost"] + cmdline)
            else:
                cmdline = [i.replace("\"", "") for i in cmdline] 
                ret.append(cmdline)
        else:
            ret.append(["ssh", n] + cmdline)
    return ret

def fix_workers(workers):
    local = os.uname()[1]
    if len(workers) == 0:
        workers.append(local)
    return workers

def execute(cmdline, logger, workers = [], use_ssh = False, backend = False):
    if not isinstance(cmdline, list):
        cmdline = cmdline.split()
    logger.info(' '.join(cmdline))
    cmdline_list = fix_cmdline(cmdline, workers, use_ssh)
    process_pool = []
    success = False
    for cmdline in cmdline_list:
        try:
            process_pool.append([cmdline, subprocess.Popen(cmdline, stdout=PIPE, stderr=STDOUT)])
        except:
            return success
    process = process_pool[0]
    if backend:
        return True
    with process[1].stdout:
        log_subprocess_output(process[1].stdout, logger, 0 if len(workers) > 0 else "")
    success = True
    for idx, process in enumerate(process_pool):
        if idx != 0:
            with process[1].stdout:
                log_subprocess_output(process[1].stdout, logger, idx)
        rc = process[1].wait()
        if rc != 0:
            logger.error(f"{process[0]} failed")
            success = False
    return success

def execute_check(cmdline, check_func, logger):
    if not isinstance(cmdline, list):
        cmdline = cmdline.split()
    logger.info(' '.join(cmdline))
    process = subprocess.Popen(cmdline, stdout=PIPE, stderr=STDOUT)
    with process.stdout:
        ret = decode_subprocess_output(process.stdout, logger)
    rc = process.wait()
    return check_func(ret)

def check_requirements(docker_name, workers, local, logger):
    # check if we can access all remote workers
    # check if sshpass is installed
    cmdline = "sshpass -V"
    if not execute(cmdline, logger):
        # install sshpass
        cmdline = get_install_cmd() + " sshpass"
        if not execute(cmdline, logger):
            logger.error("Not detect sshpass and failed in installing, please fix manually")
            return False, None
    # if docker exists in docker hub?
    def is_regist_docker_exists(ret):
        for line in ret[1:]:
            in_dickerhub_name = line.split()[1]
            if in_dickerhub_name.endswith(f"e2eaiok/{docker_name}"):
                return True
        return False
    cmdline = f"docker search e2eaiok/{docker_name}"
    if execute_check(cmdline, is_regist_docker_exists, logger):
        logger.info(f"Docker Container {docker_name} exists, skip docker build")
        return True, 'no_build_docker_no_push'

    if len(workers) == 0:
        return True, 'build_docker_no_push'
    if len(workers) == 1 and local in workers:
        return True, 'build_docker_no_push'
    for n in workers:
        if local == n:
            continue
        cmdline = f"timeout 5 ssh {n} hostname"
        if not execute(cmdline, logger):
            logger.error(f"Unable to access {n} or passwdless to {n} is not configured")
            logger.error(f"Please run bash scripts/config_passwdless_ssh.sh {n}")
            return False, None
        def check_docker_config(json_str):
            import json
            info = json.loads(json_str[0])
            if 'HttpProxy' in info or 'HttpsProxy' in info:
                if not 'NoProxy' in info and local in info['NoProxy']:
                    logger.error(f"NoProxy is not added when Proxy is configured")
                    return False
                if not f"{local}:5000" in info['RegistryConfig']['IndexConfigs']:
                    logger.error(f"{local}:5000 is not added to insecure-registries")
                    for k, v in info['RegistryConfig']['IndexConfigs'].items():
                        logger.error(f"{k}: {v}")
                    return False
            return True
        cmdline = f"ssh {n} docker info --format " + "'{{json .}}'"
        if not execute_check(cmdline, check_docker_config, logger):
            logger.error(f"Please fix docker in {n}")
            logger.error("1. add {" + f"\"insecure-registries\" : [\"{local}:5000\"]"+"} to /etc/docker/daemon.json")
            logger.error(f"2. add to Environment=\"NO_PROXY={local}\" to /etc/systemd/system/docker.service.d/http-proxy.conf")
            logger.error(f"systemctl daemon-reload & systemctl restart docker")
            return False, None
    return True, 'start_docker_registry'

def build_docker(docker_name, docker_file, logger, proxy=None, local="localhost", is_push = False):
    tag_name = 'v1'
    proxy_config = []
    if proxy:
        proxy_config.extend(["--build-arg", f"http_proxy={proxy}"])
        proxy_config.extend(["--build-arg", f"https_proxy={proxy}"])

    def is_regist_docker_exists(ret):
        return docker_name in ret[-1]

    next_step = 1 if is_push else 2

    # step 1
    if next_step == 1:
        cmdline = f"docker image ls {local}:5000/{docker_name}:{tag_name}"
        if execute_check(cmdline, is_regist_docker_exists, logger):
            logger.info(f"Docker Container {docker_name} exists, skip start")
            next_step = 5
        else:
            next_step = 2
    
    # step 2
    if next_step == 2:
        # check if docker exists
        def is_docker_exists(ret):
            return docker_name in ret[-1]

        cmdline = f"docker image ls {docker_name}"
        if execute_check(cmdline, is_docker_exists, logger):
            logger.info(f"Docker {docker_name} exists, skip docker build")
            next_step = 4 if is_push else 6
        else:
            next_step = 3

    # step 3
    if next_step == 3:
        # start to build
        cmdline = ["docker", "build",  "-t",  docker_name, "Dockerfile", "-f", f"Dockerfile/{docker_file}"] + proxy_config
        if execute(cmdline, logger):
            next_step = 4 if is_push else 6
        else:
            return False, f"{local}:5000/{docker_name}:v1"

    # step 4
    if next_step == 4:
        # tag docker
        cmdline = f"docker tag {docker_name} {local}:5000/{docker_name}:{tag_name}"
        if execute(cmdline, logger):
            next_step = 5
        else:
            return False, f"{local}:5000/{docker_name}:{tag_name}"

    # step 5
    if next_step == 5:
        # push docker
        cmdline = f"docker push {local}:5000/{docker_name}:{tag_name}"
        if execute(cmdline, logger):
            return True, f"{local}:5000/{docker_name}:{tag_name}"
        else:
            return False, f"{local}:5000/{docker_name}:{tag_name}"
    return True, f"{local}:5000/{docker_name}:v1" if is_push else f"{docker_name}"

def start_docker_registry(logger):
    def is_registry_exists(ret):
        return 'registry' in ret[-1]

    cmdline = f"docker ps -f name=registry"
    if execute_check(cmdline, is_registry_exists, logger):
        logger.info(f"Docker Container registry exists, skip start")
        return True

    cmdline = "docker run -d -e REGISTRY_HTTP_ADDR=0.0.0.0:5000 -p 5000:5000 --restart=always --name registry registry:2"
    return execute(cmdline, logger)

def run_docker(docker_name, docker_nickname, port, dataset_path, spark_shuffle_dir, logger, workers=[]):
    # prepare dataset path
    cmdline = f"mkdir -p {dataset_path}"
    execute(cmdline, logger, workers)

    # check if docker exists
    def is_docker_exists(ret):
        for line in ret[1:]:
            container_name = line.split()[-1]
            if container_name == f"{docker_nickname}":
                return True
        return False

    workers = fix_workers(workers)
    cmdline = f"docker ps -f name={docker_nickname}"
    cmdline_list = fix_cmdline(cmdline, workers)
    new_workers = []
    for worker, cmdline in zip(workers, cmdline_list):
        if not execute_check(cmdline, is_docker_exists, logger):
            new_workers.append(worker)
    workers = new_workers
    if len(workers) == 0:
        return True, port

    # run
    cmdline = ["docker", "run",  "--shm-size=300g", "--privileged",  "--network",  "host", "--device=/dev/dri", "-d", "-v", f"{dataset_path}/:/home/vmagent/app/dataset", "-v", f"{current_folder}/:/home/vmagent/app/recdp", "-v", f"{spark_shuffle_dir}/:/home/vmagent/app/spark_local_dir", "-w",  "/home/vmagent/app/", "--name", docker_nickname,  docker_name, "/bin/bash", "-c", "\"service ssh start & sleep infinity\""]

    return execute(cmdline, logger, workers), port

def build_cluster(port, workers, logger):
    if len(workers) == 0:
        return True
    file_path = os.path.dirname(os.path.abspath(__file__))
    cmdline = f"sshpass -p docker scp -P {port} -o StrictHostKeyChecking=no {file_path}/config_passwdless_ssh.sh {workers[0]}:~/"
    sleep(3)
    if not execute(cmdline, logger):
        sleep(3)
        if not execute(cmdline, logger):
            logger.error(f"please check if sshpass is installed, and is {workers[0]}:{port} has a conflict hot key in known_hosts")
            return False
    
    for n in workers:
        cmdline = f"sshpass -p docker ssh {workers[0]} -p {port} -o StrictHostKeyChecking=no bash ~/config_passwdless_ssh.sh {n}"
        if not execute(cmdline, logger):
            logger.error(f"please check if sshpass is installed, and is {workers[0]}:{port} has a conflict hot key in known_hosts")
            return False
    return True 

def build_jupyter(port, head, logger):
    cmdline = f"sshpass -p docker ssh {head} -p {port} /home/start_jupyter.sh"
    if not execute(cmdline, logger):
        logger.error(f"initiate jupyter failed, please manually execute")
        logger.error(cmdline)


def prepare_logger(log_path):
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger('e2eAIOK deployment')
    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.DEBUG)
    return rootLogger

def main(input_args):
    logger = prepare_logger(input_args.log_path)
    if not os.path.isabs(input_args.dataset_path):
        input_args.dataset_path = f"{current_folder}/{input_args.dataset_path}"
    if not os.path.isabs(input_args.spark_shuffle_dir):
        input_args.spark_shuffle_dir = f"{current_folder}/{input_args.spark_shuffle_dir}"

    hostname = os.uname()[1]
    print_success = False
    if input_args.backend == 'spark':
        docker_name = "e2eaiok-spark"
        docker_file = "DockerfileHadoopSpark"
        docker_nickname = "e2eaiok-spark"
        port = 12349

    # 0. prepare_env
    r, next_step = check_requirements(docker_name, input_args.workers, hostname, logger)
    if not r:
        logger.error(f"Failed in check_requirements, please check {input_args.log_path}")
        exit()

    # 1.1 start a local registry for other nodes to pull docker
    if next_step == "start_docker_registry":
        r = start_docker_registry(logger)
        next_step = "build_docker_and_push"
        if r:
            logger.info(f"Completed Start Docker Registry")
        else:
            logger.error(f"Failed in start docker registry, please check {input_args.log_path}")
            exit()

    # 1.2 build docker
    if next_step == "build_docker_no_push" or next_step == "build_docker_and_push":
        is_push = next_step == "build_docker_and_push"
        r, docker_name = build_docker(docker_name, docker_file, logger, input_args.proxy, local=hostname, is_push = is_push)
        if r:
            logger.info(f"Completed Docker Building")
        else:
            logger.error(f"Failed in building docker, please check {input_args.log_path}")
            exit()
    else:
        docker_name = f"e2eaiok/{docker_name}"

    # 2. start docker
    r, port = run_docker(docker_name, docker_nickname, port, input_args.dataset_path, input_args.spark_shuffle_dir, logger, input_args.workers)
    if r:
        cmd = [f"ssh {n} -p {port}" for n in input_args.workers]
        print_success = True
    else:
        logger.error(f"Failed in run docker, please check {input_args.log_path}")
        exit()

    # 3. passwdless in clustering mode
    r = build_cluster(port, input_args.workers, logger)
    if r:
        logger.info(f"passwdless among {input_args.workers} is completed")
    else:
        logger.error(f"Failed in passwdless, please check {input_args.log_path}")
        exit()
    
    # 4. start jupyter env
    r = build_jupyter(port, input_args.workers[0], logger)
    if not r:
        logger.error(f"Failed to enable jupyter, please check {input_args.log_path}")
    if print_success:
        logger.info(f"Docker Container is now running, you may access by")
        logger.info(f"{cmd}, access_code: docker")
        logger.info(f"we chose {hostname} as master, passwdless ssh is only enabled from {hostname} to others")
        

if __name__ == "__main__":
    input_args = parse_args(sys.argv[1:])
    main(input_args)

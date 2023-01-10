
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

    parser.add_argument('-b', '--backend',choices=['tensorflow', 'pytorch', 'pytorch110', 'pytorch112'],default='pytorch110')
    parser.add_argument('-dp', '--dataset_path',type=str,default="../e2eaiok_dataset",help='large capacity folder for dataset storing')
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

def execute(cmdline, logger, workers = [], use_ssh = False):
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
        cmdline = ["docker", "build",  "-t",  docker_name, "Dockerfile-ubuntu18.04", "-f", f"Dockerfile-ubuntu18.04/{docker_file}"] + proxy_config
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

def run_docker(docker_name, docker_nickname, port, dataset_path, logger, workers=[]):
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
    cmdline = ["docker", "run",  "--shm-size=100g", "--privileged",  "--network",  "host", "--device=/dev/dri", "-d", "-v", f"{dataset_path}/:/home/vmagent/app/dataset", "-v", f"{current_folder}/:/home/vmagent/app/e2eaiok",  "-w",  "/home/vmagent/app/", "--name", docker_nickname,  docker_name, "/bin/bash", "-c", "\"service ssh start & sleep infinity\""]

    return execute(cmdline, logger, workers), port

def build_cluster(port, workers, logger):
    if len(input_args.workers) == 0:
        return True
    file_path = os.path.dirname(os.path.abspath(__file__))
    cmdline = f"sshpass -p docker scp -P {port} -o StrictHostKeyChecking=no {file_path}/config_passwdless_ssh.sh {workers[0]}:~/"
    if not execute(cmdline, logger):
        sleep(3)
        execute(cmdline, logger)
    
    for n in workers:
        cmdline = f"sshpass -p docker ssh {workers[0]} -p {port} -o StrictHostKeyChecking=no bash ~/config_passwdless_ssh.sh {n}"
        if not execute(cmdline, logger):
            return False
    return True 

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

    hostname = os.uname()[1]
    print_success = False
    # latest tensorflow env
    if input_args.backend == 'tensorflow':
        docker_name = "e2eaiok-tensorflow"
        docker_file = "DockerfileTensorflow"
        docker_nickname = "e2eaiok-tensorflow"
        port = 12344
    # pytorch1.10 env
    if input_args.backend == 'pytorch110':
        docker_name = "e2eaiok-pytorch110"
        docker_file = "DockerfilePytorch110"
        docker_nickname = "e2eaiok-pytorch110"
        port = 12345
    # pytorch1.5 env
    if input_args.backend == 'pytorch':
        docker_name = "e2eaiok-pytorch"
        docker_file = "DockerfilePytorch"
        docker_nickname = "e2eaiok-pytorch"
        port = 12346
    # pytorch1.12 env
    if input_args.backend == 'pytorch112':
        docker_name = "e2eaiok-pytorch112"
        docker_file = "DockerfilePytorch112"
        docker_nickname = "e2eaiok-pytorch112"
        port = 12347

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
    r, port = run_docker(docker_name, docker_nickname, port, input_args.dataset_path, logger, input_args.workers)
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
    if print_success:
        logger.info(f"Docker Container is now running, you may access by")
        logger.info(f"{cmd}, access_code: docker")
        logger.info(f"we chose {hostname} as master, passwdless ssh is only enabled from {hostname} to others")
        

if __name__ == "__main__":
    input_args = parse_args(sys.argv[1:])
    main(input_args)

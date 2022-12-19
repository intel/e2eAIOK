import os
import subprocess

if __name__ == '__main__':
    test_dir = os.path.dirname(os.path.realpath(__file__)) # test dir
    for filename in os.listdir(test_dir):
        if filename == 'run_testting.py':
            print("Self script, continue")
            continue
        elif filename.lower().startswith("test") or filename.lower().endswith("test.py"):
            abs_path = os.path.join(test_dir,filename)
            print("Unittest file:%s"%abs_path)
            print("\n".join(subprocess.check_output("pytest -v %s"%abs_path,shell=True,stderr=subprocess.STDOUT
                                                    ).decode('utf-8').strip().split('\r\n')
                            )
                  )
        else:
            print("None test file: %s"%filename)
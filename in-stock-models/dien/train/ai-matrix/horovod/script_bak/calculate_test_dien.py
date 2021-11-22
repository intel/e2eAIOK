import os
import re

# change 1: log path
log_path = "/home2/yunfeima/logs/dien/multi-instance" 

# batch_size = 1024
batch_size = 128

postfix_list = ["tcmalloc_intel-maint-tf-1216.log"]

for postfix in postfix_list:
    for i in range(1,29):
        log_name_pattern = "^" + str(i) + r"_[0-9]{1,2}_" + str(batch_size) + "_" + postfix
        # print(log_name_pattern)

        throughput_all = 0
        for line in os.listdir(log_path):
            result = re.search(log_name_pattern, line)
            if not result:
                continue

            abs_path = os.path.join(log_path, line)
            throughput_pattern = "Approximate accelerator performance in recommendations/second is"
            with open(abs_path, 'r') as f:
                outputs = f.readlines()
            for data in outputs:
                if throughput_pattern in data:
                    value = float(data.split(' ')[-1])
                    throughput_all += value

        if throughput_all == 0:
            continue
        print("model: {}, core num: {},   throughput all: {:.3f}".format(postfix[:-4], i, throughput_all))

        output_total_throughput_filename = "total_throughput_{}cores_{}.log".format(i, postfix[:-4])
        with open(os.path.join(log_path, output_total_throughput_filename), 'w') as f:
            f.write('throughput: {} \n'.format(throughput_all))
        
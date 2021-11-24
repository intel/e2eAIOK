dir_path = "/mnt/sbc/chendi/tmp/chendi/ai-matrix/macro_benchmark/DIEN_TF2/"
p_file_name = dir_path + "previous_data/"
c_file_name = dir_path
fo = open(c_file_name + "local_train_splitByUser_diff", "w")
user_map = {}
user_map_read = {}

with open(c_file_name + "local_train_splitByUser", "r") as f_rev:
    row = 0
    for line in f_rev.readlines():
        line = line.strip()
        items = line.split("\t")
        if items[1] not in user_map:
            user_map[items[1]]= []
        user_map[items[1]].append((line))
        row += 1
        if (row % 10000 == 0):
            print("progress: %d" % row)

with open(p_file_name + "local_train_splitByUser", "r") as f:
    for line in f.readlines():
        line = line.strip()
        items = line.split("\t")
        if items[1] in user_map and items[0] == '1':
            if line != user_map[items[1]][1]:
                print("expected: " + line, file = fo)
                print("actual: " + user_map[items[1]][1], file = fo)

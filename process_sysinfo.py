import sys
import csv

res = []
cur_time = ""
titles = ['time', 'user', 'sys', 'iowait', 'idle', 'cnt']

with open(sys.argv[1]) as f:
    for line in f.readlines():
        splitted = line.split()
        if len(splitted) < 9 or splitted[2] == 'CPU':
            continue
        if splitted[0] != cur_time:
            cur_time = splitted[0]
            res.append({'time': cur_time, 'user': 0, 'sys': 0, 'iowait': 0, 'idle': 0, 'cnt': 0})
        else:
            res[-1]['user'] += float(splitted[3])
            res[-1]['sys'] += float(splitted[5])
            res[-1]['iowait'] += float(splitted[6])
            res[-1]['idle'] += float(splitted[8])
            res[-1]['cnt'] += 1

with open(sys.argv[2], 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(titles)
    for line in res:
        if line['user'] > 5:
            csvwriter.writerow(['%.2f' % line[k] if isinstance(line[k], float) else line[k] for k in titles])

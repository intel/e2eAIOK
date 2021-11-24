import sys
import time

class Shower:
    def __init__(self, period):
        self.line = ''
        self.last = time.time() + period
        self.period = period
    def show(self, msg, stats):
        if time.time() < self.last:
            return
        self.last = time.time() + self.period
        line = '%s: %s' % (msg, '  '.join('%.3f GB' % (e * 4 / 1024 / 1024) for e in stats))
        extra = len(self.line) - len(line)
        self.line = line
        if extra > 0:
            line += ' ' * extra
        sys.stdout.write('\r' + line)

def main(pid):
    fname = '/proc/%d/statm' % pid
    max_stats = []
    shower = Shower(1)
    while True:
        try:
            with open(fname) as inp:
                stats = [int(x.strip()) for x in inp.readline().split()]
        except IOError:
            break
        if not max_stats:
            max_stats = stats
        else:
            max_stats = [max(o, now) for o, now in zip(max_stats, stats)]
        time.sleep(0.1)
        shower.show('[%s]' % time.asctime(), max_stats)

    shower.show('Final', max_stats)
    print()

if __name__ == '__main__':
    try:
        pid, = sys.argv[1:]
        pid = int(pid)
    except ValueError:
        sys.exit('Usage: %s pid-to-watch' % sys.argv[0])
    main(pid)


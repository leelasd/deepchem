import sys
import re

lines = [x.strip() for x in sys.stdin.readlines()]

for index, line in enumerate(lines):
  if line.find("GPU") != -1 and line.find("PID") != -1:
    start_line = index + 2
    break

used_gpu = set()
while start_line < len(lines):
  line = lines[start_line]
  m = re.match(r".*(\d).*", line)
  if m:
    used_gpu.add(int(m.groups()[0]))
  start_line += 1

for i in range(0,8):
  if i not in used_gpu:
    print(i)
    sys.exit(0)

sys.exit(1)

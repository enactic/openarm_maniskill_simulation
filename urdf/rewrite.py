import sys
import os

with open(sys.argv[1], mode="r", encoding="utf-8") as fin:
    for line in fin:
        line = line.rstrip("\n")
        line = line.replace('package://openarm_description/', '')
        print(line)
        

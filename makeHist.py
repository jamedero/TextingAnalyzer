#!/usr/bin/env python3

"""
Quick script for understanding the /time/ distribution of text
messages. Generates a histogram of texts by hour of the day.

Author: Julie Medero
"""

import sys
import matplotlib.pyplot as plt
import datetime

lines = sys.stdin.readlines()

dts = [datetime.datetime.strptime(dt.strip(), "%m/%d/%Y %H:%M") for dt in lines]

min = dts[0]
max = dts[-1]

timerange = max - min

intervals = int(timerange / datetime.timedelta(minutes=15))

print(intervals)

n, bins, patches = plt.hist(dts, intervals, normed=0)

plt.xlabel('Time')
plt.ylabel('Number of messages')
plt.grid(True)

plt.savefig(sys.argv[1])

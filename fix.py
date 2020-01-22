#!/usr/bin/env python3

import argparse, sys
import re

parser = argparse.ArgumentParser(description='Fix variables in system.',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', dest='fix', required=True,
          help='String indicating how to fix variables, e.g., "01" fixes x_(n-1)=0 and x_n=1 (for n variables in total).')
args = parser.parse_args()

fix = args.fix

fix = fix.strip()
fix = fix.replace(" ", "")
fix = fix.replace(",", "")
fix = fix.replace(";", "")
 
fix = [int(v) for v in fix]

try:
  for data in sys.stdin:
    data = data.strip()
    if len(data) == 0:
      print(data)
      continue
    if not (data[0] == '0' or data[0] =='1'):
      if data.startswith("Number of variables"):
        m = re.match(r'[^0-9]*([0-9]*)', data)
  
        n = int(m.group(1))
  
        print("Number of variables (n) : {0}".format(n-len(fix)))
      else:
        print(data)
      continue
    
    data = data.replace(" ", "")
    data = data.replace(";", "")
    
    if not (((n*(n-1)) >> 1) + 2*n + 1 == len(data)):
      sys.stderr.write("Input length error!")
      sys.exit(-1)
    
    data = [int(v) for v in data]
    
    new_n = n-len(fix)
    
    sq = []
    lin = [0] * new_n
    const = 0
    
    for i in range(n):
      for j in range(i+1):
        val = data[0]
        data = data[1:]
    
        if i >= new_n:
          if j >= new_n:
            const ^= val & fix[i-new_n] & fix[j-new_n]
          else:
            lin[j] ^= val & fix[i-new_n]
        else:
          if j >= new_n:
            lin[i] ^= val & fix[j-new_n]
          else:
            sq.append(val)
    
    for i in range(n):
      val = data[0]
      data = data[1:]
    
      if i >= new_n:
        const ^= val & fix[i-new_n]
      else:
        lin[i] ^= val
    
    const ^= data[0]
    
    print(" ".join([str(v) for v in (sq + lin + [const])]) + ";")

except BrokenPipeError: 
  pass
 

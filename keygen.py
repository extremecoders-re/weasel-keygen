#!/usr/bin/env python3

import io
import re
import sys
import hashlib
import random
from subprocess import Popen, PIPE, STDOUT

from coefficients import coeffs


init_bits = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0]

file_header = """Galois Field : GF(2)
Number of variables (n) : 50
Number of polynomials (m) : 40
Seed : 0
Order : graded reverse lex order

*********************"""

def parse_output(bitstring):
    ks = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"

    li = list(bitstring)
    li = [list(reversed(li[i:i+5])) for i in range(0, len(li), 5)]
    
    key = ""
    for a in li:
        s = ""
        for x in a:
            s += "0" if x[0] == "0" else "1"
        i = int(s, 2)
        key += ks[i]    

    print (key[0:5], key[5:], sep="-")

def fix(fix, eqns):
    fix = [int(v) for v in fix]
    f = io.StringIO()

    try:
        for data in eqns:
            data = data.strip()
            if len(data) == 0:
                print(data, file=f)
                continue
            if not (data[0] == '0' or data[0] =='1'):
                if data.startswith("Number of variables"):
                    m = re.match(r'[^0-9]*([0-9]*)', data)
        
                    n = int(m.group(1))
        
                    print("Number of variables (n) : {0}".format(n-len(fix)), file=f)
                else:
                    print(data, file=f)
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
            
            print(" ".join([str(v) for v in (sq + lin + [const])]) + ";", file=f)

        return f
    except BrokenPipeError: 
        pass


def main(username):
    f = io.StringIO()

    hsh = hashlib.md5(username.encode("utf8")).digest()
    hash_bytes = [hsh[2], hsh[5], hsh[8], hsh[11], hsh[14]]
    target = ""

    for h in hash_bytes:
        target += "{0:08b}".format(h)[::-1]

    target_bits = list(map(int, list(target)))

    print (file_header, file=f)

    for idx, line in enumerate(coeffs):
        print (line, end = " ", file=f)
        print (init_bits[idx] ^ target_bits[idx], ";", file=f)

    fixed_bits = "{0:010b}".format(random.getrandbits(10))
    fixed_system = fix(fixed_bits, io.StringIO(f.getvalue()))
    f.close()

    p = Popen(['./guess'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate(input=bytes(fixed_system.getvalue(), 'utf8'))
    # print (stdout)
    # print (stderr)

    try:
        soln = int(stdout.split()[0].strip(), 16)
        output = '{0:040b}'.format(soln)
        print ("Username: ", username)
        print ("Password: ", end = "")
        parse_output((fixed_bits[::-1] + output)[::-1])
    except:
        print ("[-] Unable to generate password! Please retry")


if __name__ == "__main__":
    main(sys.argv[1])

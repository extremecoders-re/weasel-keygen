# Weasel keygen

This is a keygen to solve the [weasel keygenme challenge](https://forum.tuts4you.com/topic/38604-weasel-by-kao/). It is based on the GPU version of libFES available [here](http://www.polycephaly.org/projects/forcemq/index.shtml) licensed under MIT.


## Compile Instructions

To compile, open a new Python3 Colab Notebook and follow the steps below. Ensure that the project Hardware Acclerator  is set to GPU. Alternatively, use this notebook.

```
! git clone https://github.com/extremecoders-re/weasel-keygen
! chmod +x ./weasel-keygen/keygen.py
! chmod +x ./weasel-keygen/fix.py
! chmod +x ./weasel-keygen/gen_kernel.py
! cd weasel-solver && make guess
```

To generate a password, simply run keygen.py. To make the process faster, a part of the password is randomly generated while the other half is calculated. As a result, ocasionally it may fail to generate a password, in that case retry. Examples

```
! cd weasel-solver && ./keygen.py 0xec

Username:  0xec
Password: YBG3T-6EWUG
```

```
! cd weasel-solver && ./keygen.py kao
Username:  kao
Password: QL9UB-LTKYY
```

```
! cd weasel-solver && ./keygen.py extremecoders
Username:  extremecoders
Password: RAGWC-XW6VP
```



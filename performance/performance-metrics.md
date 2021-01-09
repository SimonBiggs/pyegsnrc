# Performance metrics for EGSnrc

In order to make informed decisions in rewriting EGSnrc in any language, with simulation efficiency firmly in mind, we need tools to assess the computational costs of various implementations, data structures and code logic.


## Performance measurement tools

Since we will probably investigate various languages, compilers, etc., we need performance tools that work at the kernel level, i.e., that are not dependent on language, compiler or linker. So tools such as `gprof` which depend on `gcc` for example, are typically out, as are most vendor tools. On the flip side, kernel-level tools will not port across operating systems, and we will use Linux as our development test platform.


### time

A basic performance tool is the `/urs/bin/time -v` command (usually shadowed by the shell builtin `time`; on macOS use `/usr/bin/time -lp`), which reads kernel-level counters (for _free_, since those are already updated by the kernel during normal operation) to report execution time, split into three main components:

1. `real`: wall clock duration (including other processes, also called `elapsed`)
2. `user`: time spent by current process only, in user mode only (rings 1,2,3)
3. `sys`: time spent by current process only, in kernel mode only (ring 0)

Therefore, to measure code performance on a 1 core system, one could rely on `user+sys` (which can be much less than `real` on a heavily loaded system). On multi-core systems however it becomes tricky because the kernel may dispatch system calls on different cores, hence `user+sys` int his case can become much _higher_ than the wall clock duration, so it is not a good measure of code performance. One is left with `real` as the next best metric, with the caveat that `real` wall clock time of course depends on system load. This is why `time` is not a robust metric for our purposes.

For example, below is a sample output of `time`. The leading duration data for each run is from wall clock timing measured inside code itself, to show consistency. See how the reported elapsed time (5.29 seconds) is much shorter than the `user+system` sum (5.95 seconds). This is also shown in the metric `Percent of CPU this job got: 112%` (5.95/5.29 = 112%)

```bash
/usr/bin/time -v ./py-plain.py
```

```
--------------------------------------
Python with plain numpy arrays: x, t, z, u, v, w, E
--------------------------------------
run 0: 494.865 ms
run 1: 487.721 ms
run 2: 482.606 ms
run 3: 484.568 ms
run 4: 483.066 ms
run 5: 482.658 ms
run 6: 482.752 ms
run 7: 484.822 ms
run 8: 495.538 ms
run 9: 502.441 ms
--------------------------------------
        Command being timed: "./py-plain.py"
        User time (seconds): 5.44
        System time (seconds): 0.51
        Percent of CPU this job got: 112%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:05.29
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 28320
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 0
        Minor (reclaiming a frame) page faults: 48325
        Voluntary context switches: 785
        Involuntary context switches: 356
        Swaps: 0
        File system inputs: 0
        File system outputs: 0
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0
```


### perf

The linux `perf` command provides fine-grained, processor specific hardware and software counters for timing and profiling. Typical usage is `perf stat
command`. Profiling is needed to go beyond simply measuring code speed, to understand where the bottlenecks are in order to improve performance.

The durations reported by `perf` are typically a little _longer_ than `time`, and it is unclear why. We prefer to use `perf` since it is a modern tool, has more options (for example to report cache misses), and relies on hardware counters that may be CPU make and model specific. The `perf` command is not available on macOS, which is BSD-based, so in that case `time` is the fallback solution (there is also `dtrace` on macOS, but that is the subject of an entire book!) Beyond basic timing, `perf` can generate detailed profiling reports for all sorts of system events, down to the assembler instructions, using the `perf record` command and companion `perf report` analysis tool.

Here is a sample output from `perf` (using `-d` for a few extra events, in particular cache misses). The initial printed duration data is from wall clock timing compiled in in the code itself, to show consistency.

```bash
perf stat -d cc-class
```

```
run 0: duration = 69 ms
run 1: duration = 69 ms
run 2: duration = 69 ms
run 3: duration = 69 ms
run 4: duration = 69 ms
run 5: duration = 70 ms
run 6: duration = 69 ms
run 7: duration = 69 ms
run 8: duration = 69 ms
run 9: duration = 69 ms
--------------------------------------
TOTAL = 691 ms
--------------------------------------

 Performance counter stats for 'cc-class':

            697.89 msec task-clock:u              #    0.997 CPUs utilized
                 0      context-switches:u        #    0.000 K/sec
                 0      cpu-migrations:u          #    0.000 K/sec
             1,695      page-faults:u             #    0.002 M/sec
     2,484,195,766      cycles:u                  #    3.560 GHz                      (49.86%)
     7,959,130,236      instructions:u            #    3.20  insn per cycle           (62.47%)
       779,922,230      branches:u                # 1117.547 M/sec                    (62.61%)
           687,405      branch-misses:u           #    0.09% of all branches          (62.61%)
     1,123,489,953      L1-dcache-loads:u         # 1609.843 M/sec                    (62.31%)
         8,847,457      L1-dcache-load-misses:u   #    0.79% of all L1-dcache hits    (24.93%)
           494,160      LLC-loads:u               #    0.708 M/sec                    (24.92%)
            80,811      LLC-load-misses:u         #   16.35% of all LL-cache hits     (37.39%)

       0.699676549 seconds time elapsed

       0.694354000 seconds user
       0.003996000 seconds sys

```

Note that the values reported in the comments on the left are simply ratios derived from the counter data on the left; they do not contain any additional information, but are convenient. For example, the 3.663 GHz figure is simply obtained by dividing `cycles` by `task-clock`. The `:u` flag for the reported events specifies `user mode`, but in `perf` this **includes** children system calls (to see the time spent only in system call only, use the `:k` (kernel) modifier). Note that there is a convenient option `perf stat --repeat n` (n an integer) which runs the command n times and provides uncertainties in the reported metrics.


##### Note on CPU utilization

In particular, note that the `CPUs utilized` figure in the `perf` output is just `task-clock` divided by wall clock duration (`seconds time elapsed` at the bottom), and not a true measure of number of cores used (which explains why it is not an integer). Even without explicit threading in the code, the linux kernel _will_ balance execution across multiple cores. This is confirmed by reporting stats per cpu in non-aggregated form (requires sudo privileges):

```bash
sudo perf stat --all-cpu --no-aggr ./cc-class
```

```
run 0: duration = 68 ms
run 1: duration = 68 ms
run 2: duration = 68 ms
run 3: duration = 68 ms
run 4: duration = 74 ms
run 5: duration = 71 ms
run 6: duration = 69 ms
run 7: duration = 69 ms
run 8: duration = 65 ms
run 9: duration = 65 ms
--------------------------------------
TOTAL = 685 ms
--------------------------------------

 Performance counter stats for 'system wide':

CPU0                  696.35 msec cpu-clock                 #    1.000 CPUs utilized
CPU1                  696.35 msec cpu-clock                 #    1.000 CPUs utilized
CPU2                  696.36 msec cpu-clock                 #    1.000 CPUs utilized
CPU3                  696.37 msec cpu-clock                 #    1.000 CPUs utilized
CPU4                  696.38 msec cpu-clock                 #    1.000 CPUs utilized
CPU5                  696.39 msec cpu-clock                 #    1.000 CPUs utilized
CPU6                  696.39 msec cpu-clock                 #    1.000 CPUs utilized
CPU7                  696.39 msec cpu-clock                 #    1.000 CPUs utilized
CPU8                  696.39 msec cpu-clock                 #    1.000 CPUs utilized
CPU9                  696.39 msec cpu-clock                 #    1.000 CPUs utilized
CPU10                 696.39 msec cpu-clock                 #    1.000 CPUs utilized
CPU11                 696.39 msec cpu-clock                 #    1.000 CPUs utilized
CPU0                      52      context-switches          #    0.075 K/sec
CPU1                     302      context-switches          #    0.434 K/sec
CPU2                     158      context-switches          #    0.227 K/sec
CPU3                     764      context-switches          #    0.001 M/sec
CPU4                     108      context-switches          #    0.155 K/sec
CPU5                      42      context-switches          #    0.060 K/sec
CPU6                      10      context-switches          #    0.014 K/sec
CPU7                      27      context-switches          #    0.039 K/sec
CPU8                      12      context-switches          #    0.017 K/sec
CPU9                     219      context-switches          #    0.314 K/sec
CPU10                      6      context-switches          #    0.009 K/sec
CPU11                     38      context-switches          #    0.055 K/sec
CPU0                       0      cpu-migrations            #    0.000 K/sec
CPU1                       2      cpu-migrations            #    0.003 K/sec
CPU2                       2      cpu-migrations            #    0.003 K/sec
CPU3                       2      cpu-migrations            #    0.003 K/sec
CPU4                       1      cpu-migrations            #    0.001 K/sec
CPU5                       0      cpu-migrations            #    0.000 K/sec
CPU6                       1      cpu-migrations            #    0.001 K/sec
CPU7                       4      cpu-migrations            #    0.006 K/sec
CPU8                       1      cpu-migrations            #    0.001 K/sec
CPU9                       9      cpu-migrations            #    0.013 K/sec
CPU10                      1      cpu-migrations            #    0.001 K/sec
CPU11                      1      cpu-migrations            #    0.001 K/sec
CPU0                       0      page-faults               #    0.000 K/sec
CPU1                       0      page-faults               #    0.000 K/sec
CPU2                       0      page-faults               #    0.000 K/sec
CPU3                       0      page-faults               #    0.000 K/sec
CPU4                       3      page-faults               #    0.004 K/sec
CPU5                       0      page-faults               #    0.000 K/sec
CPU6                       0      page-faults               #    0.000 K/sec
CPU7                   1,699      page-faults               #    0.002 M/sec
CPU8                       0      page-faults               #    0.000 K/sec
CPU9                       9      page-faults               #    0.013 K/sec
CPU10                      0      page-faults               #    0.000 K/sec
CPU11                      0      page-faults               #    0.000 K/sec
CPU0              33,921,387      cycles                    #    0.049 GHz
CPU1             141,985,376      cycles                    #    0.204 GHz
CPU2              20,951,434      cycles                    #    0.030 GHz
CPU3             229,663,639      cycles                    #    0.330 GHz
CPU4              42,800,758      cycles                    #    0.061 GHz
CPU5               4,350,717      cycles                    #    0.006 GHz
CPU6               1,279,280      cycles                    #    0.002 GHz
CPU7           2,546,497,896      cycles                    #    3.657 GHz
CPU8               1,298,730      cycles                    #    0.002 GHz
CPU9              97,957,271      cycles                    #    0.141 GHz
CPU10              1,036,679      cycles                    #    0.001 GHz
CPU11              6,034,177      cycles                    #    0.009 GHz
CPU0              43,642,279      instructions              #    1.29  insn per cycle
CPU1             117,801,919      instructions              #    3.47  insn per cycle
CPU2              12,086,720      instructions              #    0.36  insn per cycle
CPU3             268,457,033      instructions              #    7.91  insn per cycle
CPU4              35,042,566      instructions              #    1.03  insn per cycle
CPU5                 604,756      instructions              #    0.02  insn per cycle
CPU6                 173,831      instructions              #    0.01  insn per cycle
CPU7           7,952,229,022      instructions              #  234.43  insn per cycle
CPU8                 180,221      instructions              #    0.01  insn per cycle
CPU9              74,547,678      instructions              #    2.20  insn per cycle
CPU10                119,873      instructions              #    0.00  insn per cycle
CPU11              1,159,495      instructions              #    0.03  insn per cycle
CPU0               5,366,683      branches                  #    7.707 M/sec
CPU1              20,263,269      branches                  #   29.099 M/sec
CPU2               2,479,675      branches                  #    3.561 M/sec
CPU3              56,683,298      branches                  #   81.401 M/sec
CPU4               6,353,220      branches                  #    9.124 M/sec
CPU5                 153,870      branches                  #    0.221 M/sec
CPU6                  34,869      branches                  #    0.050 M/sec
CPU7             782,374,567      branches                  # 1123.542 M/sec
CPU8                  35,104      branches                  #    0.050 M/sec
CPU9              12,606,919      branches                  #   18.104 M/sec
CPU10                 25,072      branches                  #    0.036 M/sec
CPU11                288,356      branches                  #    0.414 M/sec
CPU0                 352,753      branch-misses             #    6.57% of all branches
CPU1                 419,275      branch-misses             #    7.81% of all branches
CPU2                  88,783      branch-misses             #    1.65% of all branches
CPU3                 884,520      branch-misses             #   16.48% of all branches
CPU4                 116,890      branch-misses             #    2.18% of all branches
CPU5                  11,468      branch-misses             #    0.21% of all branches
CPU6                   3,025      branch-misses             #    0.06% of all branches
CPU7                 728,384      branch-misses             #   13.57% of all branches
CPU8                   3,393      branch-misses             #    0.06% of all branches
CPU9                 263,089      branch-misses             #    4.90% of all branches
CPU10                  2,161      branch-misses             #    0.04% of all branches
CPU11                 22,092      branch-misses             #    0.41% of all branches

       0.696640777 seconds time elapsed
```

This output shows that most of the work is carried out on the `CPU7` core, but that a few nstructions, most likely system calls, still run on other cores. It is possible to restrict a program to run on a specific cpu or set of cpus with the `taskset` linux command, e.g., `taskset --cpu-list 0` to constrain execution to the first cpu, as much as possible (it is also possible to pass a list of cpus as in `--cpu-list 0,1`). We can see the effect of this command in the `stat --all-cpu --no-aggr` all cpu output:

```bash
sudo perf stat --all-cpu --no-aggr taskset --cpu-list 0 ./cc-class
```

```
run 0: duration = 68 ms
run 1: duration = 66 ms
run 2: duration = 66 ms
run 3: duration = 65 ms
run 4: duration = 67 ms
run 5: duration = 68 ms
run 6: duration = 68 ms
run 7: duration = 68 ms
run 8: duration = 66 ms
run 9: duration = 65 ms
--------------------------------------
TOTAL = 667 ms
--------------------------------------

 Performance counter stats for 'system wide':

CPU0                  678.84 msec cpu-clock                 #    0.999 CPUs utilized
CPU1                  678.87 msec cpu-clock                 #    0.999 CPUs utilized
CPU2                  678.88 msec cpu-clock                 #    0.999 CPUs utilized
CPU3                  678.88 msec cpu-clock                 #    0.999 CPUs utilized
CPU4                  678.89 msec cpu-clock                 #    0.999 CPUs utilized
CPU5                  678.92 msec cpu-clock                 #    0.999 CPUs utilized
CPU6                  678.92 msec cpu-clock                 #    0.999 CPUs utilized
CPU7                  678.93 msec cpu-clock                 #    0.999 CPUs utilized
CPU8                  678.93 msec cpu-clock                 #    0.999 CPUs utilized
CPU9                  678.94 msec cpu-clock                 #    1.000 CPUs utilized
CPU10                 678.94 msec cpu-clock                 #    1.000 CPUs utilized
CPU11                 678.94 msec cpu-clock                 #    1.000 CPUs utilized
CPU0                      34      context-switches          #    0.050 K/sec
CPU1                      64      context-switches          #    0.094 K/sec
CPU2                     694      context-switches          #    0.001 M/sec
CPU3                     143      context-switches          #    0.211 K/sec
CPU4                     366      context-switches          #    0.539 K/sec
CPU5                     112      context-switches          #    0.165 K/sec
CPU6                       9      context-switches          #    0.013 K/sec
CPU7                       6      context-switches          #    0.009 K/sec
CPU8                       4      context-switches          #    0.006 K/sec
CPU9                     161      context-switches          #    0.237 K/sec
CPU10                      2      context-switches          #    0.003 K/sec
CPU11                     30      context-switches          #    0.044 K/sec
CPU0                       6      cpu-migrations            #    0.009 K/sec
CPU1                       6      cpu-migrations            #    0.009 K/sec
CPU2                       5      cpu-migrations            #    0.007 K/sec
CPU3                       3      cpu-migrations            #    0.004 K/sec
CPU4                       2      cpu-migrations            #    0.003 K/sec
CPU5                       0      cpu-migrations            #    0.000 K/sec
CPU6                       1      cpu-migrations            #    0.001 K/sec
CPU7                       0      cpu-migrations            #    0.000 K/sec
CPU8                       0      cpu-migrations            #    0.000 K/sec
CPU9                       3      cpu-migrations            #    0.004 K/sec
CPU10                      0      cpu-migrations            #    0.000 K/sec
CPU11                      1      cpu-migrations            #    0.001 K/sec
CPU0                   1,700      page-faults               #    0.003 M/sec
CPU1                      30      page-faults               #    0.044 K/sec
CPU2                       3      page-faults               #    0.004 K/sec
CPU3                       3      page-faults               #    0.004 K/sec
CPU4                       9      page-faults               #    0.013 K/sec
CPU5                       0      page-faults               #    0.000 K/sec
CPU6                     173      page-faults               #    0.255 K/sec
CPU7                       0      page-faults               #    0.000 K/sec
CPU8                       0      page-faults               #    0.000 K/sec
CPU9                       4      page-faults               #    0.006 K/sec
CPU10                      0      page-faults               #    0.000 K/sec
CPU11                      0      page-faults               #    0.000 K/sec
CPU0           2,480,315,410      cycles                    #    3.654 GHz
CPU1               8,326,486      cycles                    #    0.012 GHz
CPU2             227,051,125      cycles                    #    0.334 GHz
CPU3              23,603,315      cycles                    #    0.035 GHz
CPU4             165,343,500      cycles                    #    0.244 GHz
CPU5              34,754,806      cycles                    #    0.051 GHz
CPU6               2,595,344      cycles                    #    0.004 GHz
CPU7               5,915,700      cycles                    #    0.009 GHz
CPU8               3,448,965      cycles                    #    0.005 GHz
CPU9              71,355,841      cycles                    #    0.105 GHz
CPU10              2,135,054      cycles                    #    0.003 GHz
CPU11              3,794,275      cycles                    #    0.006 GHz
CPU0           7,951,964,333      instructions              #    3.21  insn per cycle
CPU1               2,669,445      instructions              #    0.00  insn per cycle
CPU2             270,535,642      instructions              #    0.11  insn per cycle
CPU3              11,642,200      instructions              #    0.00  insn per cycle
CPU4             171,134,662      instructions              #    0.07  insn per cycle
CPU5              45,728,528      instructions              #    0.02  insn per cycle
CPU6                 772,647      instructions              #    0.00  insn per cycle
CPU7               4,272,697      instructions              #    0.00  insn per cycle
CPU8                  78,030      instructions              #    0.00  insn per cycle
CPU9              54,853,204      instructions              #    0.02  insn per cycle
CPU10                656,819      instructions              #    0.00  insn per cycle
CPU11                628,364      instructions              #    0.00  insn per cycle
CPU0             782,323,364      branches                  # 1152.444 M/sec
CPU1                 546,867      branches                  #    0.806 M/sec
CPU2              57,324,747      branches                  #   84.445 M/sec
CPU3               2,400,660      branches                  #    3.536 M/sec
CPU4              30,219,622      branches                  #   44.517 M/sec
CPU5               5,577,661      branches                  #    8.216 M/sec
CPU6                 137,019      branches                  #    0.202 M/sec
CPU7                 786,180      branches                  #    1.158 M/sec
CPU8                  15,888      branches                  #    0.023 M/sec
CPU9               8,646,478      branches                  #   12.737 M/sec
CPU10                140,101      branches                  #    0.206 M/sec
CPU11                151,344      branches                  #    0.223 M/sec
CPU0                 704,778      branch-misses             #    0.09% of all branches
CPU1                  24,622      branch-misses             #    0.00% of all branches
CPU2                 933,032      branch-misses             #    0.12% of all branches
CPU3                 106,333      branch-misses             #    0.01% of all branches
CPU4                 621,712      branch-misses             #    0.08% of all branches
CPU5                 354,724      branch-misses             #    0.05% of all branches
CPU6                   6,869      branch-misses             #    0.00% of all branches
CPU7                   8,453      branch-misses             #    0.00% of all branches
CPU8                     972      branch-misses             #    0.00% of all branches
CPU9                 202,487      branch-misses             #    0.03% of all branches
CPU10                  3,757      branch-misses             #    0.00% of all branches
CPU11                 14,539      branch-misses             #    0.00% of all branches

       0.679274935 seconds time elapsed
```

##### Note on timing python scripts with sudo privilege

There is a subtlety when timing python code: when running `perf` with `sudo` privilege, the `sys` time is markedly reduced, which skews the results for short runs. This is most likely due to the fact that under `sudo` the python version is typically different than under a user account (here I am using python 3.8 under `pyenv`, but the system version is 2.7.5). Timing could also be affected by the default niceness level under `sudo` for the python interpreter itself, or difference latencies in setting up different versions of the python interpreter (especially importing large libraries such as `numpy`, which takes about 1 second to load!). The upshot is that for very short runs a python script may runs faster overall under sudo, counterintuitively, even though the code itself runs slower (as measured by the wall clock timer inside the code), owing to the much reduces `sys` time with sudo, because the older interpreter loads faster, etc. **Therefore, be careful when comparing with `perf` output in cases where sudo privileges are required to generate the data (such as the unaggregated report above).**

This problem does not arise when using the C++ code, so it becomes tricky to compare C++ and python timings _for short runs_. Notably, with python the wall clock duration can be significantly _shorter_ than the reported `usr` time.

Without `sudo`:

```
perf stat -d ./py-plain.py
--------------------------------------
Python with plain numpy arrays: x, t, z, u, v, w, E
--------------------------------------
run 0: 61.791 ms
run 1: 58.527 ms
run 2: 58.469 ms
run 3: 58.429 ms
run 4: 58.275 ms
run 5: 58.116 ms
run 6: 57.812 ms
run 7: 57.739 ms
run 8: 57.904 ms
run 9: 58.327 ms
--------------------------------------

 Performance counter stats for './py-plain.py':

          1,611.48 msec task-clock:u              #    1.688 CPUs utilized
                 0      context-switches:u        #    0.000 K/sec
                 0      cpu-migrations:u          #    0.000 K/sec
            60,460      page-faults:u             #    0.038 M/sec
     2,619,801,085      cycles:u                  #    1.626 GHz                      (47.23%)
     6,124,717,721      instructions:u            #    2.34  insn per cycle           (59.74%)
       546,165,320      branches:u                #  338.921 M/sec                    (60.69%)
         4,530,789      branch-misses:u           #    0.83% of all branches          (60.43%)
       868,647,534      L1-dcache-loads:u         #  539.036 M/sec                    (30.96%)
        37,124,132      L1-dcache-load-misses:u   #    4.27% of all L1-dcache hits    (32.43%)
        13,913,982      LLC-loads:u               #    8.634 M/sec                    (26.45%)
         1,547,734      LLC-load-misses:u         #   11.12% of all LL-cache hits     (35.74%)

       0.954794540 seconds time elapsed

       1.088908000 seconds user
       0.536628000 seconds sys
```

With `sudo`:

```
sudo perf stat -d ./py-plain.py
--------------------------------------
Python with plain numpy arrays: x, t, z, u, v, w, E
--------------------------------------
run 0: 74.805 ms
run 1: 71.636 ms
run 2: 71.278 ms
run 3: 71.461 ms
run 4: 70.991 ms
run 5: 70.870 ms
run 6: 70.633 ms
run 7: 70.875 ms
run 8: 71.398 ms
run 9: 70.906 ms
--------------------------------------

 Performance counter stats for './py-plain.py':

            774.02 msec task-clock                #    0.995 CPUs utilized
                24      context-switches          #    0.031 K/sec
                 3      cpu-migrations            #    0.004 K/sec
            34,919      page-faults               #    0.045 M/sec
     2,757,903,253      cycles                    #    3.563 GHz                      (50.07%)
     8,218,047,917      instructions              #    2.98  insn per cycle           (62.50%)
     1,065,675,395      branches                  # 1376.812 M/sec                    (62.41%)
         2,141,441      branch-misses             #    0.20% of all branches          (62.41%)
     1,582,963,196      L1-dcache-loads           # 2045.129 M/sec                    (62.09%)
        34,021,605      L1-dcache-load-misses     #    2.15% of all L1-dcache hits    (25.06%)
         9,217,781      LLC-loads                 #   11.909 M/sec                    (25.06%)
         1,017,831      LLC-load-misses           #   11.04% of all LL-cache hits     (37.59%)

       0.777586869 seconds time elapsed

       0.727880000 seconds user
       0.046927000 seconds sys
```

Here the `sudo` run (python 2.7.5) is about 25% faster than in normal user mode (python 3.8). But the trend is **reversed** as the code runs for a longer period of time, showing that indeed there is better latency but worse run-time performance in the older interpreter.

At the end of the day, at any rate, we rely on the `task-clock` metric which is immune to system load, with **runs long enough to avoid python setup latency, which differs between versions or user privilege** Or else, in order to test using shorter runs, just subtract the python setup time estimated by running an empty script with all imports (e.g., 1 second when importing numpy).

Also keep an eye of the `CPUs utilized` comment, to compare fairly, as one program may effectively be utilizing more cores internally. For example, the python interpreter atively shares more execution time between cores, while C++ does not without explicit threading compiled inside the program. Hence, if a python program were to run twice as fast as its C++ equivalent, but using two cores, then one could in turn to the C++ threads library for a fair comparison. In that sense, `task-clock` is a robust metric, because it accumulates time spent on all cores.

If you want to look further into stabilizing a system for performance measurements, check out https://easyperf.net/blog/2019/08/02/Perf-measurement-environment-on-Linux#3-set-scaling_governor-to-performance

## Basic performance tests

In EGSnrc and Monte Carlo in general, a large amount of computing time is spent generating random numbers and updating particle data. We therefore start investigating performance of these fundamental operations with toy codes as a sort of baseline. These codes are not representative of real patterns of memory access or logical branches within a Monte Carlo simulation. For example, the arrays here are updated linearly with no dependencies between them or one cross section data tables etc., which is an ideal scenario.

Yet if these toy codes are not efficient in the proposed language and implementation, then there is little hope for the simulation code overall. It is a bare minimum test.

All tests reported below have been performed on a 6-core 12-threads Intel(R) Xeon(R) CPU E5-1650 v3 @ 3.50GHz, running linux kernel `3.10.0-1127.19.1.el7.x86_64`.

### C++ baseline

The bare minimum C++ code `cc-class.cc` and its `perf` results are shown below. We have also tried `cc-plain.cc` with flat vectors for each component of a particle, but found that an array of particles was in fact slightly more efficient, probably owing to improved cache locality (an hypothesis borne out by `perf` analysis of cache misses, not shown).

```c++
// compile with: g++ -std=c++11 -pedantic -Wall -O3 -Wextra -o cc-plain cc-plain.cc
// profile with perf stat cc-plain

// includes
#include <iostream>
#include <chrono>
#include <random>
#include <vector>

// defines
#define NUM_PARTICLES 100000
#define ITERATIONS 100
#define RUNS 10

class Particle {
public:
    double x, y, z;
    double u, v, w;
    double E;
};


// main
int main () {

    // title
    std::cout << "--------------------------------------\n";
    std::cout << "C++ with particle array\n";
    std::cout << "--------------------------------------\n";

    // particle array
    std::vector<Particle> particles(NUM_PARTICLES);

    // timing
    auto total_duration = 0;

    // random number generator
    std::mt19937 generator(1);
    std::uniform_real_distribution<double> sample(-1.0, 1.0);

    // update particle arrays a number of times
    for (int run=0; run<RUNS; run++) {

        // poor man's timere
        auto start = std::chrono::high_resolution_clock::now();

        // update particle arrays
        for (int i=0; i<ITERATIONS; i++) {

            for (auto p = particles.begin(); p != particles.end(); ++p) {
                p->x += sample(generator);
                p->y += sample(generator);
                p->z += sample(generator);
                p->u += sample(generator);
                p->v += sample(generator);
                p->w += sample(generator);
                p->E += sample(generator);
            }
        }

        // report duration
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
        total_duration += duration.count();
        std::cout << "run " << run << ": " << duration.count() << " ms" << std::endl;
    }

    std::cout << "--------------------------------------\n";
    std::cout << "TOTAL = " << total_duration << " ms\n";
    std::cout << "--------------------------------------\n";
    return EXIT_SUCCESS;
}
```

```
$ perf stat -d cc-class
--------------------------------------
C++ with particle array
--------------------------------------
run 0: 678 ms
run 1: 686 ms
run 2: 658 ms
run 3: 654 ms
run 4: 655 ms
run 5: 652 ms
run 6: 652 ms
run 7: 652 ms
run 8: 656 ms
run 9: 652 ms
--------------------------------------
TOTAL = 6595 ms
--------------------------------------

 Performance counter stats for 'cc-class':

          6,605.91 msec task-clock:u              #    1.000 CPUs utilized
                 0      context-switches:u        #    0.000 K/sec
                 0      cpu-migrations:u          #    0.000 K/sec
             1,694      page-faults:u             #    0.256 K/sec
    24,468,988,304      cycles:u                  #    3.704 GHz                      (49.99%)
    79,320,754,247      instructions:u            #    3.24  insn per cycle           (62.49%)
     7,797,910,742      branches:u                # 1180.445 M/sec                    (62.49%)
         6,768,344      branch-misses:u           #    0.09% of all branches          (62.50%)
    11,218,911,583      L1-dcache-loads:u         # 1698.315 M/sec                    (62.47%)
        87,975,686      L1-dcache-load-misses:u   #    0.78% of all L1-dcache hits    (25.01%)
         4,658,221      LLC-loads:u               #    0.705 M/sec                    (25.00%)
            93,630      LLC-load-misses:u         #    2.01% of all LL-cache hits     (37.49%)

       6.607708178 seconds time elapsed

       6.604359000 seconds user
       0.001999000 seconds sys
```

```
$ perf record cc-class
$ perf report --stdio
# (...)
# Overhead  Command   Shared Object     Symbol
# ........  ........  ................  ............................................................................
    92.68%  cc-class  cc-class          [.] std::generate_canonical<double, 53ul, std::mersenne_twister_engine (...)
     7.23%  cc-class  cc-class          [.] main
     0.07%  cc-class  [unknown]         [k] 0xffffffff9c9894ef
     0.01%  cc-class  [unknown]         [k] 0xffffffff9c992fd8
     0.01%  cc-class  ld-2.17.so        [.] do_lookup_x
     0.00%  cc-class  ld-2.17.so        [.] _dl_new_object
```

Hence we see that about 93% of the time is spent generating random numbers, which for this dummy code is what we want, so that it is spending as little time as possible on memory operations to simply shuffle data around. Note that there is aggressive optimization by the compiler here. Without the `-O3` flag, the code runs more than 10 times slower!

There is not much we can do to improve this simplistic C++ for performance (short of multithreading). We _could_ write a plain C version (or an assembler version for that matter, but we are not envisaging a plain C rewrite of EGSnrc. Hence, this constitutes our baseline performance (for 1e5 particles, 100 random updates, and 10 runs, on a 6-core 12-threads Intel(R) Xeon(R) CPU E5-1650 v3 @ 3.50GHz, tallied with `perf --repeat 10`):

```
6,637.89 msec task-clock:u              #    1.000 CPUs utilized            ( +-  0.18% )
```

The situation is quite different in python and other higher-level languages, where there are more ways today to implement the code and to structure the data, various libraries, e.g., `numpy` for efficient manipulation of numerical  data. Python as an interpreted language is typically slower than a compiled language, and we want to investigate to what extent.

### Python baseline

The interest in a python implementation of EGSnrc has been renewed because of the JAX library, which is apparently able to vectorize the code to run on GPU (with some restrictions on the data types), and with decent performance on CPU, from one and the same source code. We therefore want to study the performance of our baseline test in python.

Below is the `py-plain.py` and `perf` profile of the baseline source code, which relies on numpy (plain python is much slower, as expected). This implementation uses flat numpy arrays the default numpy random number generator, which is the same as in the C++ implementation, and one of the best generators available today (Mersenne Twister). We have verified that the timer decorator function does not incur a significant time cost.

```python
#!/usr/bin/env python

# py-plain.py: plain python/numpy implementation with flat arrays

import numpy
import time

# defines
NUM_PARTICLES = int(1e5)
ITERATIONS = 100
RUNS = 10

# title
print("--------------------------------------")
print("Python with plain numpy arrays: x, y, z, u, v, w, E")
print("--------------------------------------")

# timer decorator
def timer(f):
    def wrap(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        stop = time.time()
        duration = (stop-start)*1000.0
        print('run {}: {:.3f} ms'.format(run, duration))
        return ret
    return wrap

# runIterations
def runIterations(x, y, z, u, v, w, E):
    for j in range(ITERATIONS):
        random_normal = numpy.random.rand(NUM_PARTICLES)
        x += random_normal
        random_normal = numpy.random.rand(NUM_PARTICLES)
        y += random_normal
        random_normal = numpy.random.rand(NUM_PARTICLES)
        z += random_normal
        random_normal = numpy.random.rand(NUM_PARTICLES)
        u += random_normal
        random_normal = numpy.random.rand(NUM_PARTICLES)
        v += random_normal
        random_normal = numpy.random.rand(NUM_PARTICLES)
        w += random_normal
        random_normal = numpy.random.rand(NUM_PARTICLES)
        E += random_normal

# timed version of runIterations
runIterations = timer(runIterations)

# main function
def main(run):

    numpy.random.seed()

    x = numpy.zeros(NUM_PARTICLES)
    y = numpy.zeros(NUM_PARTICLES)
    z = numpy.zeros(NUM_PARTICLES)
    u = numpy.zeros(NUM_PARTICLES)
    v = numpy.zeros(NUM_PARTICLES)
    w = numpy.zeros(NUM_PARTICLES)
    E = numpy.zeros(NUM_PARTICLES)

    runIterations(x, y, z, u, v, w, E)

# call main function
for run in range(RUNS):
    main(run)


print("--------------------------------------")
```

```
$ perf stat -d --repeat 10 py-plain.py
# (...)
--------------------------------------
Python with plain numpy arrays: x, y, z, u, v, w, E
--------------------------------------
run 0: 481.390 ms
run 1: 480.603 ms
run 2: 480.417 ms
run 3: 480.167 ms
run 4: 480.684 ms
run 5: 482.984 ms
run 6: 479.871 ms
run 7: 479.716 ms
run 8: 479.831 ms
run 9: 479.619 ms
--------------------------------------

 Performance counter stats for 'py-plain.py' (10 runs):

          5,870.21 msec task-clock:u              #    1.135 CPUs utilized            ( +-  0.07% )
                 0      context-switches:u        #    0.000 K/sec
                 0      cpu-migrations:u          #    0.000 K/sec
            47,551      page-faults:u             #    0.008 M/sec                    ( +-  0.01% )
    18,694,000,002      cycles:u                  #    3.185 GHz                      ( +-  0.14% )  (49.55%)
    54,565,441,782      instructions:u            #    2.92  insn per cycle           ( +-  0.13% )  (62.08%)
     4,355,888,573      branches:u                #  742.033 M/sec                    ( +-  0.11% )  (62.21%)
        10,514,132      branch-misses:u           #    0.24% of all branches          ( +-  0.41% )  (62.24%)
     8,936,017,017      L1-dcache-loads:u         # 1522.265 M/sec                    ( +-  2.08% )  (46.74%)
       279,670,098      L1-dcache-load-misses:u   #    3.13% of all L1-dcache hits    ( +-  1.59% )  (34.98%)
       113,452,330      LLC-loads:u               #   19.327 M/sec                    ( +-  1.05% )  (26.15%)
           782,938      LLC-load-misses:u         #    0.69% of all LL-cache hits     ( +-  4.06% )  (38.24%)

            5.1732 +- 0.0133 seconds time elapsed  ( +-  0.26% )
```

```
# Overhead  Command   Shared Object                                     Symbol
# ........  ........  ................................................  .................................................................
#
    60.74%  python    _mt19937.cpython-38-x86_64-linux-gnu.so           [.] __pyx_f_5numpy_6random_8_mt19937_mt19937_double
    13.21%  python    [unknown]                                         [k] 0xffffffff9c992fd8
     9.09%  python    _mt19937.cpython-38-x86_64-linux-gnu.so           [.] mt19937_gen
     6.65%  python    _multiarray_umath.cpython-38-x86_64-linux-gnu.so  [.] DOUBLE_add
     5.90%  python    mtrand.cpython-38-x86_64-linux-gnu.so             [.] random_standard_uniform_fill
     0.82%  python    [unknown]                                         [k] 0xffffffff9c9894ef
     0.21%  bash      [unknown]                                         [k] 0xffffffff9c9894ef
     0.19%  python    python3.8                                         [.] _PyEval_EvalFrameDefault
     0.19%  python    libc-2.17.so                                      [.] __sched_yield
     0.13%  python    python3.8                                         [.] lookdict_unicode_nodummy
     0.11%  python    libopenblasp-r0-ae94cfde.3.9.dev.so               [.] blas_thread_server
     0.11%  bash      [unknown]                                         [k] 0xffffffff9c992fd8
     0.08%  python    python3.8                                         [.] collect.constprop.36
     0.07%  python    python3.8                                         [.] lookdict_unicode
     0.05%  python    python3.8                                         [.] _PyObject_Malloc
     0.05%  python    python3.8                                         [.] PyObject_GenericGetAttr
     0.05%  python    python3.8                                         [.] PyParser_AddToken
     0.04%  python    python3.8                                         [.] _PyType_Lookup
     0.04%  python    python3.8                                         [.] vgetargskeywords
     0.04%  python    python3.8                                         [.] visit_decref
     0.04%  python    libc-2.17.so                                      [.] __memcpy_ssse3_back
     0.03%  python    python3.8                                         [.] update_one_slot
     0.03%  python    python3.8                                         [.] _PyObject_Free
     0.03%  bash      libc-2.17.so                                      [.] __gconv_transform_utf8_internal
     0.03%  python    libc-2.17.so                                      [.] _int_malloc
     0.03%  python    libopenblasp-r0-ae94cfde.3.9.dev.so               [.] sched_yield@plt
     0.03%  python    python3.8                                         [.] _Py_Dealloc
     0.03%  python    python3.8                                         [.] frame_dealloc
     0.03%  python    python3.8                                         [.] visit_reachable
     0.03%  python    python3.8                                         [.] tupledealloc
     0.03%  python    python3.8                                         [.] r_object
     0.03%  python    ld-2.17.so                                        [.] do_lookup_x
     0.03%  python    _common.cpython-38-x86_64-linux-gnu.so            [.] __pyx_f_5numpy_6random_7_common_double_fill
     0.02%  python    python3.8                                         [.] PyDict_GetItemWithError
     0.02%  python    python3.8                                         [.] find_name_in_mro
     0.02%  python    ld-2.17.so                                        [.] _dl_lookup_symbol_x
     0.02%  python    python3.8                                         [.] PyTuple_New
     0.02%  python    python3.8                                         [.] PyType_IsSubtype
     0.02%  python    python3.8                                         [.] dict_traverse
     0.02%  python    _multiarray_umath.cpython-38-x86_64-linux-gnu.so  [.] diophantine_dfs
     0.02%  head      [unknown]                                         [k] 0xffffffff9c9894ef
     0.02%  python    python3.8                                         [.] siphash24
     0.02%  python    python3.8                                         [.] _PyDict_LoadGlobal
     0.02%  pyenv-ve  [unknown]                                         [k] 0xffffffff9c9894ef
     0.02%  bash      libc-2.17.so                                      [.] _int_malloc
     0.02%  python    python3.8                                         [.] _PyUnicode_FromUCS1
     0.02%  python    python3.8                                         [.] _PyObject_MakeTpCall
     0.02%  python    libc-2.17.so                                      [.] _int_free
     0.02%  python    _multiarray_umath.cpython-38-x86_64-linux-gnu.so  [.] PyUFunc_GenericFunction_int
     0.02%  python    mtrand.cpython-38-x86_64-linux-gnu.so             [.] PyThreadState_Get@plt
     0.02%  python    python3.8                                         [.] PyNode_AddChild
     0.02%  python    python3.8                                         [.] PyDict_SetItem
     0.01%  python    python3.8                                         [.] convertitem
     0.01%  python    python3.8                                         [.] tok_get
     0.01%  python    python3.8                                         [.] PyType_GenericAlloc
     0.01%  python    python3.8                                         [.] unsafe_latin_compare
     0.01%  python    libc-2.17.so                                      [.] malloc
     0.01%  python    python3.8                                         [.] PyCode_Optimize
```

Here we observe, a little surprisingly, that the python code runs about 12% faster than the corresponding C++ code. However it is also effectively utilizing 1.123 cpus, so per cpus it is equivalent. Of course it is nice that python provides this multicore acceleration _for free_ so to speak, but it is not a fundamental speedup. Next we note a slew of python function calls, but there don't amount to much expense, and 90% of the time is spent in the random number generator. Since this basic code does not in fact do much, it makes sense that python (with numpy) runs as fast as C++. This is a good starting point for python.

### Python class

Although we won't be able to use python classes per say (except as wrappers) for JAX vectorization, I am curious about the performance impact of packaging the particle information inside a python class, in `py-class.py`:

```python
#!/usr/bin/env python

# py-class.py: try to process particles as an array of Particle class objects

import numpy
import time

# defines
NUM_PARTICLES = int(1e5)
ITERATIONS = 10
RUNS = 10

# particle class
class Particle:

    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.u = 0
        self.v = 0
        self.w = 0
        self.E = 0

    def update(self, random_values):
        self.x += random_values[0]
        self.y += random_values[1]
        self.z += random_values[2]
        self.u += random_values[3]
        self.v += random_values[4]
        self.w += random_values[5]
        self.E += random_values[6]



# title
print("--------------------------------------")
print("Python with list of particle class: particles = [Particle() for _ in range(NUM_PARTICLES)]")
print("--------------------------------------")

# timer decorator
def timer(f):
    def wrap(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        stop = time.time()
        duration = (stop-start)*1000.0
        print('run {}: {:.3f} ms'.format(run, duration))
        return ret
    return wrap

# runIterations
def runIterations(particles):
    for j in range(ITERATIONS):
        for p in particles:
            p.update(numpy.random.rand(7))

# timed version of runIterations
runIterations = timer(runIterations)

# main function
def main(run):

    numpy.random.seed()
    particles = [Particle() for _ in range(NUM_PARTICLES)]
    runIterations(particles)

# call main function
for run in range(RUNS):
    main(run)


print("--------------------------------------")
```

Performance in this case is about **60 times slower,** presumably owing to the fact that numpy is no longer able to optimize numerical array operations. This is borne out by `perf report` which shows that most of the time is spent on array and dict operations, and random number generation is relegated to using only about 3.5% of the entire execution time. This data structuring comes at too high a price in python, and this is not really a surprise, but it means that options to structure data in python may be restricted in order to preserve performance.

### Python class-wrapped flat arrays

Flat arrays provide code efficiency, but object classes provide clearer data structure and cleaner modular code; perhaps we can simply wrap the flat arrays into a class for convenience of passing around the particle arrays as a single object reference, for example in `py-wrap.py`:

```python
#!/usr/bin/env python

# py-wrap.py: python/numpy implementation with flat arrays, but wrapped in an
# object api.

import numpy
import time

# defines
NUM_PARTICLES = int(1e5)
ITERATIONS = 100
RUNS = 10

# particle class
class ParticleArray:

    def __init__(self):
        self.x = numpy.zeros(NUM_PARTICLES)
        self.y = numpy.zeros(NUM_PARTICLES)
        self.z = numpy.zeros(NUM_PARTICLES)
        self.u = numpy.zeros(NUM_PARTICLES)
        self.v = numpy.zeros(NUM_PARTICLES)
        self.w = numpy.zeros(NUM_PARTICLES)
        self.E = numpy.zeros(NUM_PARTICLES)

    def update(self, random_values):
        self.x += random_values[0,:]
        self.y += random_values[1,:]
        self.z += random_values[2,:]
        self.u += random_values[3,:]
        self.v += random_values[4,:]
        self.w += random_values[5,:]
        self.E += random_values[6,:]

# title
print("--------------------------------------")
print("Python class wrapped plain numpy arrays: class Particle: self.x = numpy.zeros(NUM_PARTICLES) etc.")
print("--------------------------------------")

# timer decorator
def timer(f):
    def wrap(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        stop = time.time()
        duration = (stop-start)*1000.0
        print('run {}: {:.3f} ms'.format(run, duration))
        return ret
    return wrap

# runIterations
def runIterations(particles):
    for j in range(ITERATIONS):
        random_values= numpy.random.rand(7, NUM_PARTICLES)
        particles.update(random_values)

# timed version of runIterations
runIterations = timer(runIterations)

# main function
def main(run):

    numpy.random.seed()
    particles = ParticleArray()
    runIterations(particles)

# call main function
for run in range(RUNS):
    main(run)

print("--------------------------------------")
```

```
$ perf stat -d --repeat 10 py-wrap.py
# (...)
--------------------------------------
Python class wrapped plain numpy arrays: class Particle: self.x = numpy.zeros(NUM_PARTICLES) etc.
--------------------------------------
run 0: 503.983 ms
run 1: 499.932 ms
run 2: 500.239 ms
run 3: 498.597 ms
run 4: 499.120 ms
run 5: 497.420 ms
run 6: 495.669 ms
run 7: 495.976 ms
run 8: 494.994 ms
run 9: 495.801 ms
--------------------------------------

 Performance counter stats for 'py-wrap.py' (10 runs):

          6,034.76 msec task-clock:u              #    1.132 CPUs utilized            ( +-  0.04% )
                 0      context-switches:u        #    0.000 K/sec
                 0      cpu-migrations:u          #    0.000 K/sec
            53,344      page-faults:u             #    0.009 M/sec                    ( +-  0.01% )
    19,228,510,673      cycles:u                  #    3.186 GHz                      ( +-  0.16% )  (49.61%)
    54,505,892,610      instructions:u            #    2.83  insn per cycle           ( +-  0.14% )  (62.13%)
     4,348,197,973      branches:u                #  720.525 M/sec                    ( +-  0.08% )  (62.30%)
        10,565,548      branch-misses:u           #    0.24% of all branches          ( +-  0.47% )  (62.14%)
     8,618,701,498      L1-dcache-loads:u         # 1428.175 M/sec                    ( +-  2.08% )  (35.57%)
       275,279,622      L1-dcache-load-misses:u   #    3.19% of all L1-dcache hits    ( +-  0.97% )  (31.95%)
       114,026,320      LLC-loads:u               #   18.895 M/sec                    ( +-  0.57% )  (25.27%)
        14,380,536      LLC-load-misses:u         #   12.61% of all LL-cache hits     ( +-  1.05% )  (37.28%)

            5.3332 +- 0.0113 seconds time elapsed  ( +-  0.21% )
```

As expected, this is essentially as efficient (3% slower) as the flat array implementation (but cleaner and more modular): So it seems at least possible to use this approach as a compromise between code modularity and numerical efficiency in a python implementation.

### Python JAX implementation

We now turn to the JAX library, and first look at the impact of using `jax.numpy` instead of the standard `numpy` library, using the following source code `py-jaxnumpy.py`:

```python
#!/usr/bin/env python

# py-jaxnumpy.py: python/jax.numpy implementation with flat arrays
# requires the JAX library: pip install --upgrade jax jaxlib

import jax
import time

# defines
NUM_PARTICLES = int(1e5)
ITERATIONS = 100
RUNS = 10

# title
print("--------------------------------------")
print("Bare jax.numpy arrays: x = jax.numpy.zeros(NUM_PARTICLES)")
print("--------------------------------------")

# timer decorator
def timer(f):
    def wrap(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        stop = time.time()
        duration = (stop-start)*1000.0
        print('run {}: {:.3f} ms'.format(run, duration))
        return ret
    return wrap

# runIterations
def runIterations(prng_key, x, y, z, u, v, w, E):
    for j in range(ITERATIONS):

        random_values = jax.random.normal(prng_key, shape=(7, NUM_PARTICLES))
        (prng_key, ) = jax.random.split(prng_key, 1)

        x += random_values[0,:]
        y += random_values[1,:]
        z += random_values[2,:]
        u += random_values[3,:]
        v += random_values[4,:]
        w += random_values[5,:]
        E += random_values[6,:]

    return prng_key, x, y, z, u, v, w, E

# timed version of runIterations
runIterations = timer(runIterations)

# main function
def main(run):

    seed = 0
    prng_key = jax.random.PRNGKey(seed)

    x = jax.numpy.zeros(NUM_PARTICLES)
    y = jax.numpy.zeros(NUM_PARTICLES)
    z = jax.numpy.zeros(NUM_PARTICLES)
    u = jax.numpy.zeros(NUM_PARTICLES)
    v = jax.numpy.zeros(NUM_PARTICLES)
    w = jax.numpy.zeros(NUM_PARTICLES)
    E = jax.numpy.zeros(NUM_PARTICLES)

    (prng_key, x, y, z, u, v, w, E) = runIterations(prng_key, x, y, z, u, v, w, E)

# call main function
for run in range(RUNS):
    main(run)

print("--------------------------------------")
```

```
$ perf stat -d --repeat 10 py-jaxnumpy.py
# (...)
--------------------------------------
Bare jax.numpy arrays: x = jax.numpy.zeros(NUM_PARTICLES)
--------------------------------------
WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
run 0: 1682.651 ms
run 1: 1366.705 ms
run 2: 1319.231 ms
run 3: 1326.533 ms
run 4: 1337.960 ms
run 5: 1304.933 ms
run 6: 1372.836 ms
run 7: 1344.103 ms
run 8: 1341.807 ms
run 9: 1345.570 ms
--------------------------------------

 Performance counter stats for 'py-jaxnumpy-nowrap.py' (10 runs):

         28,816.96 msec task-clock:u              #    1.936 CPUs utilized            ( +-  0.18% )
                 0      context-switches:u        #    0.000 K/sec
                 0      cpu-migrations:u          #    0.000 K/sec
           562,451      page-faults:u             #    0.020 M/sec                    ( +-  2.80% )
    80,712,379,604      cycles:u                  #    2.801 GHz                      ( +-  0.23% )  (48.62%)
    74,155,590,960      instructions:u            #    0.92  insn per cycle           ( +-  0.10% )  (61.16%)
    10,545,433,364      branches:u                #  365.945 M/sec                    ( +-  0.08% )  (60.96%)
       256,488,797      branch-misses:u           #    2.43% of all branches          ( +-  0.30% )  (60.85%)
    18,352,073,999      L1-dcache-loads:u         #  636.850 M/sec                    ( +-  0.33% )  (41.10%)
     2,086,293,451      L1-dcache-load-misses:u   #   11.37% of all L1-dcache hits    ( +-  0.27% )  (28.50%)
       862,397,166      LLC-loads:u               #   29.927 M/sec                    ( +-  0.26% )  (26.28%)
        36,681,347      LLC-load-misses:u         #    4.25% of all LL-cache hits     ( +-  2.59% )  (36.82%)

           14.8841 +- 0.0498 seconds time elapsed  ( +-  0.33% )
```

We now find that the code runs nearly 5 times slower (looking at the `task-clock` metric only). There is better utilization of the cores (1.942 cpus utilized, so the wall clock is only about twice that of the C++ reference). So there is a performance hit in the way JAX manipulates the data internally. This is confirmed by the `perf report` (after generating with `perf record py-jaxnumpy.py`): we notice a very long slew of `xla` and `memcpy` operations (too much to list here) taking up lots of processor time. The gambit here is to accept this loss of performance for the moment, and check whether it can be compensated by JIT and GPU acceleration down the line, through the JAX library.

Note that the numpy arrays are not class-wrapped anymore, because the point of using JAX is to compile for the GP device using `jax.jit` compilation, and `jit` will complain if the `runIterations` function takes non-basic types as argument. However, [@SimonBiggs explained how](https://github.com/nrc-cnrc/EGSnrc/discussions/658#discussioncomment-257722) it is nevertheless possible to package particles neatly inside a dictionary for some level of encapsulation with JAX, e.g.

```python
particles: Particles = {
      "x": jax.numpy.zeros((3, NUM_PARTICLES)),
      "y": jax.numpy.zeros((3, NUM_PARTICLES)),
      "z": jax.numpy.zeros((1, NUM_PARTICLES)),
      # etc.
  }
```

### Python using @jax.jit compilation (on the CPU)

Let's now simply turn on the JIT compilation in `py-jit.py` by adding the `@jax.jit` decorator to the `runIterations` function (or, equivalently, redefining `runIterations = jax.jit(runIterations)). We find the following performance results:

```
$ perf stat -d py-jaxjit.py
--------------------------------------
Bare jax.numpy arrays: x = jax.numpy.zeros(NUM_PARTICLES)
--------------------------------------
WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
run 0: 40057.053 ms
run 1: 0.795 ms
run 2: 656.621 ms
run 3: 659.458 ms
run 4: 656.339 ms
run 5: 659.087 ms
run 6: 657.162 ms
run 7: 661.003 ms
run 8: 666.534 ms
run 9: 661.401 ms
--------------------------------------

 Performance counter stats for 'py-jaxjit.py':

         66,797.09 msec task-clock:u              #    1.428 CPUs utilized
                 0      context-switches:u        #    0.000 K/sec
                 0      cpu-migrations:u          #    0.000 K/sec
         2,053,662      page-faults:u             #    0.031 M/sec
   222,853,805,967      cycles:u                  #    3.336 GHz                      (48.89%)
   286,339,544,211      instructions:u            #    1.28  insn per cycle           (61.42%)
    51,326,962,581      branches:u                #  768.401 M/sec                    (61.24%)
       691,003,649      branch-misses:u           #    1.35% of all branches          (61.00%)
    71,307,548,279      L1-dcache-loads:u         # 1067.525 M/sec                    (30.18%)
     5,767,628,926      L1-dcache-load-misses:u   #    8.09% of all L1-dcache hits    (34.12%)
     1,537,240,955      LLC-loads:u               #   23.014 M/sec                    (25.64%)
       248,774,646      LLC-load-misses:u         #   16.18% of all LL-cache hits     (36.91%)

      46.781237677 seconds time elapsed

      62.687542000 seconds user
       5.373824000 seconds sys
```

Interestingly, there is now a huge latency cost, the reason of which is unclear: presumably to setup and initialize `jit`, perhaps a costly operation when there is no GPU available? Note however that after the first run, the timing is only about 30% worse than the C++ reference for each run. At any rate we will need to be able to compile specifically for the CPU _or_ the GPU But this is easily achieved by importing the appropriate libaries depending on the context, for example in `py-jaxornot.py` shown below. The conditional on a `USE_JAX` global is inelegant, of course, but the point is that it seems possible, from a single source code, to select a CPU `numpy` run with near C++ performance, or jit compilation for a GPU target.

```python
#!/usr/bin/env python

# py-jaxornot.py: python/jax.numpy implementation with flat arrays
# requires the JAX library: pip install --upgrade jax jaxlib

# use jax or not?
USE_JAX = False

import time

if (USE_JAX):
    import jax
    import jax.numpy as numpy
    import jax.random as random
else:
    import numpy
    import numpy.random as random

# defines
NUM_PARTICLES = int(1e5)
ITERATIONS = 100
RUNS = 10

# title
print("--------------------------------------")
print("Bare jax.numpy arrays: x = jax.numpy.zeros(NUM_PARTICLES)")
print("--------------------------------------")

# timer decorator
def timer(f):
    def wrap(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        stop = time.time()
        duration = (stop-start)*1000.0
        print('run {}: {:.3f} ms'.format(run, duration))
        return ret
    return wrap

# runIterations
def runIterations(prng_key, x, y, z, u, v, w, E):
    for j in range(ITERATIONS):

        if (USE_JAX):
            random_values = random.normal(prng_key, shape=(7, NUM_PARTICLES))
            (prng_key, ) = random.split(prng_key, 1)
        else:
            random_values= numpy.random.rand(7, NUM_PARTICLES)

        x += random_values[0,:]
        y += random_values[1,:]
        z += random_values[2,:]
        u += random_values[3,:]
        v += random_values[4,:]
        w += random_values[5,:]
        E += random_values[6,:]

    return prng_key, x, y, z, u, v, w, E

# jit compilation if using jax
if (USE_JAX):
    runIterations = jax.jit(runIterations)

# timed version of runIterations
runIterations = timer(runIterations)

# main function
def main(run):

    seed = 0
    if (USE_JAX):
        prng_key = random.PRNGKey(seed)
    else:
        prng_key = random.seed(seed)

    x = numpy.zeros(NUM_PARTICLES)
    y = numpy.zeros(NUM_PARTICLES)
    z = numpy.zeros(NUM_PARTICLES)
    u = numpy.zeros(NUM_PARTICLES)
    v = numpy.zeros(NUM_PARTICLES)
    w = numpy.zeros(NUM_PARTICLES)
    E = numpy.zeros(NUM_PARTICLES)

    (prng_key, x, y, z, u, v, w, E) = runIterations(prng_key, x, y, z, u, v, w, E)

# call main function
for run in range(RUNS):
    main(run)

print("--------------------------------------")
```

### Python using @jax.jit compilation for the GPU

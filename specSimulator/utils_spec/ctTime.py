import numpy as np
import matplotlib.pyplot as plt
import coloralf as c
from time import time
from contextlib import contextmanager



class ctTime():

    def __init__(self, name='CTTIME', unit=None, verbose=True, nbLoop=None):

        t0 = time()

        self.defineFactPrec(unit)
        self.times = dict()
        self.t0 = dict()
        self.w2rank = dict()
        self.rank2time = dict()
        self.begin = None
        self.verbose = verbose
        self.nbLoop = 0
        self.nbLoopTotal = nbLoop
        self.running = False

        self.total = list()

        self.selfTime = time() - t0

    def o(self, who, rank="None"):

        t0 = time()

        if who not in self.times.keys(): 
            self.times[who] = [0.0]
            self.w2rank[who] = rank
            self.t0[who] = None
            if rank not in self.rank2time.keys() : self.rank2time[rank] = [[0.0],  [who]]
            else : self.rank2time[rank][1].append(who)
        if self.t0[who] is None:
            self.t0[who] = time()
        else:
            print(f"{c.r}WARNING : {who} is already on.{c.d}")

        self.selfTime += time() - t0

    def c(self, who):

        t0 = time()
        if who in self.times.keys():
            toAdd = time() - self.t0[who]
            self.times[who][-1] += toAdd
            self.rank2time[self.w2rank[who]][0][-1] += toAdd
            self.t0[who] = None
        else:
            print(f"{c.r}WARNING : {who} not exists.")
        self.selfTime += time() - t0

    @contextmanager
    def go(self, who, rank="None"):

        self.o(who, rank)
        yield
        self.c(who)


    def run(self):

        t0 = time()

        self.running = True
        self.begin = time()

        self.selfTime += time() - t0


    def newLoop(self):

        t0 = time()

        if self.begin is not None:

            self.total.append(time() - self.begin)
            self.nbLoop += 1

            for k in self.times.keys():
                self.times[k].append(0.0)

            for rank in self.rank2time.keys():
                self.rank2time[rank][0].append(0.0)

        else:
            self.running = True

        self.begin = time()
        self.selfTime += time() - t0


    def defineFactPrec(self, unit):

        if   unit is None  : self.unit, self.fact, self.prec = None, None, None
        elif unit == "h"   : self.fect, self.prec = 1/3600, 2
        elif unit == "min" : self.fact, self.prec = 1/60,   2
        elif unit == "sec" : self.fact, self.prec = 1.0,    2
        elif unit == "ms"  : self.fact, self.prec = 1e3,    1
        elif unit == "µs"  : self.fact, self.prec = 1e6,    1
        elif unit == "ns"  : self.fact, self.prec = 1e9,    1
        else:
            print(f"{c.r}WARNING : {unit} not exist. Try `h`, `min`, `sec`, `ms`, `µs`.{c.d}")
            self.unit, self.fact, self.prec = None, None, None

    def defineUnit(self, t):

        if  t > 1800.0 : self.unit = "h"
        elif t > 600.0 : self.unit = "min"
        elif t > 5.0   : self.unit = "sec"
        elif t > 5e-3  : self.unit = "ms"
        else : self.unit = "µs"

        self.defineFactPrec(self.unit)

    def time2str(self, t):

        return f"{np.mean(t)*self.fact:.{self.prec}f} ~ {np.std(t)*self.fact:.{self.prec}f} {self.unit}"

    def time2pc(self, t, tt):

        return f"{np.sum(t)/np.sum(tt)*100:4.1f}%"


    def result(self):

        self.total.append(time() - self.begin)
        self.t = np.sum(self.total)
        self.nbLoop += 1
        
        if self.unit is None : self.defineUnit(np.mean(self.total))
        print(f"{c.lm}Result of ctTime with {self.nbLoop} loop : {self.time2str(self.total)}{c.d}")
        print(f"{c.m}Total selfTime   : {self.selfTime*1e3:.2f} ms / {self.selfTime*1e3/self.nbLoop:.2f} ms per loop{c.d}")

        if self.verbose:

            for rank, rankTime in self.rank2time.items():

                print(f"\n{c.g}For rank {c.lg}{rank}{c.g}, total={self.time2str(rankTime[0])}:{c.d}")

                maxLen = max([len(w) for w in rankTime[1]])

                for who in rankTime[1]:
                    print(f"{c.ly}* {who:{maxLen}} : {c.y}[{self.time2pc(self.times[who], self.total)}]{c.ly}[{self.time2pc(self.times[who], rankTime[0])}] {self.time2str(self.times[who])}{c.d}")



def delete_ctt(file):

    print(f"{c.y}Re-write {file}.py to {file}_true.py{c.d}")

    with open(f"./SpecSimulator/{file}.py", "r") as f:

        lines = f.read().split("\n")

    newPy = list()
    onDelation = False
    indentation = None

    for line in lines:

        if "self.ctt" in line:

            pass

        else:

            newPy.append(line)

    with open(f"./SpecSimulator/{file}_true.py", "w") as f:
        f.write("\n".join(newPy))







# coding: utf-8
import RRGA as GAs
import random


if __name__ == "__main__":
    seed = 2
    random.seed(seed)
    nvariable = 16
    nrange = 4
    ngeneration = 1000
    npop = 200
    f_types = {1:"Sphere",2:"Rastrigin",3:"Ackley",4:"Rosenbrock",5:"Rosenbrock_star",6:"Rosenbrock_chain"}
    f_type = 4
    f_name = f_types[f_type]
    
    ga = GAs.RRGA(nvariable,nrange,ngeneration,npop,f_type,f_name,seed)
    ga.initFILEs()
    ga.optimize()

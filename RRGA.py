# coding: utf-8
import csv
import copy
import math
import time
import random
import numpy as np
import os
import datetime
import collections
import shutil

class RRGA:
    def __init__(self,nvariable,nrange,ngeneration,npop,f_type,f_name,seed):
        self.seed = seed
        random.seed(self.seed)

        #-- parameters used in the optimization process --#
        # parameters related to the solution of individuals
        self.nvariable=nvariable
        self.nbit=5
        self.totalbit=self.nvariable*self.nbit
        
        # parameters related to the search ranges
        self.UB=0.99
        self.LB=0.2
        self.ppp=math.sqrt(-2*math.log(self.LB))
        self.pp=math.sqrt(-2*math.log(self.LB+(self.UB-self.LB)/(pow(2,self.nbit-1)-1)))
        self.nrange=4
        self.mrange = -1
        self.aaafit = 0.0

        # parameter related to individuals
        self.npop=npop
        self.ncross=int(self.npop*0.3)
        self.imin = -1

        # upper and lower limits for each parameter
        self.smin=1.0e-6
        self.smax=3
        self.indmax=700
        self.funcmax=1
        self.variablemax=256

        # other paraneters
        self.immigrant=10               # the number of individuals mutate
        self.ngeneration=ngeneration    # the number of generation
        self.g = 0                      # generations has passed since optimization starts
        self.GR=[110+i for i in range(20)]


        #-- lists used in the optimization process --#
        # values that each individual has
        self.func=[0.0 for i in range(self.indmax*self.funcmax)]        # function value of each individual
        self.fitness=[0.0 for i in range(self.indmax*self.funcmax)]     # fitness value of each individual
        self.x=[0.0 for i in range(self.indmax*self.variablemax)]       # the phenotype solution that each individual has
        self.ix=[0 for i in range(self.indmax*self.variablemax)]         
        self.ws=[0 for i in range(self.indmax*self.totalbit)]
        self.xstd=[0 for i in range(nvariable)]                         # the standard diviation of all individuals
        self.tave=[0 for i in range(nvariable)]                         # the average of design variable values for all individuals
        self.prange=[-1 for i in range(self.indmax)]                    # the index of each search range to which each individual belong
        self.xgreat=[0 for i in range(nvariable)]                       # design variable value of the best individual
        self.fgreat=1.0e33                                              # function value of the best individual
        self.roulette=[0 for i in range(self.indmax)]                   # value on the roulette board of each individual
        self.zflg=[0 for i in range(self.npop)]

        # parameter related to each search range
        self.xave=[0 for i in range(nvariable*self.nrange)]             # the center of each search range
        self.tstd=[0 for i in range(nvariable*nrange)]                  # the standard diviation of individuals in each search range
        self.xrstd=[0 for i in range(nvariable*nrange)]                 # the standard diviation of individuals in each search range
        self.xlstd=[0 for i in range(nvariable*nrange)]                 # the standard diviation of individuals in each search range
        self.rangepop=[0 for i in range(nrange)]                        # the number of individuals in each search range
        self.age=[0 for i in range(nrange)]                             # the age of each search range
        self.whoisit=[-1 for i in range(nrange)]                        # the index of the best individual in each search range
        self.rangetop=[1.0e33 for i in range(nrange)]                   # function value of the best individual in each search range
        self.rangewho=[-1 for i in range(nrange)]                       # the index of the best individual in each search range
        self.minpop=int(npop/nrange*0.5)                                # the minimum number of individuals in a seach range
        self.rangenext=[1.0e33 for i in range(nrange)]                  # function value of the best individual in each search range
        self.ws2=[1.0e33 for i in range(npop+self.ncross)]              # the list for coping the other lists
        self.greatAge = 0                                               # Time elapsed since the last best solution was obtained
        self.fgreat_range = [1.0e+33 for i in range(nrange)]
        self.elite_range = [-1 for i in range(nrange)]

        # lists related to Hoxgene of each individual
        self.hgncross=self.ncross*2
        self.hoximin=-1
        self.hoxfgreat=-1.0e33
        self.hoxgreat=[-1 for i in range(nvariable)]
        self.ws3=[-1 for i in range(self.npop)]                         # the index of parents in which each child inherits the gene
        self.ws4=[1.0e33 for i in range(npop+self.hgncross)]
        self.HGfitness=[0.0 for i in range(self.npop+self.hgncross)]
        self.HGphenotype=[0.0 for i in range(self.npop+self.hgncross)]
        self.hoxzflg=[0 for i in range(self.npop)]
        self.pfitness=[0.0 for i in range(self.npop)]
        self.hoxflg=False
        self.nonhoxflg=False
        self.hoxage=0
        self.nonhoxage=0
        self.hoxbegin=300
        self.hoxcnt=0
        self.hoxterm=30
        self.nonhoxterm=200
        self.upperT1=4
        self.lowerT1=2
        self.usualgene=[0 for i in range(self.totalbit*(self.npop+self.ncross))]
        self.realtype_x=[0.0 for i in range(self.npop*self.nvariable)]
        self.f_value=0.0
        # self.Hoxgene=[random.randint(0,1) for i in range(nvariable*(self.npop+self.hgncross))]
        self.Hoxgene=[0 for i in range(nvariable*(self.npop+self.hgncross))]
        self.realHox=[[] for i in range(self.npop+self.hgncross)]
        for ipop in range(self.npop+self.hgncross):
            sflg=[0 for i in range(self.nvariable)]
            cnt=0
            standnum=random.randint(1,3)
            while cnt < standnum: 
                key = random.randint(0,self.nvariable-1)
                if sflg[key] == 1:
                    pass
                else:
                    sflg[key]=1
                    cnt+=1
                    self.Hoxgene[ipop*self.nvariable+key]=1
        
        # parameters related to the search considering dependencies between variables
        self.npair = 3                              # the number of variable pairs
        self.selectedHox=[]
        self.P = []                                 # the set of variable pairs that have dependencies on each other
        self.Pi = -1                                # the variable pair we are referencing now
        self.psearch_st=False                       # this is the flag indicate the begining of the search considering dependencies between variables
        self.psearch_flg=False                      # this flag indicates whether the search considering dependensies between variables is being conducted
        self.psearch_age=0                          # this age indicates the generation that have passed since the pair search was started.
        self.pgreatAge=0
        self.newcnt=0
        self.pcnt=0
        self.locrange=-1
        self.llmax=10
        self.updateTHRESH=1
        self.termlimit=200                          # the limit of generations to continue searching for a pair 
        # variables for saving the value of each parameter at the beging of the search
        # information about the specific individual
        self.xgreat_init=[0.0 for i in range(self.nvariable)]
        self.fgreat_init=1.0e+33
        # information about all individuals
        self.xstd_init=[0.0 for i in range(self.nvariable)]
        self.tave_init=[0.0 for i in range(self.nvariable)]
        self.fitness_init=[0.0 for i in range(self.npop+self.ncross)]
        self.func_init=[0.0 for i in range(self.npop+self.ncross)]
        self.x_init=[0.0 for i in range(self.nvariable*(self.npop+self.ncross))]
        self.GAstring_init=[0 for i in range(self.totalbit*(self.npop+self.ncross))]
        # information about search range
        self.xave_init=[0.0 for i in range(self.nvariable*self.nrange)]
        self.xrstd_init=[0.0 for i in range(self.nvariable*self.nrange)]
        self.xlstd_init=[0.0 for i in range(self.nvariable*self.nrange)]
        self.tstd_init=[0.0 for i in range(self.nvariable*self.nrange)]

        
        # parameters used to judge the convergence
        self.pxstd=[0.0 for i in range(self.nvariable)]
        self.pxstd_sum=[0.0 for i in range(self.nvariable)]
        self.inside=[0.0 for i in range(self.nvariable)]
        self.threshold=[0.0 for i in range(self.nvariable)]
        self.error_worst=[0.0 for i in range(self.nvariable)]
        self.judgement=[False for i in range(self.nvariable)]
        self.allgenetypes=self.generateGrayArray()
        self.PASTXSTD=[0.0 for i in range(self.nvariable)]
        self.PASTXSTDSUM=[0.0 for i in range(self.nvariable)]
        self.CONVTHRESH=[0.0 for i in range(self.nvariable)]
        self.CONVJUDGE=[0.0 for i in range(self.nvariable)]
        self.deltaF=[]
        self.deltaFi=[]
        self.PAIR=[]

        # debug
        self.s_time=0


        # lists for recoding the data in the optimization process
        self.fgreats = []
        self.aaafits = []


        # the information about Evaluation function
        self.f_type = f_type
        self.f_name = f_name

        # upper and lower limits for each design variable
        self.xmax, self.xmin=self.getlim()                        # the upper and lower limits for each design variable
        self.kkmin=0.5
        
        #-- generate initial individuals --#
        initialpoint = self.generate_initxave()
        for irange in range(self.nrange):
            for ivar in range(self.nvariable):
                self.xave[self.nvariable*irange+ivar]=initialpoint[irange][ivar]
                self.xrstd[self.nvariable*irange+ivar]=(self.xmax[ivar]-self.xave[nvariable*irange+ivar])/self.ppp
                self.xlstd[self.nvariable*irange+ivar]=(self.xave[self.nvariable*irange+ivar]-self.xmin[ivar])/self.ppp
        
        self.GAstring=[random.randint(0,1) for i in range(self.totalbit*(self.indmax))]     # the genotype solution that each individual has
        # print(self.GAstring[0:self.totalbit*self.npop])
        for ipop in range(self.npop):
            self.prange[ipop]=random.randint(0,self.nrange-1)
            self.rangepop[self.prange[ipop]]+=1
            self.ARange(ipop)
        
        for ipop in range(npop):
            self.functioncall(ipop)
        
        for i in range(self.nvariable):
            self.tave[i]=0
            for ipop in range(self.npop):
                self.tave[i]+=self.x[self.nvariable*ipop+i]
            self.tave[i]/=self.npop
            self.xstd[i]=0
            for ipop in range(self.npop):
                self.xstd[i]+=pow(self.x[self.nvariable*ipop+i]-self.tave[i],2)
            self.xstd[i]/=self.npop
            self.xstd[i]=math.sqrt(self.xstd[i])
        

        for ipop in range(self.npop+self.hgncross):
            sflg=[0 for i in range(self.nvariable)]
            cnt=0
            standnum=random.randint(2,4)
            while cnt < standnum: 
                key = random.randint(0,self.nvariable-1)
                if sflg[key] == 1:
                    pass
                else:
                    sflg[key]=1
                    cnt+=1
                    self.Hoxgene[ipop*self.nvariable+key]=1
    

    # optimization
    def optimize(self):
        for generation in range(self.ngeneration):
            start=time.time()
            # start=time.time()
            self.writeEachIter()
            # print("writeEach="+str(time.time()-start))

            # start=time.time()
            self.writeDebug()
            # print("writeDebug="+str(time.time()-start))

            # start=time.time()
            self.HoxSwitch(generation)
            # print("Hoxswitch="+str(time.time()-start))

            # start=time.time()
            self.CrossOverPHASE()
            # print("CrossOver="+str(time.time()-start))

            # start=time.time()
            self.NaturalSelectPHASE()
            # print("NaturalSelect="+str(time.time()-start))

            # start=time.time()
            self.MutationPHASE()
            # print("Mutation="+str(time.time()-start))

            # start=time.time()
            self.PSwitch()
            # print("Pswitch="+str(time.time()-start))

            # start=time.time()
            self.UpdateRangePhase()
            print("Generation="+str(self.g)+"fgreatest="+str(self.fgreat)+" minimum="+str(self.fitness[self.imin])+" min(t) = "+str(min(self.fitness[0:self.npop]))+" average="+str(self.aaafit)+" newborn="+str(self.greatAge))
            # print("UpdateRange="+str(time.time()-start))

            # print("elapsedtime="+str(time.time()-start))
        self.writeResult()



    # function : define the upper and lower limits for each design variable
    def getlim(self):
        xmax = []
        xmin = []
        for ivar in range(self.nvariable):
            # Sphere
            if self.f_type == 1:
                xmax.append(5)
                xmin.append(-5)
            
            # Rastrigin
            elif self.f_type == 2:
                xmax.append(5.12)
                xmin.append(-5.12)
            
            # Ackley
            elif self.f_type == 3:
                xmax.append(32.768)
                xmin.append(-32.768)
            
            # Rosenbrock(A)
            elif self.f_type == 4:
                xmax.append(2)
                xmin.append(-2)

            # Rosenbrock(star)
            elif self.f_type == 5:
                xmax.append(2)
                xmin.append(-2)

            # Rosenbrock(chain)
            elif self.f_type == 6:
                xmax.append(2)
                xmin.append(-2)
        
        return xmax, xmin
    


    # sampling
    def sampling(self, whole_x, nvariable, nsample):
        sample = list()
        i_s = 0

        while i_s < nsample:
            s = list()
            for ivar in range(nvariable):
                rows = range(len(whole_x[ivar]))
                r = random.sample(rows,1)[0]

                s.append(whole_x[ivar][r])

            if s in sample:
                i_s += 0
            else:
                sample.append(s)
                i_s += 1
        # print("sample >>>   " + str(sample))
        
        return sample

    
    # Generate the initial center point of each search area in the solution space in a well-balanced manner
    def generate_initxave(self):
        whole_data = list()
        s_point = list()
        sum_n = 0

        for ivar in range(self.nvariable):
            whole_data.append(list(np.arange(self.xmin[ivar],self.xmax[ivar],0.01))) # 解空間内の全個体を擬似的(粗く)に取得
            sum_n += len(np.arange(self.xmin[ivar],self.xmax[ivar],0.01))
        
        nsample = sum_n * 0.8
        whole_x_t = np.array(whole_data).T
        sample = self.sampling(whole_data, self.nvariable, nsample)                  # 初期点候補集団の解を取得

        sample = np.matrix(sample)
        sample_t = sample.T

        # center point of all samples
        centroid = np.mean(sample, axis=0)

        # distance between centroid and all samples
        dist = sample - centroid
        dist_c = np.linalg.norm(sample - centroid, ord = 2, axis = 1)

        # select the point closest to the centroid as a 1st initial point
        p1_i = np.argmin(dist_c)
        p1 = np.array(sample[p1_i])
        # print("p1 >>>   " + str(p1))


        s_point.append(list(p1[0]))

        # delete the point selected
        sample = np.delete(sample, p1_i, 0)


        # calculate distance between p1 and others
        dist = sample - np.array(p1)
        dist_p1 = np.linalg.norm(dist, ord = 2, axis = 1)


        # get the point fathest from p1
        p2_i = np.argmax(dist_p1)
        p2 = np.array(sample[p2_i])

        # print("p2 >>>   " + str(p2))
        s_point.append(list(p2[0]))

        # delete the point selected
        sample = np.delete(sample, p2_i, 0)


        p_n = self.nrange - len(s_point)

        for iter in range(p_n):
            D = list()
            for i in range(len(s_point)):
                d = list(np.linalg.norm(sample - np.array(s_point[i]), ord = 2, axis = 1))
                D.append(d)
            
            D = np.matrix(D)
            D_t = D.T

            D_min = D_t.min(axis = 1)

            p_i = np.argmax(D_min)
            p = np.array(sample[p_i])

            s_point.append(list(p[0]))
    
        
        return s_point



    def writeDebug(self):
        seed_dir=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"seed"+str(self.seed)))
        result_dir=os.path.join(seed_dir,"result")
        hox_dir=os.path.join(result_dir,"Hox")
        
        hoxeval=os.path.join(hox_dir,"HGfitness.csv")
        with open(hoxeval,"a") as f:
            writer=csv.writer(f,lineterminator='\n')
            writer.writerow(self.HGfitness)
    

    def writeDebug2(self,tmpdata0,tmpdata1,state,npop):
        seed_dir=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"seed"+str(self.seed)))
        result_dir=os.path.join(seed_dir,"result")
        hox_dir=os.path.join(result_dir,"Hox")

        hoxeval_sorted=os.path.join(hox_dir,"HGfitness_sorted.csv")
        with open(hoxeval_sorted,"a") as f:
            writer=csv.writer(f,lineterminator='\n')
            items=[state,"n="+str(len(tmpdata0)),"nt="+str(npop)]
            writer.writerow(items)
            data0=["pair",""]
            data0.extend(tmpdata0)
            writer.writerow(data0)
            data1=["fit",""]
            data1.extend(tmpdata1)
            writer.writerow(data1)
            writer.writerow([""])
    

    # This function generates all n bit Gray Array
    def generateGrayArray(self):
    
        # base case
        if (self.nbit <= 0):
            return
    
        # 'arr' will store all generated codes
        arr = list()
    
        # start with one-bit pattern
        arr.append("0")
        arr.append("1")
    
        # Every iteration of this loop generates
        # 2*i codes from previously generated i codes.
        i = 2
        j = 0
        while(True):
    
            if i >= 1 << self.nbit:
                break
        
            # Enter the prviously generated codes
            # again in arr[] in reverse order.
            # Nor arr[] has double number of codes.
            for j in range(i - 1, -1, -1):
                arr.append(arr[j])
            # print(arr)
            # append 0 to the first half
            for j in range(i):
                arr[j] = "0" + arr[j]
    
            # append 1 to the second half
            for j in range(i, 2 * i):
                arr[j] = "1" + arr[j]
            i = i << 1
    
        # prcontents of arr[]
        # print("arr >>> " + str(arr))
        bit = list()
        bit_rev = list()
        for row in arr:
            bitrow = list(row)
            introw = [int(v) for v in bitrow]
            bit.append(introw)
            rowcpy = copy.deepcopy(introw)
            rowcpy.reverse()
            bit_rev.append(rowcpy)
        
        # print("bit >>> " + str(bit))
        # print("bitrev >>> " + str(bit_rev))

        return bit_rev


    # evaluate the initial Hoxgene
    def init_Hoxgene(self):
        ppfitness=[0.0 for i in range(self.npop)]
        ccfitness=[0.0 for i in range(self.npop)]
        ## save pfitness ##
        for ipop in range(self.npop):
            ppfitness[ipop]=self.fitness[ipop]
        

        ## copy usual gene of each individual ##
        usualgene=[0 for i in range(self.npop*self.totalbit)]
        for ipop in range(self.npop):
            for ibit in range(self.totalbit):
                usualgene[ipop*self.totalbit+ibit]=self.GAstring[ipop*self.totalbit+ibit]
        

        ## copy realtype solution ##
        realtype_x=[0.0 for i in range(self.npop*self.nvariable)]
        realtype_x_k=[0.0 for i in range(self.npop*self.nvariable)]
        diff_x=[0.0 for i in range(self.npop)]
        for ipop in range(self.npop):
            for ivar in range(self.nvariable):
                realtype_x[ipop*self.nvariable+ivar]=self.x[ipop*self.nvariable+ivar]
                realtype_x_k[ipop*self.nvariable+ivar]=self.x[ipop*self.nvariable+ivar]

        ## temporally Cross Over ##
        self.data_pare=[0.0 for i in range(self.npop+self.hgncross)]
        self.data_chil=[0.0 for i in range(self.npop+self.hgncross)]
        self.data_calc=[0.0 for i in range(self.npop+self.hgncross)]
        for i in range(int(0.5*self.npop)):
            pi0=2*i
            pi1=2*i+1

            ## copy another's Hoxgene ##
            # key = random.randint(1,self.nvariable-2)    # cross over point
            # for i in range(key):
            #     # child 0
            #     self.Hoxgene[self.nvariable*pi0+i]=0
            #     # child 1
            #     self.Hoxgene[self.nvariable*pi1+i]=0
            # for i in range(key,self.nvariable):
            #     # child 0
            #     self.Hoxgene[self.nvariable*pi0+i]=self.Hoxgene[self.nvariable*pi1+i]
            #     # child 1
            #     self.Hoxgene[self.nvariable*pi1+i]=self.Hoxgene[self.nvariable*pi0+i]
            
            
            ## guarantee the minimum of standing genes
            # for an individual pi0
            if 1 in self.Hoxgene[self.nvariable*pi0:self.nvariable*pi0+self.nvariable]:
                pass
            else:
                bn=random.randint(1,3)
                sflg=[0 for i in range(self.nvariable)]
                for ibit in range(bn):
                    aflg=True
                    while aflg:
                        kk=random.randint(0,self.nvariable-1)
                        if sflg[kk]==1:
                            aflg=True
                        else:
                            self.Hoxgene[pi0*self.nvariable+kk]=1
                            sflg[kk]=1
                            aflg=False
            
            # for an individual pi1
            if 1 in self.Hoxgene[self.nvariable*pi1:self.nvariable*pi1+self.nvariable]:
                pass
            else:
                bn=random.randint(1,3)
                sflg=[0 for i in range(self.nvariable)]
                for ibit in range(bn):
                    aflg=True
                    while aflg:
                        kk=random.randint(0,self.nvariable-1)
                        if sflg[kk]==1:
                            aflg=True
                        else:
                            self.Hoxgene[pi1*self.nvariable+kk]=1
                            sflg[kk]=1
                            aflg=False


            ## update usual gene ##
            for ivar in range(self.nvariable):
                # about pi0
                if self.Hoxgene[self.nvariable*pi0+ivar] == 0:
                    pass
                elif self.Hoxgene[self.nvariable*pi0+ivar] == 1:
                    for ibit in range(self.nbit):
                        usualgene[self.totalbit*pi0+ivar*self.nbit+ibit]=usualgene[self.totalbit*pi1+ivar*self.nbit+ibit]
                
                # about pi1
                if self.Hoxgene[self.nvariable*pi1+ivar] == 0:
                    pass
                elif self.Hoxgene[self.nvariable*pi1+ivar] == 1:
                    for ibit in range(self.nbit):
                        usualgene[self.totalbit*pi1+ivar*self.nbit+ibit]=usualgene[self.totalbit*pi0+ivar*self.nbit+ibit]
            

            ## update realtype_x ##
            ps=[pi0,pi1]
            for ipop in ps:
                d=0
                for ivar in range(self.nvariable):
                    # about pi0
                    if self.Hoxgene[self.nvariable*ipop+ivar] == 0:
                        pass
                    elif self.Hoxgene[self.nvariable*ipop+ivar] == 1:
                        for j in range(self.nbit):
                            self.ws[j]=usualgene[self.totalbit*ipop+ivar*self.nbit+j]
                        ix=self.idecoding(self.nbit,self.nbit-1)
                        if ix<pow(2,self.nbit-1):
                            realtype_x[self.nvariable*ipop+ivar]=self.xave[self.nvariable*self.prange[ipop]+ivar]-self.xlstd[self.nvariable*self.prange[ipop]+ivar]*math.sqrt(-2*math.log(self.LB+(self.UB-self.LB)*ix/(pow(2,self.nbit-1)-1)))
                        else:
                            realtype_x[self.nvariable*ipop+ivar]=self.xave[self.nvariable*self.prange[ipop]+ivar]+self.xrstd[self.nvariable*self.prange[ipop]+ivar]*math.sqrt(-2*math.log(self.UB-(self.UB-self.LB)*(ix-pow(2,self.nbit-1))/(pow(2,self.nbit-1)-1)))
                    d+=((realtype_x[self.nvariable*ipop+ivar]-realtype_x_k[self.nvariable*ipop+ivar])**2)
                diff_x[ipop]=math.sqrt(d)


            ## update ccfitness ##
            for ipop in ps:
                s=0
                # Sphere function
                if self.f_type == 1:
                    for ivar in range(self.nvariable):
                        s += (realtype_x[ipop*self.nvariable+ivar])**2
                
                # Rastrigin function
                elif self.f_type == 2:
                    for ivar in range(self.nvariable):
                        s += (realtype_x[ipop*self.nvariable+ivar]**2-10*np.cos(2*np.pi*realtype_x[ipop*self.nvariable+ivar]))
                    s += 10*self.nvariable
                
                # Ackley function
                elif self.f_type == 3:
                    t1 = 0
                    t2 = 0
                    for ivar in range(self.nvariable):
                        t1 += (realtype_x[ipop*self.nvariable+ivar])**2
                        t2 += np.cos(2*np.pi*realtype_x[ipop*self.nvariable+ivar])
                    s = 20-20*np.exp(-0.2*np.sqrt((1/self.nvariable)*t1))+np.e-np.exp((1/self.nvariable)*t2)
                
                # Rosenbrock function (type of A)
                elif self.f_type == 4:
                    for ivar in range(int(self.nvariable/2)):
                        x1=realtype_x[ipop*self.nvariable+2*ivar]
                        x2=realtype_x[ipop*self.nvariable+2*ivar+1]
                        s+=100*(x2-x1*x1)*(x2-x1*x1)+(x1-1)*(x1-1)
                
                # Rosenbrock function (type of star)
                elif self.f_type == 5:
                    for ivar in range(1,self.nvariable):
                        x1=realtype_x[ipop*self.nvariable]
                        x2=realtype_x[ipop*self.nvariable+ivar]
                        s+=100*(x1-x2*x2)*(x1-x2*x2)+(x2-1)*(x2-1)
                
                # Rosenbrock function (type of chain)
                elif self.f_type == 6:
                    for ivar in range(self.nvariable-1):
                        x0 = realtype_x[ipop*self.nvariable+ivar]
                        x1 = realtype_x[ipop*self.nvariable+ivar+1]
                        s+=100*(x1-x0*x0)*(x1-x0*x0)+(x0-1)*(x0-1)
                
                ccfitness[ipop]=copy.deepcopy(s)
                eval=(self.fgreat-ccfitness[ipop])/(diff_x[ipop]+1)
                self.data_pare[ipop]=self.fgreat
                self.data_chil[ipop]=ccfitness[ipop]
                self.data_calc[ipop]=eval
                self.HGfitness[ipop]=eval



    # revaluate the Hoxgene
    def reval_Hoxgene(self):
        ppfitness=[0.0 for i in range(self.npop)]
        ccfitness=[0.0 for i in range(self.npop)]
        ## save the present fitness ##
        for ipop in range(self.npop):
            ppfitness[ipop]=self.fitness[ipop]
        
        ## copy usual gene of each individual ##
        usualgene=[0 for i in range(self.npop*self.totalbit)]
        for ipop in range(self.npop):
            for ibit in range(self.totalbit):
                usualgene[ipop*self.totalbit+ibit]=self.GAstring[ipop*self.totalbit+ibit]
        
        ## copy realtype solution ##
        realtype_x=[0.0 for i in range(self.npop*self.nvariable)]
        for ipop in range(self.npop):
            for ivar in range(self.nvariable):
                realtype_x[ipop*self.nvariable+ivar]=self.x[ipop*self.nvariable+ivar]

        ## temporally Cross Over ##
        self.data_pare=[0.0 for i in range(self.npop+self.hgncross)]
        self.data_chil=[0.0 for i in range(self.npop+self.hgncross)]
        self.data_calc=[0.0 for i in range(self.npop+self.hgncross)]
        for i in range(int(0.5*self.npop)):
            pi0=2*i
            pi1=2*i+1

            ## update usual gene ##
            for ivar in range(self.nvariable):
                # about pi0
                if self.Hoxgene[self.nvariable*pi0+ivar] == 0:
                    pass
                elif self.Hoxgene[self.nvariable*pi0+ivar] == 1:
                    for ibit in range(self.nbit):
                        usualgene[self.totalbit*pi0+ivar*self.nbit+ibit]=usualgene[self.totalbit*pi1+ivar*self.nbit+ibit]
                
                # about pi1
                if self.Hoxgene[self.nvariable*pi1+ivar] == 0:
                    pass
                elif self.Hoxgene[self.nvariable*pi1+ivar] == 1:
                    for ibit in range(self.nbit):
                        usualgene[self.totalbit*pi1+ivar*self.nbit+ibit]=usualgene[self.totalbit*pi1+ivar*self.nbit+ibit]
            

            ## update realtype_x ##
            ps=[pi0,pi1]
            for ipop in ps:
                for ivar in range(self.nvariable):
                    # about pi0
                    if self.Hoxgene[self.nvariable*ipop+ivar] == 0:
                        pass
                    elif self.Hoxgene[self.nvariable*ipop+ivar] == 1:
                        for j in range(self.nbit):
                            self.ws[j]=usualgene[self.totalbit*ipop+ivar*self.nbit+j]
                        ix=self.idecoding(self.nbit,self.nbit-1)
                        if ix<pow(2,self.nbit-1):
                            realtype_x[self.nvariable*ipop+ivar]=self.xave[self.nvariable*self.prange[ipop]+ivar]-self.xlstd[self.nvariable*self.prange[ipop]+ivar]*math.sqrt(-2*math.log(self.LB+(self.UB-self.LB)*ix/(pow(2,self.nbit-1)-1)))
                        else:
                            realtype_x[self.nvariable*ipop+ivar]=self.xave[self.nvariable*self.prange[ipop]+ivar]+self.xrstd[self.nvariable*self.prange[ipop]+ivar]*math.sqrt(-2*math.log(self.UB-(self.UB-self.LB)*(ix-pow(2,self.nbit-1))/(pow(2,self.nbit-1)-1)))


            ## update ccfitness ##
            for ipop in ps:
                s=0
                # Sphere function
                if self.f_type == 1:
                    for ivar in range(self.nvariable):
                        s += (realtype_x[ipop*self.nvariable+ivar])**2
                
                # Rastrigin function
                elif self.f_type == 2:
                    for ivar in range(self.nvariable):
                        s += (realtype_x[ipop*self.nvariable+ivar]**2-10*np.cos(2*np.pi*realtype_x[ipop*self.nvariable+ivar]))
                    s += 10*self.nvariable
                
                # Ackley function
                elif self.f_type == 3:
                    t1 = 0
                    t2 = 0
                    for ivar in range(self.nvariable):
                        t1 += (realtype_x[ipop*self.nvariable+ivar])**2
                        t2 += np.cos(2*np.pi*realtype_x[ipop*self.nvariable+ivar])
                    s = 20-20*np.exp(-0.2*np.sqrt((1/self.nvariable)*t1))+np.e-np.exp((1/self.nvariable)*t2)
                
                # Rosenbrock function (type of A)
                elif self.f_type == 4:
                    for ivar in range(int(self.nvariable/2)):
                        x1=realtype_x[ipop*self.nvariable+2*ivar]
                        x2=realtype_x[ipop*self.nvariable+2*ivar+1]
                        s+=100*(x2-x1*x1)*(x2-x1*x1)+(x1-1)*(x1-1)
                
                # Rosenbrock function (type of star)
                elif self.f_type == 5:
                    for ivar in range(1,self.nvariable):
                        x1=realtype_x[ipop*self.nvariable]
                        x2=realtype_x[ipop*self.nvariable+ivar]
                        s+=100*(x1-x2*x2)*(x1-x2*x2)+(x2-1)*(x2-1)
                
                # Rosenbrock function (type of chain)
                elif self.f_type == 6:
                    for ivar in range(self.nvariable-1):
                        x0 = realtype_x[ipop*self.nvariable+ivar]
                        x1 = realtype_x[ipop*self.nvariable+ivar+1]
                        s+=100*(x1-x0*x0)*(x1-x0*x0)+(x0-1)*(x0-1)
                
                ccfitness[ipop]=copy.deepcopy(s)
                eval=abs(ppfitness[ipop]-ccfitness[ipop])/abs(ppfitness[ipop])*100
                self.data_pare[ipop]=ppfitness[ipop]
                self.data_chil[ipop]=ccfitness[ipop]
                self.data_calc[ipop]=eval
                self.HGfitness[ipop]=eval

    

    # convert genotype to phenotype
    def idecoding(self, pbit, ibit):
        if ibit==0:
            r=self.ws[0]
        else:
            if self.ws[ibit]==0:
                r=self.idecoding(pbit,ibit-1)
            else:
                r=pow(2,ibit+1)-self.idecoding(pbit,ibit-1)-1
        return r



    # give each individual the real type solution (convert phenotype to realtype)
    def ARange(self, ipop):
        pbit=int(self.totalbit/self.nvariable)
        for ivariable in range(self.nvariable):
            if not self.psearch_flg:
                for j in range(pbit):
                    self.ws[j]=self.GAstring[self.totalbit*ipop+ivariable*pbit+j]
                ix=self.idecoding(pbit,pbit-1)
                if ix<pow(2,pbit-1):
                    self.x[self.nvariable*ipop+ivariable]=self.xave[self.nvariable*self.prange[ipop]+ivariable]-self.xlstd[self.nvariable*self.prange[ipop]+ivariable]*math.sqrt(-2*math.log(self.LB+(self.UB-self.LB)*ix/(pow(2,pbit-1)-1)))
                else:
                    self.x[self.nvariable*ipop+ivariable]=self.xave[self.nvariable*self.prange[ipop]+ivariable]+self.xrstd[self.nvariable*self.prange[ipop]+ivariable]*math.sqrt(-2*math.log(self.UB-(self.UB-self.LB)*(ix-pow(2,pbit-1))/(pow(2,pbit-1)-1)))
                #print("ix="+str(ix)+"xlstd="+str(xlstd[ivariable])+"xrstd="+str(xrstd[ivariable])+"UB="+str(UB)+"LB="+str(LB)+" xave="+str(xave[ivariable])+" x="+str(x[nvariable*ipop+ivariable]))
                #a=input()
            else:
                if ivariable in self.Pi:
                    for j in range(pbit):
                        self.ws[j]=self.GAstring[self.totalbit*ipop+ivariable*pbit+j]
                    ix=self.idecoding(pbit,pbit-1)
                    if ix<pow(2,pbit-1):
                        self.x[self.nvariable*ipop+ivariable]=self.xave[self.nvariable*self.prange[ipop]+ivariable]-self.xlstd[self.nvariable*self.prange[ipop]+ivariable]*math.sqrt(-2*math.log(self.LB+(self.UB-self.LB)*ix/(pow(2,pbit-1)-1)))
                    else:
                        self.x[self.nvariable*ipop+ivariable]=self.xave[self.nvariable*self.prange[ipop]+ivariable]+self.xrstd[self.nvariable*self.prange[ipop]+ivariable]*math.sqrt(-2*math.log(self.UB-(self.UB-self.LB)*(ix-pow(2,pbit-1))/(pow(2,pbit-1)-1)))
                else:
                    self.x[self.nvariable*ipop+ivariable] = self.xgreat[ivariable]
    
    def relocArange(self, ipop):
        pbit=int(self.totalbit/self.nvariable)
        for ivariable in range(self.nvariable):
            if ivariable in self.Pi:
                for j in range(pbit):
                    self.ws[j]=self.GAstring[self.totalbit*ipop+ivariable*pbit+j]
                ix=self.idecoding(pbit,pbit-1)
                if ix<pow(2,pbit-1):
                    self.x[self.nvariable*ipop+ivariable]=self.xgreat[ivariable]-self.xlstd[self.nvariable*self.mrange+ivariable]*math.sqrt(-2*math.log(self.LB+(self.UB-self.LB)*ix/(pow(2,pbit-1)-1)))
                else:
                    self.x[self.nvariable*ipop+ivariable]=self.xgreat[ivariable]+self.xrstd[self.nvariable*self.mrange+ivariable]*math.sqrt(-2*math.log(self.UB-(self.UB-self.LB)*(ix-pow(2,pbit-1))/(pow(2,pbit-1)-1)))
            else:
                self.x[self.nvariable*ipop+ivariable] = self.xgreat[ivariable]
    
    def relocArange2(self, ipop):
        pbit=int(self.totalbit/self.nvariable)
        for ivariable in range(self.nvariable):
            for j in range(pbit):
                self.ws[j]=self.GAstring[self.totalbit*ipop+ivariable*pbit+j]
            ix=self.idecoding(pbit,pbit-1)
            if ix<pow(2,pbit-1):
                self.x[self.nvariable*ipop+ivariable]=self.xgreat[ivariable]-0.5*self.xlstd[self.nvariable*self.mrange+ivariable]*math.sqrt(-2*math.log(self.LB+(self.UB-self.LB)*ix/(pow(2,pbit-1)-1)))
            else:
                self.x[self.nvariable*ipop+ivariable]=self.xgreat[ivariable]+0.5*self.xrstd[self.nvariable*self.mrange+ivariable]*math.sqrt(-2*math.log(self.UB-(self.UB-self.LB)*(ix-pow(2,pbit-1))/(pow(2,pbit-1)-1)))
    
    def locArange(self, ipop):
        pbit=int(self.totalbit/self.nvariable)
        for ivariable in range(self.nvariable):
            if not self.psearch_flg:
                for j in range(pbit):
                    self.ws[j]=self.GAstring[self.totalbit*ipop+ivariable*pbit+j]
                ix=self.idecoding(pbit,pbit-1)
                if ix<pow(2,pbit-1):
                    self.x[self.nvariable*ipop+ivariable]=self.xgreat[ivariable]-self.kkmin*self.xlstd[self.nvariable*self.prange[ipop]+ivariable]*math.sqrt(-2*math.log(self.LB+(self.UB-self.LB)*ix/(pow(2,pbit-1)-1)))
                else:
                    self.x[self.nvariable*ipop+ivariable]=self.xgreat[ivariable]+self.kkmin*self.xrstd[self.nvariable*self.prange[ipop]+ivariable]*math.sqrt(-2*math.log(self.UB-(self.UB-self.LB)*(ix-pow(2,pbit-1))/(pow(2,pbit-1)-1)))
            else:
                if ivariable in self.Pi:
                    for j in range(pbit):
                        self.ws[j]=self.GAstring[self.totalbit*ipop+ivariable*pbit+j]
                    ix=self.idecoding(pbit,pbit-1)
                    if ix<pow(2,pbit-1):
                        self.x[self.nvariable*ipop+ivariable]=self.xgreat[ivariable]-self.kkmin*self.xlstd[self.nvariable*self.prange[ipop]+ivariable]*math.sqrt(-2*math.log(self.LB+(self.UB-self.LB)*ix/(pow(2,pbit-1)-1)))
                    else:
                        self.x[self.nvariable*ipop+ivariable]=self.xgreat[ivariable]+self.kkmin*self.xrstd[self.nvariable*self.prange[ipop]+ivariable]*math.sqrt(-2*math.log(self.UB-(self.UB-self.LB)*(ix-pow(2,pbit-1))/(pow(2,pbit-1)-1)))
                else:
                    self.x[self.nvariable*ipop+ivariable] = self.xgreat[ivariable]


    def HGARange(self,ipop,cipop):
        hoxgene=self.Hoxgene[self.nvariable*ipop:self.nvariable*ipop+self.nvariable]    # Hoxgene of an individual we are currently referencing 
        pipop=self.ws3[cipop]   # the index of an individual's parent
        pbit=int(self.totalbit/self.nvariable)
        for ivariable in range(self.nvariable):
            if hoxgene[ivariable]==0:
                self.x[self.nvariable*ipop+ivariable]=self.xgreat[ivariable]
            else:
                for j in range(pbit):
                    self.ws[j]=self.GAstring[self.totalbit*ipop+ivariable*pbit+j]
                ix=self.idecoding(pbit,pbit-1)
                # if ix<pow(2,pbit-1):
                #     self.x[self.nvariable*ipop+ivariable]=self.xave[self.nvariable*self.prange[ipop]+ivariable]-self.xlstd[self.nvariable*self.prange[ipop]+ivariable]*math.sqrt(-2*math.log(self.LB+(self.UB-self.LB)*ix/(pow(2,pbit-1)-1)))
                # else:
                #     self.x[self.nvariable*ipop+ivariable]=self.xave[self.nvariable*self.prange[ipop]+ivariable]+self.xrstd[self.nvariable*self.prange[ipop]+ivariable]*math.sqrt(-2*math.log(self.UB-(self.UB-self.LB)*(ix-pow(2,pbit-1))/(pow(2,pbit-1)-1)))
                if ix<pow(2,pbit-1):
                    self.x[self.nvariable*ipop+ivariable]=self.xgreat[ivariable]-self.xstd[ivariable]*math.sqrt(-2*math.log(self.LB+(self.UB-self.LB)*ix/(pow(2,pbit-1)-1)))
                else:
                    self.x[self.nvariable*ipop+ivariable]=self.xgreat[ivariable]+self.xstd[ivariable]*math.sqrt(-2*math.log(self.UB-(self.UB-self.LB)*(ix-pow(2,pbit-1))/(pow(2,pbit-1)-1)))
    
    def toReal(self,ipop):
        start=time.time()
        hoxgene=self.Hoxgene[self.nvariable*ipop:self.nvariable*ipop+self.nvariable]    # Hoxgene of an individual we are currently referencing 
        pbit=int(self.totalbit/self.nvariable)
        for ivariable in range(self.nvariable):
            if hoxgene[ivariable]==0:
                self.realtype_x[ivariable]=self.xgreat[ivariable]
            else:
                for j in range(pbit):
                    self.ws[j]=self.usualgene[ivariable*pbit+j]
                ix=self.idecoding(pbit,pbit-1)
                # if ix<pow(2,pbit-1):
                #     self.realtype_x[ivariable]=self.xave[self.nvariable*self.prange[ipop]+ivariable]-self.xlstd[self.nvariable*self.prange[ipop]+ivariable]*math.sqrt(-2*math.log(self.LB+(self.UB-self.LB)*ix/(pow(2,pbit-1)-1)))
                # else:
                #     self.realtype_x[ivariable]=self.xave[self.nvariable*self.prange[ipop]+ivariable]+self.xrstd[self.nvariable*self.prange[ipop]+ivariable]*math.sqrt(-2*math.log(self.UB-(self.UB-self.LB)*(ix-pow(2,pbit-1))/(pow(2,pbit-1)-1)))
                if ix<pow(2,pbit-1):
                    self.realtype_x[ivariable]=self.xgreat[ivariable]-self.xstd[ivariable]*math.sqrt(-2*math.log(self.LB+(self.UB-self.LB)*ix/(pow(2,pbit-1)-1)))
                else:
                    self.realtype_x[ivariable]=self.xgreat[ivariable]+self.xstd[ivariable]*math.sqrt(-2*math.log(self.UB-(self.UB-self.LB)*(ix-pow(2,pbit-1))/(pow(2,pbit-1)-1)))
        # print("--> to realtype="+str(time.time()-start))

        self.s_time=time.time()


    def toFunc(self):
        s=0
        start=time.time()
        # Sphere function
        if self.f_type == 1:
            for ivar in range(self.nvariable):
                s += (self.realtype_x[ivar])**2
        
        # Rastrigin function
        elif self.f_type == 2:
            for ivar in range(self.nvariable):
                s += (self.realtype_x[ivar]**2-10*np.cos(2*np.pi*self.realtype_x[ivar]))
            s += 10*self.nvariable
        
        # Ackley function
        elif self.f_type == 3:
            t1 = 0
            t2 = 0
            for ivar in range(self.nvariable):
                t1 += (self.realtype_x[ivar])**2
                t2 += np.cos(2*np.pi*self.realtype_x[ivar])
            s = 20-20*np.exp(-0.2*np.sqrt((1/self.nvariable)*t1))+np.e-np.exp((1/self.nvariable)*t2)
        
        # Rosenbrock function (type of A)
        elif self.f_type == 4:
            for ivar in range(int(self.nvariable/2)):
                x1=self.realtype_x[2*ivar]
                x2=self.realtype_x[2*ivar+1]
                s+=100*(x2-x1*x1)*(x2-x1*x1)+(x1-1)*(x1-1)
        
        # Rosenbrock function (type of star)
        elif self.f_type == 5:
            for ivar in range(1,self.nvariable):
                x1=self.realtype_x[0]
                x2=self.realtype_x[ivar]
                s+=100*(x1-x2*x2)*(x1-x2*x2)+(x2-1)*(x2-1)
        
        # Rosenbrock function (type of chain)
        elif self.f_type == 6:
            for ivar in range(self.nvariable-1):
                x0 = self.realtype_x[ivar]
                x1 = self.realtype_x[ivar+1]
                s+=100*(x1-x0*x0)*(x1-x0*x0)+(x0-1)*(x0-1)
        # print("--> tofunc time="+str(time.time()-start))

        self.s_time=time.time()
        self.f_value=s
        

    def evalHoxgene(self,ipop,cipop):
        p_fitness=self.pfitness[cipop]
        c_fitness=self.fitness[ipop]
        
        # caluculate Evaluation value(s)
        # s=np.exp((p_fitness-c_fitness)/abs(p_fitness))
        # give a bias
        # if s < 1:
        #     s = s*(1e-2)
        # else:
        #     s = s*(1e+2)

        s=(self.fgreat-c_fitness)
        self.HGfitness[ipop]=s
    
    def evalHoxAve(self,npop,state):
        self.geneToReal(npop)
        seen = []
        # get the unique data from realHox
        unique = [v for v in self.realHox[0:npop] if self.realHox[0:npop].count(v) >= 1 and not seen.append(v) and seen.count(v) == 1]

        tmpdata0 = []
        tmpdata1 = []
        for pi in unique:
            pi_ind = [i for i, v in enumerate(self.realHox[0:npop]) if v == pi]
            pi_num = len(pi_ind)

            # calculate HGfitness average
            aaahgfit = 0
            for i in pi_ind:
                aaahgfit += self.HGfitness[i]
            aaahgfit /= pi_num

            pistr="["
            for i in pi:
                pistr += (str(i)+",")
            pistr=pistr.rstrip(",")
            pistr+="]"

            for i in pi_ind:
                self.HGfitness[i] = aaahgfit
                tmpdata0.append(pistr)
                tmpdata1.append(aaahgfit)
        
        self.writeDebug2(tmpdata0,tmpdata1,state,npop)
    
    # convert genotype hoxgene to realtype hoxgene
    def geneToReal(self,npop):
        for ipop in range(npop):
            pi = []
            for ivar in range(self.nvariable):
                if self.Hoxgene[ipop*self.nvariable+ivar] == 1:
                    pi.append(ivar)
                else:
                    pass
            self.realHox[ipop]=pi



    def functioncall(self, ipop):
        s=0
        # Sphere function
        if self.f_type == 1:
            for ivar in range(self.nvariable):
                s += (self.x[ipop*self.nvariable+ivar])**2
        
        # Rastrigin function
        elif self.f_type == 2:
            for ivar in range(self.nvariable):
                s += (self.x[ipop*self.nvariable+ivar]**2-10*np.cos(2*np.pi*self.x[ipop*self.nvariable+ivar]))
            s += 10*self.nvariable
        
        # Ackley function
        elif self.f_type == 3:
            t1 = 0
            t2 = 0
            for ivar in range(self.nvariable):
                t1 += (self.x[ipop*self.nvariable+ivar])**2
                t2 += np.cos(2*np.pi*self.x[ipop*self.nvariable+ivar])
            s = 20-20*np.exp(-0.2*np.sqrt((1/self.nvariable)*t1))+np.e-np.exp((1/self.nvariable)*t2)
        
        # Rosenbrock function (type of A)
        elif self.f_type == 4:
            for ivar in range(int(self.nvariable/2)):
                x1=self.x[ipop*self.nvariable+2*ivar]
                x2=self.x[ipop*self.nvariable+2*ivar+1]
                s+=100*(x2-x1*x1)*(x2-x1*x1)+(x1-1)*(x1-1)
        
        # Rosenbrock function (type of star)
        elif self.f_type == 5:
            for ivar in range(1,self.nvariable):
                x1=self.x[ipop*self.nvariable]
                x2=self.x[ipop*self.nvariable+ivar]
                s+=100*(x1-x2*x2)*(x1-x2*x2)+(x2-1)*(x2-1)
        
        # Rosenbrock function (type of chain)
        elif self.f_type == 6:
            for ivar in range(self.nvariable-1):
                x0 = self.x[ipop*self.nvariable+ivar]
                x1 = self.x[ipop*self.nvariable+ivar+1]
                s+=100*(x1-x0*x0)*(x1-x0*x0)+(x0-1)*(x0-1)
        
        self.func[ipop]=s
        self.fitness[ipop]=s



    def CrossOver(self):
        # print("CrossOver")
        ### Get information about each search range ###
        # update information about each search ranges
        for i in range(self.nrange):
            self.age[i]+=1           # update the age of each search range
            self.rangepop[i]=0       # initialize the population in each search range
            self.rangetop[i]=1.0e33  # initialize the best f value in each search range
        # get the best f value and the index of individual with the best f value in each search range 
        for ipop in range(self.npop):
            k=self.prange[ipop]
            if self.fitness[ipop]<self.rangetop[k]:
                self.rangetop[k]=self.fitness[ipop]     # the best f value in each search range
                self.whoisit[k]=ipop                    # the index of the search range with the best f value
        

        ### Cross Over ###
        #--- ルーレット盤の作成(make a roulette board for parent selection) ---#
        roulettetotal=0
        roulette=[0 for i in range(self.npop+self.ncross)]            # initialize a roulette board
        maxfit=min(self.fitness[0:self.npop])                         # the best f value of all individuals
        minfit=max(self.fitness[0:self.npop])                         # the worst f value of all individuals
        for irange in range(self.nrange):
            icnt = self.prange[0:self.npop].count(irange)
            self.rangepop[irange]=icnt                               # count the population in each search range

        # make a roulette board
        start=time.time()
        for i in range(self.npop):
            # roulette[i]=1+9*(self.fitness[i]-minfit)/(maxfit-minfit)*(1-(self.rangepop[self.prange[i]]-self.npop/self.nrange)/self.npop*5)#変更ポイント＃1 最終的には９９
            roulette[i]=1+9*(self.fitness[i]-minfit)/(maxfit-minfit)
            roulettetotal+=roulette[i]
        # print("mkroulette(C)="+str(time.time()-start))
        
        # if self.psearch_flg:
        #     print("fitness[0,self.npop] >>> " + str(self.fitness[0:self.npop]))
        #     print("roulette[0,self.npop] >>> " + str(roulette[0:self.npop]))
        #     print("roulettetotal >>> " + str(roulettetotal))
        
        #--- 親の選択(select parents) ---#
        start=time.time()
        ws=[-1 for i in range(self.ncross)]  # the list for recording the index of the selected parents 
        for i in range(self.ncross):
            if i%2==0:
                key=random.uniform(0,roulettetotal)
                j=0
                hit=0
                while hit<key:
                    hit+=roulette[j]
                    if hit<key:
                        j+=1
                roulettetotal-=roulette[j]
                ws[i]=j
                roulette[j]=0
            else:
                if i%4==1:
                    key=random.uniform(0,roulettetotal)
                    j=0
                    hit=0
                    while hit<key:
                        hit+=roulette[j]
                        if hit<key:
                            j+=1
                    roulettetotal-=roulette[j]
                    ws[i]=j
                    roulette[j]=0
                else:
                    wflg=0
                    while wflg==0:
                        key=random.uniform(0,roulettetotal)
                        j=0
                        hit=0
                        while hit<key:
                            hit+=roulette[j]
                            if hit<key:
                                j+=1
                        if self.prange[j]==self.prange[ws[i-1]]:
                            wflg=1
                    roulettetotal-=roulette[j]
                    ws[i]=j
                    roulette[j]=0
        # print("selectP="+str(time.time()-start))

        #--- 交叉(Cross over : generate children inherits the gene of the selected parents)
        if not self.psearch_flg:
            start=time.time()
            for i in range(int(0.5*self.ncross)):
                mcros=1+random.randint(1,3)         # the number of Cross over points 
                randseed=2*self.totalbit/mcros*1.5
                k=0
                cstart=0
                tab=0
                for ii in range(mcros):
                    crossend=random.randint(2,int(randseed+tab))
                    if cstart+crossend>self.totalbit:
                        crossend=self.totalbit-cstart
                    if k%2==0:
                        for m in range(crossend):
                            self.GAstring[self.totalbit*(self.npop+2*i)+m+cstart]=self.GAstring[self.totalbit*ws[2*i+1]+m+cstart]
                            self.GAstring[self.totalbit*(self.npop+2*i+1)+m+cstart]=self.GAstring[self.totalbit*ws[2*i]+m+cstart]
                    else:
                        for m in range(crossend):
                            self.GAstring[self.totalbit*(self.npop+2*i)+m+cstart]=self.GAstring[self.totalbit*ws[2*i]+m+cstart]
                            self.GAstring[self.totalbit*(self.npop+2*i+1)+m+cstart]=self.GAstring[self.totalbit*ws[2*i+1]+m+cstart]
                    k+=1
                    cstart+=crossend
                tab=randseed*(ii+1)-crossend
                if k%2==0:
                    for m in range(self.totalbit-cstart):
                        self.GAstring[self.totalbit*(self.npop+2*i)+m+cstart]=self.GAstring[self.totalbit*ws[2*i+1]+m+cstart]
                        self.GAstring[self.totalbit*(self.npop+2*i+1)+m+cstart]=self.GAstring[self.totalbit*ws[2*i]+m+cstart]
                else:
                    for m in range(self.totalbit-cstart):
                        self.GAstring[self.totalbit*(self.npop+2*i)+m+cstart]=self.GAstring[self.totalbit*ws[2*i]+m+cstart]
                        self.GAstring[self.totalbit*(self.npop+2*i+1)+m+cstart]=self.GAstring[self.totalbit*ws[2*i+1]+m+cstart]
                self.prange[self.npop+2*i]=self.prange[ws[2*i]]
                self.prange[self.npop+2*i+1]=self.prange[ws[2*i+1]]
            # print("crossOver="+str(time.time()-start))
        # when the search considering dependencies between variables is taking place
        else:
            start=time.time()
            for i in range(int(0.5*self.ncross)):
                p0=ws[2*i]          # parents' index
                p1=ws[2*i+1]
                c0=self.npop+2*i    # children's index
                c1=self.npop+2*i+1
                
                for ivar in self.Pi:
                    mcross=1+random.randint(0,1)    # the number of cross over point
                    cp=[]                           # cross over point
                    for ii in range(mcross):
                        pflg=True
                        while pflg:
                            ip=random.randint(1,self.nbit-1)
                            if ip in cp:
                                pass
                            else:
                                cp.append(ip)
                                pflg=False
                    
                    # one-point crossover
                    if len(cp) == 1:
                        cpi=cp[0]       # crossover point we are referencing now
                        for ibit in range(cpi):
                            self.GAstring[self.totalbit*c0+self.nbit*ivar+ibit]=self.GAstring[self.totalbit*p0+self.nbit*ivar+ibit]
                            self.GAstring[self.totalbit*c1+self.nbit*ivar+ibit]=self.GAstring[self.totalbit*p1+self.nbit*ivar+ibit]
                        for ibit in range(cpi,self.nbit):
                            self.GAstring[self.totalbit*c0+self.nbit*ivar+ibit]=self.GAstring[self.totalbit*p1+self.nbit*ivar+ibit]
                            self.GAstring[self.totalbit*c1+self.nbit*ivar+ibit]=self.GAstring[self.totalbit*p0+self.nbit*ivar+ibit]
                    
                    # two-point crossover
                    if len(cp) == 2:
                        cp.sort()
                        stp=cp[0]
                        edp=cp[1]
                        for ibit in range(stp):
                            self.GAstring[self.totalbit*c0+self.nbit*ivar+ibit]=self.GAstring[self.totalbit*p0+self.nbit*ivar+ibit]
                            self.GAstring[self.totalbit*c1+self.nbit*ivar+ibit]=self.GAstring[self.totalbit*p1+self.nbit*ivar+ibit]
                        for ibit in range(stp,edp+1):
                            self.GAstring[self.totalbit*c0+self.nbit*ivar+ibit]=self.GAstring[self.totalbit*p1+self.nbit*ivar+ibit]
                            self.GAstring[self.totalbit*c1+self.nbit*ivar+ibit]=self.GAstring[self.totalbit*p0+self.nbit*ivar+ibit]
                        for ibit in range(edp+1,self.nbit):
                            self.GAstring[self.totalbit*c0+self.nbit*ivar+ibit]=self.GAstring[self.totalbit*p0+self.nbit*ivar+ibit]
                            self.GAstring[self.totalbit*c1+self.nbit*ivar+ibit]=self.GAstring[self.totalbit*p1+self.nbit*ivar+ibit]

                self.prange[self.npop+2*i]=self.prange[ws[2*i]]
                self.prange[self.npop+2*i+1]=self.prange[ws[2*i+1]]
            # print("(crossOver)="+str(time.time()-start))

        ## Evaluation ##
        start=time.time()
        #--- 子個体の評価(evaluate the generated children) ---#
        for ipop in range(self.ncross):
            # convert a genotyped solution of children to a real-numbered solution
            self.ARange(ipop+self.npop)
            # evaluate children's solution
            self.functioncall(ipop+self.npop)
        # print("evalC="+str(time.time()-start))

        # get the best f value and the index of individual with the best f value in each search range 
        for ipop in range(self.npop+self.ncross):
            k=self.prange[ipop]
            if self.rangetop[k]>self.fitness[ipop]:
                self.rangetop[k]=self.fitness[ipop]     # the best f value in each search range
                self.rangewho[k]=ipop                   # the index of the individual with the best f value in each search range    


    def strTypePtmp(self,ipop):
        pstr="["
        for ivar in range(self.nvariable):
            if self.Hoxgene[self.nvariable*ipop+ivar] == 1:
                pstr+=(str(ivar)+",")
            else:
                pass
        pstr=pstr.rstrip(",")
        pstr+="]"

        return pstr



    # Hoxgene Cross Over
    def HGCrossOver(self):
        # print("HGCrossOver")
        # self.writeDebug4("before HGCrossOver")

        ### Get information about each search range ###
        # update information about each search ranges
        # for i in range(self.nrange):
        #     self.age[i]+=1           # update the age of each search range
        #     self.rangepop[i]=0       # initialize the population in each search range
        #     self.rangetop[i]=1.0e33  # initialize the best f value in each search range
        
        # get the best f value and the index of individual with the best f value in each search range 
        # for ipop in range(self.npop):
        #     k=self.prange[ipop]
        #     if self.fitness[ipop]<self.rangetop[k]:
        #         self.rangetop[k]=self.fitness[ipop]   # the best f value in each search range
        #         self.whoisit[k]=ipop             # the index of the search range with the best f value

        ### Hoxgene Cross Over ###
        #--- ルーレット盤の作成(make a roulette board for parent selection) ---#
        roulettetotal=0
        roulette=[0 for i in range(self.npop+self.hgncross)]              # initialize a roulette board
        maxfit=max(self.HGfitness[0:self.npop])                           # the best f value of all individuals
        minfit=min(self.HGfitness[0:self.npop])                           # the worst f value of all individuals
        # for irange in range(self.nrange):
        #     icnt=self.prange[0:self.npop].count(irange)
        #     self.rangepop[irange]=icnt
        
        # make a roulette board
        start=time.time()
        for i in range(self.npop):
            roulette[i]=3+9*(self.HGfitness[i]-minfit+5)/(maxfit-minfit+5)
            roulettetotal+=roulette[i]
        # print("HGmkroulette(C)="+str(time.time()-start)) 

        start=time.time()
        #--- 親の選択(select parents) ---#
        ws=[-1 for i in range(self.ncross)]  # the list for recording the index of the selected parents 
        for i in range(self.ncross):
            key=random.uniform(0,roulettetotal)
            j=0
            hit=0
            while hit<key:
                hit+=roulette[j]
                if hit<key:
                    j+=1
            roulettetotal-=roulette[j]
            ws[i]=j
            roulette[j]=0
        # print("HGselectP="+str(time.time()-start))                  
        
        ## initialize pfitness and ws3 ##
        for ipop in range(self.npop):
            self.ws3[ipop]=-1           # a list for recoding the index of a parent
            self.pfitness[ipop]=1.0e33


        #--- HG交叉(Hoxgene Cross over : generate children inherits the gene of the selected parents)
        start=time.time()
        P_tmp=[]
        pfit_tmp=[]
        cfit_tmp=[]
        diff_tmp=[]
        x_parent=[0.0 for i in range(self.hgncross*self.nvariable)]
        x_diff=[0.0 for ipop in range(self.hgncross)]
        for i in range(int(self.ncross*0.5)):
            i0=2*i
            i1=2*i+1
            cci0=4*i
            cci1=4*i+1
            cci2=4*i+2
            cci3=4*i+3

            # the indexs of parents
            pi0=ws[i0]
            pi1=ws[i1]

            # the indexs of children
            ci00=self.npop+cci0
            ci10=self.npop+cci1
            ci01=self.npop+cci2
            ci11=self.npop+cci3


            ## Cross Over point 1 ##
            ihoxgene=self.Hoxgene[self.nvariable*pi0:self.nvariable*pi0+self.nvariable]
            standBits=[i for i,v in enumerate(ihoxgene) if v==1]
            key=-1                                        # crossover point
            if len(standBits)==0:
                key=random.randint(1,self.nvariable-2)    # crossover point
            elif len(standBits)==1:
                key=random.randint(1,self.nvariable-2)    # crossover point
            else:
                del standBits[-1]
                key=random.choice(standBits)+1
            for ii in range(key):
                # child 0
                self.Hoxgene[self.nvariable*ci00+ii]=0
                # child 1
                self.Hoxgene[self.nvariable*ci10+ii]=self.Hoxgene[self.nvariable*pi0+ii]
            for ii in range(key,self.nvariable):
                # child 0
                self.Hoxgene[self.nvariable*ci00+ii]=self.Hoxgene[self.nvariable*pi0+ii]
                # child 1
                self.Hoxgene[self.nvariable*ci10+ii]=0
            
            
            ## guarantee a minimum of standing genes
            # about a child00
            # case0 : Hoxgene doesn't take 1
            if 1 not in self.Hoxgene[self.nvariable*ci00:self.nvariable*ci00+self.nvariable]:
                # random method
                ntake1=random.randint(2,4)  # select the No of bits taking 1 randomly
                stbits=random.sample(range(0,self.nvariable),ntake1)    # select the bits taking 1 randomly

                # give 1s to the selected bits
                for ibit in stbits:
                    self.Hoxgene[self.nvariable*ci00+ibit]=1
            
            # case1 : Hoxgene takes only one 1
            elif self.Hoxgene[self.nvariable*ci00:self.nvariable*ci00+self.nvariable].count(1) == 1:
                # random methods
                ntake1=random.randint(1,3)  # select the No of bits taking 1 randomly
                stbits=random.sample(range(0,self.nvariable),ntake1)    # select the bits taking 1 randomly

                # give 1s to the selected bits
                for ibit in stbits:
                    self.Hoxgene[self.nvariable*ci00+ibit]=1

            # case2 : Hoxgene takes 5 or more 1s
            elif self.Hoxgene[self.nvariable*ci00:self.nvariable*ci00+self.nvariable].count(1) > 4:
                # random methods
                ibits=[i for i,v in enumerate(self.Hoxgene[self.nvariable*ci00:self.nvariable*ci00+self.nvariable]) if v == 1]     # the indexs of bits taking 1
                cntake1=len(ibits)          # current No of bits taking 1
                ntake1=random.randint(2,4)  # the No. of bits taking 1 in this Hoxgene
                removebits=random.sample(ibits,cntake1-ntake1)  # select the bits taking 0

                # remove 1s and replace 0s from the selected bits
                for ibit in removebits:
                    self.Hoxgene[self.nvariable*ci00+ibit]=0
            

            # about a child10
            # case0 : Hoxgene doesn't take 1
            if 1 not in self.Hoxgene[self.nvariable*ci10:self.nvariable*ci10+self.nvariable]:
                # random method
                ntake1=random.randint(2,4)  # select the No of bits taking 1 randomly
                stbits=random.sample(range(0,self.nvariable),ntake1)    # select the bits taking 1 randomly

                # give 1s to the selected bits
                for ibit in stbits:
                    self.Hoxgene[self.nvariable*ci10+ibit]=1
            
            # case1 : Hoxgene takes only one 1
            elif self.Hoxgene[self.nvariable*ci10:self.nvariable*ci10+self.nvariable].count(1) == 1:
                # random methods
                ntake1=random.randint(1,3)  # select the No of bits taking 1 randomly
                stbits=random.sample(range(0,self.nvariable),ntake1)    # select the bits taking 1 randomly

                # give 1s to the selected bits
                for ibit in stbits:
                    self.Hoxgene[self.nvariable*ci10+ibit]=1
            
            # case2 : Hoxgene takes 5 or more 1s
            elif self.Hoxgene[self.nvariable*ci10:self.nvariable*ci10+self.nvariable].count(1) > 4:
                # random methods
                ibits=[i for i,v in enumerate(self.Hoxgene[self.nvariable*ci10:self.nvariable*ci10+self.nvariable]) if v == 1]     # the indexs of bits taking 1
                cntake1=len(ibits)          # current No of bits taking 1
                ntake1=random.randint(2,4)  # the No. of bits taking 1 in this Hoxgene
                removebits=random.sample(ibits,cntake1-ntake1)  # select the bits taking 0

                # remove 1s and replace 0s from the selected bits
                for ibit in removebits:
                    self.Hoxgene[self.nvariable*ci10+ibit]=0
            

            P_tmp.append(self.strTypePtmp(ci00))
            pfit_tmp.append(self.fgreat)
            P_tmp.append(self.strTypePtmp(ci10))
            pfit_tmp.append(self.fgreat)
            

            ## generate children's usual gene(GAstring) ##
            # unify gene arrays given to children
            key0=random.randint(0,self.nvariable-1)
            copygene0=self.GAstring[self.totalbit*pi0+key0*self.nbit:self.totalbit*pi0+key0*self.nbit+self.nbit]
            key1=random.randint(0,self.nvariable-1)
            copygene1=self.GAstring[self.totalbit*pi1+key1*self.nbit:self.totalbit*pi1+key1*self.nbit+self.nbit]
            cp=random.randint(1,self.nbit-1)
            unifiedgene0=copygene0[0:cp]
            unifiedgene0.extend(copygene1[cp:self.nbit])
            unifiedgene1=copygene1[0:cp]
            unifiedgene1.extend(copygene0[cp:self.nbit])
            for ivar in range(self.nvariable):
                # about child00
                self.ws3[cci0]=pi0
                if self.Hoxgene[self.nvariable*ci00+ivar] == 0:
                    for ibit in range(self.nbit):
                        self.GAstring[self.totalbit*ci00+ivar*self.nbit+ibit] = self.GAstring[self.totalbit*pi0+ivar*self.nbit+ibit]
                else:
                    for ibit in range(self.nbit):
                        self.GAstring[self.totalbit*ci00+ivar*self.nbit+ibit] = unifiedgene0[ibit]


                # about child 10
                self.ws3[cci1]=pi1
                if self.Hoxgene[self.nvariable*ci10+ivar] == 0:
                    for ibit in range(self.nbit):
                        self.GAstring[self.totalbit*ci10+ivar*self.nbit+ibit] = self.GAstring[self.totalbit*pi1+ivar*self.nbit+ibit]
                else:
                    for ibit in range(self.nbit):
                        self.GAstring[self.totalbit*ci10+ivar*self.nbit+ibit] = unifiedgene1[ibit]
            
            self.prange[ci00]=self.prange[pi0]
            self.prange[ci10]=self.prange[pi1]
            self.pfitness[cci0]=self.fitness[pi0]
            self.pfitness[cci1]=self.fitness[pi1]


            ## Cross Over point 2 ##
            ihoxgene=self.Hoxgene[self.nvariable*pi0:self.nvariable*pi0+self.nvariable]
            standBits=[i for i,v in enumerate(ihoxgene) if v==1]
            key=-1                                      # crossover point
            if len(standBits)==0:
                key=random.randint(1,self.nvariable-2)    # crossover point
            elif len(standBits)==1:
                key=random.randint(1,self.nvariable-2)    # crossover point
            else:
                del standBits[-1]
                key=random.choice(standBits)+1
            for ii in range(key):
                # child 0
                self.Hoxgene[self.nvariable*ci01+ii]=self.Hoxgene[self.nvariable*pi1+ii]
                # child 1
                self.Hoxgene[self.nvariable*ci11+ii]=0
            for ii in range(key,self.nvariable):
                # child 0
                self.Hoxgene[self.nvariable*ci01+ii]=0
                # child 1
                self.Hoxgene[self.nvariable*ci11+ii]=self.Hoxgene[self.nvariable*pi1+ii]
            
            ## guarantee a minimum of standing genes
            # about a child01
            # case0 : Hoxgene doesn't take 1
            if 1 not in self.Hoxgene[self.nvariable*ci01:self.nvariable*ci01+self.nvariable]:
                # random method
                ntake1=random.randint(2,4)  # select the No of bits taking 1 randomly
                stbits=random.sample(range(0,self.nvariable),ntake1)    # select the bits taking 1 randomly

                # give 1s to the selected bits
                for ibit in stbits:
                    self.Hoxgene[self.nvariable*ci01+ibit]=1
            
            # case1 : Hoxgene takes only one 1
            elif self.Hoxgene[self.nvariable*ci01:self.nvariable*ci01+self.nvariable].count(1) == 1:
                # random methods
                ntake1=random.randint(1,3)  # select the No of bits taking 1 randomly
                stbits=random.sample(range(0,self.nvariable),ntake1)    # select the bits taking 1 randomly

                # give 1s to the selected bits
                for ibit in stbits:
                    self.Hoxgene[self.nvariable*ci01+ibit]=1
            
            # case2 : Hoxgene takes 5 or more 1s
            elif self.Hoxgene[self.nvariable*ci01:self.nvariable*ci01+self.nvariable].count(1) > 4:
                # random methods
                ibits=[i for i,v in enumerate(self.Hoxgene[self.nvariable*ci01:self.nvariable*ci01+self.nvariable]) if v == 1]     # the indexs of bits taking 1
                cntake1=len(ibits)          # current No of bits taking 1
                ntake1=random.randint(2,4)  # the No. of bits taking 1 in this Hoxgene
                removebits=random.sample(ibits,cntake1-ntake1)  # select the bits taking 0

                # remove 1s and replace 0s from the selected bits
                for ibit in removebits:
                    self.Hoxgene[self.nvariable*ci01+ibit]=0

            
            # about a child10
            # case0 : Hoxgene doesn't take 1
            if 1 not in self.Hoxgene[self.nvariable*ci11:self.nvariable*ci11+self.nvariable]:
                # random method
                ntake1=random.randint(2,4)  # select the No of bits taking 1 randomly
                stbits=random.sample(range(0,self.nvariable),ntake1)    # select the bits taking 1 randomly

                # give 1s to the selected bits
                for ibit in stbits:
                    self.Hoxgene[self.nvariable*ci11+ibit]=1

            # case1 : Hoxgene takes only one 1
            elif self.Hoxgene[self.nvariable*ci11:self.nvariable*ci11+self.nvariable].count(1) == 1:
                # random methods
                ntake1=random.randint(1,3)  # select the No of bits taking 1 randomly
                stbits=random.sample(range(0,self.nvariable),ntake1)    # select the bits taking 1 randomly

                # give 1s to the selected bits
                for ibit in stbits:
                    self.Hoxgene[self.nvariable*ci11+ibit]=1

            # case2 : Hoxgene takes 4 or more 1s
            elif self.Hoxgene[self.nvariable*ci11:self.nvariable*ci11+self.nvariable].count(1) > 4:
                # random methods
                ibits=[i for i,v in enumerate(self.Hoxgene[self.nvariable*ci11:self.nvariable*ci11+self.nvariable]) if v == 1]     # the indexs of bits taking 1
                cntake1=len(ibits)          # current No of bits taking 1
                ntake1=random.randint(2,4)  # the No. of bits taking 1 in this Hoxgene
                removebits=random.sample(ibits,cntake1-ntake1)  # select the bits taking 0

                # remove 1s and replace 0s from the selected bits
                for ibit in removebits:
                    self.Hoxgene[self.nvariable*ci11+ibit]=0
            
            
            P_tmp.append(self.strTypePtmp(ci01))
            pfit_tmp.append(self.fgreat)
            P_tmp.append(self.strTypePtmp(ci11))
            pfit_tmp.append(self.fgreat)

            ## generate children's usual gene(GAstring) ##
            # unify gene arrays given to children
            key0=random.randint(0,self.nvariable-1)
            copygene0=self.GAstring[self.totalbit*pi0+key0*self.nbit:self.totalbit*pi0+key0*self.nbit+self.nbit]
            key1=random.randint(0,self.nvariable-1)
            copygene1=self.GAstring[self.totalbit*pi1+key1*self.nbit:self.totalbit*pi1+key1*self.nbit+self.nbit]
            cp=random.randint(1,self.nbit-1)
            unifiedgene0=copygene0[0:cp]
            unifiedgene0.extend(copygene1[cp:self.nbit])
            unifiedgene1=copygene1[0:cp]
            unifiedgene1.extend(copygene0[cp:self.nbit])
            for ivar in range(self.nvariable):
                # about child00
                self.ws3[cci2]=pi0
                if self.Hoxgene[self.nvariable*ci01+ivar] == 0:
                    for ibit in range(self.nbit):
                        self.GAstring[self.totalbit*ci01+ivar*self.nbit+ibit] = self.GAstring[self.totalbit*pi0+ivar*self.nbit+ibit]
                else:
                    for ibit in range(self.nbit):
                        self.GAstring[self.totalbit*ci01+ivar*self.nbit+ibit] = unifiedgene0[ibit]

                # about child 10
                self.ws3[cci3]=pi1
                if self.Hoxgene[self.nvariable*ci11+ivar] == 0:
                    for ibit in range(self.nbit):
                        self.GAstring[self.totalbit*ci11+ivar*self.nbit+ibit] = self.GAstring[self.totalbit*pi1+ivar*self.nbit+ibit]
                else:
                    for ibit in range(self.nbit):
                        self.GAstring[self.totalbit*ci11+ivar*self.nbit+ibit] = unifiedgene1[ibit]
            
            self.prange[ci01]=self.prange[pi0]
            self.prange[ci11]=self.prange[pi1]
            self.pfitness[cci2]=self.fitness[pi0]
            self.pfitness[cci3]=self.fitness[pi1]
        # print("HGcrossOver="+str(time.time()-start))

        ## Evaluation ##
        start=time.time()
        #--- 子個体の評価(evaluate the generated children) ---#
        for ipop in range(self.hgncross):
            # convert a genotyped solution of children to a real-numbered solution
            self.HGARange(ipop+self.npop,ipop)
            # evaluate children's solution
            self.functioncall(ipop+self.npop)
            # evaluate children's Hoxgene
            self.evalHoxgene(ipop+self.npop,ipop)

            diff=0
            for ivar in range(self.nvariable):
                diff+=((self.x[(ipop+self.npop)*self.nvariable+ivar]-self.xgreat[ivar])**2)
            diff=math.sqrt(diff)

            self.HGfitness[ipop+self.npop]=self.HGfitness[ipop+self.npop]/(diff+1)

            # for debuging
            self.data_pare[ipop]=self.fgreat
            self.data_chil[ipop]=self.fitness[ipop+self.npop]
            self.data_calc[ipop]=self.HGfitness[ipop+self.npop]
            cfit_tmp.append(self.fitness[ipop+self.npop])
            diff_tmp.append(self.fgreat-cfit_tmp[ipop])
        # print("HGCeval="+str(time.time()-start))
        
        self.writeFitChange("CrossOver",P_tmp,pfit_tmp,cfit_tmp,diff_tmp)
        
        # self.evalHoxAve(self.npop+self.hgncross,"CrossOver")

        # get the best f value and the index of individual with the best f value in each search range 
        # for ipop in range(self.npop+self.hgncross):
        #     k=self.prange[ipop]
        #     if self.rangetop[k]>self.fitness[ipop]:
        #         self.rangetop[k]=self.fitness[ipop]     # the best f value in each search range
        #         self.rangewho[k]=ipop                   # the index of the individual with the best f value in each search range
        

        # self.writeDebug4("after HGCrossOver")
    

    def writeFitChange(self,timing,P_tmp,pfit_tmp,cfit_tmp,diff_tmp):
        seed_dir=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"seed"+str(self.seed)))
        result_dir=os.path.join(seed_dir,"result")

        HGFitChange=os.path.normpath(os.path.join(result_dir,"FitChange.csv"))
        with open(HGFitChange,"a") as f:
            writer=csv.writer(f,lineterminator='\n')
            writer.writerow(["g="+str(self.g),timing])
            pdat=["pair"]
            pdat.extend(P_tmp)
            writer.writerow(pdat)
            pfit=["pfit"]
            pfit.extend(pfit_tmp)
            writer.writerow(pfit)
            cfit=["cfit"]
            cfit.extend(cfit_tmp)
            writer.writerow(cfit)
            diff=["diff"]
            diff.extend(diff_tmp)
            writer.writerow(diff)
            writer.writerow([""])


    def CrossOverPHASE(self):
        if not self.hoxflg:
            self.writeDebug5("no estimation")
            self.CrossOver()
        elif self.hoxflg:
            self.writeDebug5("estimating..")
            self.HGCrossOver()
    
    def writeDebug5(self,cstr):
        seed_dir=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"seed"+str(self.seed)))
        result_dir=os.path.join(seed_dir,"result")

        datapath=os.path.join(result_dir,"estimate_time.csv")
        with open(datapath,"a") as f:
            writer = csv.writer(f,lineterminator='\n')
            writer.writerow([self.g,cstr,self.hoxcnt])
    

    def NaturalSelect(self):
        # print("NaturalSelect")
        # self.writeDebug4("before Naturalselect")

        ### Natural Select ###
        #--- ルーレット盤の作成(make a roulette board for selecting individuals in next generation) ---#
        roulettetotal=0
        roulette=[0 for i in range(self.npop+self.ncross)]
        maxfit=min(self.fitness[0:self.npop+self.ncross])
        minfit=max(self.fitness[0:self.npop+self.ncross])
        self.imin=self.fitness[0:self.npop+self.ncross].index(maxfit)

        
        # update the best f value ever (fgreat) and the best solution ever (xgreat)
        # debug point mmfit
        if self.fgreat>maxfit:
            self.fgreat=maxfit
            self.greatAge = 0
            if self.psearch_flg:
                self.pgreatAge = 0
                self.newcnt+=1
            for i in range(self.nvariable):
                self.xgreat[i]=self.x[self.nvariable*self.imin+i]


        # make a roulette board
        start=time.time()
        for i in range(self.npop+self.ncross):
            roulette[i]=1+999*(self.fitness[i]-minfit)/(maxfit-minfit)
            roulettetotal+=roulette[i]
        # print("mkroulette(N)="+str(time.time()-start))
        
        # guarantee a minimum number of individuals in each search range to avoid annihilation of individuals in the seach range
        # debug point ws2
        start=time.time()
        for i in range(self.npop+self.ncross):
            self.ws2[i]=roulette[i]
        
        for i in range(self.nrange):
            for j in range(self.minpop):
                tttp=0
                it = 0
                for ipop in range(self.npop+self.ncross):
                    if self.prange[ipop]==i:
                        if tttp<self.ws2[ipop]:
                            tttp=self.ws2[ipop]
                            it=ipop
                self.ws2[it]=0
            self.rangenext[i]=tttp
        
        # self.writeDebug4("middle1 Naturalselect")

        # guarantee a minimum number of individuals in each search range
        for irange in range(self.nrange):
            self.rangepop[irange]=0
        ii=0
        ws=[-1 for i in range(self.ncross+self.npop)]
        # guarantee the individual with the best solution
        ws[0] = self.imin
        ii += 1
        roulettetotal-=roulette[self.imin]
        roulette[self.imin]=0
        self.rangepop[self.prange[self.imin]]+=1
        # select individuals except the individual with the best solution
        for j in range(self.npop+self.ncross):
            if roulette[j]>self.rangenext[self.prange[j]] and self.rangepop[self.prange[j]]<self.minpop:
                ws[ii]=j
                ii+=1
                roulettetotal-=roulette[j]
                roulette[j]=0
                self.rangepop[self.prange[j]]+=1
        
        # self.writeDebug4("middle2 Naturalselect")

        #--- 残りの個体を選択(select individuals that will survive to the next generation from the rest of individuals) ---#
        for i in range(self.npop-ii):
            key=random.uniform(0,roulettetotal)
            j=0
            hit=0
            while hit<key:
                hit+=roulette[j]
                if hit<key:
                    j+=1
            roulettetotal-=roulette[j]
            ws[i+ii]=j
            roulette[j]=0
        # print("selectIND="+str(time.time()-start))

        # print("surviving individuals:"+str(ws[0:self.npop]))
        #--- 親の並び替え(sort surviving individuals) ---#
        start=time.time()
        lst=[-1 for i in range(self.npop)]
        for i in range(self.npop):
            minv=1.0e33
            for j in range(self.npop):
                if minv>ws[j]:
                    minv=ws[j]
                    jj=j
            lst[i]=minv
            ws[jj]=self.npop+self.ncross+1
        ffitness=0
        mmfitness=1.0e33
        # debug point sort
        for i in range(self.npop):
            j=lst[i]
            self.prange[i]=self.prange[j]
            for k in range(self.totalbit):
                self.GAstring[self.totalbit*i+k]=self.GAstring[self.totalbit*j+k]
            for k in range(self.nvariable):
                self.x[self.nvariable*i+k]=self.x[self.nvariable*j+k]
            for k in range(self.funcmax):
                self.func[self.funcmax*i+k]=self.func[self.funcmax*j+k]
            self.fitness[i]=self.fitness[j]
            if mmfitness>self.fitness[i]:
                mmfitness=self.fitness[i]
                self.imin=i
            ffitness+=self.fitness[i]
        ffitness/=self.npop
        # print("sortIND="+str(time.time()-start))

        # self.writeDebug4("after Naturalselect")



    def HGNaturalSelect(self):
        # print("HGNaturalselect")
        ### Natural Select ###
        # self.writeDebug4("before HGNaturalselect")
        #--- ルーレット盤の作成(make a roulette board for selecting individuals in next generation) ---#
        roulettetotal=0
        roulette=[0 for i in range(self.npop+self.hgncross)]
        maxfit=min(self.fitness[0:self.npop+self.hgncross])
        minfit=max(self.fitness[0:self.npop+self.hgncross])
        self.imin=self.fitness[0:self.npop+self.hgncross].index(maxfit)
        
        # update the best f value ever (fgreat) and the best solution ever (xgreat)
        # debug point mmfit
        if self.fgreat>maxfit:
            self.fgreat=maxfit
            self.greatAge = 0
            if self.psearch_flg:
                self.pgreatAge = 0
                self.newcnt+=1
            for i in range(self.nvariable):
                self.xgreat[i]=self.x[self.nvariable*self.imin+i]


        # make a roulette board
        for i in range(self.npop+self.hgncross):
            roulette[i]=1+999*(self.fitness[i]-minfit)/(maxfit-minfit)
            roulettetotal+=roulette[i]
        
        # guarantee a minimum number of individuals in each search range to avoid annihilation of individuals in the seach range
        # debug point ws2
        for i in range(self.npop+self.hgncross):
            self.ws4[i]=roulette[i]
        
        for i in range(self.nrange):
            for j in range(self.minpop):
                tttp=0
                for ipop in range(self.npop+self.hgncross):
                    if self.prange[ipop]==i:
                        if tttp<self.ws4[ipop]:
                            tttp=self.ws4[ipop]
                            it=ipop
                self.ws4[it]=0
            self.rangenext[i]=tttp
        
        # self.writeDebug4("middle1 HGNaturalselect")

        # guarantee a minimum number of individuals in each search range
        for irange in range(self.nrange):
            self.rangepop[irange]=0
        ii=0
        ws=[-1 for i in range(self.hgncross+self.npop)]
        # guarantee the individual with the best solution
        ws[0] = self.imin
        ii += 1
        roulettetotal-=roulette[self.imin]
        roulette[self.imin]=0
        self.rangepop[self.prange[self.imin]]+=1
        # select individuals except the individual with the best solution
        for j in range(self.npop+self.hgncross):
            if roulette[j]>self.rangenext[self.prange[j]] and self.rangepop[self.prange[j]]<self.minpop:
                ws[ii]=j
                ii+=1
                roulettetotal-=roulette[j]
                roulette[j]=0
                self.rangepop[self.prange[j]]+=1
        
        # self.writeDebug4("middle2 HGNaturalselect")

        #--- 残りの個体を選択(select individuals that will survive to the next generation from the rest of individuals) ---#
        for i in range(self.npop-ii):
            key=random.uniform(0,roulettetotal)
            j=0
            hit=0
            while hit<key:
                hit+=roulette[j]
                if hit<key:
                    j+=1
            roulettetotal-=roulette[j]
            ws[i+ii]=j
            roulette[j]=0

        # print("surviving individuals:"+str(ws[0:self.npop]))

        #--- 親の並び替え(sort surviving individuals) ---#
        lst=[-1 for i in range(self.npop)]
        for i in range(self.npop):
            minv=1.0e33
            for j in range(self.npop):
                if minv>ws[j]:
                    minv=ws[j]
                    jj=j
            lst[i]=minv
            ws[jj]=self.npop+self.hgncross+1
        ffitness=0
        mmfitness=1.0e33
        
        # debug point sort
        for i in range(self.npop):
            j=lst[i]
            self.prange[i]=self.prange[j]
            #print(str(j)+" "+str(prange[i]))
            for k in range(self.totalbit):
                self.GAstring[self.totalbit*i+k]=self.GAstring[self.totalbit*j+k]
            for k in range(self.nvariable):
                self.x[self.nvariable*i+k]=self.x[self.nvariable*j+k]
            for k in range(self.funcmax):
                self.func[self.funcmax*i+k]=self.func[self.funcmax*j+k]
            self.fitness[i]=self.fitness[j]
            if mmfitness>self.fitness[i]:
                mmfitness=self.fitness[i]
                self.imin=i
            ffitness+=self.fitness[i]
        ffitness/=self.npop
        
        # self.writeDebug4("middle HGNaturalselect")

        ### Hoxgene Natural Select ###
        #--- ルーレット盤の作成(make a roulette board for selecting individuals in next generation) ---#
        roulettetotal=0
        roulette=[0 for i in range(self.npop+self.hgncross)]
        maxfit=max(self.HGfitness[0:self.npop+self.hgncross])
        minfit=min(self.HGfitness[0:self.npop+self.hgncross])
        self.hoximin=self.HGfitness[0:self.npop+self.hgncross].index(maxfit)

        
        # update the best f value ever (fgreat) and the best solution ever (xgreat)
        # debug point mmfit
        if self.hoxfgreat<maxfit:
            self.hoxfgreat=maxfit
            for i in range(self.nvariable):
                self.hoxgreat[i]=self.Hoxgene[self.nvariable*self.hoximin+i]

        # make a roulette board
        start=time.time()
        for i in range(self.npop+self.hgncross):
            roulette[i]=1+100000*(self.HGfitness[i]-minfit+5)/(maxfit-minfit+5)
            roulettetotal+=roulette[i]
        # print("HGmkroulette(N)="+str(time.time()-start))

        start=time.time()
        ii=0
        ws=[-1 for i in range(self.hgncross+self.npop)]
        # guarantee surviving of the individual with the best solution
        # ws[0] = self.hoximin
        # ii += 1
        # roulettetotal-=roulette[self.hoximin]
        # roulette[self.hoximin]=0
        # self.rangepop[self.prange[self.hoximin]]+=1

        self.mkpair3("before select")
        # guarantee TopN Hoxgenes for the next generation
        # topN=int(self.npop*0.1)
        topN=int(self.npop*0.1)
        # print(roulette)
        for ipop in range(topN):
            ind = roulette.index(max(roulette))
            ws[ii] = ind
            ii += 1
            roulettetotal-=roulette[ind]
            roulette[ind]=0
        
        #--- 生き残るHox遺伝子選択(select Hoxgenes that will survive to the next generation) ---#
        for i in range(self.npop-ii):
            key=random.uniform(0,roulettetotal)
            j=0
            hit=0
            while hit<key:
                hit+=roulette[j]
                if hit<key:
                    j+=1
            roulettetotal-=roulette[j]
            ws[i+ii]=j
            roulette[j]=0
        # print("HGselectIND="+str(time.time()-start))

        #--- 親の並び替え(sort surviving Hoxgenes) ---#
        start=time.time()
        lst=[-1 for i in range(self.npop)]
        for i in range(self.npop):
            minv=1.0e33
            for j in range(self.npop):
                if minv>ws[j]:
                    minv=ws[j]
                    jj=j
            lst[i]=minv
            ws[jj]=self.npop+self.hgncross+1
        ffitness=0
        mmfitness=-1.0e33
        # print(lst)
        # debug point sort
        for i in range(self.npop):
            j=lst[i]
            for k in range(self.nvariable):
                self.Hoxgene[self.nvariable*i+k]=self.Hoxgene[self.nvariable*j+k]
            self.HGfitness[i]=self.HGfitness[j]
            if mmfitness<self.HGfitness[i]:
                mmfitness=self.HGfitness[i]
                self.hoximin=i
        # print("HGsortIND="+str(time.time()-start))
        # self.writeDebug4("after HGNaturalselect")
        self.mkpair3("after select")


    def NaturalSelectPHASE(self):
        if not self.hoxflg:
            self.NaturalSelect()
        elif self.hoxflg:
            self.HGNaturalSelect()
    

    def mkpair3(self,state):
        # initialize the set P
        if self.P != []:
            while self.P != []:
                del self.P[0]
        
        hoxgenes=0
        fitnesss=0
        if "before" in state:
            hoxgenes=copy.deepcopy(self.Hoxgene[0:(self.npop+self.hgncross)*self.nvariable])
            fitnesss=copy.deepcopy(self.HGfitness[0:self.npop+self.hgncross])
        else:
            hoxgenes=self.Hoxgene[0:self.npop*self.nvariable]
            fitnesss = self.HGfitness[0:self.npop]
        pairfit=[]
        for i in range(int(self.nvariable*2)):
            eflg = True
            while eflg:
                igene = fitnesss.index(max(fitnesss))
                # print("igene="+str(igene))
                ihoxgene = hoxgenes[igene*self.nvariable:igene*self.nvariable+self.nvariable]
                
                Pi = []
                for ivar in range(self.nvariable):
                    ig = ihoxgene[ivar]
                    if ig == 1:
                        Pi.append(ivar)
                    else:
                        pass

                self.P.append(Pi)
                pairfit.append(fitnesss[igene])
                eflg = False
                # update the lists for detemining variable pairs
                del fitnesss[igene]
                del hoxgenes[igene*self.nvariable:igene*self.nvariable+self.nvariable]
        
        self.writePAIR3(pairfit,state)
        while self.P != []:
            del self.P[0]
    

    def writePAIR3(self,pairfit,state):
        seed_dir=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"seed"+str(self.seed)))
        result_dir=os.path.join(seed_dir,"result")
        PAIR_dir = os.path.join(result_dir,"PAIR")

        arr_tmp = []
        for pi in self.P:
            strpi = "["
            for pii in pi:
                strpi += (str(pii)+",")
            strpi = strpi.rstrip(",")
            strpi += "]"

            arr_tmp.append(strpi)
        
        fit=0
        if "before" in state:
            fit=copy.deepcopy(self.HGfitness[0:self.npop+self.hgncross])
        else:
            fit=copy.deepcopy(self.HGfitness[0:self.npop])
        pairdataN = os.path.join(PAIR_dir,"PAIRdataNaturalSelect.csv")
        with open(pairdataN,"a") as f:
            writer = csv.writer(f, lineterminator = '\n')
            data=[self.g,self.npair,state]
            data.extend(arr_tmp)
            fitdata=["","",""]
            fitdata.extend(pairfit)
            tfitdata=["","",""]
            tfitdata.extend(sorted(fit, reverse=True))

            writer.writerow(data)
            writer.writerow(fitdata)
            writer.writerow(tfitdata)


    def Mutation(self):
        # print("Mutation")
        ### Mutation ###
        lflg=0
        tflg=0
        #if fgreat*1.25<mmfitness and generation>500:
        #    tflg=1
        #if newborn>50:
        #    tflg=1
        #print(str(tflg))

        change=int(0.05*self.npop)           # the number of individuals cause mutation 
        #change+=int(generation/50)
        for ipop in range(self.npop):
            self.zflg[ipop]=0   # this flag express whether individuals cause mutation or not
        bigchange=0

        #--- 突然変異(let some individuals mutate) ---#
        if not self.psearch_flg:
            start=time.time()
            for ipop in range(self.npop):
                if ipop==self.imin:
                    self.zflg[ipop]=1
                else:
                    z=random.randint(0,self.npop)
                    if z<change:
                        km=random.randint(1,2)
                        for ii in range(km):
                            kk=random.randint(0,self.totalbit-1)
                            if self.GAstring[self.totalbit*ipop+kk]==0:
                                self.GAstring[self.totalbit*ipop+kk]=1
                            else:
                                self.GAstring[self.totalbit*ipop+kk]=0
                    else:
                        self.zflg[ipop]=1
            # print("zflg="+str(self.zflg))
            # print("Mutation="+str(time.time()-start))
            

            #--- 外来種突然変異(let some individuals Exotic species mutate) ---#       
            start=time.time()
            for i in range(int(self.immigrant)):
                km=self.imin
                while km==self.imin:
                    km=random.randint(0,self.npop-1)
                self.zflg[km]=0
                for ii in range(self.totalbit):
                    self.GAstring[self.totalbit*km+ii]=random.randint(0,1)
            # print("zflg="+str(self.zflg))
            # print("AlianMutation="+str(time.time()-start))
        else:
            start=time.time()
            change=int(0.2*self.npop)  
            for ipop in range(self.npop):
                if ipop==self.imin:
                    self.zflg[ipop]=1
                else:
                    z=random.randint(0,self.npop)
                    if z<change:
                        ndim=len(self.Pi)
                        km=random.randint(1,ndim)       # the number of variables that cause mutation
                        vars=random.sample(self.Pi,km)
                        for ivar in vars:
                            kk=random.randint(0,self.nbit-1)
                            if self.GAstring[self.totalbit*ipop+self.nbit*ivar+kk]==0:
                                self.GAstring[self.totalbit*ipop+self.nbit*ivar+kk]=1
                            else:
                                self.GAstring[self.totalbit*ipop+self.nbit*ivar+kk]=0
                    else:
                        self.zflg[ipop]=1
            # print("zflg="+str(self.zflg))
            # print("Mutation="+str(time.time()-start))

            #--- 外来種突然変異(let some individuals Exotic species mutate) ---#
            start=time.time()     
            for i in range(int(self.immigrant)):
                km=self.imin
                while km==self.imin:
                    km=random.randint(0,self.npop-1)
                self.zflg[km]=0
                for ivar in self.Pi:
                    for ibit in range(self.nbit):
                        self.GAstring[self.totalbit*km+self.nbit*ivar+ibit]=random.randint(0,1)
            # print("zflg="+str(self.zflg))
            # print("AlianMutation="+str(time.time()-start))

        #--- 突然変異，外来種突然変異を起こした個体の評価(Evaluate individuals cause mutation and exotic species mutation) ---#
        data=[]
        start=time.time()
        for i in range(self.npop):
            if self.zflg[i]==0:
                self.ARange(i)
                self.functioncall(i)
                data.append(self.fitness[i])
        # print("fitness="+str(data))
        # print("evalM="+str(time.time()-start))

        # upadate information about individuals survive to the next generation and search ranges
        for irange in range(self.nrange):
            self.rangepop[irange] = 0
        
        mafit=1.0e33
        self.aaafit=0.0
        # the population of indivuduals in each seach range
        for ipop in range(self.npop):
            self.rangepop[self.prange[ipop]]+=1
            if mafit>self.fitness[ipop]:
                mafit=self.fitness[ipop]
                self.imin=ipop
                self.mrange=self.prange[ipop]
            self.aaafit+=(self.fitness[ipop]/self.npop)
        

        # 最良値の確認(confirm the best f value and the best solution)
        if mafit<self.fgreat:
            self.fgreat=mafit
            self.greatAge = 0
            if self.psearch_flg:
                self.pgreatAge = 0
                self.newcnt+=1
            for j in range(self.nvariable):
                self.xgreat[j]=self.x[self.nvariable*ipop+j]
        
        # define the search range with the individual with the best f value as mrange      
        self.age[self.mrange]=0

        # set individuals in mrange not to be updated
        for ipop in range(self.npop):
            self.zflg[ipop]=0
            if self.prange[ipop]==self.mrange:
                self.zflg[ipop]=1
        
        # record the greatest indivisuals of each search range
        for irange in range(self.nrange):
            self.fgreat_range[irange] = 1.0e+33
            self.elite_range[irange] = -1
        for ipop in range(self.npop):
            k = self.prange[ipop]
            s = self.fitness[ipop]
            if s < self.fgreat_range[k]:
                self.fgreat_range[k] = s
                self.elite_range[k] = ipop

        # caluculate the average of all design variables and stadard diviation of all design variables
        for i in range(self.nvariable):
            self.tave[i]=0
            for ipop in range(self.npop):
                self.tave[i]+=self.x[self.nvariable*ipop+i]
            self.tave[i]/=self.npop
            self.xstd[i]=0
            for ipop in range(self.npop):
                self.xstd[i]+=pow(self.x[self.nvariable*ipop+i]-self.tave[i],2)
            self.xstd[i]/=self.npop
            self.xstd[i]=math.sqrt(self.xstd[i])
        # print("tave="+str(self.tave[0:self.nvariable]))
        # print("xstd="+str(self.xstd[0:self.nvariable]))
    


    def HGMutation(self):
        # print("HGMutation")
        # self.writeDebug4("before HGMutation")

        ### Mutation ###
        lflg=0
        tflg=0

        change=int(0.05*self.npop)           # the number of individuals cause mutation 
        #change+=int(generation/50)
        for ipop in range(self.npop):
            self.hoxzflg[ipop]=0   # this flag express whether individuals cause mutation or not
        bigchange=0

        for ipop in range(self.npop):
            self.ws3[ipop]=-1
            self.pfitness[ipop]=1.0e33

        self.mkpair3("a Mutation")
        #--- 突然変異(let some individuals mutate) ---#
        start=time.time()
        x_k=[]
        cipop=0
        for ipop in range(self.npop):
            if ipop==self.hoximin:
                self.hoxzflg[ipop]=1
            elif ipop==self.imin:
                self.hoxzflg[ipop]=1
            else:
                z=random.randint(0,self.npop)
                if z<change:
                    ihoxgene=self.Hoxgene[self.nvariable*ipop:self.nvariable*ipop+self.nvariable]
                    scnt=ihoxgene.count(1)
                    sind=[i for i, x in enumerate(ihoxgene) if x == 1]
                    for ivar in range(self.nvariable):
                        x_k.append(self.x[ipop*self.nvariable+ivar])
                    
                    # case0 : Hoxgene doesn't take 1
                    if 1 not in self.Hoxgene[self.nvariable*ipop:self.nvariable*ipop+self.nvariable]:
                        # random method
                        ntake1=random.randint(2,4)
                        stbits=random.sample(range(0,self.nvariable),ntake1)

                        # give 1 to the bits
                        for ibit in stbits:
                            self.Hoxgene[self.nvariable*ipop+ibit]=1

                        self.ws3[cipop]=ipop
                        self.pfitness[cipop]=copy.deepcopy(self.fitness[ipop])
                        cipop+=1
                    
                    # case1 : Hoxgene takes only one 1
                    elif self.Hoxgene[self.nvariable*ipop:self.nvariable*ipop+self.nvariable].count(1) == 1:
                        # random method
                        ntake1=random.randint(1,3)
                        stbits=random.sample(range(0,self.nvariable),ntake1)

                        # give 1 to the bits
                        for ibit in stbits:
                            self.Hoxgene[self.nvariable*ipop+ibit]=1

                        self.ws3[cipop]=ipop
                        self.pfitness[cipop]=copy.deepcopy(self.fitness[ipop])
                        cipop+=1
                    
                    # case2 : Hoxgene takes two 1s
                    elif self.Hoxgene[self.nvariable*ipop:self.nvariable*ipop+self.nvariable].count(1) == 2:
                        key=random.randint(0,1)
                        
                        # case2-0 : preserve current bits context
                        if key == 0:
                            self.ws3[cipop]=ipop
                            self.pfitness[cipop]=copy.deepcopy(self.fitness[ipop])
                            cipop+=1
                        # case2-1 : change a bit value
                        else:
                            # random method
                            itake0=[i for i,v in enumerate(self.Hoxgene[self.nvariable*ipop:self.nvariable*ipop+self.nvariable]) if v == 0]
                            stbit=random.sample(itake0,1)[0]

                            # give 1 to the selected bit
                            self.Hoxgene[self.nvariable*ipop+stbit]=1

                            self.ws3[cipop]=ipop
                            self.pfitness[cipop]=copy.deepcopy(self.fitness[ipop])
                            cipop+=1
                    
                    # case3 : Hoxgene takes three 1s
                    elif self.Hoxgene[self.nvariable*ipop:self.nvariable*ipop+self.nvariable].count(1) == 3:
                        key=random.randint(0,1)

                        # case3-0 : preserve current bits context
                        if key == 0:
                            self.ws3[cipop]=ipop
                            self.pfitness[cipop]=copy.deepcopy(self.fitness[ipop])
                            cipop+=1
                        
                        # case3-1
                        else:
                            kbit=random.randint(0,self.nvariable)

                            if self.Hoxgene[self.nvariable*ipop+kbit] == 1:
                                self.Hoxgene[self.nvariable*ipop+kbit]=0
                            else:
                                self.Hoxgene[self.nvariable*ipop+kbit]=1
                            
                            self.ws3[cipop]=ipop
                            self.pfitness[cipop]=copy.deepcopy(self.fitness[ipop])
                            cipop+=1
                    
                    # case4 : Hoxgene takes four 1s
                    elif self.Hoxgene[self.nvariable*ipop:self.nvariable*ipop+self.nvariable].count(1) == 4:
                        key=random.randint(0,1)

                        # case4-0 : preserve current bits context
                        if key == 0:
                            self.ws3[cipop]=ipop
                            self.pfitness[cipop]=copy.deepcopy(self.fitness[ipop])
                            cipop+=1
                        
                        # case4-1
                        else:
                            # random method
                            itake1=[i for i,v in enumerate(self.Hoxgene[self.nvariable*ipop:self.nvariable*ipop+self.nvariable]) if v == 1]
                            stbit=random.sample(itake1,1)[0]

                            # remove 1 and replace 0 
                            self.Hoxgene[self.nvariable*ipop+stbit]=0

                            self.ws3[cipop]=ipop
                            self.pfitness[cipop]=copy.deepcopy(self.fitness[ipop])
                            cipop+=1
                    
                    # case5 : Hoxgene takes five or more 1s
                    else:
                        # random method
                        itake1=[i for i,v in enumerate(self.Hoxgene[self.nvariable*ipop:self.nvariable*ipop+self.nvariable]) if v == 1]
                        cntake1=len(itake1)          # current No of bits taking 1
                        ntake1=random.randint(2,4)  # the No. of bits taking 1 in this Hoxgene
                        removebits=random.sample(itake1,cntake1-ntake1)  # select the bits taking 0

                        # remove 1s and replace 0s from the selected bits
                        for ibit in removebits:
                            self.Hoxgene[self.nvariable*ipop+ibit]=0
                        
                        self.ws3[cipop]=ipop
                        self.pfitness[cipop]=copy.deepcopy(self.fitness[ipop])
                        cipop+=1

                else:
                    self.hoxzflg[ipop]=1
        # print("HGMutation="+str(time.time()-start))

        #--- 突然変異，外来種突然変異を起こした個体の評価(Evaluate individuals cause mutation and exotic species mutation) ---#
        estart=time.time()
        P_tmp=[]
        pfit_tmp=[]
        cfit_tmp=[]
        diff_tmp=[]
        cipop=0
        # usualgene=copy.deepcopy(self.GAstring)
        # realtype_x=copy.deepcopy(self.x)
        for i in range(self.npop):
            if self.hoxzflg[i]==0:
                # copy data
                for ibit in range(self.nvariable*self.nbit):
                    self.usualgene[ibit]=self.GAstring[i*self.totalbit+ibit]
                for ivar in range(self.nvariable):
                    self.realtype_x[ivar]=self.x[i*self.nvariable+ivar]
                self.f_value=0.0

                # unified array value
                # key0=random.randint(0,self.nvariable-1)
                # copygene0=self.GAstring[self.totalbit*pi0+key0*self.nbit:self.totalbit*pi0+key0*self.nbit+self.nbit]
                # key1=random.randint(0,self.nvariable-1)
                # unifiedgene0=copygene0[0:cp]
                # unifiedgene0.extend(copygene1[cp:self.nbit])
                # unifiedgene1=copygene1[0:cp]
                # unifiedgene1.extend(copygene0[cp:self.nbit])

                start=time.time()
                P_tmp.append(self.strTypePtmp(i))
                # print("tostr="+str(time.time()-start))

                pfit_tmp.append(self.fgreat)
                
                start=time.time()
                # update usual gene(GAstring)
                for ivar in range(self.nvariable):
                    ihox=self.Hoxgene[self.nvariable*i+ivar]
                    if ihox==0:
                        pass
                    if ihox==1:
                        kk=random.randint(0,self.nbit-1)
                        if self.usualgene[ivar*self.nbit+kk] == 0:
                            self.usualgene[ivar*self.nbit+kk]=1
                        else:
                            self.usualgene[ivar*self.nbit+kk]=0
                # print("chg ugene="+str(time.time()-start))
                
                # update realtype solution
                start=time.time()
                self.toReal(i)
                # print("--> return="+str(time.time()-self.s_time))
                # print("to real="+str(time.time()-start))

                # update function value
                start=time.time()
                self.toFunc()
                # print("--> return="+str(time.time()-self.s_time))
                # print("x to func="+str(time.time()-start))

                # update evaluation value of Hoxgene
                # print("before:"+str(self.HGfitness[i]))
                self.HGfitness[i]=self.fgreat-self.f_value
                # print("after1:"+str(self.HGfitness[i]))

                start=time.time()
                diff=0
                for ivar in range(self.nvariable):
                    diff+=((x_k[cipop*self.nvariable+ivar]-self.xgreat[ivar])**2)
                diff=math.sqrt(diff)
                # print("calc norm="+str(time.time()-start))
                
                self.HGfitness[i]=self.HGfitness[i]/(diff+1)
                # print("after2:"+str(self.HGfitness[i]))

                # for debuging
                start=time.time()
                self.data_pare[cipop]=self.fgreat
                self.data_chil[cipop]=self.f_value
                self.data_calc[cipop]=self.HGfitness[i]
                cfit_tmp.append(self.f_value)
                diff_tmp.append(self.fgreat-self.f_value)
                # print("debug="+str(time.time()-start))

                cipop+=1
        # print("HGevalM="+str(time.time()-estart))
        self.writeFitChange("Mutation",P_tmp,pfit_tmp,cfit_tmp,diff_tmp)
        # self.evalHoxAve(self.npop,"Mutation")

        # self.writeDebug4("after HGMutation")

        # upadate information about individuals survive to the next generation and search ranges
        for irange in range(self.nrange):
            self.rangepop[irange] = 0

        hoxmafit=-1.0e33
        mafit=1.0e33
        self.aaafit=0.0
        # the population of indivuduals in each seach range
        for ipop in range(self.npop):
            self.rangepop[self.prange[ipop]]+=1
            if hoxmafit<self.HGfitness[ipop]:
                hoxmafit=self.HGfitness[ipop]
                self.hoximin=ipop
            if mafit>self.fitness[ipop]:
                mafit=self.fitness[ipop]
                self.imin=ipop
                self.mrange=self.prange[ipop]
            self.aaafit+=(self.fitness[ipop]/self.npop)

        # 最良値の確認(confirm the best f value and the best solution)
        if hoxmafit>self.hoxfgreat:
            self.hoxfgreat=hoxmafit
            for j in range(self.nvariable):
                self.hoxgreat[j]=self.Hoxgene[self.nvariable*ipop+j]
        
        if mafit<self.fgreat:
            self.fgreat=mafit
            self.greatAge = 0
            if self.psearch_flg:
                self.pgreatAge = 0
                self.newcnt+=1
            for j in range(self.nvariable):
                self.xgreat[j]=self.x[self.nvariable*ipop+j]
        
        # print(">>> after Hoxgene Mutation")
        # print("fgreat="+str(self.fgreat)+" f_imin="+str(self.fitness[self.imin])+" f_teacher="+str(min(self.fitness[0:self.npop])))

        # set individuals in mrange not to be updated
        for ipop in range(self.npop):
            self.hoxzflg[ipop]=0
        
        # define the search range with the individual with the best f value as mrange      
        self.age[self.mrange]=0

        # set individuals in mrange not to be updated
        for ipop in range(self.npop):
            self.zflg[ipop]=0
            if self.prange[ipop]==self.mrange:
                self.zflg[ipop]=1
        
        # record the greatest indivisuals of each search range
        for irange in range(self.nrange):
            self.fgreat_range[irange] = 1.0e+33
            self.elite_range[irange] = -1
        for ipop in range(self.npop):
            k = self.prange[ipop]
            s = self.fitness[ipop]
            if s < self.fgreat_range[k]:
                self.fgreat_range[k] = s
                self.elite_range[k] = ipop

        # caluculate the average of all design variables and stadard diviation of all design variables
        for i in range(self.nvariable):
            self.tave[i]=0
            for ipop in range(self.npop):
                self.tave[i]+=self.x[self.nvariable*ipop+i]
            self.tave[i]/=self.npop
            self.xstd[i]=0
            for ipop in range(self.npop):
                self.xstd[i]+=pow(self.x[self.nvariable*ipop+i]-self.tave[i],2)
            self.xstd[i]/=self.npop
            self.xstd[i]=math.sqrt(self.xstd[i])
        self.mkpair3("b Mutation")

    def MutationPHASE(self):
        if not self.hoxflg:    
            self.Mutation()
        elif self.hoxflg:
            self.HGMutation()
            self.mkpair2("est end")


    def UpdateRange(self):
        # print("updataRange")
        ### update each search range except mrange ###            
        refresh=int(self.npop/self.nrange+10)*1.5
        #a=input()
        # self.writeDebug4("upadate Range")
        for irange in range(self.nrange):
            if irange!=self.mrange:
                
                #--- Update ---#
                if not self.psearch_flg:
                    for i in range(self.nvariable):
                        self.xave[irange*self.nvariable+i]=0
                        for j in range(self.npop):
                            if self.prange[j]==irange:
                                self.xave[irange*self.nvariable+i]+=self.x[j*self.nvariable+i]
                        self.xave[irange*self.nvariable+i]/=self.rangepop[irange]
                        if self.xave[irange*self.nvariable+i]>self.xmax[i]-self.ppp*self.smin:
                            self.xave[irange*self.nvariable+i]=self.xmax[i]-self.ppp*self.smin
                        if self.xave[irange*self.nvariable+i]<self.xmin[i]+self.ppp*self.smin:
                            self.xave[irange*self.nvariable+i]=self.xmin[i]+self.ppp*self.smin
                        self.tstd[irange*self.nvariable+i]=0
                        for j in range(self.npop):
                            if self.prange[j]==irange:
                                self.tstd[irange*self.nvariable+i]+=pow(self.x[j*self.nvariable+i]-self.xave[irange*self.nvariable+i],2)
                        self.tstd[irange*self.nvariable+i]/=self.rangepop[irange]
                        self.tstd[irange*self.nvariable+i]=math.sqrt(self.tstd[irange*self.nvariable+i])
                        if self.xmax[i]<self.xave[irange*self.nvariable+i]+self.ppp*self.tstd[irange*self.nvariable+i]:
                            self.xrstd[irange*self.nvariable+i]=(self.xmax[i]-self.xave[irange*self.nvariable+i])/self.ppp
                        else:
                            self.xrstd[irange*self.nvariable+i]=self.tstd[irange*self.nvariable+i]
                        if self.xmin[i]>self.xave[irange*self.nvariable+i]-self.ppp*self.tstd[irange*self.nvariable+i]:
                            self.xlstd[irange*self.nvariable+i]=(-self.xmin[i]+self.xave[irange*self.nvariable+i])/self.ppp
                        else:
                            self.xlstd[irange*self.nvariable+i]=self.tstd[irange*self.nvariable+i]
                else:
                    if irange == self.locrange:
                        for i in range(self.nvariable):
                            self.xave[irange*self.nvariable+i]=0
                            for j in range(self.npop):
                                if self.prange[j]==irange:
                                    self.xave[irange*self.nvariable+i]+=self.x[j*self.nvariable+i]
                            self.xave[irange*self.nvariable+i]/=self.rangepop[irange]
                            if self.xave[irange*self.nvariable+i]>self.xmax[i]-self.ppp*self.smin:
                                self.xave[irange*self.nvariable+i]=self.xmax[i]-self.ppp*self.smin
                            if self.xave[irange*self.nvariable+i]<self.xmin[i]+self.ppp*self.smin:
                                self.xave[irange*self.nvariable+i]=self.xmin[i]+self.ppp*self.smin
                            self.tstd[irange*self.nvariable+i]=0
                            for j in range(self.npop):
                                if self.prange[j]==irange:
                                    self.tstd[irange*self.nvariable+i]+=pow(self.x[j*self.nvariable+i]-self.xave[irange*self.nvariable+i],2)
                            self.tstd[irange*self.nvariable+i]/=self.rangepop[irange]
                            self.tstd[irange*self.nvariable+i]=math.sqrt(self.tstd[irange*self.nvariable+i])
                            if self.xmax[i]<self.xave[irange*self.nvariable+i]+self.ppp*self.tstd[irange*self.nvariable+i]:
                                self.xrstd[irange*self.nvariable+i]=(self.xmax[i]-self.xave[irange*self.nvariable+i])/self.ppp
                            else:
                                self.xrstd[irange*self.nvariable+i]=self.tstd[irange*self.nvariable+i]
                            if self.xmin[i]>self.xave[irange*self.nvariable+i]-self.ppp*self.tstd[irange*self.nvariable+i]:
                                self.xlstd[irange*self.nvariable+i]=(-self.xmin[i]+self.xave[irange*self.nvariable+i])/self.ppp
                            else:
                                self.xlstd[irange*self.nvariable+i]=self.tstd[irange*self.nvariable+i]
                    else:
                        for i in range(self.nvariable):
                            self.xave[irange*self.nvariable+i]=0
                            for j in range(self.npop):
                                if self.prange[j]==irange:
                                    self.xave[irange*self.nvariable+i]+=self.x[j*self.nvariable+i]
                            self.xave[irange*self.nvariable+i]/=self.rangepop[irange]
                            if self.xave[irange*self.nvariable+i]>self.xmax[i]-self.ppp*self.smin:
                                self.xave[irange*self.nvariable+i]=self.xmax[i]-self.ppp*self.smin
                            if self.xave[irange*self.nvariable+i]<self.xmin[i]+self.ppp*self.smin:
                                self.xave[irange*self.nvariable+i]=self.xmin[i]+self.ppp*self.smin
                            self.tstd[irange*self.nvariable+i]=0
                            for j in range(self.npop):
                                if self.prange[j]==irange:
                                    self.tstd[irange*self.nvariable+i]+=pow(self.x[j*self.nvariable+i]-self.xave[irange*self.nvariable+i],2)
                            self.tstd[irange*self.nvariable+i]/=self.rangepop[irange]
                            self.tstd[irange*self.nvariable+i]=math.sqrt(self.tstd[irange*self.nvariable+i])

                            self.xrstd[irange*self.nvariable+i] = self.tstd[irange*self.nvariable+i]
                            xu=self.xave[irange*self.nvariable+i]+self.ppp*self.xrstd[irange*self.nvariable+i]
                            if self.xmax[i] < xu:
                                self.xrstd[irange*self.nvariable+i]=(self.xmax[i]-self.xave[irange*self.nvariable+i])/self.ppp
                            else:
                                if xu < self.xgreat[i] and self.xgreat[i] < self.xave[irange*self.nvariable+i]:
                                    self.xrstd[irange*self.nvariable+i]=(self.xgreat[i]-self.xave[irange*self.nvariable+i])/self.ppp
                                    if self.xrstd[irange*self.nvariable+i] > (self.xmax[i]-self.xave[irange*self.nvariable+i])/self.ppp:
                                        self.xrstd[irange*self.nvariable+i]=(self.xmax[i]-self.xave[irange*self.nvariable+i])/self.ppp
                            
                            self.xlstd[irange*self.nvariable+i] = self.tstd[irange*self.nvariable+i]
                            xl=self.xave[irange*self.nvariable+i]-self.ppp*self.xlstd[irange*self.nvariable+i]
                            if self.xmin[i] > self.xave[irange*self.nvariable+i]-self.ppp*self.xlstd[irange*self.nvariable+i]:
                                self.xlstd[irange*self.nvariable+i]=(-self.xmin[i]+self.xave[irange*self.nvariable+i])/self.ppp
                            else:
                                if xl > self.xgreat[i] and self.xgreat[i] < self.xave[irange*self.nvariable+i]:
                                    self.xlstd[irange*self.nvariable+i]=(self.xave[irange*self.nvariable+i]-self.xgreat[i])/self.ppp
                                    if self.xlstd[irange*self.nvariable+i] > (self.xave[irange*self.nvariable+i] - self.xmin[i])/self.ppp:
                                        self.xlstd[irange*self.nvariable+i]=(self.xave[irange*self.nvariable+i] - self.xmin[i])/self.ppp


                #--- kill and regenerate search range that hasn't improved the greatest f value for several generations---#
                if self.age[irange]>50:
                    for i in range(self.nvariable):
                        if self.xstd[i]<self.nrange*self.tstd[irange*self.nvariable+i]:                         # regeneration type 1
                            # print("a"+str(i)+"   "+str(xgreat[i])+" "+str(xave[i])+" "+str(xstd[i])+" "+str(nrange*tstd[irange*nvariable+i]))
                            self.xave[self.nvariable*irange+i]=self.xgreat[i]
                            self.xrstd[self.nvariable*irange+i]=self.smin
                            self.xlstd[self.nvariable*irange+i]=self.smin

                        else:
                            if abs(self.xave[self.nvariable*irange+i]-self.xave[self.nvariable*self.mrange+i])<0.1:  # regeneration type 2
                                self.xave[self.nvariable*irange+i]=self.xave[self.nvariable*self.mrange+i]
                            else:
                                self.xave[self.nvariable*irange+i]=self.tave[i]
                            self.xrstd[self.nvariable*irange+i]=self.xstd[i]
                            self.xlstd[self.nvariable*irange+i]=self.xstd[i]


                    for ipop in range(self.npop):
                        if self.prange[ipop]==irange:
                            if random.randint(0,9)<5:
                                if ipop != self.elite_range[irange]:
                                    for ivar in range(self.nvariable):
                                        for i in range(self.totalbit):
                                            self.GAstring[self.totalbit*ipop+ivar*self.nbit+i]=random.randint(0,1)
                                self.zflg[ipop]=2       
                    self.age[irange]=-10
        
        maxfit=1.0e33
        # if self.greatAge>15 or self.g > 1000:
        #     llmax=int(self.npop*0.05)
        for ipop in range(self.npop):
            if self.zflg[ipop]==2:
                if not self.psearch_flg:
                    self.ARange(ipop)
                else:
                    if random.randint(0,self.npop) < self.llmax:
                        self.locArange(ipop)
                    else:
                        self.ARange(ipop)
                self.functioncall(ipop)
            self.zflg[ipop]=0

            if maxfit > self.fitness[ipop]:
                maxfit=self.fitness[ipop]
                self.imin=ipop

                if self.fgreat>maxfit:
                    self.fgreat=maxfit
                    self.greatAge=0
                    if self.psearch_flg:
                        self.pgreatAge=0
                        self.newcnt+=1
                    for ivar in range(self.nvariable):
                        self.xgreat[ivar]=self.x[self.nvariable*ipop+ivar]
        
        



        # print("key = "+str(key))
        self.greatAge += 1
        self.g += 1
        if not self.psearch_flg:
            self.nonhoxage += 1
        self.fgreats.append([self.fgreat])
        self.aaafits.append([self.aaafit])
    

    def UpdateRangePhase(self):
        if not self.hoxflg:    
            self.UpdateRange()
        elif self.hoxflg:
            self.g += 1
            self.hoxage += 1


    def writeDebug4(self,state):
        seed_dir=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"seed"+str(self.seed)))
        result_dir=os.path.join(seed_dir,"result")
        debug_dir=os.path.join(result_dir,"debug")

        rangepop_path=os.path.join(debug_dir,"rangepop.csv")
        with open(rangepop_path,"a") as f:
            writer = csv.writer(f,lineterminator = '\n')
            data = copy.deepcopy(self.rangepop)
            data.append(state)
            writer.writerow(data)
    

    def writeHoxEnd(self):
        seed_dir=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"seed"+str(self.seed)))
        result_dir=os.path.join(seed_dir,"result")
        
        hoxresult=os.path.normpath(os.path.join(result_dir,"Hoxgene.csv"))
        with open(hoxresult,"a") as f:
            writer=csv.writer(f, lineterminator='\n')
            writer.writerow(["generation="+str(self.g)])
            for ipop in range(self.npop):
                data=[self.HGfitness[ipop]]
                data.extend(self.Hoxgene[ipop*self.nvariable:ipop*self.nvariable+self.nvariable])
                writer.writerow(data)
            writer.writerow([""])
        
    
        EndRealHox=os.path.join(result_dir,"EndRealHox.csv")
        with open(EndRealHox,"a") as f:
            writer=csv.writer(f,lineterminator = '\n')
            writer.writerow(["generation="+str(self.g)])
            strRealHox=self.sortRealHox()
            writer.writerow(strRealHox)
            writer.writerow([""])
        
        hox_dir=os.path.join(result_dir,"Hox")
        allzeros=os.path.join(hox_dir,"allzeros.csv")
        with open(allzeros,"a") as f:
            writer=csv.writer(f, lineterminator='\n')
            writer.writerow(["generation="+str(self.g)])
            for ipop in range(self.npop):
                aflg="all zero"
                for ivar in range(self.nvariable):
                    if self.Hoxgene[ipop*self.nvariable+ivar] == 1:
                        aflg="exist 1"
                    else:
                        pass
                writer.writerow([aflg])
            writer.writerow([""])



    def HoxSwitch(self,generation):
        if self.g > self.hoxbegin and self.nonhoxage > self.nonhoxterm and self.greatAge > 10 and not self.psearch_flg and not self.hoxflg:
            print(">>> Hox term begin")
            self.hoxflg=True
            self.hoxage=0
            if self.hoxcnt==0:
                self.init_Hoxgene()
                # self.evalHoxAve(self.npop,"initHoxgene")
            self.mkpair2("est start")
            self.hoxcnt+=1
        
        elif self.hoxflg and self.hoxage >= self.hoxterm:
            self.hoxflg=False
            self.psearch_st=True
    

    def initFILEs(self):
        seed_dir=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"seed"+str(self.seed)))
        if os.path.exists(seed_dir):
            pass
        else:
            os.mkdir(seed_dir)
        
        result_dir=os.path.join(seed_dir,"result")
        if os.path.exists(result_dir):
            pass
        else:
            os.mkdir(result_dir)

        aaaHGFit=os.path.join(result_dir,"aaaHGFit.csv")
        if os.path.exists(aaaHGFit):
            os.remove(aaaHGFit)
        
        fitnesspath = os.path.join(result_dir,"Fitness.csv")
        with open(fitnesspath,"w") as f:
            writer=csv.writer(f,lineterminator='\n')
            writer.writerow(["g","hoxflg","","aaafit","","fitness"])
        
        confirmAge=os.path.join(result_dir,"confirmAge.csv")
        with open(confirmAge,'w') as f:
            writer=csv.writer(f,lineterminator='\n')
            writer.writerow(["greatAge","pgAge","psearch_age"])
        
        confirmAge2=os.path.join(result_dir,"confirmAge2.csv")
        with open(confirmAge2,'w') as f:
            writer=csv.writer(f,lineterminator='\n')
            writer.writerow(["pgAge","psearch_age"])
        
        HGFitChange=os.path.normpath(os.path.join(result_dir,"FitChange.csv"))
        if os.path.exists(HGFitChange):
            os.remove(HGFitChange)

        fg_imin=os.path.normpath(os.path.join(result_dir,"fg_imin.csv"))
        if os.path.exists(fg_imin):
            os.remove(fg_imin)
        
        imin_t=os.path.normpath(os.path.join(result_dir,"imin_t.csv"))
        if os.path.exists(imin_t):
            os.remove(imin_t)
        
        hoxresult=os.path.normpath(os.path.join(result_dir,"Hoxgene.csv"))
        if os.path.exists(hoxresult):
            os.remove(hoxresult)
        
        ALLRealHox = os.path.join(result_dir,"ALLRealHox.csv")
        if os.path.exists(ALLRealHox):
            os.remove(ALLRealHox)
        
        EndRealHox = os.path.join(result_dir,"EndRealHox.csv")
        if os.path.exists(EndRealHox):
            os.remove(EndRealHox)
        
        hox_dir=os.path.join(result_dir,"Hox")
        if os.path.exists(hox_dir):
            pass
        else:
            os.mkdir(hox_dir)
        
        datapath=os.path.join(result_dir,"estimate_time.csv")
        if os.path.exists(datapath):
            os.remove(datapath)
        
        datapath=os.path.join(result_dir,"revalFromRSLT.csv")
        if os.path.exists(datapath):
            os.remove(datapath)
        
        judgedata=os.path.join(result_dir,"convjudge.csv")
        with open(judgedata,"w") as f:
            writer=csv.writer(f, lineterminator='\n')
            writer.writerow([i for i in range(self.nvariable)])
        
        hoxeval=os.path.join(hox_dir,"HGfitness.csv")
        if os.path.exists(hoxeval):
            os.remove(hoxeval)
        
        hoxeval_sorted=os.path.join(hox_dir,"HGfitness_sorted.csv")
        if os.path.exists(hoxeval_sorted):
            os.remove(hoxeval_sorted)
        
        allzeros=os.path.join(hox_dir,"allzeros.csv")
        if os.path.exists(allzeros):
            os.remove(allzeros)
        
        debug_dir=os.path.join(result_dir,"debug")
        if os.path.exists(debug_dir):
            pass
        else:
            os.mkdir(debug_dir)
        
        rangepop_path=os.path.join(debug_dir,"rangepop.csv")
        if os.path.exists(rangepop_path):
            os.remove(rangepop_path)
        
        sparcexave=os.path.join(debug_dir,"sparcexave.csv")
        if os.path.exists(sparcexave):
            os.remove(sparcexave)
        
        pair_dir = os.path.join(debug_dir,"PAIR")
        if os.path.exists(pair_dir):
            pass
        else:
            os.mkdir(pair_dir)

        for ivar in range(self.nvariable):
            datapath=os.path.join(pair_dir,"var"+str(ivar)+".csv")
            if os.path.exists(datapath):
                os.remove(datapath)
        
        for ivar in range(self.nvariable):
            datapath=os.path.join(pair_dir,"var"+str(ivar)+".csv")
            if os.path.exists(datapath):
                os.remove(datapath)
        
        hf_pf_path = os.path.join(result_dir,"hflg_psflg.csv")
        with open(hf_pf_path,"w") as f:
            writer = csv.writer(f, lineterminator = '\n')
            writer.writerow(["hoxflg","","psflg","","match"])
        

        released_x = os.path.join(hox_dir,"released_x")
        if os.path.exists(released_x):
            shutil.rmtree(released_x)
            os.mkdir(released_x)
        else:
            os.mkdir(released_x)
        
        fgreat_psearch=os.path.join(hox_dir,"fgreat_psearch.csv")
        if os.path.exists(fgreat_psearch):
            os.remove(fgreat_psearch)
        
        fgreat_psearch_analy=os.path.join(hox_dir,"fgreat_psearch_analy.csv")
        if os.path.exists(fgreat_psearch_analy):
            os.remove(fgreat_psearch_analy)
        
        whole_dir = os.path.join(result_dir,"WHOLE")
        if os.path.exists(whole_dir):
            pass
        else:
            os.mkdir(whole_dir)
        
        wholex_dir=os.path.join(whole_dir,"x")
        if os.path.exists(wholex_dir):
            pass
        else:
            os.mkdir(wholex_dir)
        
        wholex_analy=os.path.join(whole_dir,"analysis")
        if os.path.exists(wholex_analy):
            pass
        else:
            os.mkdir(wholex_analy)
        
        wholes_analy=os.path.join(wholex_analy,"SearchRange")
        if os.path.exists(wholes_analy):
            pass
        else:
            os.mkdir(wholes_analy)
        
        wholedata=os.path.join(whole_dir,"wholedata.csv")
        with open(wholedata,"w") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(["fgreat","hox","psearch","pair"])
        
        for ivar in range(self.nvariable):
            wholedata_xi=os.path.join(wholex_dir,"x"+str(ivar)+"w.csv")
            with open(wholedata_xi,"w") as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(["fgreat","hox","psearch","pair","","ave_xi","ind"])
            
            wholedata_xi_analy=os.path.join(wholex_analy,"x"+str(ivar)+"w_a.csv")
            with open(wholedata_xi_analy,"w") as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(["generation","fgreat","hox","psearch","pair","","p_flg","ave_xi","std_xi","ind"])
            
            xiave_path=os.path.join(wholes_analy,"x"+str(ivar)+"ave.csv")
            with open(xiave_path,"w") as f:
                writer = csv.writer(f,lineterminator = '\n')
                writer.writerow(["range0","range1","range2","range3"])
            
            xirstd_path=os.path.join(wholes_analy,"x"+str(ivar)+"rstd.csv")
            with open(xirstd_path,"w") as f:
                writer = csv.writer(f, lineterminator = '\n')
                writer.writerow(["range0","range1","range2","range3"])
            
            xilstd_path=os.path.join(wholes_analy,"x"+str(ivar)+"lstd.csv")
            with open(xilstd_path,"w") as f:
                writer = csv.writer(f, lineterminator = '\n')
                writer.writerow(["range0","range1","range2","range3"])
        
        PAIR_dir = os.path.join(result_dir,"PAIR")
        if os.path.exists(PAIR_dir):
            pass
        else:
            os.mkdir(PAIR_dir)
        
        pairdata = os.path.join(PAIR_dir,"pair.csv")
        if os.path.exists(pairdata):
            os.remove(pairdata)

        pairdata2 = os.path.join(PAIR_dir,"PAIRdataE.csv")
        if os.path.exists(pairdata2):
            os.remove(pairdata2)
        
        pairdata3=os.path.join(PAIR_dir,"PAIRdata.csv")
        if os.path.exists(pairdata3):
            os.remove(pairdata3)
        
        pairdataN = os.path.join(PAIR_dir,"PAIRdataNaturalSelect.csv")
        if os.path.exists(pairdataN):
            os.remove(pairdataN)

        
        
    def writeResult(self):
        seed_dir=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"seed"+str(self.seed)))
        result_dir=os.path.join(seed_dir,"result")
        
        # write the greatest f value at each generation to csv file
        fgreat_path = os.path.normpath(os.path.join(result_dir,"fgreat.csv"))
        with open(fgreat_path,"w") as f:
            writer=csv.writer(f, lineterminator='\n')
            writer.writerows(self.fgreats)
        
        # write the average of f value at each generation to csv file
        aaafit_path = os.path.normpath(os.path.join(result_dir,"aaafit.csv"))
        with open(aaafit_path,"w") as f:
            writer=csv.writer(f, lineterminator='\n')
            writer.writerows(self.aaafits)
        
        # write Hoxgenes
        # hoxgene_path = os.path.normpath(os.path.join(result_dir,"hoxgene_final.csv"))
        # with open(hoxgene_path,"w") as f:
        #     writer=csv.writer(f)
        #     for ipop in range(self.npop):
        #         data=[self.HGfitness[ipop]]
        #         data.extend(self.Hoxgene[self.nvariable*ipop:self.nvariable*ipop+self.nvariable])
        #         writer.writerow(data)
        

    def toStrRealHox(self,real):
        strRealHox = []
        for pi in real:
            strpi = "["
            for v in pi:
                strpi += (str(v)+",")
            strpi = strpi.rstrip(",")
            strpi += "]"

            strRealHox.append(strpi)

        return strRealHox


    def sortRealHox(self):
        sortind=np.argsort(self.HGfitness[0:self.npop])
        RealHox_sorted=[[] for i in range(self.npop+self.hgncross)]

        j=0
        for i in sortind:
            RealHox_sorted[j] = self.realHox[j]
            j+=1
        
        strRealHox = self.toStrRealHox(RealHox_sorted)

        return strRealHox

    

    def writeEachIter(self):
        seed_dir=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"seed"+str(self.seed)))
        result_dir=os.path.join(seed_dir,"result")
        debug_dir=os.path.join(result_dir,"debug")
        pair_dir = os.path.join(debug_dir,"PAIR")
        whole_dir = os.path.join(result_dir,"WHOLE")
        wholex_dir=os.path.join(whole_dir,"x")
        wholex_analy=os.path.join(whole_dir,"analysis")
        wholes_analy=os.path.join(wholex_analy,"SearchRange")

        for ivar in range(self.nvariable):
            datapath=os.path.join(pair_dir,"var"+str(ivar)+".csv")

            with open(datapath,"a") as f:
                writer=csv.writer(f, lineterminator='\n')
                data = 0
                if self.Pi == -1:
                    data=["nonfixed"]
                else:
                    data = ["dofixed"]
                    data.extend(copy.deepcopy(self.Pi))
                    data.append(self.pgreatAge)
                    data.append(self.psearch_age)
                writer.writerow(data)
                x_data=[]
                for ipop in range(self.npop):
                    x_data.append(self.x[ipop*self.nvariable+ivar])
                writer.writerow(x_data)
        

        ALLRealHox = os.path.join(result_dir,"ALLRealHox.csv")
        strRealHox = self.sortRealHox()
        with open(ALLRealHox,"a") as f:
            writer = csv.writer(f, lineterminator = '\n')
            writer.writerow(strRealHox)
        
        fitnesspath = os.path.join(result_dir,"Fitness.csv")
        with open(fitnesspath,"a") as f:
            writer=csv.writer(f,lineterminator='\n')
            data=[self.g,self.hoxflg,"",self.aaafit,""]
            data.extend(self.fitness[0:self.npop])
            writer.writerow(data)
        

        # wholedata=os.path.join(whole_dir,"wholedata.csv")
        # with open(wholedata,"a") as f:
        #     writer=csv.writer(f, lineterminator="\n")
        #     data=[self.fgreat]
        #     h_flg="-"
        #     if self.hoxflg:
        #         h_flg="estimation"
        #     p_flg="-"
        #     strp="-"
        #     if self.psearch_flg:
        #         p_flg="searching"
        #         strp="["
        #         for i in self.Pi:
        #             strp+=(str(i)+",")
        #         strp=strp.rstrip(",")
        #         strp+="]"
        #     data.extend([h_flg,p_flg,strp])
        #     writer.writerow(data)
        

        for ivar in range(self.nvariable):
            wholedata_xi=os.path.join(wholex_dir,"x"+str(ivar)+"w.csv")
            with open(wholedata_xi,"a") as f:
                writer=csv.writer(f,lineterminator='\n')
                data=[self.fgreat]
                h_flg="-"
                if self.hoxflg:
                    h_flg="estimation"
                p_flg="-"
                strp="-"
                if self.psearch_flg:
                    p_flg="searching"
                    strp="["
                    for i in self.Pi:
                        strp+=(str(i)+",")
                    strp=strp.rstrip(",")
                    strp+="]"
                data.extend([h_flg,p_flg,strp,""])

                x_data=[]
                for ipop in range(self.npop):
                    x_data.append(self.x[ipop*self.nvariable+ivar])
                aaaxi=(sum(x_data)/self.npop)

                data.append(aaaxi)
                data.extend(x_data)

                writer.writerow(data)



            wholedata_xi_analy=os.path.join(wholex_analy,"x"+str(ivar)+"w_a.csv")
            with open(wholedata_xi_analy,"a") as f:
                writer=csv.writer(f,lineterminator='\n')
                data=["g="+str(self.g),self.fgreat]
                h_flg=False
                if self.hoxflg:
                    h_flg=True
                p_flg=False
                pp_flg = False
                strp="-"
                if self.psearch_flg:
                    p_flg=True
                    strp="["
                    for i in self.Pi:
                        strp+=(str(i)+",")
                    strp=strp.rstrip(",")
                    strp+="]"
                    if ivar in self.Pi:
                        pp_flg = True
                data.extend([h_flg,p_flg,strp,"",pp_flg])

                x_data=[]
                for ipop in range(self.npop):
                    x_data.append(self.x[ipop*self.nvariable+ivar])
                aaaxi=(sum(x_data)/self.npop)

                powsum=0
                for ipop in range(self.npop):
                    powsum+=((self.x[ipop*self.nvariable+ivar]-aaaxi)**2)
                std=math.sqrt(powsum/self.npop)


                data.append(aaaxi)
                data.append(std)
                data.extend(x_data)

                writer.writerow(data)
            
            xave_tmp = []
            xrstd_tmp = []
            xlstd_tmp = []
            for irange in range(self.nrange):
                xave_tmp.append(self.xave[irange*self.nvariable+ivar])
                xrstd_tmp.append(self.xrstd[irange*self.nvariable+ivar])
                xlstd_tmp.append(self.xlstd[irange*self.nvariable+ivar])
            
            xiave_path=os.path.join(wholes_analy,"x"+str(ivar)+"ave.csv")
            with open(xiave_path, "a") as f:
                writer = csv.writer(f, lineterminator = '\n')
                writer.writerow(xave_tmp)

            xirstd_path=os.path.join(wholes_analy,"x"+str(ivar)+"rstd.csv")
            with open(xirstd_path,"a") as f:
                writer = csv.writer(f, lineterminator = '\n')
                writer.writerow(xrstd_tmp)
            
            xilstd_path=os.path.join(wholes_analy,"x"+str(ivar)+"lstd.csv")
            with open(xilstd_path,"a") as f:
                writer = csv.writer(f, lineterminator = '\n')
                writer.writerow(xlstd_tmp)
            
        
        confirmAge=os.path.join(result_dir,"confirmAge.csv")
        with open(confirmAge,'a') as f:
            writer=csv.writer(f,lineterminator='\n')
            writer.writerow([self.greatAge,self.pgreatAge,self.psearch_age])
        

        if self.g > self.hoxbegin+1:
            aaaHGFit=os.path.join(result_dir,"aaaHGFit.csv")
            with open(aaaHGFit,"a") as f:
                writer=csv.writer(f, lineterminator='\n')
                writer.writerow([sum(self.HGfitness[0:self.npop])/self.npop])

        
        
        # hf_pf_path = os.path.join(result_dir,"hflg_psflg.csv")
        # with open(hf_pf_path,"a") as f:
        #     writer = csv.writer(f, lineterminator = '\n')
        #     data = [self.hoxflg,"",self.psearch_flg,""]

        #     if self.hoxflg == self.psearch_flg:
        #         data.append("match "+str(self.hoxflg))
        #     else:
        #         data.append("mismatch")
            
        #     writer.writerow(data)


    ### function to search considering the dependenceis between variables ### 
    # function for making variable pairs
    def mkpair(self):
        # initialize the set P
        if self.P != []:
            while self.P != []:
                del self.P[0]
        
        while self.PAIR != []:
            del self.PAIR[0]
        
        
        while self.deltaF != []:
            del self.deltaF[0]



        
        hoxgenes = copy.deepcopy(self.Hoxgene[0:self.npop*self.nvariable])
        fitnesss = self.HGfitness[0:self.npop]
        hoxinds = [i for i in range(self.npop)]
        
        for i in range(self.npair):
            eflg = True
            while eflg:
                igene = fitnesss.index(max(fitnesss))
                ind = hoxinds[igene]
                
                # print("igene="+str(igene))
                ihoxgene = hoxgenes[igene*self.nvariable:igene*self.nvariable+self.nvariable]
                
                Pi = []
                for ivar in range(self.nvariable):
                    ig = ihoxgene[ivar]
                    if ig == 1:
                        Pi.append(ivar)
                    else:
                        pass
                
                if Pi in self.P:
                    eflg = True
                    # update the lists for detemining variable pairs
                    del fitnesss[igene]
                    del hoxgenes[igene*self.nvariable:igene*self.nvariable+self.nvariable]
                
                else:
                    self.P.append(Pi)
                    self.PAIR.append(Pi)
                    self.selectedHox.append(ind)
                    eflg = False
                    # update the lists for detemining variable pairs
                    del fitnesss[igene]
                    del hoxgenes[igene*self.nvariable:igene*self.nvariable+self.nvariable]
                    del hoxinds[igene]
    

    def writePAIR(self):
        seed_dir=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"seed"+str(self.seed)))
        result_dir=os.path.join(seed_dir,"result")
        PAIR_dir = os.path.join(result_dir,"PAIR")

        arr_tmp = []
        for pi in self.P:
            strpi = "["
            for pii in pi:
                strpi += (str(pii)+",")
            strpi = strpi.rstrip(",")
            strpi += "]"

            arr_tmp.append(strpi)
        
        pairdata = os.path.join(PAIR_dir,"pair.csv")
        with open(pairdata,"a") as f:
            writer = csv.writer(f, lineterminator = '\n')
            writer.writerow(pairdata)
            writer.writerow([""])
    

    def mkpair2(self,state):
        # initialize the set P
        if self.P != []:
            while self.P != []:
                del self.P[0]
        
        hoxgenes = copy.deepcopy(self.Hoxgene[0:self.npop*self.nvariable])
        fitnesss = self.HGfitness[0:self.npop]
        pairfit=[]
        for i in range(int(self.nvariable*2)):
            eflg = True
            while eflg:
                igene = fitnesss.index(max(fitnesss))
                # print("igene="+str(igene))
                ihoxgene = hoxgenes[igene*self.nvariable:igene*self.nvariable+self.nvariable]
                
                Pi = []
                for ivar in range(self.nvariable):
                    ig = ihoxgene[ivar]
                    if ig == 1:
                        Pi.append(ivar)
                    else:
                        pass
                
                if Pi in self.P:
                    eflg = True
                    # update the lists for detemining variable pairs
                    del fitnesss[igene]
                    del hoxgenes[igene*self.nvariable:igene*self.nvariable+self.nvariable]
                
                else:
                    self.P.append(Pi)
                    pairfit.append(fitnesss[igene])
                    eflg = False
                    # update the lists for detemining variable pairs
                    del fitnesss[igene]
                    del hoxgenes[igene*self.nvariable:igene*self.nvariable+self.nvariable]
        
        self.writePAIR2(pairfit,state)
        while self.P != []:
            del self.P[0]



    def writePAIR2(self,pairfit,state):
        seed_dir=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"seed"+str(self.seed)))
        result_dir=os.path.join(seed_dir,"result")
        PAIR_dir = os.path.join(result_dir,"PAIR")

        arr_tmp = []
        for pi in self.P:
            strpi = "["
            for pii in pi:
                strpi += (str(pii)+",")
            strpi = strpi.rstrip(",")
            strpi += "]"

            arr_tmp.append(strpi)
        
        pairdata2 = os.path.join(PAIR_dir,"PAIRdataE.csv")
        with open(pairdata2,"a") as f:
            writer = csv.writer(f, lineterminator = '\n')
            data=[self.g,self.npair,state]
            data.extend(arr_tmp)
            fitdata=["","",""]
            fitdata.extend(pairfit)

            writer.writerow(data)
            writer.writerow(fitdata)
        
        if state == "start psearch" or state == "end of psearch":
            pairdata3=os.path.join(PAIR_dir,"PAIRdata.csv")
            with open(pairdata3,"a") as f:
                writer=csv.writer(f,lineterminator='\n')
                data=[self.g,self.npair,state]
                data.extend(arr_tmp)
                fitdata=["","",""]
                fitdata.extend(pairfit)

                writer.writerow(data)
                writer.writerow(fitdata)
    
    
    def getSparseXave(self,mxave):
        whole_data = list()
        s_point = list()
        sum_n = 0

        for ivar in range(len(self.Pi)):
            whole_data.append(list(np.arange(self.xmin[ivar],self.xmax[ivar],0.01))) # 解空間内の全個体を擬似的(粗く)に取得
            sum_n += len(np.arange(self.xmin[ivar],self.xmax[ivar],0.01))
        
        nsample = sum_n * 0.8
        sample = self.sampling(whole_data, len(self.Pi), nsample)                  # 初期点候補集団の解を取得

        sample = np.matrix(sample)

        # select the point closest to the centroid as a 1st initial point
        mxave = np.matrix(mxave)
        p1 = np.array(mxave)
        print("p1 >>>   " + str(p1))


        s_point.append(list(p1[0]))



        # calculate distance between p1 and others
        dist = sample - np.array(p1)
        dist_p1 = np.linalg.norm(dist, ord = 2, axis = 1)


        # get the point fathest from p1
        p2_i = np.argmax(dist_p1)
        p2 = np.array(sample[p2_i])

        # print("p2 >>>   " + str(p2))
        s_point.append(list(p2[0]))

        # delete the point selected
        sample = np.delete(sample, p2_i, 0)


        p_n = len(self.Pi)*10 - len(s_point)

        for iter in range(p_n):
            D = list()
            for i in range(len(s_point)):
                d = list(np.linalg.norm(sample - np.array(s_point[i]), ord = 2, axis = 1))
                D.append(d)
            
            D = np.matrix(D)
            D_t = D.T

            D_min = D_t.min(axis = 1)

            p_i = np.argmax(D_min)
            p = np.array(sample[p_i])

            s_point.append(list(p[0]))
    
        print("s_point >>> " + str(s_point))
        return s_point

    

    def relocateSearchRange(self):
        mxave = [self.xave[self.mrange*self.nvariable+ivar] for ivar in self.Pi]
        sparce_xave = self.getSparseXave(mxave)
        self.writeDebug3(sparce_xave)
        
        iivar = 0
        self.locrange = random.randint(1,3)      # the index of the search range searching around the best solution
        for ivar in self.Pi:
            iirange = 1
            for irange in range(self.nrange):
                # relocate each search range
                if irange == self.mrange:
                    # set the width of the search range with the individual with the best solution
                    self.xrstd[irange*self.nvariable+ivar] = self.xstd[ivar]/5
                    if self.xmax[ivar] < self.xave[irange*self.nvariable+ivar]+self.ppp*self.xrstd[irange*self.nvariable+ivar]:
                        self.xrstd[irange*self.nvariable+ivar]=(self.xmax[ivar]-self.xave[irange*self.nvariable+ivar])/self.ppp
                    self.xlstd[irange*self.nvariable+ivar] = self.xstd[ivar]/5
                    if self.xmin[ivar] > self.xave[irange*self.nvariable+ivar]-self.ppp*self.xlstd[irange*self.nvariable+ivar]:
                        self.xlstd[irange*self.nvariable+ivar]=(-self.xmin[ivar]+self.xave[irange*self.nvariable+ivar])/self.ppp
                else:
                    # case: the search range searches around the best solution 
                    if iirange == self.locrange:
                        # set the center of the search range
                        self.xave[irange*self.nvariable+ivar] = self.xgreat[ivar]

                        # set the width of the search range with the individual with the best solution
                        self.xrstd[irange*self.nvariable+ivar] = self.xstd[ivar]/5
                        if self.xmax[ivar] < self.xave[irange*self.nvariable+ivar]+self.ppp*self.xrstd[irange*self.nvariable+ivar]:
                            self.xrstd[irange*self.nvariable+ivar]=(self.xmax[ivar]-self.xave[irange*self.nvariable+ivar])/self.ppp
                        self.xlstd[irange*self.nvariable+ivar] = self.xstd[ivar]/5
                        if self.xmin[ivar] > self.xave[irange*self.nvariable+ivar]-self.ppp*self.xlstd[irange*self.nvariable+ivar]:
                            self.xlstd[irange*self.nvariable+ivar]=(-self.xmin[ivar]+self.xave[irange*self.nvariable+ivar])/self.ppp
                        iirange += 1
                    
                    # case : the search range searches globally
                    else:
                        # set the center of each search range
                        self.xave[irange*self.nvariable+ivar] = sparce_xave[iirange][iivar]

                        # set the width of the search ranges
                        self.xrstd[irange*self.nvariable+ivar] = self.xstd_init[ivar]*2
                        xu=self.xave[irange*self.nvariable+ivar]+self.ppp*self.xrstd[irange*self.nvariable+ivar]
                        if self.xmax[ivar] < xu:
                            self.xrstd[irange*self.nvariable+ivar]=(self.xmax[ivar]-self.xave[irange*self.nvariable+ivar])/self.ppp
                        else:
                            if xu < self.xgreat[ivar] and self.xgreat[ivar] < self.xave[irange*self.nvariable+ivar]:
                                self.xrstd[irange*self.nvariable+ivar]=(self.xgreat[ivar]-self.xave[irange*self.nvariable+ivar])/self.ppp
                                if self.xrstd[irange*self.nvariable+ivar] > (self.xmax[ivar]-self.xave[irange*self.nvariable+ivar])/self.ppp:
                                    self.xrstd[irange*self.nvariable+ivar]=(self.xmax[ivar]-self.xave[irange*self.nvariable+ivar])/self.ppp
                        
                        self.xlstd[irange*self.nvariable+ivar] = self.xstd_init[ivar]*2
                        xl=self.xave[irange*self.nvariable+ivar]-self.ppp*self.xlstd[irange*self.nvariable+ivar]
                        if self.xmin[ivar] > self.xave[irange*self.nvariable+ivar]-self.ppp*self.xlstd[irange*self.nvariable+ivar]:
                            self.xlstd[irange*self.nvariable+ivar]=(-self.xmin[ivar]+self.xave[irange*self.nvariable+ivar])/self.ppp
                        else:
                            if xl > self.xgreat[ivar] and self.xgreat[ivar] < self.xave[irange*self.nvariable+ivar]:
                                self.xlstd[irange*self.nvariable+ivar]=(self.xave[irange*self.nvariable+ivar]-self.xgreat[ivar])/self.ppp
                                if self.xlstd[irange*self.nvariable+ivar] > (self.xave[irange*self.nvariable+ivar] - self.xmin[ivar])/self.ppp:
                                    self.xlstd[irange*self.nvariable+ivar]=(self.xave[irange*self.nvariable+ivar] - self.xmin[ivar])/self.ppp

                        iirange += 1
            iivar += 1


    def writeDebug3(self,s_xave):
        seed_dir=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"seed"+str(self.seed)))
        result_dir=os.path.join(seed_dir,"result")
        debug_dir=os.path.join(result_dir,"debug")
        sparcexave=os.path.join(debug_dir,"sparcexave.csv")

        with open(sparcexave,"a") as f:
            writer = csv.writer(f, lineterminator = '\n')
            items = ["pair"]
            items.extend(self.Pi)
            writer.writerow(items)
            writer.writerow(["before relocate"])
            datas = []
            for irange in range(self.nrange):
                data = []
                for ivar in self.Pi:
                    data.append(self.xave[irange*self.nvariable+ivar])
                if irange == self.mrange:
                    data.append("mrange")
                
                datas.append(data)
            writer.writerows(datas)

            writer.writerow(["after relocate"])
            datas = []
            for i in range(len(s_xave)):
                data = []
                for ivar in range(len(self.Pi)):
                    data.append(s_xave[i][ivar])
                if i == 0:
                    data.append("mrange?")
                
                datas.append(data)
            writer.writerows(datas)
            writer.writerow("")

                        
    
    def CNV_judge(self,PXSTD,CENTER):
        ngenetypes=len(self.allgenetypes)
        middle=int(ngenetypes/2)-1
        for ibit in range(self.nbit):
            self.ws[ibit]=self.allgenetypes[middle][ibit]
        iphenotype=self.idecoding(self.nbit,self.nbit-1)

        for ivar in range(self.nvariable):
            self.inside[ivar]=PXSTD[ivar]*math.sqrt(-2*math.log(self.LB+(self.UB-self.LB)*iphenotype/(pow(2,self.nbit-1)-1)))

        for ivar in range(self.nvariable):
            self.threshold[ivar]=(abs(self.inside[ivar])/(self.xmax[ivar]-self.xmin[ivar]))*100
        
        # judge
        for ivar in range(self.nvariable):
            self.judgement[ivar] = False
        for ivar in range(self.nvariable):
            errors=[]
            for ipop in range(self.npop):
                x_i = self.x[ipop*self.nvariable+ivar]
                ce_i = CENTER[ivar]
                error = (abs(x_i-ce_i)/(self.xmax[ivar]-self.xmin[ivar]))*100       # calculate error rate
                errors.append(error)
            worst=max(errors)
            self.error_worst[ivar]=worst

            if self.threshold[ivar] > self.error_worst[ivar]:
                self.judgement[ivar] = True
            else:
                pass

    

    def psearch(self):
        # the processing before the search begins & initialize the parameter used in the search 
        st=False
        if self.psearch_st:
            print(">>> start the search considering dependencies between variables")
            self.psearch_st=False
            st=True
            self.psearch_flg=True
            self.pgreatAge = self.updateTHRESH+1
            self.Pi=-1

            self.fgreat_init=self.fgreat

            # save the value of each parameter at the begining of search
            for ivar in range(self.nvariable):
                self.xstd_init[ivar]=self.xstd[ivar]
                self.tave_init[ivar]=self.tave[ivar]
                self.xgreat_init[ivar]=self.xgreat[ivar]

            for irange in range(self.nrange):
                for ivar in range(self.nvariable):
                    self.xave_init[self.nvariable*irange+ivar]=self.xave[self.nvariable*irange+ivar]
                    self.xrstd_init[self.nvariable*irange+ivar]=self.xrstd[self.nvariable*irange+ivar]
                    self.xlstd_init[self.nvariable*irange+ivar]=self.xlstd[self.nvariable*irange+ivar]
                    self.tstd_init[self.nvariable*irange+ivar]=self.tstd[self.nvariable*irange+ivar]

            for ipop in range(self.npop):
                self.func_init[ipop]=self.func[ipop]
                self.fitness_init[ipop]=self.fitness[ipop]
                for ivar in range(self.nvariable):
                    self.x_init[self.nvariable*ipop+ivar]=self.x[self.nvariable*ipop+ivar]
                for ibit in range(self.totalbit):
                    self.GAstring_init[self.totalbit*ipop+ibit]=self.GAstring[self.totalbit*ipop+ibit]    


        # determine the variable pair we are referencing now and the search begins
        if self.psearch_flg and len(self.P) != 0 and self.pgreatAge>self.updateTHRESH and self.psearch_age!=self.pgreatAge and not st:
            self.psearch_age=-1
            self.pgreatAge=-1
        if self.psearch_flg and len(self.P) != 0 and (self.pgreatAge>self.updateTHRESH or self.psearch_age>self.termlimit):
            if self.Pi != -1:
                if self.fgreat_init > self.fgreat:
                    diff=0
                    for ivar in range(self.nvariable):
                        diff+=((self.xgreat_init[ivar]-self.xgreat[ivar])**2)
                        self.xgreat_init[ivar]=self.xgreat[ivar]
                    diff=math.sqrt(diff)
                    self.deltaFi.append((self.fgreat_init-self.fgreat)/(diff+1))
                    self.fgreat_init=self.fgreat
                if self.newcnt==0:
                    self.deltaF.append(0)
                else:
                    self.deltaF.append(np.mean(np.array(self.deltaFi)))
                self.pcnt+=1
                for ivar in range(self.nvariable):
                    self.pxstd_sum[ivar]+=self.xstd[ivar]
                    self.pxstd[ivar]=copy.deepcopy(self.pxstd_sum[ivar])/self.pcnt
                self.CNV_judge(self.pxstd,self.xgreat)
                self.writeChange(-1)
            
            self.psearch_age=0
            self.pgreatAge=0
            self.newcnt=0

            # update x_init value
            if self.Pi!=-1:
                # update x_init value
                topfit=np.argsort(self.fitness[0:self.npop])
                llmax=int(self.npop*0.2)
                for ipop in range(self.npop):
                    for ivar in range(self.nvariable):
                        if ipop == self.imin:
                            self.x_init[self.nvariable*ipop+ivar]=self.xgreat[ivar]
                        elif ipop in topfit[0:llmax]:
                            self.x_init[self.nvariable*ipop+ivar]=self.x[self.nvariable*ipop+ivar]
            

            # select the variable pair
            self.Pi=self.P[0]
            del self.P[0]
            print(">>> search for pair="+str(self.Pi))

            self.fgreat_init=self.fgreat
            while self.deltaFi != []:
                del self.deltaFi[0]


            # relocate each search range
            # self.relocateSearchRange()

            # initialize parameter to search 
            for irange in range(self.nrange):
                self.age[irange]=0
            
            # allow the values of variables selected as the variable pair to change, and fix the values of others
            for ipop in range(self.npop):
                if ipop == self.imin:
                    pass
                else:
                    for ivar in range(self.nvariable):
                        if ivar in self.Pi:
                            self.x[self.nvariable*ipop+ivar]=self.x_init[self.nvariable*ipop+ivar]
                        else:
                            self.x[self.nvariable*ipop+ivar]=self.xgreat[ivar]
                    # if self.prange[ipop] == self.locrange or self.prange[ipop] == self.mrange:
                    #     self.ARange(ipop)
                    # else:
                    #     km=random.randint(0,self.rangepop[self.prange[ipop]])
                    #     if km <= self.llmax:
                    #         self.relocArange(ipop)
                    #     else:
                    #         self.ARange(ipop)
                    self.functioncall(ipop)
            
            # recalculate the average
            for ivar in range(self.nvariable):
                self.tave[ivar]=0
                for ipop in range(self.npop):
                    self.tave[ivar]+=self.x[ipop*self.nvariable+ivar]
                self.tave[ivar]/=self.npop
                
                self.xstd[ivar]=0
                for ipop in range(self.npop):
                    self.xstd[ivar]+=pow(self.x[self.nvariable*ipop+ivar]-self.tave[ivar],2)
                self.xstd[ivar]/=self.npop
                self.xstd[ivar]=math.sqrt(self.xstd[ivar])


            # initialize
            self.pcnt=1
            for ivar in range(self.nvariable):
                self.pxstd_sum[ivar]=self.xstd[ivar]
                self.pxstd[ivar]=copy.deepcopy(self.pxstd_sum[ivar])/self.pcnt
            self.CNV_judge(self.pxstd,self.xgreat)

            self.writeChange(1)

        
        # continue to search considering dependencies between variables
        elif self.psearch_flg and len(self.P) != 0 and self.pgreatAge<=self.updateTHRESH and self.psearch_age<=self.termlimit:
            if self.fgreat_init > self.fgreat:
                diff=0
                for ivar in range(self.nvariable):
                    diff+=((self.xgreat_init[ivar]-self.xgreat[ivar])**2)
                    self.xgreat_init[ivar]=self.xgreat[ivar]
                diff=math.sqrt(diff)
                self.deltaFi.append((self.fgreat_init-self.fgreat)/(diff+1))
                self.fgreat_init=self.fgreat
            self.pcnt+=1
            for ivar in range(self.nvariable):
                self.pxstd_sum[ivar]+=self.xstd[ivar]
                self.pxstd[ivar]=copy.deepcopy(self.pxstd_sum[ivar])/self.pcnt
            self.CNV_judge(self.pxstd,self.xgreat)
            self.writeChange(0)
            self.pgreatAge+=1
            self.psearch_age+=1
        

        # If all variable pairs were referenced, 
        # → End the serach considering dependecies
        # determine the variable pair we are referencing now and the search begins
        if self.psearch_flg and len(self.P) == 0 and self.pgreatAge>self.updateTHRESH and self.psearch_age!=self.pgreatAge:
            self.psearch_age=-1
            self.pgreatAge=-1
        if self.psearch_flg and len(self.P) == 0  and (self.pgreatAge>self.updateTHRESH or self.psearch_age>self.termlimit):
            if self.fgreat_init > self.fgreat:
                diff=0
                for ivar in range(self.nvariable):
                    diff+=((self.xgreat_init[ivar]-self.xgreat[ivar])**2)
                    self.xgreat_init[ivar]=self.xgreat[ivar]
                diff=math.sqrt(diff)
                self.deltaFi.append((self.fgreat_init-self.fgreat)/(diff+1))
                self.fgreat_init=self.fgreat
            if self.newcnt==0:
                    self.deltaF.append(0)
            else:
                self.deltaF.append(np.mean(np.array(self.deltaFi)))
            self.pcnt+=1
            for ivar in range(self.nvariable):
                self.pxstd_sum[ivar]+=self.xstd[ivar]
                self.pxstd[ivar]=copy.deepcopy(self.pxstd_sum[ivar])/self.pcnt
            self.CNV_judge(self.pxstd,self.xgreat)
            
            self.writeChange(-1)
            

            # update x_init value
            topfit=np.argsort(self.fitness[0:self.npop])
            llmax=int(self.npop*0.2)
            for ipop in range(self.npop):
                for ivar in range(self.nvariable):
                    if ipop == self.imin:
                        self.x_init[self.nvariable*ipop+ivar]=self.xgreat[ivar]
                    elif ipop in topfit[0:llmax]:
                        self.x_init[self.nvariable*ipop+ivar]=self.x[self.nvariable*ipop+ivar]

            print(">>> end of the search considering dependencies between variables")
            self.psearch_flg=False
            self.pgreatAge=0
            self.psearch_age=0
            self.newcnt=0
            self.nonhoxage=0
            self.Pi=-1

            

            # initialize parameter to restart the search considering dependencies between all variables
            for irange in range(self.nrange):
                self.age[irange]=0
            
            for ipop in range(self.npop):
                if ipop == self.imin:
                    pass
                else:
                    if random.randint(0,9) < 2:
                        self.relocArange2(ipop)
                    else:
                        for ivar in range(self.nvariable):
                            self.x[self.nvariable*ipop+ivar]=self.x_init[self.nvariable*ipop+ivar]
                self.functioncall(ipop)
            
            # revaluate each Hoxgene for the search from now on
            # self.reval_Hoxgene()
            self.evalFromResult()
            # self.evalHoxAve(self.npop,"revalHoxgene")
            self.mkpair2("end of psearch")

            self.greatAge=-20


        # continue to search considering dependencies between variables
        elif self.psearch_flg and len(self.P) == 0 and self.pgreatAge<=self.updateTHRESH and self.psearch_age<=self.termlimit:
            if self.fgreat_init > self.fgreat:
                diff=0
                for ivar in range(self.nvariable):
                    diff+=((self.xgreat_init[ivar]-self.xgreat[ivar])**2)
                    self.xgreat_init[ivar]=self.xgreat[ivar]
                diff=math.sqrt(diff)
                self.deltaFi.append((self.fgreat_init-self.fgreat)/(diff+1))
                self.fgreat_init=self.fgreat
            self.pcnt+=1
            for ivar in range(self.nvariable):
                self.pxstd_sum[ivar]+=self.xstd[ivar]
                self.pxstd[ivar]=copy.deepcopy(self.pxstd_sum[ivar])/self.pcnt
            self.CNV_judge(self.pxstd,self.xgreat)
            self.writeChange(0)
            self.pgreatAge+=1
            self.psearch_age+=1
        
        self.writeDebug6()
    

    def writeDebug6(self):
        seed_dir=os.path.normpath(os.path.join(os.path.dirname(__file__),"seed"+str(self.seed)))
        result_dir=os.path.join(seed_dir,"result")

        confirmAge2=os.path.join(result_dir,"confirmAge2.csv")
        with open(confirmAge2,"a") as f:
            writer=csv.writer(f, lineterminator='\n')
            writer.writerow([self.pgreatAge,self.psearch_age])


    def evalFromResult(self):
        self.writeChange1(0)

        p_i=0
        # for ipair in self.PAIR:
        #     for ipop in range(self.npop):
        #         ihoxgene=self.Hoxgene[ipop*self.nvariable:ipop*self.nvariable+self.nvariable]
        #         pflg=True
        #         for ivar in range(self.nvariable):
        #             if ivar in ipair:
        #                 if ihoxgene[ivar] == 0:
        #                     pflg=False
        #             else:
        #                 if ihoxgene[ivar] == 1:
        #                     pflg=False
        #         if pflg:
        #             s=self.deltaF[p_i]
        #             # give a bias
        #             # if s < 1:
        #             #     s = s*(1e-2)
        #             # else:
        #             #     s = s*(1e+2)
        #             self.HGfitness[ipop]=s
        for ipop in self.selectedHox:
            s=self.deltaF[p_i]
            # give a bias
            # if s < 1:
            #     s = s*(1e-2)
            # else:
            #     s = s*(1e+2)
            self.HGfitness[ipop]=s
            p_i+=1
        
        while self.selectedHox != []:
            del self.selectedHox[0]
        self.writeChange1(1)


    def writeChange1(self,state):
        seed_dir=os.path.normpath(os.path.join(os.path.dirname(__file__),"seed"+str(self.seed)))
        result_dir=os.path.join(seed_dir,"result")

        datapath=os.path.join(result_dir,"revalFromRSLT.csv")
        items=[]
        data=[]
        for ipair in self.PAIR:
            p_str="["
            for i_p in ipair:
                p_str+=(str(i_p)+",")
            p_str=p_str.rstrip(",")
            p_str+="]"
            for ipop in range(self.npop):
                ihoxgene=self.Hoxgene[ipop*self.nvariable:ipop*self.nvariable+self.nvariable]
                pflg=True
                for ivar in range(self.nvariable):
                    if ivar in ipair:
                        if ihoxgene[ivar] == 0:
                            pflg=False
                    else:
                        if ihoxgene[ivar] == 1:
                            pflg=False
                if pflg:
                    items.append(p_str)
                    data.append(self.HGfitness[ipop])
        if state==0:
            with open(datapath,"a") as f:
                writer=csv.writer(f, lineterminator="\n")
                writer.writerow(["generation="+str(self.g)])
                writer.writerow(items)
                writer.writerow(data)
        
        elif state==1:
            with open(datapath,"a") as f:
                writer=csv.writer(f, lineterminator="\n")
                writer.writerow(["generation="+str(self.g)])
                writer.writerow(items)
                writer.writerow(data)
                writer.writerow([""])
    

    def PSwitch(self):
        if not self.psearch_flg:
            for ivar in range(self.nvariable):
                self.PASTXSTDSUM[ivar]+=self.xstd[ivar]
                self.PASTXSTD[ivar]=(self.PASTXSTDSUM[ivar]/(self.g+1))
        
        # convergence judgement
        self.judgeConvergence()
        self.writeJudge()

        cnt=self.CONVJUDGE.count(True)
        if not self.psearch_flg and self.psearch_st and not self.hoxflg:
            self.mkpair2("start psearch")
            self.mkpair()
            self.psearch()
        
        elif self.psearch_flg:
            self.psearch()
    


    def judgeConvergence(self):
        xave_ranges=[0.0 for i in range(self.nvariable)]
        for ivar in range(self.nvariable):
            xave_sum = 0
            r_n = 0         # the number of ranges
            weight = 5
            for irange in range(self.nrange):
                if irange == self.mrange:
                    xave_sum += (weight*self.xave[irange*self.nvariable+ivar])
                    r_n += weight
                else:
                    xave_sum += self.xave[irange*self.nvariable+ivar]
                    r_n += 1
            xave_ranges[ivar] = xave_sum/r_n

        ngenetypes=len(self.allgenetypes)
        middle=int(ngenetypes/2)-1
        for ibit in range(self.nbit):
            self.ws[ibit]=self.allgenetypes[middle][ibit]
        iphenotype=self.idecoding(self.nbit,self.nbit-1)

        innerwidth=[0.0 for ivar in range(self.nvariable)]
        for ivar in range(self.nvariable):
            innerwidth[ivar]=self.PASTXSTD[ivar]*math.sqrt(-2*math.log(self.LB+(self.UB-self.LB)*iphenotype/(pow(2,self.nbit-1)-1)))
        
        for ivar in range(self.nvariable):
            self.CONVTHRESH[ivar]=(abs(innerwidth[ivar])/(self.xmax[ivar]-self.xmin[ivar]))*100
        
        mostfar=[0.0 for i in range(self.nvariable)]
        for ivar in range(self.nvariable):
            errors=list()
            for ipop in range(self.npop):
                x_i = copy.deepcopy(self.x[ipop*self.nvariable+ivar])
                ce_i = copy.deepcopy(xave_ranges[ivar])
                error = (abs(x_i-ce_i)/(self.xmax[ivar]-self.xmin[ivar]))*100       # calculate error rate
                errors.append(error)

            worst = max(errors)
            mostfar[ivar] = worst    # maximum distance between the center of each search range and each individual

        # judgement
        for ivar in range(self.nvariable):
            if self.CONVTHRESH[ivar] > mostfar[ivar]:
                self.CONVJUDGE[ivar]=True
            else:
                self.CONVJUDGE[ivar]=False
    


    def writeJudge(self):
        seed_dir=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"seed"+str(self.seed)))
        result_dir=os.path.join(seed_dir,"result")

        judgedata=os.path.join(result_dir,"convjudge.csv")
        with open(judgedata,"a") as f:
            writer=csv.writer(f, lineterminator='\n')
            data=copy.deepcopy(self.CONVJUDGE)
            data.append(self.psearch_flg)
            writer.writerow(data)




    def writeChange(self,state):
        seed_dir=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"seed"+str(self.seed)))
        result_dir=os.path.join(seed_dir,"result")
        hox_dir=os.path.join(result_dir,"Hox")
        released_x=os.path.join(hox_dir,"released_x")

        if state == 1:
            if self.Pi != -1:
                p_str="["
                for i in self.Pi:
                    p_str += (str(i)+",")
                p_str = p_str.rstrip(",")
                p_str+="]"

            for ivar in self.Pi:
                rxi_path=os.path.join(released_x,"x"+str(ivar)+"_released.csv")

                xi_dat = []
                for ipop in range(self.npop):
                    xi_dat.append(self.x[self.nvariable*ipop+ivar])
                aaaxi=sum(xi_dat)/self.npop

                with open(rxi_path,"a") as f:
                    writer = csv.writer(f, lineterminator = '\n')
                    writer.writerow(["g="+str(self.g),p_str,"hoxcnt="+str(self.hoxcnt)])
                    writer.writerow(["fgreat","","xi_ave","ind"])
                    data = [self.fgreat,"",aaaxi]
                    data.extend(xi_dat)
                    writer.writerow(data)
        
        elif state == 0:
            for ivar in self.Pi:
                rxi_path=os.path.join(released_x,"x"+str(ivar)+"_released.csv")

                xi_dat = []
                for ipop in range(self.npop):
                    xi_dat.append(self.x[self.nvariable*ipop+ivar])
                aaaxi=sum(xi_dat)/self.npop
                
                with open(rxi_path,"a") as f:
                    writer = csv.writer(f, lineterminator = '\n')
                    data = [self.fgreat,"",aaaxi]
                    data.extend(xi_dat)
                    writer.writerow(data)
        
        elif state == -1:
            for ivar in self.Pi:
                rxi_path=os.path.join(released_x,"x"+str(ivar)+"_released.csv")

                xi_dat = []
                for ipop in range(self.npop):
                    xi_dat.append(self.x[self.nvariable*ipop+ivar])
                aaaxi=sum(xi_dat)/self.npop
                
                with open(rxi_path,"a") as f:
                    writer = csv.writer(f, lineterminator = '\n')
                    data = [self.fgreat,"",aaaxi]
                    data.extend(xi_dat)
                    writer.writerow(data)
                    writer.writerow([""])
        

        fgreat_psearch=os.path.join(hox_dir,"fgreat_psearch.csv")
        p_str="["
        for i in self.Pi:
            p_str += (str(i)+",")
        p_str = p_str.rstrip(",")
        p_str+="]"

        c_state=""
        if self.greatAge == 1:
            c_state="change"
        
        if state == 1:
            with open(fgreat_psearch,"a") as f:
                writer=csv.writer(f,lineterminator='\n')
                items=["generation","fgreat","pair","c_state",""]
                Xg_str = ["xgreat"+str(i) for i in self.Pi]
                items.extend(Xg_str)
                items.append("")
                STD_str = ["std"+str(i) for i in self.Pi]
                STD_str.append("")
                items.extend(STD_str)
                PSTD_str = ["pstd"+str(i) for i in self.Pi]
                PSTD_str.append("")
                items.extend(PSTD_str)
                cnvflg_str = ["cnv_"+str(i) for i in self.Pi]
                cnvflg_str.append("")
                items.extend(cnvflg_str)
                writer.writerow(items)
                dat=[self.g,self.fgreat,p_str,"",""]
                dat.extend([self.xgreat[i] for i in self.Pi])
                dat.append("")
                dat.extend([self.xstd[i] for i in self.Pi])
                dat.append("")
                dat.extend(self.pxstd[i] for i in self.Pi)
                dat.append("")
                dat.extend([self.judgement[i] for i in self.Pi])
                writer.writerow(dat)
        
        elif state == 0:
            with open(fgreat_psearch,"a") as f:
                writer=csv.writer(f,lineterminator='\n')
                dat=[self.g,self.fgreat,p_str,c_state,""]
                dat.extend([self.xgreat[i] for i in self.Pi])
                dat.append("")
                dat.extend([self.xstd[i] for i in self.Pi])
                dat.append("")
                dat.extend(self.pxstd[i] for i in self.Pi)
                dat.append("")
                dat.extend([self.judgement[i] for i in self.Pi])
                writer.writerow(dat)
        
        elif state == -1:
            with open(fgreat_psearch,"a") as f:
                writer=csv.writer(f,lineterminator='\n')
                dat=[self.g,self.fgreat,p_str,c_state,""]
                dat.extend([self.xgreat[i] for i in self.Pi])
                dat.append("")
                dat.extend([self.xstd[i] for i in self.Pi])
                dat.append("")
                dat.extend(self.pxstd[i] for i in self.Pi)
                dat.append("")
                dat.extend([self.judgement[i] for i in self.Pi])
                writer.writerow(dat)
                writer.writerow([""])
        

        fgreat_psearch_analy=os.path.join(hox_dir,"fgreat_psearch_analy.csv")
        with open(fgreat_psearch_analy,"a") as f:
            writer=csv.writer(f,lineterminator='\n')
            dat=[self.fgreat]
            writer.writerow(dat)
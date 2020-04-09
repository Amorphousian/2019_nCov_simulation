# -*- coding: utf-8 -*-

"""
Created on Tue Feb  4 09:26:56 2020

@author: Administrator
"""

import numpy as np
from matplotlib import pyplot

import matplotlib
matplotlib.rcParams['font.family']='Microsoft YaHei'
matplotlib.rcParams['font.size']=10


class City():
    def __init__(self,N=1000,T=100,D=2,J=1,S=100,distri='N',H=0,R=3):
        self.N = N      #number of citizens
        self.T = T      #number of steps
        self.D = D      #infection distance
        self.J = J      #citizen movement        
        self.S = S      #size of city
        self.distri = distri  #distibution of initial position
        self.Ht = H     #total hospital size
        self.H = H      #current hospital size
        self.R = R      #response time to infected
        self.pos = None     #position of citizens
        self.sta = None     #infected
        self.staN = None    #not infected
        self.day = None    #days since infection

    def initialize(self):
        #initial position of citizens
        if self.distri == 'N':
            self.pos = np.random.randn(self.N,2) / 4 * self.S
        elif self.distri == 'U':
            self.pos = np.random.rand(self.N,2) * self.S
        #initial state of citizens
        self.sta = np.zeros(self.N,bool)
        self.day = np.ones(self.N,int)*(-1)
        lucky = np.random.randint(self.N)
        self.sta[lucky] = True
        self.day[lucky] = 1
        self.staN = (1 - self.sta).astype(bool)
        
        
    def __distance(self):
        infect = self.pos[self.sta,:]
        for i in range(len(infect)):
           dist = np.sqrt((self.pos[:,0]-infect[i,0])**2+(self.pos[:,1]-infect[i,1])**2)
           expo = np.argwhere(dist < self.D).ravel()
           prob = np.random.random(len(expo))
           newinfect = prob > (dist[expo]/self.D)
           for j in expo[newinfect]:
               if self.day[j] < 0:
                   self.day[j] = 1
           self.sta[expo] = (newinfect) + self.sta[expo]
           self.staN = (1-self.sta).astype(bool)
       
    def __move(self):
        #next step position
        step = np.random.randn(len(self.pos),2) * self.J
        self.pos += step
        self.day[np.argwhere(self.day>0).ravel()] += 1
        hos = np.argwhere(self.day == self.R).ravel()
        day=np.ones(10,int)
        if self.H >= len(hos):
            self.pos = np.delete(self.pos,hos,axis=0) 
            self.sta = np.delete(self.sta,hos,axis=0)
            self.staN = (1 - self.sta).astype(bool) 
            self.day = np.delete(self.day,hos,axis=0) 
            self.__distance()
            self.H -=len(hos)

    def show(self):
        pyplot.figure(1,clear=True)
        pyplot.plot(self.pos[self.staN,0],self.pos[self.staN,1],'b.',label='未感染')
        pyplot.plot(self.pos[self.sta,0],self.pos[self.sta,1],'r.',label='已感染')
        pyplot.legend(loc='upper right')
        for t in range(self.T):
            pyplot.ion()
            pyplot.cla()
            self.__move()
            pyplot.plot(self.pos[self.staN,0],self.pos[self.staN,1],'b.',label='未感染')
            pyplot.plot(self.pos[self.sta,0],self.pos[self.sta,1],'r.',label='已感染')
            pyplot.legend(loc='best')
            if self.J > 1:
                tmp = '高'
            else:
                tmp = '低'
            pyplot.title('第{0}天, 感染人数:{1}/{2}人, 医院床位:{3}/{4}个, 人员流动率:{5}'
                         .format(t,sum(self.sta),len(self.pos),self.H,self.Ht,tmp))
            pyplot.pause(.1)
        pyplot.pause(1)
        pyplot.ioff()


if __name__ == '__main__':
    #city(citizens, steps, infection distance, jump, city size,
    #     distribution, hospital size, response time)
    city1=City(1000,100,3,5,
               100,'U')
    city1.initialize()
    city1.show()
    
    city2=City(1000,100,3,.5,100,
               'U')
    city2.initialize()
    city2.show()
    
    city3=City(1000,100,3,5,100,
               'U',100,7)
    city3.initialize()
    city3.show()
    
    city4=City(1000,100,3,.5,100,
               'U',100,7)
    city4.initialize()
    city4.show()

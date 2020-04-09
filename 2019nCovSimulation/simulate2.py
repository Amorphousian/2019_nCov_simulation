# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 09:32:15 2020

@author: Fangjian

This is a Python script, please open in .py format or any IDE.
"""

import numpy as np
import matplotlib
from matplotlib import pyplot

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['font.size'] = 10


class City(object):
    '''
    Create a city object for contagion simulation.
    '''

    def __init__(self, N=1000, T=100, D=2, J=1, S=100, distri='Normal',
                 H=0, R=3, C=None, B=0.5, P=0.67):
        '''
        Parameters
        ----------
        N : INT, optional, default is 1000.
            Number of citizens.
        T : INT, optional, default is 100.
            Number of days(steps).
        D : FLOAT, optional, default is 2.0.
            Infection distance.
        J : FLOAT, optional, default is 1.0.
            Citizen movement(jump).
        S : FLOAT, optional, default is 100.
            Size of city.
        distri : STR, optional, default is 'N'.
            Distibution of initial position,
            'U' for Uniform, 'N' for Normal.
        H : INT, optional, default is 0.
            Current hospital size.
        R : INT, optional, default is 3.
            Response time to infected.
        C : TUPLE, optional, default is None.
            Incubation period.
        B : FLOAT, optional, default is 0.5.
            Contagious coefficient of incubated.
        P : FLOAT, optional, default is 0.67.
            Transition probability from incubated to infected.

        '''
        self.N = N      # Number of citizens
        self.T = T      # Number of days(steps)
        self.D = D      # Infection distance
        self.J = J      # Citizen movement(jump)
        self.S = S      # Size of city
        self.distri = distri        # Distibution of initial position
        self.Ht = H     # Total hospital size
        self.H = H      # Current hospital size
        self.R = R      # Response time to infected
        self.C = C      # Incubation period
        self.B = B      # Contagious coefficient of incubated
        self.P = P      # Transition probability
        self.pos = None     # Position of citizens
        self.ifc = None     # Infected bool
        self.ifcN = None    # Not infected bool
        self.icb = None     # Incubated bool
        self.day = None     # Days since infection
        self.incu = None    # Days left of incubated
        self.infect_status = []    # Record infection status
        if self.Ht > 0:     # Record hospital status if exists
            self.hospit_status = []
        if self.C:     # Record incubation status if exists
            self.incuba_status = []

    def __repr__(self):
        return('Population = {0},'
               '\nDays = {1},'
               '\nInfection distance = {2},'
               '\nCitizen movement = {3},'
               '\nCity size = {4},'
               '\nInitial position = {5},'
               '\nHospital size = {6},'
               '\nResponse time = {7},'
               '\nIncubation period = {8},'
               '\nIncubation contagious coef. = {9},'
               '\nIncubation transition prob = {10}'
               .format(self.N, self.T, self.D, self.J, self.S,
                       self.distri, self.Ht, self.R, self.C,
                       self.B, self.P))

    def initialize(self):
        '''
        Initalize a city.
        '''
        # Initial position of citizens
        if self.distri == 'Normal':
            self.pos = np.random.randn(self.N, 2) / 4 * self.S
        elif self.distri == 'Uniform':
            self.pos = np.random.rand(self.N, 2) * self.S
        # First 'lucky' guy
        lucky = np.random.randint(self.N)
        # Initial state of citizens
        self.ifc = np.zeros(self.N, bool)
        self.ifc[lucky] = True
        self.ifcN = (1 - self.ifc).astype(bool)
        # Initial hospital response
        if self.Ht > 0:
            self.day = np.ones(self.N, int) * (-1)
            self.day[lucky] = 1
        # Initial incubation days
        if self.C:
            self.icb = np.zeros(self.N, bool)
            self.incu = np.ones(self.N, int) * (-1)

    def __distance(self):
        '''
        Calculate distance to every infected and simulate contagion.
        '''
        if self.C:    # Incubation period exists
            # Incubated first since infected creates more incubated
            # Contagious incubated
            self.__distance_incu_incuba()
            # Contagious infected
            self.__distance_incu_infect()
        else:   # Incubation period does not exist
            # Contagious infected
            self.__distance_ifc_infect()

    def __distance_incu_incuba(self):
        '''
        Simulate contagious status for incubated if incubation exists.
        '''
        incuba = self.pos[self.icb, :]
        # Loop through all incubated
        for i in range(incuba.shape[0]):
            # Calulate Euclidean distance to current incubated
            dist = np.sqrt((self.pos[:, 0] - incuba[i, 0]) ** 2
                           + (self.pos[:, 1] - incuba[i, 1]) ** 2)
            # Identify exposure to current incubated
            expo = np.argwhere(dist < self.D).ravel()
            # Random sample from U(0,1)
            prob = np.random.rand(expo.size) * self.B
            # Simulate contagion
            probB = prob > (dist[expo] / self.D)
            # Potential incubated
            proincuba = np.zeros(self.ifcN.size, bool)
            proincuba[expo[probB]] = True
            # New incubated (potential && not infected)
            newincuba = proincuba * self.ifcN
            # Set incubation period for new incubated
            self.incu[newincuba] = (np.random.rand(newincuba.sum())
                                    * (self.C[1] + 1 - self.C[0])
                                    + self.C[0]).astype(int)
            # self.incu[newincuba] = np.random.randint(
            #     self.C[0], self.C[1] + 1, newincuba.sum())
            # Set incubation status for new incubated
            self.icb[newincuba] = True
            # Set not infected status for new incubated
            self.ifcN[newincuba] = False

    def __distance_incu_infect(self):
        '''
        Simulate contagious status for infected if incubation exists.
        '''
        infect = self.pos[self.ifc, :]
        # Loop through all infected
        for i in range(infect.shape[0]):
            # Calulate Euclidean distance to current infected
            dist = np.sqrt((self.pos[:, 0] - infect[i, 0]) ** 2
                           + (self.pos[:, 1] - infect[i, 1]) ** 2)
            # Identify exposure to current infected
            expo = np.argwhere(dist < self.D).ravel()
            # Random sample from U(0,1)
            prob = np.random.rand(expo.size)
            # Simulate contagion
            probB = prob > (dist[expo] / self.D)
            # Potential incubated
            proincuba = np.zeros(self.ifcN.size, bool)
            proincuba[expo[probB]] = True
            # New incubated (potential && not infected)
            newincuba = proincuba * self.ifcN
            # Set incubation period for new incubated
            self.incu[newincuba] = (np.random.rand(newincuba.sum())
                                    * (self.C[1] + 1 - self.C[0])
                                    + self.C[0]).astype(int)
            # Set incubation status for new incubated
            self.icb[newincuba] = True
            # Set not infected status for new incubated
            self.ifcN[newincuba] = False

    def __distance_ifc_infect(self):
        '''
        Simulate contagious status for infected if incubation DNE.
        '''
        infect = self.pos[self.ifc, :]
        # Loop through all infected
        for i in range(infect.shape[0]):
            # Calulate Euclidean distance to current infected
            dist = np.sqrt((self.pos[:, 0] - infect[i, 0]) ** 2
                           + (self.pos[:, 1] - infect[i, 1]) ** 2)
            # Identify exposure to current infected
            expo = np.argwhere(dist < self.D).ravel()
            # Random sample from U(0,1)
            prob = np.random.rand(expo.size)
            # Simulate contagion
            probB = prob > (dist[expo] / self.D)
            # Potential incubated
            proinfect = np.zeros(self.ifcN.size, bool)
            proinfect[expo[probB]] = True
            # New infect (potential && not infected)
            newinfect = proinfect * self.ifcN
            self.ifc[newinfect] = True
            self.ifcN[newinfect] = False
            if self.Ht > 0:
                self.day[newinfect] = 1

    def __hospital(self):
        '''
        Hospitalize/quarantine infected.
        '''
        # Infection day adds one for all infected
        self.day[np.argwhere(self.day > 0).ravel()] += 1
        # Identify infected needed to quarantine
        hos = np.argwhere(self.day == self.R).ravel()
        # Quarantine/delete identified infected, if hospital has enough space
        if self.H >= hos.size:
            self.pos = np.delete(self.pos, hos, axis=0)
            self.ifc = np.delete(self.ifc, hos, axis=0)
            self.ifcN = np.delete(self.ifcN, hos, axis=0)
            self.day = np.delete(self.day, hos, axis=0)
            self.icb = np.delete(self.icb, hos, axis=0)
            self.incu = np.delete(self.incu, hos, axis=0)
            self.H -= hos.size
        elif self.H > 0:
            self.pos = np.delete(self.pos, hos[:self.H], axis=0)
            self.ifc = np.delete(self.ifc, hos[:self.H], axis=0)
            self.ifcN = np.delete(self.ifcN, hos[:self.H], axis=0)
            self.day = np.delete(self.day, hos[:self.H], axis=0)
            self.icb = np.delete(self.icb, hos[:self.H], axis=0)
            self.incu = np.delete(self.incu, hos[:self.H], axis=0)
            self.H = 0

    def __incubation(self):
        '''
        Transition from incubated to infected.
        '''
        # Incubation lesses one day for all incubated
        self.incu[np.argwhere(self.incu > -1).ravel()] -= 1
        # Identify transition from incubated to infected
        trans = np.argwhere(self.incu == 0).ravel()
        # Simulate probable transition
        prob1 = np.random.rand(trans.size) < self.P
        prob0 = (1-prob1).astype(bool)
        # Transition to infected
        self.icb[trans[prob1]] = False
        self.ifc[trans[prob1]] = True
        self.day[trans[prob1]] = 1
        # Transition to normal
        self.icb[trans[prob0]] = False
        self.ifcN[trans[prob0]] = True
        self.incu[trans[prob0]] = -1

    def __move(self):
        '''
        Move to next day.
        '''
        # Stimulate random walks for all citizens
        step = np.random.randn(self.pos.shape[0], 2) * self.J
        # Stimulate next step positions
        self.pos += step
        if self.C:    # If incubation exists
            self.__incubation()    # Simulate transition from incubation
        if self.H > 0:    # If hospital has vacant spot
            self.__hospital()    # Quarantine infected
        self.__distance()    # Simulate contagion

    def show(self):
        '''
        Plot graph/animation.
        '''
        pyplot.figure(1, clear=True)
        if self.C:
            self.__show_incubation()
        else:
            self.__show_noincubation()

    def __show_incubation(self):
        '''
        Plot animation if incubation exists.
        '''
        pyplot.plot(self.pos[self.ifc, 0], self.pos[self.ifc, 1],
                    'r*', label='已感染')
        pyplot.plot(self.pos[self.icb, 0], self.pos[self.icb, 1],
                    'y.', label='潜伏期')
        pyplot.plot(self.pos[self.ifcN, 0], self.pos[self.ifcN, 1],
                    'b.', label='未感染')
        pyplot.legend(loc='upper left')
        # pyplot.xlim((-self.S, 2*self.S))
        # pyplot.ylim((-self.S, 2*self.S))
        pyplot.ion()
        t = 0
        while self.ifc.sum() > 0 or self.icb.sum() > 0:
            pyplot.clf()
            self.__move()
            pyplot.plot(self.pos[self.ifc, 0], self.pos[self.ifc, 1],
                        'r*', label='已感染')
            pyplot.plot(self.pos[self.icb, 0], self.pos[self.icb, 1],
                        'y.', label='潜伏期')
            pyplot.plot(self.pos[self.ifcN, 0], self.pos[self.ifcN, 1],
                        'b.', label='未感染')
            pyplot.legend(loc='upper left')
            # pyplot.xlim((-self.S, 2*self.S))
            # pyplot.ylim((-self.S, 2*self.S))
            if self.J > 1:
                tmp = '快'
            else:
                tmp = '慢'
            pyplot.title('第{0}天, 感染人数:{1}/{2}人, 潜伏人数:{6}人,'
                         ' 医院床位:{3}/{4}个, 人员流动:{5}'
                         .format(t, self.ifc.sum(), self.pos.shape[0],
                                 self.H, self.Ht, tmp, self.icb.sum()))
            pyplot.pause(.1)
            self.infect_status.append(self.ifc.sum())
            self.incuba_status.append(self.icb.sum())
            self.hospit_status.append(self.Ht - self.H)
            t += 1
            if t >= self.T:
                break
        pyplot.pause(1)
        pyplot.ioff()

    def __show_noincubation(self):
        '''
        Plot animation if incubation DNE.
        '''
        pyplot.plot(self.pos[self.ifc, 0], self.pos[self.ifc, 1],
                    'r*', label='已感染')
        pyplot.plot(self.pos[self.ifcN, 0], self.pos[self.ifcN, 1],
                    'b.', label='未感染')
        pyplot.legend(loc='upper left')
        # pyplot.xlim((-self.S, 2*self.S))
        # pyplot.ylim((-self.S, 2*self.S))
        pyplot.ion()
        t = 0
        while self.ifc.sum() > 0 or self.icb.sum() > 0:
            pyplot.clf()
            self.__move()
            pyplot.plot(self.pos[self.ifc, 0], self.pos[self.ifc, 1],
                        'r*', label='已感染')
            pyplot.plot(self.pos[self.ifcN, 0], self.pos[self.ifcN, 1],
                        'b.', label='未感染')
            pyplot.legend(loc='upper left')
            # pyplot.xlim((-self.S, 2*self.S))
            # pyplot.ylim((-self.S, 2*self.S))
            if self.J > 1:
                tmp = '快'
            else:
                tmp = '慢'
            pyplot.title(
                '第{0}天, 感染人数:{1}/{2}人, 医院床位:{3}/{4}个, 人员流动:{5}'
                .format(t, self.ifc.sum(), self.pos.shape[0],
                        self.H, self.Ht, tmp))
            pyplot.pause(.1)
            self.infect_status.append(self.ifc.sum())
            self.hospit_status.append(self.Ht - self.H)
            t += 1
            if t > self.T:
                break
        pyplot.pause(1)
        pyplot.ioff()

    def trend(self):
        '''
        Plot infected and incubated status of the City Object.
        '''
        pyplot.figure(2, clear=True)
        pyplot.plot(self.infect_status, 'r-', label='感染人数')
        pyplot.plot(self.incuba_status, 'b-', label='疑似人数')
        pyplot.legend(loc='best')
        pyplot.xlabel('天数')
        pyplot.ylabel('人数')
        pyplot.show()


if __name__ == '__main__':
    # City(citizens,days,dist,jump,size,distri,
    #      hospital,response,(incu),incuCoef,incuTrans)
    city0 = City(1000, 100, 3, 3, 100, 'Uniform',
                 100, 7, (4, 7), 0.8, 0.6)
    city0.initialize()
    city0.show()
    city0.trend()

    city1 = City(1000, 100, 3, .8, 100, 'Uniform',
                 100, 7, (4, 7), 0.8, 0.6)
    city1.initialize()
    city1.show()
    city1.trend()

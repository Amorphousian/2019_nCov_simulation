# -*- coding: utf-8 -*-

# ###############################################
from simulate2 import City

# City(citizens,days,dist,jump,size,distri,
#     hospital,response,(incu),incuCoef,incuTrans)
city0 = City(1000, 100, 3, 3, 100, 'Uniform',
             100, 7, (4, 7), 0.8, 0.6)
city0.initialize()
city0.show()
city0.trend()

city1 = City(1000, 100, 3, .8, 100, 'Normal',
             100, 7, (4, 7), 0.8, 0.6)
city1.initialize()
city1.show()
city1.trend()

# ################################################
from record import get_data
print(get_data())

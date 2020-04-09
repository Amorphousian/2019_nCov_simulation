# -*- coding: utf-8 -*-

# GetData#########################################################
import requests as rq
from bs4 import BeautifulSoup as bs


def get_data():
    page0 = rq.get('https://3g.dxy.cn/newh5/view/pneumonia')
    page0.encoding = 'utf-8'
    page1 = bs(page0.text, features="html.parser").get_text().split(',')

    for j, i in enumerate(page1):
        if "countRemark" in i:
            # print(i)
            # print(j)
            output = ''
            for k in range(6):
                output += (page1[j+1+k] + '\n')
            return output

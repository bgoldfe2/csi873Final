# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 19:21:36 2017

@author: Bruce
"""
from math import sqrt

def ci95(cor,tot):
    p = cor/tot
    sigma = sqrt((p*(1-p))/tot)
    mu = 1.96 * sigma
    ci_low =format(p - mu,'.4f')
    ci_hi = format(p + mu,'.4f')
    print("[",ci_low,",",ci_hi,"]")

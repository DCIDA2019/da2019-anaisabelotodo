#!/usr/bin/env python
from __future__ import absolute_import, unicode_literals, print_function
import numpy
from numpy import pi, cos
from pymultinest.solve import solve
import os
import pymultinest
if not os.path.exists("chains"): os.mkdir("chains")
from scipy.special import factorial

# probability function, taken from the eggbox problem.

file = numpy.loadtxt('data.txt') 
data = file
x = data[0]
y = data[1]

def myprior(cube):
    cube[0] = 5*cube[0]
    cube[1] = 5*cube[1]
    return cube 

def modelo(theta):
    return theta[0]*x + theta[1] # Lineal

def myloglike(cube): # Función que calcula el likelihood
    model = modelo([cube[0],cube[1]]) # Calculamos el modelo utilizando la función modelo
    log_likelihood = y*numpy.log(model)-model-numpy.log(factorial(y))  # Obtenemos L
    return log_likelihood.sum() # Regresamos la suma

# number of dimensions our problem has
parameters = ["a", "b"]
n_params = len(parameters)
# name of the output files
prefix = "chains/3-"

# run MultiNest
result = solve(LogLikelihood=myloglike, Prior=myprior, 
	n_dims=n_params, outputfiles_basename=prefix, verbose=True)

print()
#print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')
for name, col in zip(parameters, result['samples'].transpose()):
	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

# lets analyse the results
a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename=prefix)
s = a.get_stats()

# make marginal plots by running:
# $ python multinest_marginals.py chains/3-
# For that, we need to store the parameter names:
import json
with open('%sparams.json' % prefix, 'w') as f:
	json.dump(parameters, f, indent=2)


with open('%sstats.json' %  a.outputfiles_basename, mode='w') as f:
	json.dump(s, f, indent=2)
print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))


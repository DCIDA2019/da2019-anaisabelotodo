#!/usr/bin/env python
from __future__ import absolute_import, unicode_literals, print_function
import numpy
from numpy import pi, cos
from pymultinest.solve import solve
import os
if not os.path.exists("chains"): os.mkdir("chains")
from colossus.cosmology import cosmology


# probability function, taken from the eggbox problem.

file = numpy.loadtxt('GilMarin_2016_CMASSDR12_measurement_monopole_post_recon.txt').T 
data = file
k = data[0]

def myprior(cube):
    cube[0]=0.7*cube[0]+0.1
    cube[1] = 3*cube[1]+0.1
    return cube 


def Pk_Om(Om_,b,k):
    cosmo = cosmology.setCosmology('planck15') # Generamos los parámetros default del experimento planck15
    z = 0.57 
    cosmo.Om0 = Om_ # Modificamos la densidad de materia
    return b**2*cosmo.matterPowerSpectrum(k,z) # Se regresa la función matterPowerSpectrum multiplicada por el bias al cuadrado


def myloglike(cube):
    Om = cube[0]
    b = cube[1]
    equis = data[0] # Llamamos x al primer conjunto de datos
    ye = data[1] # Llamamos y al segundo conjunto de datos
    yerr = data[2]  # Llamamos yerror al tercer conjunto de datos
    model = Pk_Om(Om, b,equis) # Calculamos el modelo utilizando la función Pk_om
    chisq2 = (ye-model)**2/(yerr**2) + numpy.log(yerr**2) # Obtenemos chi^2
    return -0.5*chisq2.sum()


# number of dimensions our problem has
parameters = ["Om0", "b"]
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

# make marginal plots by running:
# $ python multinest_marginals.py chains/3-
# For that, we need to store the parameter names:
import json
with open('%sparams.json' % prefix, 'w') as f:
	json.dump(parameters, f, indent=2)


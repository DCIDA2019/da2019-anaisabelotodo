#!/usr/bin/env python
# coding: utf-8

# In[8]:


import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt
import math
import random

from colossus.cosmology import cosmology

import emcee
import tqdm


# In[9]:


cosmo = cosmology.setCosmology('planck15')


# In[10]:


pk_cmasdr12 = np.loadtxt('GilMarin_2016_CMASSDR12_measurement_monopole_post_recon.txt').T


# In[14]:


def Pk_Om(Om_,b,beta,k):
    cosmo = cosmology.setCosmology('planck15') # Generamos los parámetros default del experimento planck15
    z = 0.57 
    cosmo.Om0 = Om_ # Modificamos la densidad de materia
    return b**2*(1+beta)*cosmo.matterPowerSpectrum(k,z) # Se regresa la función matterPowerSpectrum multiplicada por el bias al cuadrado


# In[15]:


def log_likelihood(theta,data): # Función que calcula el logaritmo natural del likelihood
    Om = theta[0]
    b = theta[1]
    beta = theta[2]
    equis = data[0] # Llamamos x al primer conjunto de datos
    ye = data[1] # Llamamos y al segundo conjunto de datos
    yerr = data[2]  # Llamamos yerror al tercer conjunto de datos
    model = Pk_Om(Om, b, beta,equis) # Calculamos el modelo utilizando la función Pk_om
    chisq2 = (ye-model)**2/(yerr**2) + np.log(yerr**2) # Obtenemos chi^2
    return chisq2.sum() # Regresamos la suma de todos los chi^2, que da como resultado el ln del likelihood.


# In[16]:


np.random.seed(1)


# In[18]:


ini_point = (0.3,1.6,0.1)

#chisq_ = lambda *args: chisq(*args)
data = pk_cmasdr12

min_sol = op.minimize(log_likelihood, ini_point, args = data, method = 'L-BFGS-B', bounds=((0,1),(0,5),(0,1)))


# In[20]:


def log_prior(theta):
    Om0, b, beta = theta
    if 0.05 < Om0 < 0.8 and 0.1 < b < 3 and 0.05 < beta < 0.8:
        return 0.0
    return -np.inf


# In[21]:


x, y, yerr = pk_cmasdr12[0], pk_cmasdr12[1], pk_cmasdr12[2]
def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + -0.5*log_likelihood(theta, [x, y, yerr])


# In[ ]:


pos = min_sol.x + 1e-4 * np.random.randn(32, 3)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
sampler.run_mcmc(pos, 50000, progress=True);


# In[22]:


fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["Om0", "b", "beta"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig('numbervsiterations.png')
#plt.show()


# In[ ]:


flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)


# In[ ]:


import corner
fig1 = corner.corner(
    flat_samples, labels=labels, truths=min_sol.x
);
fig1.savefig('corner.png')
plt.show


# In[ ]:


from IPython.display import display, Math
median = []
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    median.append(mcmc[1])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))


fig2 = plt.figure(figsize=(10,10))
plt.loglog()
k = 10**np.linspace(-5,6,100000)
cosmo.Om0 = median[0]
Pk = median[1]**2*(1+median[2])*cosmo.matterPowerSpectrum(k,0.57)

plt.plot(k, Pk, '-', label = 'Om0 = 0.35, b= 1.849, beta = 0.338') 
plt.errorbar(pk_cmasdr12[0], pk_cmasdr12[1], yerr = pk_cmasdr12[2], fmt = '.b')

plt.xlim(1e-3,1)
plt.ylim(100,3e6)
plt.legend()
plt.xlabel('k(Mpc/h)')
plt.ylabel('P(k)')
fig2 = plt.savefig('result.png')
plt.show()


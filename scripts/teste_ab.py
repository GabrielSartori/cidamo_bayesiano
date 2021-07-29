#----------------------------------------------------------------------------------------------
#' Latex Fórmula

# https://towardsdatascience.com/conjugate-prior-explained-75957dc80bfb


#----------------------------------------------------------------------------------------------
#' Parâmetros 

# binomial, n_ensaio  de bernoulli
#+ p = probabilidade de ocorrência
#+ n = número de ensaios

#+ n = Naturais Positivos
#+ x = Espaço paramétrico =[0, N]
#+ p [0, 1]

# beta 
#+ \alpha - sucesso 
#+ \beta  - falha

# \[
#   f(x) = {{N}\choose{x}} \cdot \theta^{x}(1-\theta)^{N-x}
# \]

# \[
# g(\theta) = \frac{\Gamma(\alpha +
# \beta)}{\Gamma(\alpha)\Gamma(\beta)}\theta^{\alpha - 1}(1 - \theta)^{\beta - 1}
# \]

# beta 
#+ \alpha - x + \alpha
#+ \beta  - (n - x) + \beta


#----------------------------------------------------------------------------------------------
#' Versão Python Base
 
import pandas as pd
import numpy as np
from scipy.stats import beta, gamma
import matplotlib.pyplot as plt
import seaborn as sns


#* Base Python
priori_alpha = 1
priori_beta = 1

beta_conversao_priori = beta(1, 1)
beta_conversao_priori_rand = beta_conversao_priori.rvs(100000)

plt.hist(beta_conversao_priori_rand)
plt.show()

#* Posteriori

# Até 30 dias
conversion_a = 130
sample_a = 500

posterior_a = beta(
      priori_alpha + conversion_a
    , priori_beta + sample_a - conversion_a
    )

conversion_b = 110
sample_b = 480

posterior_b = beta(
        priori_alpha + conversion_b
    , priori_beta + sample_b - conversion_b
    )

simulate_a = posterior_a.rvs(10000)
simulate_b = posterior_b.rvs(10000)

plt.figure(figsize = (10,6))
sns.histplot(simulate_a, color = '#fee6ce')
sns.histplot(simulate_b, color = '#d0d1e6')
sns.despine()
plt.legend(title='Smoker', loc='upper left', labels=['Simulated A', 'Simulated B'])
plt.ylabel("")
plt.show()

#* Métricas de Incerteza

#+ a probabilidade da campanha do tratamento ser melhor que a campanha do controle (better) 
#+ é a perda esperada se escolhermos a campanha do tratamento como melhor e na verdade ela for pior (loss)
#+ é a melhora esperada se escolhermos a campanha tratamento como vencedora e ela realmente for melhor (uplift)

# Better
better_trat = np.sum(simulate_a > simulate_b)
pct_better_trat = (better_trat/10000) * 100
pct_better_trat

# Loss
loss_trat = np.mean((simulate_b < simulate_a) * (simulate_b - simulate_a)/simulate_a)
loss_trat * 100

# Uplift
uplift_trat = np.mean((simulate_b - simulate_a)/simulate_a) 
uplift_trat * 100


# https://medium.com/loftbr/masp-nosso-framework-de-marketing-digital-3ec46bfc2f96
#----------------------------------------------------------------------------------------------
#' Versão Python pymc3
import pymc3 as pm
from scipy.stats import beta, gamma, bernoulli

theta_unk = 0.3 # unknown theta, that's the parameter we want to estimate
nb_data = n = 40
data = bernoulli.rvs(theta_unk, size=nb_data)

def create_model_pymc3(data):
    with pm.Model() as model: 
        theta = pm.Beta('theta', alpha=1, beta=1)
        bernoulli = pm.Bernoulli('bernoulli',p=theta, observed=data)
    return model

# def create_model_pymc3_(data):
#     with pm.Model() as model: 
#         theta = pm.Beta('theta', alpha=1, beta=1)
#         n = len(data)
#         k = np.sum(data)
#         p = k/n
#         binominal = pm.Binomial('binominal', n, p, observed=k)
#     return model

model = create_model_pymc3(data)
# model = create_model_pymc3_(data)

# map_estimate = pm.find_MAP(model=model)

with model:
    p_trace = pm.sample(10000)

theta_trace = p_trace.posterior.theta.values

hist, bins = np.histogram(theta_trace, bins=np.arange(0.,1.01,0.01), normed=True)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.xlim([0., 1.0])
plt.ylim([0., 6.0])
plt.show()

#----------------------------------------------------------------------------------------------
#' Caso Teste A\B Bayesiano

+ Temos duas campanhas de Marketing
+ Campanha A 
+ Campanha B






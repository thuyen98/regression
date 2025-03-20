#!/usr/bin/env python
# coding: utf-8

# # LIBRAIRIES & FONCTIONS UTILES

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from numpy.linalg import matrix_rank

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression, Ridge,enet_path, Lasso, LassoCV, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def backward_one_step(X,
                      Y,
                      estimator,
                      scoring):
    # -------------------------------------------------------------------------
    # /!\ on suppose ici que la premiere colonne de X correspond à l'intercept
    # -------------------------------------------------------------------------
    # on recupere la dimension de X
    p = X.shape[1]
    # vecteur des listes de scores
    all_scores = []
    # => on parcourt toutes les colonnes de X sauf la première
    for j in range(1,p):
        # On crée une matrice temporaire X_tmp qui contient
        # toutes le colonnes de X sauf la j
        # (on supprime la colonne j de X)
        X_tmp = np.delete(np.array(X), j, axis=1)
        # On fit un "estimator" (ici la regression lineaire) de Y vs X_tmp
        estimator.fit(X_tmp,Y)
        # On predit la reponse sur les donnees X_tmp
        Y_pred_tmp = estimator.predict(X_tmp)
        # On calcule le score
        score_tmp = scoring(Y,Y_pred_tmp)
        # On stocke le score
        all_scores += [score_tmp]
    # On retourne la colonne pour laquelle le score est "maximal"
    # /!\ comme on parcourt de 1 à p -> il faut ajouter 1 ...
    j_opt = np.argmax(all_scores)+1
    return j_opt

def backward_stepwise(X,Y,estimator,scoring,random_state=42):
    # -------------------------------------------------------------------------
    # /!\ on suppose ici que la premiere colonne de X correspond à l'intercept
    # -------------------------------------------------------------------------
    p = X.shape[1]
    # creation d'une liste correspondant aux indices initiaux des colonnes de X
    list_index_X = list(range(p))
    # Split Echantillon d'apprentissage / echantillon de test
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                      test_size=0.2,
                                                      random_state = random_state)
    # Vecteur des scores calculés à chaque étape (sur X_val)
    all_scores = []
    # Vecteur des composantes à enlever à chaque étape
    removed_list = []

    # Application de la stepwise procedure
    # On ne fera que p-1 iterations (car on laisse l'intercept)
    for j in range(p-1):
        print(f"{j}",end="\r")

        # Sélection de la colonne a enlever
        j_iter = backward_one_step(X_train,
                                   Y_train,
                                   estimator,
                                   scoring)

        # On recupere l'indice j_iter de la colonne a enlever
        removed_list += [list_index_X[j_iter]]
        # On supprime l'indice de list_index_X
        list_index_X = np.delete(list_index_X, j_iter)

        # On supprime la colonne j_iter de X_train & X_val
        X_train = np.delete(np.array(X_train), j_iter, axis=1)
        X_val = np.delete(np.array(X_val), j_iter, axis=1)

        # On fait la regression de Y_train vs X_train
        estimator.fit(X_train,Y_train)
        # On prédit & calcule le score sur X_val
        Y_pred_val = estimator.predict(X_val)
        # On stocke le score
        all_scores += [scoring(Y_val,Y_pred_val)]

    return all_scores,removed_list

## TEST

# from sklearn.datasets import make_regression
# XX,YY,coef = make_regression(n_samples=50,
#                         n_features=10,n_informative=2,random_state=42,
#                         coef=True)
# XX_I = pd.DataFrame(XX)
# XX_I.insert(0,"Intercept",1)

# res = backward_stepwise(XX_I,
#                         YY,
#                         LinearRegression(fit_intercept=False),
#                         r2_score)
# coef[np.array(res[1])-1]


# # EXERCICE
# 
# * Des données de spectroscopie Infra Rouge ont été recueillies sur différents échantillons de viande
# * Pour chaque observation des mesures précises de conteneur en gras ont aussi été réalisées.
# Comme les mesures de taux de gras sont compliquées à réaliser, on aimerait construire un prédicteur à partir des données de spectroscopie qui sont plus faciles à obtenir.
# ___

# ___

# ## Lire et explorer les données
# * Quelle est la dimension de la matrice $X$
# * Faire une régression avec toutes les variables (validez avec echantillon de test/apprentissage)
# 
# ___

# In[2]:


data = pd.read_csv("meatspec.csv")
data.drop(columns="Unnamed: 0", inplace=True)


# In[3]:


data.head()


# In[5]:


X = data.drop("fat", axis = 1)
intercept = np.ones(215)
Y = intercept + data["fat"]


# In[6]:


X.info()


# In[7]:


X.shape


# In[ ]:





# In[8]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                      test_size=0.2,
                                                      random_state = 42)


# In[9]:


LR = LinearRegression()


LR.fit(X_train, Y_train)


Y_pred = LR.predict(X_test)


mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f'R² est {r2}')
print(f'Les paramètres estimés {LR.coef_}')
print(f'L intercept est {LR.intercept_}')


# ___
# 
# ## Régressions et Sélection de variables
# 
# La matrice $X$ étant de dimension relativement élevée, nous allons étudier la possibilité de réduire sa dimension en ne selectionnant que certaines variables.
# Dans toutes les approches étudiées par la suite, on fera bien attention à utiliser une base de donnée d'apprentissage et une base de donnée de test.
# 
# ___

# In[10]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[11]:


#faire un ACP

from sklearn.decomposition import PCA

X -= X.mean(axis=0)
X /= X.std(axis=0)

#ACP
pca = PCA()
pca.fit_transform(X)
n = pca.n_components_


# In[12]:


_ = plt.bar(range(1,len(pca.explained_variance_ratio_)+1),
        pca.explained_variance_ratio_.cumsum())
_ = plt.title("Explained Variance Ratio (Cummulative Sum)")
_ = plt.xlabel("Number Of PCA")
_ = plt.ylabel("Explained Variance Ratio")


# ___
# 
# ### APPROCHE ALEATOIRE
# 
# * On sélectionne de façon aléatoire $K$ composantes et on effectue la régression de Y sur ces $K$ composantes. On calcule le $R^2$ sur une base de données de test.
# * On effectuera cette opération plusieurs fois de façon à avoir une valeur moyenne de $R^2$.
# 
# Cette valeur servira de point de repère entre un modèle "naïf" (sélection de façon aléatoire) et un modèle où des variables auront été sélectionnées selon une stratégie donnée.
# 
# Dans ce qui suit, nous pourrons sélectionner: $K = 5,10,20$
# 
# ___

# In[13]:


list_comp = list(X.columns)


# In[14]:


K_values = [5, 10, 20]
n_iterations = 100
results = {}

for K in K_values:
    r2_scores = []

    for _ in range(n_iterations):
        selected_features = np.random.choice(X.columns, size=K, replace=False)

        X_train, X_test, y_train, y_test = train_test_split(X[selected_features], Y, test_size=0.2, random_state=42)


        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)

    results[K] = np.mean(r2_scores)

for K, r2_mean in results.items():
    print(f"Moyenne du R² pour K={K}: {r2_mean:.2f}")


# ___
# 
# ### APPROCHE STEPWISE
# * On applique la procédure stepwise "backward" de façon sélectionner $K$ variables.
# * On effectue la régression sur le $K$ variables sélectionnées.
# Quelles remarques pouvez vous faire ?
# 
# ___

# In[15]:


estimator = LinearRegression()
scoring = r2_score


# In[16]:


print(backward_one_step(X,Y,estimator,scoring))
print(backward_stepwise(X,Y,estimator,scoring,random_state=42))


# In[17]:


res =  backward_stepwise(X,Y,estimator,scoring,random_state=42)
plt.figure(figsize=(25, 8))
_ = plt.plot(res[0],'.-')

j_opt = np.argmax(res[0])
max_r2_score = res[0][j_opt]
# plt.scatter(j_opt, max_r2_score, color='red', s=100, zorder=3)


_ = plt.xticks(ticks=range(len(res[1])), labels=res[1])


# ___
# 
# ### APPROCHE LASSO
# * On applique la procédure LASSO de façon sélectionner $K$ variables.
# * On effectue la régression sur le $K$ variables sélectionnées.
# Quelles remarques pouvez vous faire ?
# 
# ___

# In[18]:


K_values = [5,10,20]
alpha = 0.1
coefs = []


selected_features = np.random.choice(X.columns, size=5, replace=False)
lasso = Lasso(alpha=alpha)
lasso.fit(X[selected_features], Y)
coefs.append(lasso.coef_)

for i in range(X[selected_features].shape[1]):
  print(np.array(coefs)[:,i])


# In[19]:


n_alphas = 200
alphas = np.logspace(-1, 2, n_alphas)
coefs  = []
for a in alphas:
    lasso = Lasso(alpha=a)
    lasso.fit(X, Y)
    coefs.append(lasso.coef_)


# In[20]:


for i in range(X.shape[1]):
  plt.plot(alphas,np.array(coefs)[:,i],linestyle="--")

plt.gca().set_xscale('log')


# In[21]:


# plt.figure(figsize=(10, 6))
# plt.plot(alphas, coefs)
# plt.xscale("log")  # Échelle logarithmique
# plt.xlabel("Alpha")
# plt.ylabel("Valeur des coefficients")
# plt.title("Évolution des coefficients du Lasso en fonction d'Alpha")
# plt.show()


# ___
# 
# ### AUTRE(S) APPROCHES
# * Quelles techniques avez vous vu de façon à réduire la dimension de la matrice $X$ ?
# * Que pouvez-vous proposer ?
# * Comparer avec les autres approches
# 
# ___

# In[22]:


estim = LinearRegression()
score = make_scorer(r2_score)


# In[23]:


score = make_scorer(r2_score)
selection = RFECV(estim, cv=5, scoring=score)


# In[24]:


_fit = selection.fit(X,Y)


# **RIDGE**

# In[25]:


n_alphas = 200
alphas = np.logspace(-1, 2, n_alphas)

coefs = []
for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X, Y)
    coefs.append(ridge.coef_)


# In[26]:


for i in range(X.shape[1]):
  plt.plot(alphas,np.array(coefs)[:,i],linestyle="--")

plt.gca().set_xscale('log')

ridge_cv = RidgeCV(alphas=alphas).fit(X, Y)
plt.vlines(ridge_cv.alpha_,np.min(coefs),np.max(coefs),color="k")


# In[ ]:





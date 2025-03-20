# regression
+ Développement d’un modèle de régression pour estimer la base de donnéés.
+ Implémentation et comparaison de méthodes de sélection de variables *(stepwise, LASSO)}* pour optimiser la prédiction.
+ Évaluation des performances par validation croisée et exploration d’approches de réduction de dimension *(ACP, sélection aléatoire, RIDGE).*
## Méthode
-----
### 1. Stepwise
La méthode Stepwise est une technique itérative pour sélectionner les variables dans un modèle de régression.
1. Sélection Forward et Backward: Il existe deux variantes principales : la sélection forward (ajout progressif de variables) et la sélection backward (élimination progressive de variables).
2. Critères de Sélection: Les variables sont ajoutées ou supprimées en fonction de critères statistiques comme le **R² ajusté**, le critère d'information d'Akaike **(AIC)** ou le critère d'information bayésien **(BIC)**.
3. Automatisation: La méthode est automatisée, ce qui permet de tester rapidement de nombreuses combinaisons de variables
-----
### 2. LASSO (Least Absolute Shrinkage and Selection Operator)
LASSO est une méthode de régularisation qui effectue à la fois la sélection de variables et la régularisation pour améliorer la précision des prédictions et l'interprétabilité du modèle.
1. Régularisation **L1**: LASSO utilise une pénalisation L1 qui contraint la somme des valeurs absolues des coefficients de régression, ce qui force certains coefficients à être exactement zéro, éliminant ainsi certaines variables .
2. Réduction de la Variance: En réduisant la complexité du modèle, LASSO aide à prévenir le sur-ajustement (overfitting) .
3. Applications Diverses: Bien que principalement utilisé pour la régression linéaire, LASSO peut être étendu à d'autres modèles statistiques comme les modèles linéaires généralisés .
-----
### 3. Ridge (Ridge Regression)
La régression Ridge est une méthode de régularisation qui ajoute une pénalité aux coefficients de régression pour réduire leur magnitude et ainsi limiter la complexité du modèle.
1. Régularisation **L2**: Ridge utilise une pénalisation L2 qui ajoute le carré des coefficients à la fonction de coût, ce qui réduit la magnitude des coefficients **mais ne les force pas à être exactement zéro** .
2. Gestion de la Multicolinéarité: Ridge est particulièrement utile pour *traiter les problèmes de multicolinéarité* où les variables indépendantes sont fortement corrélées .
3. Stabilité des Estimations: En réduisant la variance des estimations, Ridge améliore la stabilité et la robustesse du modèle .

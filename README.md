# Prédiction de l'attrition bancaire (churn)

Projet de data science qui prédit si un client va quitter la banque, à partir de données synthétiques.

## Ce que fait le projet

- Génère 2 000 clients avec un taux de départ réaliste (~20-25 %)
- Cible tirée de façon probabiliste (Bernoulli), donc non triviale à prédire
- Compare 3 modèles par validation croisée (ROC-AUC) : régression logistique, Random Forest, Gradient Boosting
- Gère le déséquilibre de classes (`class_weight="balanced"`)
- Produit un tableau de bord : matrice de confusion, coefficients, courbes ROC, distribution des probabilités

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python churn_prediction.py
python churn_prediction.py --n-samples 5000 --output dashboard.png --no-show
```

## Limites

Les données sont synthétiques : seules `Age`, `Balance`, `IsActiveMember` et `NumOfProducts` influencent le départ. Les résultats illustrent la démarche, pas la performance sur un cas réel. Une piste d'évolution est d'utiliser le dataset Kaggle « Churn Modelling » (avec `Geography` et `Gender`).

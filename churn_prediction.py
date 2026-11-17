"""
Prédiction de l'attrition (churn) bancaire sur données synthétiques.

Améliorations par rapport à la v1 :
- cible tirée de façon probabiliste (loi de Bernoulli) au lieu d'un seuil déterministe
- déséquilibre de classes réaliste (~20 % de départs) géré via class_weight
- métriques adaptées : ROC-AUC, precision/recall de la classe "Parti"
- comparaison de 3 modèles avec validation croisée stratifiée
- tableau de bord Matplotlib à 4 graphiques
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def generate_synthetic_bank_data(n_samples=2000, random_state=RANDOM_STATE):
    """Génère un jeu de données bancaire synthétique pour simuler l'attrition."""
    rng = np.random.default_rng(random_state)

    df = pd.DataFrame({
        "CreditScore": rng.integers(350, 850, size=n_samples),
        "Age": rng.integers(18, 70, size=n_samples),
        "Tenure": rng.integers(0, 10, size=n_samples),  # ancienneté (années)
        "Balance": rng.uniform(0, 150_000, size=n_samples),
        "NumOfProducts": rng.integers(1, 4, size=n_samples),
        "HasCrCard": rng.integers(0, 2, size=n_samples),
        "IsActiveMember": rng.integers(0, 2, size=n_samples),
        "EstimatedSalary": rng.uniform(20_000, 180_000, size=n_samples),
    })

    # Logique métier : seules 4 variables influencent réellement le départ.
    # CreditScore, Tenure, HasCrCard et EstimatedSalary sont du "bruit".
    z = (
        -1.0                                         # intercept -> ~20 % de départs
        + 0.05 * (df["Age"] - 40)
        - 0.000004 * df["Balance"]
        - 1.2 * df["IsActiveMember"]
        + 1.0 * (df["NumOfProducts"] > 2).astype(int)
        + rng.normal(0, 0.5, size=n_samples)         # bruit individuel
    )
    probabilities = 1 / (1 + np.exp(-z))
    # Tirage de Bernoulli : le résultat reste incertain, comme dans la réalité
    df["Exited"] = rng.binomial(1, probabilities)
    return df


def build_models():
    """Retourne les modèles à comparer (la normalisation est incluse dans le pipeline)."""
    return {
        "Régression logistique": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def main(n_samples, output_filename, show):
    print("--- 1. Génération et préparation des données ---")
    df = generate_synthetic_bank_data(n_samples=n_samples)
    print(f"Nombre de clients : {len(df)} | Taux de départ global : {df['Exited'].mean() * 100:.2f}%\n")

    X = df.drop(columns=["Exited"])
    y = df["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("--- 2. Comparaison des modèles (validation croisée 5 plis, ROC-AUC) ---")
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        print(f"{name:<24} AUC = {scores.mean():.3f} (+/- {scores.std():.3f})")

    print("\n--- 3. Évaluation sur le jeu de test ---")
    roc_data = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        roc_data[name] = (*roc_curve(y_test, proba)[:2], auc)
        print(f"{name:<24} AUC test = {auc:.3f}")

    # Modèle de référence pour l'interprétation : la régression logistique
    best_name = "Régression logistique"
    best_model = models[best_name]
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print(f"\nRapport de classification ({best_name}, seuil 0.5) :")
    print(classification_report(y_test, y_pred, target_names=["Resté", "Parti"]))

    print("--- 4. Génération des visualisations ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Resté (0)", "Parti (1)"]).plot(
        ax=axes[0, 0], cmap="Blues", colorbar=False
    )
    axes[0, 0].set_title(f"1. Matrice de confusion ({best_name})", fontweight="bold")

    # 2. Coefficients (variables standardisées => comparables entre eux)
    coefs = pd.Series(
        best_model.named_steps["logisticregression"].coef_[0], index=X.columns
    ).sort_values()
    colors = ["crimson" if c > 0 else "teal" for c in coefs]
    axes[0, 1].barh(coefs.index, coefs.values, color=colors, alpha=0.8)
    axes[0, 1].axvline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("2. Impact des variables (rouge = favorise le départ)", fontweight="bold")
    axes[0, 1].set_xlabel("Coefficient de régression logistique")
    axes[0, 1].grid(axis="x", linestyle="--", alpha=0.6)

    # 3. Courbes ROC des trois modèles
    for name, (fpr, tpr, auc) in roc_data.items():
        axes[1, 0].plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    axes[1, 0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Hasard")
    axes[1, 0].set_title("3. Courbes ROC", fontweight="bold")
    axes[1, 0].set_xlabel("Taux de faux positifs")
    axes[1, 0].set_ylabel("Taux de vrais positifs")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    # 4. Distribution des probabilités prédites selon la réalité
    axes[1, 1].hist(y_prob[y_test == 0], bins=20, alpha=0.6, label="Clients restés", color="forestgreen")
    axes[1, 1].hist(y_prob[y_test == 1], bins=20, alpha=0.6, label="Clients partis", color="crimson")
    axes[1, 1].axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Seuil (0.5)")
    axes[1, 1].set_title("4. Distribution des probabilités prédites", fontweight="bold")
    axes[1, 1].set_xlabel("Probabilité prédite de départ")
    axes[1, 1].set_ylabel("Nombre de clients")
    axes[1, 1].legend(loc="upper center")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Graphique sauvegardé sous : '{output_filename}'")
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédiction de churn bancaire (données synthétiques)")
    parser.add_argument("--n-samples", type=int, default=2000, help="Nombre de clients générés")
    parser.add_argument("--output", default="churn_analysis_dashboard.png", help="Fichier image de sortie")
    parser.add_argument("--no-show", action="store_true", help="Ne pas afficher la figure à l'écran")
    args = parser.parse_args()
    main(args.n_samples, args.output, show=not args.no_show)

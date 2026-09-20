# Explainable AI & Predictive Modeling with SHAP

**Python • scikit-learn • Random Forest • SHAP • Cross-Validation • Model Auditing**

An end-to-end explainable-AI case study that validates a predictive model **before** interpreting it, then tests whether the resulting explanations are internally faithful, stable, and appropriately bounded.

[Portfolio](https://roxannelyle.github.io) • [GitHub Profile](https://github.com/roxannelyle)

---

## Project Snapshot

| Item | Result |
|---|---|
| Dataset | California Housing |
| Observations | 20,640 |
| Predictors | 8 numeric features |
| Primary model | Random Forest Regressor, 100 trees |
| Evaluation design | 80/20 holdout + 5-fold CV on training data |
| Held-out MAE | **0.3275** |
| Held-out RMSE | **0.5053** |
| Held-out R² | **0.8051** |
| 5-fold CV R² | **0.8045 ± 0.0068** |
| Maximum SHAP reconstruction error | **2.21 × 10⁻¹³** |
| Highest global SHAP feature | **MedInc — 38.09% of total mean absolute attribution** |

## Why This Project Matters

Explainability is only useful when it is paired with model validation and careful interpretation. A SHAP plot can faithfully explain a prediction even when the prediction itself is wrong.

This project therefore treats explainability as an **audit layer**, not as proof that a model is accurate, causal, fair, or ready for deployment.

The workflow asks four practical questions:

1. Does the Random Forest outperform simple predictive baselines?
2. What features drive predictions globally and locally?
3. Are the explanation patterns stable across methods and Random Forest seeds?
4. Where should interpretation stop because the evidence does not support a stronger claim?

---

## 1. Predictive Validation Before Explanation

The Random Forest substantially outperformed both a mean Dummy Regressor and Linear Regression on the held-out test set.

- **MAE:** 0.3275
- **RMSE:** 0.5053
- **R²:** 0.8051
- **RMSE reduction vs. mean baseline:** 55.86%
- **RMSE reduction vs. Linear Regression:** 32.22%

Five-fold cross-validation produced similar performance:

- **Mean CV RMSE:** 0.5111 ± 0.0128
- **Mean CV R²:** 0.8045 ± 0.0068

The similarity between holdout and cross-validation results supports the model comparison within this random-fold design, while not establishing geographic or temporal generalization.

### Error behavior

![Residual and error analysis](images/residual_error_analysis.png)

Error was not uniform across the target distribution. The highest-value quartile had the largest error and was underpredicted on average, showing why overall metrics alone are not enough.

---

## 2. Global SHAP Explanation

![Global SHAP feature importance](images/global_shap_importance.png)

`MedInc` was the strongest global SHAP feature:

- **MedInc:** 38.09%
- **Latitude:** 18.06%
- **AveOccup:** 16.44%
- **Longitude:** 14.69%

Together, those four features represented approximately **87.3%** of total mean absolute SHAP attribution.

These percentages describe **model-attribution magnitude**. They are not causal percentages of real-world housing value.

### Direction and observation-level variation

![SHAP summary plot](images/shap_summary_direction.png)

The SHAP summary distribution shows both contribution magnitude and direction. Higher `MedInc` values generally pushed predictions upward relative to the model baseline, while lower values pushed them downward.

---

## 3. Feature Dependence

![Median income SHAP dependence](images/medinc_dependence.png)

The Spearman correlation between `MedInc` and its own SHAP contribution was **0.9811**, indicating a strong monotonic association inside the fitted Random Forest.

The relationship is nonlinear and appears to flatten at the upper end of the observed income range. This is evidence about **model behavior**, not a causal intervention effect.

---

## 4. Local Explanations: Fidelity Is Not Accuracy

Three test observations were selected systematically:

- a typical-error case,
- the largest underprediction,
- the largest overprediction.

### Typical-error case

![Typical-error local SHAP waterfall](images/local_typical_error_waterfall.png)

### Largest underprediction

![Largest underprediction local SHAP waterfall](images/local_largest_underprediction_waterfall.png)

### Largest overprediction

![Largest overprediction local SHAP waterfall](images/local_largest_overprediction_waterfall.png)

The two most extreme prediction errors were each wrong by more than **3.04 target units**, yet SHAP still decomposed the model output correctly. This is the clearest practical lesson in the project:

> **A faithful explanation can explain a bad prediction.**

---

## 5. Explanation Fidelity

SHAP additivity was tested directly.

- Mean absolute reconstruction error: approximately **1.82 × 10⁻¹⁴**
- Maximum reconstruction error: approximately **2.21 × 10⁻¹³**

At ordinary numerical precision, the SHAP baseline plus feature contributions reproduced the Random Forest predictions.

That validates the explanation arithmetic for the executed explainer. It does **not** validate the real-world correctness, fairness, or causality of the model.

---

## 6. Comparing Importance Methods

![Comparison of feature-importance methods](images/importance_method_comparison.png)

SHAP was compared with:

- Random Forest impurity importance
- permutation importance

Rank agreement was high:

- SHAP vs. permutation: **Spearman 0.9762**
- SHAP vs. Random Forest importance: **Spearman 0.9524**

The methods broadly agreed on the most important features, but secondary rankings differed because each method defines “importance” differently.

---

## 7. Feature Dependence and Attribution Limits

![Feature correlation matrix](images/feature_correlation_matrix.png)

Two strong correlations were especially important:

- `Latitude` vs. `Longitude`: **r = −0.9245**
- `AveRooms` vs. `AveBedrms`: **r = 0.8362**

Correlated predictors can complicate how attribution is distributed among features. SHAP remains useful here, but individual feature credit should not be treated as uniquely or causally assigned.

---

## 8. Explanation Stability

The global SHAP ranking was tested across Random Forest seeds **7, 42, and 101** using the same data split and a fixed 1,000-row explanation sample.

- Held-out R² ranged from **0.8048 to 0.8067**
- Global feature ranks were identical across all three seeds
- Pairwise Spearman rank correlation was **1.0000**

This supports seed-level ranking stability under the tested setup. It does not establish stability across different data samples, model families, geographic regions, or SHAP dependence assumptions.

---

## Technical Stack

- Python
- pandas
- NumPy
- scikit-learn
- SHAP
- Matplotlib
- Jupyter / Google Colab

---

## Reproducibility

The executed notebook recorded:

- NumPy 2.1.3
- pandas 2.2.3
- scikit-learn 1.6.1
- SHAP 0.52.0

The California Housing dataset is loaded directly through `sklearn.datasets.fetch_california_housing()`, so no proprietary dataset is included in this repository.

To reproduce the workflow:

```bash
pip install -r requirements.txt
jupyter notebook explainable_ai_shap_case_study.ipynb
```

---

## Key Limitations

This analysis does **not** establish:

- causal housing determinants,
- current California market validity,
- fairness,
- regulatory compliance,
- deployment readiness,
- generalization to unseen geographic regions.

A stronger next validation phase would include geographic holdouts, alternative model families, explanation-stability comparison across resampled datasets, and carefully specified SHAP dependence assumptions.

---

## Repository Contents

```text
explainable-ai-shap/
├── README.md
├── explainable_ai_shap_case_study.ipynb
├── requirements.txt
├── .gitignore
└── images/
    ├── residual_error_analysis.png
    ├── global_shap_importance.png
    ├── shap_summary_direction.png
    ├── medinc_dependence.png
    ├── local_typical_error_waterfall.png
    ├── local_largest_underprediction_waterfall.png
    ├── local_largest_overprediction_waterfall.png
    ├── importance_method_comparison.png
    └── feature_correlation_matrix.png
```

## Author

**Roxanne Lyle**  
AI & Data Analytics • Python • Machine Learning • NLP • Responsible AI  
[Portfolio](https://roxannelyle.github.io) • [LinkedIn](https://www.linkedin.com/in/roxannelyle)

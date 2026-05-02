"""Glossary data for the Streamlit terminology tab."""

from __future__ import annotations

from typing import TypeAlias, TypedDict

ResourceLink: TypeAlias = tuple[str, str]


class TerminologyEntry(TypedDict):
    """One glossary entry with short explanations and external resources."""

    plain: str
    in_app: str
    links: list[ResourceLink]


TERMINOLOGY: dict[str, TerminologyEntry] = {
    "Surrogate model": {
        "plain": (
            "A fast approximation of an expensive process, such as a simulation "
            "solver, physical test, or optimization loop."
        ),
        "in_app": (
            "Here, the surrogate predicts airfoil noise from five experimental "
            "inputs instead of running a wind-tunnel experiment."
        ),
        "links": [
            (
                "Wikipedia: Surrogate model",
                "https://en.wikipedia.org/wiki/Surrogate_model",
            ),
        ],
    },
    "Random forest regressor": {
        "plain": (
            "A model that averages many decision trees. Each tree sees a "
            "slightly different view of the training data, which makes the "
            "average more stable than one tree."
        ),
        "in_app": (
            "The random forest is the default trust-panel model because tree "
            "disagreement gives a useful uncertainty proxy."
        ),
        "links": [
            (
                "IBM: What is random forest?",
                "https://www.ibm.com/think/topics/random-forest",
            ),
            (
                "scikit-learn: Random forests",
                "https://scikit-learn.org/stable/modules/ensemble.html"
                "#forests-of-randomized-trees",
            ),
        ],
    },
    "Gradient boosting": {
        "plain": (
            "A model built in stages. Each stage adds a new small model that "
            "focuses on the previous stage's remaining errors."
        ),
        "in_app": (
            "The gradient boosting model is included because boosted trees are a "
            "common high-performing choice for tabular regression."
        ),
        "links": [
            (
                "IBM: What is gradient boosting?",
                "https://www.ibm.com/think/topics/gradient-boosting",
            ),
            (
                "scikit-learn: Gradient-boosted trees",
                "https://scikit-learn.org/stable/modules/ensemble.html"
                "#gradient-boosted-trees",
            ),
        ],
    },
    "Linear regression": {
        "plain": (
            "A model that predicts a number from a weighted sum of input "
            "features. It is easy to explain and useful as a baseline."
        ),
        "in_app": (
            "If linear regression is much weaker than tree models, the data has "
            "nonlinear structure that a straight-line model cannot capture well."
        ),
        "links": [
            (
                "Google ML Crash Course: Linear regression",
                "https://developers.google.com/machine-learning/crash-course/"
                "linear-regression",
            ),
            (
                "scikit-learn: Linear models",
                "https://scikit-learn.org/stable/modules/linear_model.html",
            ),
        ],
    },
    "MAE": {
        "plain": (
            "Mean absolute error. Average absolute difference between predicted "
            "and actual values. In this app, it is measured in dB."
        ),
        "in_app": "If MAE is 2.0, the prediction is off by about 2 dB on average.",
        "links": [
            (
                "Google ML Crash Course: Loss",
                "https://developers.google.com/machine-learning/crash-course/"
                "linear-regression/loss",
            ),
            (
                "scikit-learn: Regression metrics",
                "https://scikit-learn.org/stable/modules/model_evaluation.html"
                "#regression-metrics",
            ),
        ],
    },
    "RMSE": {
        "plain": (
            "Root mean squared error. Similar to MAE, but large errors count "
            "more heavily because errors are squared before averaging."
        ),
        "in_app": (
            "RMSE being much larger than MAE is a hint that a model has some "
            "large misses, not just small steady errors."
        ),
        "links": [
            (
                "scikit-learn: Regression metrics",
                "https://scikit-learn.org/stable/modules/model_evaluation.html"
                "#regression-metrics",
            ),
            (
                "Wikipedia: Root-mean-square deviation",
                "https://en.wikipedia.org/wiki/Root-mean-square_deviation",
            ),
        ],
    },
    "R2": {
        "plain": (
            "Coefficient of determination. Roughly, how much of the target's "
            "variation the model explains. 1.0 is perfect; 0.0 means it is no "
            "better than predicting the average."
        ),
        "in_app": (
            "A high R2 on a random split can still be misleading if the model "
            "fails on a physical holdout."
        ),
        "links": [
            (
                "scikit-learn: R2 score",
                "https://scikit-learn.org/stable/modules/model_evaluation.html"
                "#r2-score",
            ),
            (
                "Wikipedia: Coefficient of determination",
                "https://en.wikipedia.org/wiki/Coefficient_of_determination",
            ),
        ],
    },
    "Scaler / StandardScaler": {
        "plain": (
            "A preprocessing step that puts numeric features on comparable "
            "scales, usually by subtracting the mean and dividing by standard "
            "deviation."
        ),
        "in_app": (
            "Scaling matters for nearest-neighbor distance: otherwise frequency "
            "in Hz would dominate tiny meter-valued features."
        ),
        "links": [
            (
                "scikit-learn: Standardization",
                "https://scikit-learn.org/stable/modules/preprocessing.html"
                "#standardization-or-mean-removal-and-variance-scaling",
            ),
            (
                "Google ML Crash Course: Numerical data",
                "https://developers.google.com/machine-learning/crash-course/"
                "numerical-data",
            ),
        ],
    },
    "Train/test split": {
        "plain": (
            "Training data is used to fit the model. Test data is held back so "
            "model behavior can be estimated on examples it did not see."
        ),
        "in_app": (
            "The random split measures interpolation-like performance. The "
            "physical holdout asks a tougher extrapolation question."
        ),
        "links": [
            (
                "scikit-learn: Cross-validation",
                "https://scikit-learn.org/stable/modules/cross_validation.html",
            ),
            (
                "Google ML Crash Course: Datasets, generalization, overfitting",
                "https://developers.google.com/machine-learning/crash-course/"
                "overfitting/overfitting",
            ),
        ],
    },
    "Generalization": {
        "plain": (
            "The ability of a model to work on new examples that are similar to "
            "the training data."
        ),
        "in_app": (
            "Random test performance is a first approximation of "
            "generalization, but it may overstate performance for new physical "
            "regimes."
        ),
        "links": [
            (
                "Google ML Crash Course: Generalization",
                "https://developers.google.com/machine-learning/crash-course/"
                "overfitting/generalization",
            ),
        ],
    },
    "Extrapolation": {
        "plain": (
            "Prediction outside the region covered by training data. This is "
            "where engineering surrogates become risky."
        ),
        "in_app": (
            "Testing only on high angles of attack shows how performance changes "
            "when the model is pushed into a less familiar region."
        ),
        "links": [
            ("Wikipedia: Extrapolation", "https://en.wikipedia.org/wiki/Extrapolation"),
        ],
    },
    "Out-of-distribution input": {
        "plain": (
            "An input that is meaningfully different from what the model saw "
            "during training."
        ),
        "in_app": (
            "The trust panel flags values outside the training min/max and "
            "values far from the nearest training examples."
        ),
        "links": [
            (
                "Google ML Crash Course: Generalization",
                "https://developers.google.com/machine-learning/crash-course/"
                "overfitting/generalization",
            ),
        ],
    },
    "Residual": {
        "plain": (
            "The difference between the actual value and the model prediction. "
            "Residual plots show patterns in errors."
        ),
        "in_app": (
            "A residual histogram shows whether errors are centered and whether "
            "there are long tails."
        ),
        "links": [
            (
                "Wikipedia: Errors and residuals",
                "https://en.wikipedia.org/wiki/Errors_and_residuals",
            ),
        ],
    },
    "Nearest-neighbor distance": {
        "plain": (
            "Distance from a new input to the closest training examples after "
            "scaling the features."
        ),
        "in_app": (
            "A large distance means the input is in a sparse region, even if "
            "each individual value is technically inside the min/max range."
        ),
        "links": [
            (
                "scikit-learn: Nearest Neighbors",
                "https://scikit-learn.org/stable/modules/neighbors.html",
            ),
        ],
    },
    "Uncertainty proxy": {
        "plain": (
            "A practical signal that hints at confidence, without pretending to "
            "be a rigorous uncertainty estimate."
        ),
        "in_app": (
            "Tree disagreement in the random forest is used as a quick proxy: "
            "if trees disagree more, treat the prediction more carefully."
        ),
        "links": [
            (
                "scikit-learn: Probability calibration",
                "https://scikit-learn.org/stable/modules/calibration.html",
            ),
        ],
    },
}

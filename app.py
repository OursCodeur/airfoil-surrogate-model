"""Streamlit app: surrogate-model workflow for airfoil noise.

This app makes the full surrogate-model workflow visible: data inspection,
model choice, evaluation, extrapolation, and trust signals.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, TypeAlias, TypedDict, TypeVar
from urllib.request import urlopen

# The sandboxed environment may not allow writes to ~/.config/matplotlib.
# Setting this before importing matplotlib keeps cache files local to the app.
MATPLOTLIB_CONFIG_ENV_VAR = "".join(("MPL", "CONFIG", "DIR"))
os.environ.setdefault(
    MATPLOTLIB_CONFIG_ENV_VAR,
    str(Path(__file__).parent / ".matplotlib-cache"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from streamlit.runtime.scriptrunner import get_script_run_ctx

from terminology import TERMINOLOGY

FittedModel: TypeAlias = (
    Pipeline | RandomForestRegressor | HistGradientBoostingRegressor
)
ModelFactory: TypeAlias = Callable[[], FittedModel]
CachedFunction = TypeVar("CachedFunction", bound=Callable[..., Any])


def cache_data_in_streamlit(
    **cache_options: Any,
) -> Callable[[CachedFunction], CachedFunction]:
    """Use Streamlit caching in the UI without noisy plain-Python imports.

    Streamlit logs "bare mode" cache warnings when a module with cached
    functions is imported outside `streamlit run`. The pure functions remain
    useful for command-line smoke tests, so this wrapper only applies
    `st.cache_data` when the script is running under Streamlit.
    """

    def passthrough(function: CachedFunction) -> CachedFunction:
        return function

    if get_script_run_ctx(suppress_warning=True) is None:
        return passthrough

    return st.cache_data(**cache_options)


def cache_resource_in_streamlit(
    **cache_options: Any,
) -> Callable[[CachedFunction], CachedFunction]:
    """Use Streamlit resource caching only when running under Streamlit."""

    def passthrough(function: CachedFunction) -> CachedFunction:
        return function

    if get_script_run_ctx(suppress_warning=True) is None:
        return passthrough

    return st.cache_resource(**cache_options)


class SliderConfig(TypedDict):
    """Display settings for one numeric sidebar control."""

    step: float
    format: str


# The UCI Airfoil Self-Noise dataset is small enough to load quickly. The app
# prefers a local copy so it can run offline after setup, but it can also fetch
# the dataset directly from UCI if the file is missing.
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/"
    "airfoil_self_noise.dat"
)
UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/291/airfoil+self+noise"
LOCAL_DATA_PATH = Path(__file__).parent / "data" / "airfoil_self_noise.dat"


FEATURE_COLUMNS = [
    "frequency_hz",
    "angle_of_attack_deg",
    "chord_length_m",
    "free_stream_velocity_mps",
    "suction_side_displacement_thickness_m",
]
TARGET_COLUMN = "sound_pressure_level_db"


# Human-readable labels keep the UI clearer while the code still uses stable
# column names that are easy to pass through scikit-learn.
FEATURE_LABELS = {
    "frequency_hz": "Frequency (Hz)",
    "angle_of_attack_deg": "Angle of attack (deg)",
    "chord_length_m": "Chord length (m)",
    "free_stream_velocity_mps": "Free-stream velocity (m/s)",
    "suction_side_displacement_thickness_m": (
        "Suction-side displacement thickness (m)"
    ),
}


# Sidebar labels need to stay short enough to avoid awkward wrapping in the
# control column. The full names remain available in the data tables.
SLIDER_LABELS = {
    "frequency_hz": "Frequency (Hz)",
    "angle_of_attack_deg": "Angle of attack (deg)",
    "chord_length_m": "Chord (m)",
    "free_stream_velocity_mps": "Velocity (m/s)",
    "suction_side_displacement_thickness_m": "Displacement thickness (m)",
}


FEATURE_EXPLANATIONS = {
    "frequency_hz": (
        "Acoustic frequency measured in the wind-tunnel experiment. The target "
        "is sound pressure level at this frequency."
    ),
    "angle_of_attack_deg": (
        "Angle between incoming flow and the airfoil chord line. Higher angles "
        "can create more separation and different noise behavior."
    ),
    "chord_length_m": (
        "Characteristic airfoil length. It changes the scale of the flow over "
        "the airfoil."
    ),
    "free_stream_velocity_mps": (
        "Incoming air speed before it reaches the airfoil. This is one of the "
        "main physical drivers of aerodynamic noise."
    ),
    "suction_side_displacement_thickness_m": (
        "A boundary-layer quantity on the suction side of the airfoil. In plain "
        "terms, it captures something about how much the near-wall flow has "
        "been slowed by viscosity."
    ),
}


# Slider ranges can extend beyond the observed data so extrapolation is easy to
# inspect, but physical quantities that cannot be negative stay non-negative.
FEATURE_LOWER_BOUNDS = {
    "frequency_hz": 0.0,
    "angle_of_attack_deg": 0.0,
    "chord_length_m": 0.0,
    "free_stream_velocity_mps": 0.0,
    "suction_side_displacement_thickness_m": 0.0,
}


SLIDER_CONFIG: dict[str, SliderConfig] = {
    "frequency_hz": {"step": 1.0, "format": "%.0f"},
    "angle_of_attack_deg": {"step": 0.1, "format": "%.1f"},
    "chord_length_m": {"step": 0.001, "format": "%.3f"},
    "free_stream_velocity_mps": {"step": 0.1, "format": "%.1f"},
    "suction_side_displacement_thickness_m": {"step": 0.001, "format": "%.3f"},
}


# All matplotlib plots use the same height so side-by-side charts line up.
STANDARD_FIGURE_SIZE = (7.2, 4.4)


MODEL_NOTES = {
    "Linear regression": (
        "A simple baseline. It tries to fit one weighted sum of the inputs. If "
        "it performs poorly, that is a useful sign that the relationship is not "
        "well described by a straight line."
    ),
    "Random forest": (
        "A collection of decision trees trained with randomness. It is strong "
        "on small tabular datasets and gives a simple uncertainty proxy by "
        "looking at disagreement across trees."
    ),
    "Gradient boosting": (
        "A sequence of small trees where each new tree tries to correct the "
        "errors left by the previous ones. Often a strong default for tabular "
        "regression."
    ),
}


class EvaluationResult(NamedTuple):
    """Container for a model's predictions and metrics on one evaluation split."""

    model_name: str
    split_name: str
    y_true: pd.Series
    y_pred: np.ndarray
    x_test: pd.DataFrame
    metrics: dict[str, float]


class Experiment(TypedDict):
    """All fitted objects and evaluation results used by the Streamlit UI."""

    random_models: dict[str, FittedModel]
    random_results: dict[str, EvaluationResult]
    random_train: pd.DataFrame
    random_y_train: pd.Series
    holdout_models: dict[str, FittedModel]
    holdout_results: dict[str, EvaluationResult]
    holdout_threshold: float
    holdout_train_size: int
    holdout_test_size: int


class TrustSummary(TypedDict):
    """Trust indicators for one interactive prediction."""

    prediction: float
    tree_std: float
    nearest_distance: float
    sparse_threshold: float
    sparse_region: bool
    out_of_range_features: list[str]
    recommendation: str
    bounds: pd.DataFrame


def model_factories(random_state: int) -> dict[str, ModelFactory]:
    """Return fresh model instances.

    scikit-learn estimators are stateful after fitting. Returning factories
    instead of pre-built objects prevents accidental reuse between the random
    split and the physical holdout split.
    """

    return {
        "Linear regression": lambda: Pipeline(
            steps=[
                # Linear regression does not require scaling mathematically, but
                # the scaled pipeline reflects a common real-world workflow and
                # makes coefficients easier to compare.
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Random forest": lambda: RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient boosting": lambda: HistGradientBoostingRegressor(
            max_iter=350,
            learning_rate=0.045,
            max_leaf_nodes=31,
            random_state=random_state,
        ),
    }


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute common regression metrics.

    MAE and RMSE are in the same unit as the target, so here they are dB.
    R2 is unitless and easier to compare across datasets.
    """

    actual = y_true.to_numpy(dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residuals = predicted - actual

    mae = float(np.mean(np.abs(residuals)))
    root_mean_squared_error = float(np.sqrt(np.mean(np.square(residuals))))

    total_variance = float(np.sum(np.square(actual - np.mean(actual))))
    residual_variance = float(np.sum(np.square(residuals)))
    r2 = 1.0 - residual_variance / total_variance if total_variance else 0.0

    return {"MAE": mae, "RMSE": root_mean_squared_error, "R2": r2}


def airfoil_dataframe_from_array(data: np.ndarray) -> pd.DataFrame:
    """Convert the numeric airfoil table into a named DataFrame."""

    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    return pd.DataFrame(data, columns=columns)


def read_airfoil_data() -> pd.DataFrame:
    """Load the airfoil dataset from disk, falling back to UCI if needed."""

    if LOCAL_DATA_PATH.exists():
        return airfoil_dataframe_from_array(np.loadtxt(LOCAL_DATA_PATH))

    # Fallback for a fresh checkout where data/ was not downloaded. Streamlit
    # will show a clear error if there is no network access.
    with urlopen(DATA_URL, timeout=20) as response:
        remote_data = np.loadtxt(response)
    return airfoil_dataframe_from_array(remote_data)


@cache_data_in_streamlit(show_spinner=False)
def load_airfoil_data() -> pd.DataFrame:
    """Cached Streamlit wrapper around the pure dataset loader."""

    return read_airfoil_data()


def fit_and_evaluate(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    split_name: str,
    random_state: int,
) -> tuple[dict[str, FittedModel], dict[str, EvaluationResult]]:
    """Fit all models and return both fitted models and evaluation results."""

    fitted_models: dict[str, FittedModel] = {}
    results: dict[str, EvaluationResult] = {}

    for model_name, make_model in model_factories(random_state).items():
        model = make_model()
        model.fit(x_train, y_train)
        predictions = np.asarray(model.predict(x_test), dtype=float)

        fitted_models[model_name] = model
        results[model_name] = EvaluationResult(
            model_name=model_name,
            split_name=split_name,
            y_true=y_test,
            y_pred=predictions,
            x_test=x_test,
            metrics=compute_metrics(y_test, predictions),
        )

    return fitted_models, results


def physical_holdout_split(
    df: pd.DataFrame,
    holdout_feature: str,
    holdout_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Split out the highest feature values by row rank.

    A strict value threshold can produce an empty test set when many rows share
    the same maximum value. Ranking rows keeps the slider behavior stable while
    still creating a deliberately harder high-region holdout.
    """

    row_count = len(df)
    if row_count < 2:
        raise ValueError("Physical holdout requires at least two rows.")

    requested_test_size = int(np.ceil(row_count * (1.0 - holdout_quantile)))
    test_size = min(max(1, requested_test_size), row_count - 1)

    sorted_rows = df.sort_values(
        holdout_feature,
        ascending=False,
        kind="mergesort",
    )
    holdout_test = sorted_rows.head(test_size).sort_index()
    holdout_train = df.drop(index=holdout_test.index)
    threshold = float(holdout_test[holdout_feature].min())

    return holdout_train, holdout_test, threshold


def run_experiment(
    df: pd.DataFrame,
    random_state: int,
    random_test_size: float,
    holdout_feature: str,
    holdout_quantile: float,
) -> Experiment:
    """Build both the random-split and physical-holdout experiments.

    The random split answers: how well does the model interpolate across a
    shuffled sample of the same distribution?

    The physical holdout answers: how well does the model handle a region of
    the input space that was deliberately excluded from training?
    """

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=random_test_size,
        random_state=random_state,
    )
    random_models, random_results = fit_and_evaluate(
        x_train,
        x_test,
        y_train,
        y_test,
        "Random train/test split",
        random_state,
    )

    # The holdout is rank-based rather than threshold-based. This avoids empty
    # splits for features with repeated maximum values while preserving the
    # intent: hold out the highest physical region for testing.
    holdout_train, holdout_test, threshold = physical_holdout_split(
        df=df,
        holdout_feature=holdout_feature,
        holdout_quantile=holdout_quantile,
    )

    holdout_models, holdout_results = fit_and_evaluate(
        holdout_train[FEATURE_COLUMNS],
        holdout_test[FEATURE_COLUMNS],
        holdout_train[TARGET_COLUMN],
        holdout_test[TARGET_COLUMN],
        f"Physical holdout: high {FEATURE_LABELS[holdout_feature]}",
        random_state,
    )

    return {
        "random_models": random_models,
        "random_results": random_results,
        "random_train": x_train,
        "random_y_train": y_train,
        "holdout_models": holdout_models,
        "holdout_results": holdout_results,
        "holdout_threshold": threshold,
        "holdout_train_size": len(holdout_train),
        "holdout_test_size": len(holdout_test),
    }


@cache_resource_in_streamlit(show_spinner="Training models...")
def build_experiment(
    df: pd.DataFrame,
    random_state: int,
    random_test_size: float,
    holdout_feature: str,
    holdout_quantile: float,
) -> Experiment:
    """Cached Streamlit wrapper around the pure experiment builder."""

    return run_experiment(
        df=df,
        random_state=random_state,
        random_test_size=random_test_size,
        holdout_feature=holdout_feature,
        holdout_quantile=holdout_quantile,
    )


def metrics_table(results: dict[str, EvaluationResult]) -> pd.DataFrame:
    """Format model metrics as a Streamlit-friendly table."""

    rows = []
    for result in results.values():
        rows.append(
            {
                "Model": result.model_name,
                "MAE (dB)": result.metrics["MAE"],
                "RMSE (dB)": result.metrics["RMSE"],
                "R2": result.metrics["R2"],
            }
        )
    return pd.DataFrame(rows).set_index("Model")


def add_prediction_columns(result: EvaluationResult) -> pd.DataFrame:
    """Return a copy of the test set with actual, predicted, and error columns."""

    analysis = result.x_test.copy()
    analysis["actual_db"] = result.y_true
    analysis["predicted_db"] = result.y_pred
    analysis["error_db"] = analysis["predicted_db"] - analysis["actual_db"]
    analysis["absolute_error_db"] = analysis["error_db"].abs()
    return analysis


def render_matplotlib_figure(fig: plt.Figure) -> None:
    """Render a matplotlib figure and close it to avoid memory growth."""

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def render_predicted_vs_actual(result: EvaluationResult) -> None:
    """Scatter plot: perfect predictions would sit on the diagonal line."""

    y_true = result.y_true
    y_pred = result.y_pred

    fig, ax = plt.subplots(figsize=STANDARD_FIGURE_SIZE)
    ax.scatter(y_true, y_pred, alpha=0.65)

    min_value = min(float(y_true.min()), float(np.min(y_pred)))
    max_value = max(float(y_true.max()), float(np.max(y_pred)))
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    ax.set_xlabel("Actual sound pressure level (dB)")
    ax.set_ylabel("Predicted sound pressure level (dB)")
    ax.set_title(f"{result.model_name}: predicted vs actual")
    ax.grid(True, alpha=0.25)
    render_matplotlib_figure(fig)


def render_error_histogram(result: EvaluationResult) -> None:
    """Histogram of signed errors. Zero means perfect prediction."""

    errors = result.y_pred - result.y_true.to_numpy()

    fig, ax = plt.subplots(figsize=STANDARD_FIGURE_SIZE)
    ax.hist(errors, bins=28, edgecolor="black", alpha=0.75)
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("Prediction error (predicted - actual, dB)")
    ax.set_ylabel("Count")
    ax.set_title(f"{result.model_name}: residual distribution")
    ax.grid(True, axis="y", alpha=0.25)
    render_matplotlib_figure(fig)


def format_feature_value(feature: str, value: float) -> str:
    """Format feature values with the same precision as the related slider."""

    number_format = str(SLIDER_CONFIG[feature]["format"])
    return number_format % value


def format_interval_label(feature: str, interval: pd.Interval) -> str:
    """Create compact labels for quantile-binned intervals on plots."""

    return (
        f"{format_feature_value(feature, float(interval.left))}-"
        f"{format_feature_value(feature, float(interval.right))}"
    )


def render_error_by_feature(result: EvaluationResult, feature: str) -> None:
    """Show whether one physical range has systematically worse errors."""

    analysis = add_prediction_columns(result)

    # Quantile binning creates bins with roughly equal counts. Repeated feature
    # values can collapse adjacent edges, so unique edges are used.
    bin_edges = np.unique(np.quantile(analysis[feature], np.linspace(0.0, 1.0, 6)))
    if len(bin_edges) < 2:
        st.info("This feature does not vary enough to create error bins.")
        return

    analysis["feature_bin"] = pd.cut(
        analysis[feature],
        bins=bin_edges,
        include_lowest=True,
    )
    grouped = (
        analysis.groupby("feature_bin", observed=True)["absolute_error_db"]
        .mean()
        .reset_index()
    )
    grouped["feature_bin"] = grouped["feature_bin"].map(
        lambda interval: format_interval_label(feature, interval)
    )

    fig, ax = plt.subplots(figsize=STANDARD_FIGURE_SIZE)
    ax.bar(grouped["feature_bin"], grouped["absolute_error_db"])
    ax.set_xlabel(FEATURE_LABELS[feature])
    ax.set_ylabel("Mean absolute error (dB)")
    ax.set_title("Where does the model make larger errors?")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    render_matplotlib_figure(fig)


def render_feature_ranges(df: pd.DataFrame) -> None:
    """Show min/median/max for the observed design space."""

    rows = []
    for column in FEATURE_COLUMNS:
        rows.append(
            {
                "Feature": FEATURE_LABELS[column],
                "Min": df[column].min(),
                "Median": df[column].median(),
                "Max": df[column].max(),
                "Meaning": FEATURE_EXPLANATIONS[column],
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Meaning": st.column_config.TextColumn(width="large"),
        },
    )


def central_training_point(x_train: pd.DataFrame) -> pd.Series:
    """Return a representative training row near the center of feature space.

    Using an actual training row keeps the initial prediction squarely in an
    interpolation case: the nearest-neighbor distance should start at zero.
    """

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(x_train)
    row_index = int(np.argmin(np.linalg.norm(scaled_train, axis=1)))
    return x_train.iloc[row_index]


def step_decimals(step: float) -> int:
    """Return a safe decimal count for rounding values at a given step."""

    step_text = f"{step:.10f}".rstrip("0")
    if "." not in step_text:
        return 0
    return len(step_text.split(".", maxsplit=1)[1])


def round_to_step(value: float, step: float) -> float:
    """Round a value to the nearest natural slider increment."""

    if step <= 0:
        return value
    return round(round(value / step) * step, step_decimals(step))


def floor_to_step(value: float, step: float) -> float:
    """Round a lower slider bound outward to the nearest step."""

    if step <= 0:
        return value
    return round(np.floor(value / step) * step, step_decimals(step))


def ceil_to_step(value: float, step: float) -> float:
    """Round an upper slider bound outward to the nearest step."""

    if step <= 0:
        return value
    return round(np.ceil(value / step) * step, step_decimals(step))


def build_user_input(
    df: pd.DataFrame,
    default_values: pd.Series,
) -> pd.DataFrame:
    """Create sidebar sliders for a single hypothetical design point."""

    st.sidebar.header("Single prediction")
    st.sidebar.caption(
        "Defaults use a central training example, so the initial state is an "
        "interpolation case."
    )

    values: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        observed_min = float(df[column].min())
        observed_max = float(df[column].max())
        default_value = float(default_values[column])
        slider_config = SLIDER_CONFIG[column]
        slider_step = float(slider_config["step"])

        # The slider extends 15% beyond the observed range, which makes
        # extrapolation cases available without arbitrary typed values. Lower
        # bounds preserve basic physical constraints such as non-negative
        # frequency.
        span = observed_max - observed_min
        slider_min = max(
            FEATURE_LOWER_BOUNDS[column],
            floor_to_step(observed_min - 0.15 * span, slider_step),
        )
        slider_max = ceil_to_step(observed_max + 0.15 * span, slider_step)
        default_value = round_to_step(default_value, slider_step)
        default_value = min(max(default_value, slider_min), slider_max)

        values[column] = st.sidebar.slider(
            SLIDER_LABELS[column],
            min_value=float(slider_min),
            max_value=float(slider_max),
            value=float(default_value),
            step=slider_step,
            format=str(slider_config["format"]),
            help=FEATURE_EXPLANATIONS[column],
            key=f"single_prediction_{column}_v2",
        )

    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def random_forest_tree_predictions(
    model: RandomForestRegressor,
    user_input: pd.DataFrame,
) -> np.ndarray:
    """Predict with every tree in a fitted random forest."""

    # Individual trees inside a RandomForestRegressor are fitted on NumPy arrays
    # rather than DataFrames, so we pass NumPy here to avoid feature-name noise.
    input_array = user_input.to_numpy()
    return np.array([tree.predict(input_array)[0] for tree in model.estimators_])


def trust_signal(
    user_input: pd.DataFrame,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    model: RandomForestRegressor,
) -> tuple[TrustSummary, pd.DataFrame]:
    """Compute simple product-style trust indicators for one prediction.

    This is not rigorous uncertainty quantification. It combines practical
    signals that are easy to explain: range checks, nearest-neighbor distance,
    and random-forest disagreement. Those signals are enough to make model
    trust and limits visible.
    """

    prediction = float(model.predict(user_input)[0])

    tree_predictions = random_forest_tree_predictions(model, user_input)
    tree_std = float(np.std(tree_predictions))

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(x_train)
    scaled_input = scaler.transform(user_input)

    nearest = NearestNeighbors(n_neighbors=5)
    nearest.fit(scaled_train)
    distances, indices = nearest.kneighbors(scaled_input)

    train_min = x_train.min()
    train_max = x_train.max()

    bounds_rows = []
    out_of_range_features = []
    for column in FEATURE_COLUMNS:
        value = float(user_input.iloc[0][column])
        minimum = float(train_min[column])
        maximum = float(train_max[column])
        inside = minimum <= value <= maximum
        if not inside:
            out_of_range_features.append(column)
        bounds_rows.append(
            {
                "Feature": FEATURE_LABELS[column],
                "Input": value,
                "Training min": minimum,
                "Training max": maximum,
                "Inside range": inside,
            }
        )

    neighbor_rows = x_train.iloc[indices[0]].copy()
    neighbor_rows[TARGET_COLUMN] = y_train.iloc[indices[0]].to_numpy()
    neighbor_rows["scaled_distance"] = distances[0]

    # A rough density threshold: if the nearest case is above the 90th
    # percentile of all second-nearest distances in training, call it sparse.
    # We use the second neighbor for training points because the first neighbor
    # of a training point is itself with distance zero.
    train_distances, _ = nearest.kneighbors(scaled_train, n_neighbors=2)
    sparse_threshold = float(np.quantile(train_distances[:, 1], 0.90))
    nearest_distance = float(distances[0][0])
    sparse_region = nearest_distance > sparse_threshold

    if out_of_range_features:
        recommendation = (
            "Outside observed range: validate with simulation or expert review."
        )
    elif sparse_region or tree_std > 2.5:
        recommendation = (
            "Sparse or uncertain region: useful for exploration, but validate "
            "before relying on it."
        )
    else:
        recommendation = (
            "Interpolation-like case: still validate for decisions with real "
            "cost or safety impact."
        )

    summary: TrustSummary = {
        "prediction": prediction,
        "tree_std": tree_std,
        "nearest_distance": nearest_distance,
        "sparse_threshold": sparse_threshold,
        "sparse_region": sparse_region,
        "out_of_range_features": out_of_range_features,
        "recommendation": recommendation,
        "bounds": pd.DataFrame(bounds_rows),
    }

    return summary, neighbor_rows


def render_trust_panel(
    user_input: pd.DataFrame,
    experiment: Experiment,
) -> None:
    """Render the interactive prediction and trust-signal section."""

    model = experiment["random_models"]["Random forest"]
    if not isinstance(model, RandomForestRegressor):
        raise TypeError("Expected the random-forest model for trust signals.")
    x_train = experiment["random_train"]
    y_train = experiment["random_y_train"]

    summary, neighbors = trust_signal(
        user_input=user_input,
        x_train=x_train,
        y_train=y_train,
        model=model,
    )

    st.subheader("Interactive prediction and trust signal")
    st.write(
        "This panel turns a raw model prediction into the kind of product signal "
        "an engineer could reason about."
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Predicted noise", f"{summary['prediction']:.2f} dB")
    col_b.metric("Tree disagreement", f"{summary['tree_std']:.2f} dB")
    col_c.metric("Nearest scaled distance", f"{summary['nearest_distance']:.2f}")

    if summary["out_of_range_features"]:
        readable = [
            FEATURE_LABELS[column] for column in summary["out_of_range_features"]
        ]
        st.error("Out of observed training range: " + ", ".join(readable))
    elif summary["sparse_region"]:
        st.warning(summary["recommendation"])
    else:
        st.info(summary["recommendation"])

    st.markdown("**Range check**")
    st.dataframe(summary["bounds"], width="stretch", hide_index=True)

    st.markdown("**Closest comparable training cases**")
    neighbor_display = neighbors.rename(
        columns={**FEATURE_LABELS, TARGET_COLUMN: "Actual dB"}
    )
    st.dataframe(neighbor_display, width="stretch", hide_index=True)


def render_terminology() -> None:
    """Render glossary entries with resource links."""

    st.subheader("Terminology lookup")
    st.write(
        "Vocabulary map for the concepts used in the workflow. The links favor "
        "established, readable references over deep implementation detail."
    )

    query = st.text_input(
        "Filter terms",
        placeholder="Example: RMSE, scaler, boosting",
    )
    query = query or ""
    normalized_query = query.strip().lower()

    for term, content in TERMINOLOGY.items():
        searchable = " ".join(
            [
                term,
                content["plain"],
                content["in_app"],
                " ".join(label for label, _ in content["links"]),
            ]
        ).lower()

        if normalized_query and normalized_query not in searchable:
            continue

        with st.expander(term):
            st.markdown(f"**Plain meaning:** {content['plain']}")
            st.markdown(f"**In this app:** {content['in_app']}")
            st.markdown("**Look up next:**")
            for label, url in content["links"]:
                st.markdown(f"- [{label}]({url})")


def render_project_notes() -> None:
    """Summarize the technical conclusions from the workflow."""

    st.subheader("Technical notes")
    st.markdown(
        """
**What this workflow demonstrates**

- A surrogate model can replace a slow or expensive process for fast exploration.
- A random train/test score is useful, but it can hide extrapolation risk.
- Error distribution matters more than one headline metric.
- A product should expose trust signals, not just predictions.
- For engineering workflows, model output should be tied to validation,
  domain of validity, and decision risk.
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="Airfoil Surrogate Model",
        page_icon=None,
        layout="wide",
    )

    st.title("Airfoil Surrogate Model")
    st.caption(
        "Surrogate modeling, validation, extrapolation, and trust signals on "
        "airfoil self-noise data."
    )

    df = load_airfoil_data()

    st.sidebar.header("Experiment settings")
    random_state = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=10_000,
        value=42,
        step=1,
    )
    random_test_size = st.sidebar.slider(
        "Random test split",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
    )
    holdout_feature = st.sidebar.selectbox(
        "Physical holdout feature",
        FEATURE_COLUMNS,
        index=1,
        format_func=lambda column: FEATURE_LABELS[column],
    )
    holdout_quantile = st.sidebar.slider(
        "Hold out rows above this quantile",
        min_value=0.60,
        max_value=0.90,
        value=0.80,
        step=0.05,
    )

    experiment = build_experiment(
        df=df,
        random_state=int(random_state),
        random_test_size=float(random_test_size),
        holdout_feature=holdout_feature,
        holdout_quantile=float(holdout_quantile),
    )
    user_input = build_user_input(
        df=df,
        default_values=central_training_point(experiment["random_train"]),
    )

    tabs = st.tabs(
        [
            "1. Data",
            "2. Models",
            "3. Extrapolation",
            "4. Trust signal",
            "5. Terminology",
            "6. Notes",
        ]
    )

    with tabs[0]:
        st.subheader("Dataset")
        st.write(
            "The dataset contains airfoil self-noise measurements from NASA "
            "wind-tunnel experiments. The target is sound pressure level in dB."
        )

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Rows", f"{len(df):,}")
        col_b.metric("Input features", len(FEATURE_COLUMNS))
        col_c.metric("Target", "Sound pressure level")

        st.markdown(f"[UCI dataset page]({UCI_DATASET_PAGE})")
        st.markdown("**Feature ranges**")
        render_feature_ranges(df)

        st.markdown("**Raw sample**")
        st.dataframe(df.head(20), width="stretch")

    with tabs[1]:
        st.subheader("Baseline surrogate models")
        st.write(
            "This first experiment uses a random train/test split. It is a "
            "reasonable baseline, but it can be too optimistic for engineering "
            "use cases where new designs may sit in less familiar regions."
        )

        st.markdown("**Model comparison**")
        st.dataframe(
            metrics_table(experiment["random_results"]).style.format(
                {"MAE (dB)": "{:.3f}", "RMSE (dB)": "{:.3f}", "R2": "{:.3f}"}
            ),
            width="stretch",
        )

        selected_model = st.selectbox(
            "Inspect model",
            list(experiment["random_results"].keys()),
            key="random_model",
        )
        st.info(MODEL_NOTES[selected_model])
        selected_result = experiment["random_results"][selected_model]

        col_left, col_right = st.columns(2)
        with col_left:
            render_predicted_vs_actual(selected_result)
        with col_right:
            render_error_histogram(selected_result)

        feature_for_error = st.selectbox(
            "Show error by feature",
            FEATURE_COLUMNS,
            index=1,
            format_func=lambda column: FEATURE_LABELS[column],
        )
        render_error_by_feature(selected_result, feature_for_error)

        st.markdown("**Worst predictions**")
        worst = (
            add_prediction_columns(selected_result)
            .sort_values("absolute_error_db", ascending=False)
            .head(10)
            .rename(columns={**FEATURE_LABELS})
        )
        st.dataframe(worst, width="stretch")

    with tabs[2]:
        st.subheader("Random split vs physical holdout")
        st.write(
            "A model can look good when train and test data are randomly mixed, "
            "then degrade when the test set is a specific physical region."
        )

        threshold = experiment["holdout_threshold"]
        heldout_percent = 100.0 * (1.0 - float(holdout_quantile))
        st.markdown(
            f"""
**Current holdout**

- Feature: `{FEATURE_LABELS[holdout_feature]}`
- Test region: highest `{heldout_percent:.0f}%` of rows by feature value
- Lowest test-set value: `{threshold:.5g}`
- Training rows: `{experiment["holdout_train_size"]:,}`
- Test rows: `{experiment["holdout_test_size"]:,}`
            """
        )

        random_metrics = metrics_table(experiment["random_results"]).add_prefix(
            "Random "
        )
        holdout_metrics = metrics_table(experiment["holdout_results"]).add_prefix(
            "Holdout "
        )
        comparison = pd.concat([random_metrics, holdout_metrics], axis=1)
        st.dataframe(
            comparison.style.format("{:.3f}"),
            width="stretch",
        )

        holdout_model = st.selectbox(
            "Inspect holdout model",
            list(experiment["holdout_results"].keys()),
            index=1,
            key="holdout_model",
        )
        holdout_result = experiment["holdout_results"][holdout_model]

        col_left, col_right = st.columns(2)
        with col_left:
            render_predicted_vs_actual(holdout_result)
        with col_right:
            render_error_histogram(holdout_result)

        st.markdown(
            """
**Interpretation**

If the holdout score is worse than the random split, that is not a failure of
the exercise. It shows that validation for a surrogate model has to match the
way new designs will actually be explored.
            """
        )

    with tabs[3]:
        render_trust_panel(user_input, experiment)

    with tabs[4]:
        render_terminology()

    with tabs[5]:
        render_project_notes()


if __name__ == "__main__":
    main()

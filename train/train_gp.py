import matplotlib.pyplot as plt
import numpy as np
import GPy

from colorama import Fore
from typing import Any, Sequence

from utils.criterion import Criterion


FEATURE_COLUMNS = slice(2, 6)
OUTPUT_DIMENSIONS = 2
DT = 0.05


def get_pre_processed_data(
    number_of_data: int,
    number_of_window: int,
    path: str,
    gap_window: int,
    select: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a data file and convert a history of measurements into GP features.

    Features are built from [feedback_v, feedback_w, command_v, command_w]
    over the previous `number_of_window - gap_window` samples.

    Returns:
        features: Shape (N, 4 * feature_window), or
                  (N, len(select)) when `select` is provided.
        responses: Shape (N, 2).
        additional_data: Shape (N, 2). Preserved for compatibility with the
                          original function; currently left unpopulated.
    """
    feature_window = number_of_window - gap_window

    if feature_window < 1:
        print(
            Fore.RED + "[ERROR]" + Fore.RESET +
            ": The window size is too small, please retry with a larger windows size!"
        )
        return None, None, None

    data = np.loadtxt(path, delimiter=" ")
    additional_feature_count = 0
    feature_width = 4 + additional_feature_count

    if number_of_data > data.shape[0]:
        number_of_data = data.shape[0] - number_of_window
        print(
            Fore.YELLOW + "[Warning]" + Fore.RESET +
            f": The data {path} has only {data.shape[0]} data!"
        )

    if number_of_data < 0:
        number_of_data = 0

    # Equivalent to the original nested loops:
    # data[i + feature_window - j - 1, 2:6]
    history = np.stack(
        [
            data[
                np.arange(feature_window - 1, -1, -1)
                + i,
                FEATURE_COLUMNS,
            ]
            for i in range(number_of_data)
        ],
        axis=0,
    ) if number_of_data > 0 else np.empty(
        (0, feature_window, feature_width),
        dtype=float,
    )

    features = history.reshape(number_of_data, feature_window * feature_width)
    responses = data[
        np.arange(number_of_data) + number_of_window,
        :OUTPUT_DIMENSIONS,
    ].copy()

    # Kept because the original function returns this value.
    additional_data = np.empty((number_of_data, OUTPUT_DIMENSIONS))

    if select:
        selected_features = features[:, list(select)]
        print(
            "new feature shape: ",
            selected_features.shape,
            ", respond shape: ",
            responses.shape,
        )
        return selected_features, responses, additional_data

    print(
        "feature shape: ",
        features.shape,
        ", respond shape: ",
        responses.shape,
    )
    return features, responses, additional_data


def _get_training_data(param_set: dict[str, Any], select_index: Sequence[int] | None = None):
    """Load training features and select the requested output dimension."""
    features, responses, _ = get_pre_processed_data(
        param_set["train_data_number"],
        param_set["input_window"],
        param_set["train_file_path"],
        param_set["gap_window"],
        select=select_index,
    )
    return features, responses[:, param_set.get("_output_id", 0)]


def _get_test_data(param_set: dict[str, Any], output_id: int, select_index=None):
    """Load test features and select the requested output dimension."""
    features, responses, _ = get_pre_processed_data(
        param_set["test_data_number"],
        param_set["input_window"],
        param_set["test_file_path"],
        param_set["gap_window"],
        select=select_index,
    )
    return features, responses[:, output_id]


def _create_full_kernel(input_dim: int):
    ard_length_scale = np.full(input_dim, 0.5)

    rbf_kernel_1 = GPy.kern.RBF(
        input_dim=input_dim,
        variance=3.08**2,
        lengthscale=ard_length_scale,
        ARD=True,
    )
    rbf_kernel_2 = GPy.kern.RBF(
        input_dim=input_dim,
        variance=0.204**2,
        lengthscale=0.0037,
    )

    return rbf_kernel_1 + rbf_kernel_2


def _create_sparse_kernel(input_dim: int):
    ard_length_scale = np.full(input_dim, 0.5)

    return GPy.kern.RBF(
        input_dim=input_dim,
        variance=3.08**2,
        lengthscale=ard_length_scale,
        ARD=True,
    )


def _predict(model, features: np.ndarray):
    """Return prediction mean and 95% confidence interval width."""
    mu, cov = model.predict(
        features,
        full_cov=True,
        include_likelihood=False,
    )
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    return mu, uncertainty


def _print_full_model_info(model):
    print(model)
    print(
        model["Gaussian_noise.variance"].values[0],
        ", ",
        model["sum.rbf_1.variance"].values[0],
        ", ",
        model["sum.rbf_1.lengthscale"].values[0],
        ", ",
        model["sum.rbf.variance"].values[0],
        ", ",
        ", ".join(str(value) for value in model["sum.rbf.lengthscale"].values),
        sep="",
    )


def _print_sparse_model_info(model):
    print(
        model["Gaussian_noise.variance"].values[0],
        ", ",
        model["rbf.variance"].values[0],
        ", ",
        ", ".join(str(value) for value in model["rbf.lengthscale"].values),
        sep="",
    )


def _print_metrics(
    criterion: Criterion,
    features: np.ndarray,
    responses: np.ndarray,
    prediction: np.ndarray,
    output_id: int,
    dt: float,
    sparse: bool = False,
):
    col_width = 15
    headers = ["Criterion", "No-Predict", "MPC-Predict", "GPR-Predict"]

    print(
        f"{headers[0]:<{col_width}} "
        f"{headers[1]:<{col_width}} "
        f"{headers[2]:<{col_width}} "
        f"{headers[3]:<{col_width}}"
    )

    for criterion_name in ("MAE", "RMSE"):
        no_predict = criterion.compute(
            criterion_name,
            output=features[:, -2 + output_id],
            target=features[:, -6 + output_id],
        )

        if sparse:
            mpc_predict = criterion.compute(
                criterion_name,
                output=responses * dt,
                target=np.zeros(responses.shape),
            )
            gpr_predict = (
                criterion.compute(
                    criterion_name,
                    output=responses.squeeze(),
                    target=prediction.squeeze(),
                )
                * dt
            )
        else:
            mpc_predict = criterion.compute(
                criterion_name,
                output=responses,
                target=np.zeros(responses.shape),
            )
            gpr_predict = (
                criterion.compute(
                    criterion_name,
                    output=responses.squeeze(),
                    target=prediction.squeeze(),
                )
                / dt
            )

        print(
            f"{criterion_name:<{col_width}} "
            f"{no_predict:<{col_width}.6f} "
            f"{mpc_predict:<{col_width}.6f} "
            f"{gpr_predict:<{col_width}.6f}"
        )


def _plot_predictions(prediction, responses, uncertainty):
    indices = np.arange(len(responses))

    plt.plot(prediction, label="predict")
    plt.plot(responses, label="real")
    plt.fill_between(
        indices,
        prediction.reshape(-1) + uncertainty,
        prediction.reshape(-1) - uncertainty,
        alpha=0.2,
    )
    plt.legend()
    plt.show()


def GPy_test(
    param_set: dict[str, Any],
    print_enable: bool,
    plot_enable: bool,
    optimize: bool,
    write_data: bool,
    output_id: int,
):
    criterion = Criterion()
    select_index: list[int] = []
    normalize_y = True
    noise_percent = 1e-2

    # Load training data before any calculation that depends on it.
    features, responses = get_pre_processed_data(
        param_set["train_data_number"],
        param_set["input_window"],
        param_set["train_file_path"],
        param_set["gap_window"],
        select=select_index,
    )[:2]
    responses = responses[:, output_id]

    alpha = (
        (np.std(responses) ** 2 if not normalize_y else 1.0)
        * noise_percent
    )
    _ = alpha  # Preserved from the original implementation; currently unused.

    input_dim = 4 * (
        param_set["input_window"] - param_set["gap_window"]
    )
    kernel = _create_full_kernel(input_dim)

    model = GPy.models.GPRegression(
        features,
        responses.reshape(-1, 1),
        kernel=kernel,
        normalizer=normalize_y,
    )

    if optimize:
        model.optimize_restarts(num_restarts=5)

    test_features, test_responses = get_pre_processed_data(
        param_set["test_data_number"],
        param_set["input_window"],
        param_set["test_file_path"],
        param_set["gap_window"],
        select=select_index,
    )[:2]
    test_responses = test_responses[:, output_id]

    prediction, uncertainty = _predict(model, test_features)

    if print_enable:
        _print_full_model_info(model)
        _print_metrics(
            criterion,
            test_features,
            test_responses,
            prediction,
            output_id,
            DT,
            sparse=False,
        )

    if plot_enable:
        _plot_predictions(prediction, test_responses, uncertainty)

    if write_data:
        feature_size = features.shape[1]
        with open("../../Log/fixed_data/0.txt", "w") as file:
            for row in features[: param_set["train_data_number"], :feature_size]:
                file.write(" ".join(map(str, row)) + "\n")


def Sparse_GPy_test(
    param_set: dict[str, Any],
    print_enable: bool,
    plot_enable: bool,
    optimize: bool,
    write_data: bool,
    output_id: int,
):
    criterion = Criterion()
    select_index: list[int] = []
    normalize_y = False
    sparse_method = "FITC"

    print(
        f"{Fore.BLUE}[INFO]:{Fore.RESET} "
        f"Train the sparse gp with method {sparse_method} "
        f"and normalization is {normalize_y}\n"
    )

    features, responses = get_pre_processed_data(
        param_set["train_data_number"],
        param_set["input_window"],
        param_set["train_file_path"],
        param_set["gap_window"],
        select=select_index,
    )[:2]
    responses = responses[:, output_id]

    input_dim = 4 * (
        param_set["input_window"] - param_set["gap_window"]
    )
    kernel = _create_sparse_kernel(input_dim)

    model = GPy.models.SparseGPRegression(
        features,
        responses.reshape(-1, 1),
        kernel=kernel,
        normalizer=normalize_y,
        num_inducing=param_set["num_inducing"],
    )

    if sparse_method == "FITC":
        model.inference_method = GPy.inference.latent_function_inference.FITC()
        model.parameters_changed()

    if optimize:
        model.optimize_restarts(num_restarts=10, verbose=False)

    if print_enable:
        test_features, test_responses = get_pre_processed_data(
            param_set["test_data_number"],
            param_set["input_window"],
            param_set["test_file_path"],
            param_set["gap_window"],
            select=select_index,
        )[:2]
        test_responses = test_responses[:, output_id]

        prediction, uncertainty = _predict(model, test_features)

        _print_sparse_model_info(model)
        print(
            test_features[:, -2 + output_id].shape,
            test_responses.shape,
            prediction.shape,
        )
        _print_metrics(
            criterion,
            test_features,
            test_responses,
            prediction,
            output_id,
            DT,
            sparse=True,
        )

        if plot_enable:
            _plot_predictions(prediction, test_responses, uncertainty)

    if write_data:
        file_path = param_set["fixed_data_path"]

        with open(file_path, "a") as file:
            for inducing_input in model.inducing_inputs[: param_set["num_inducing"]]:
                file.write(" ".join(map(str, inducing_input)) + "\n")


def write_train_data(param_set: dict[str, Any]):
    file_path = param_set["fixed_data_path"]

    features, responses = get_pre_processed_data(
        param_set["train_data_number"],
        param_set["input_window"],
        param_set["train_file_path"],
        param_set["gap_window"],
    )[:2]

    with open(file_path, "w") as file:
        for feature_row, response_row in zip(features, responses):
            feature_string = " ".join(map(str, feature_row))
            response_string = " ".join(map(str, response_row))
            file.write(f"{feature_string} {response_string}\n")


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "video_features_yolo.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "lab8"


@dataclass
class TrainingResult:
    weights: np.ndarray
    epochs: int
    errors: list[float]
    predictions: np.ndarray


def summation_unit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(inputs, dtype=float) @ np.asarray(weights, dtype=float)


def step_activation(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values) >= 0).astype(float)


def bipolar_step_activation(values: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(values) >= 0, 1.0, -1.0)


def sigmoid_activation(values: np.ndarray) -> np.ndarray:
    clipped_values = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-clipped_values))


def tanh_activation(values: np.ndarray) -> np.ndarray:
    return np.tanh(values)


def relu_activation(values: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, values)


def leaky_relu_activation(values: np.ndarray, slope: float = 0.01) -> np.ndarray:
    return np.where(np.asarray(values) > 0, values, slope * np.asarray(values))


def comparator_unit(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)


def add_bias_column(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    return np.c_[np.ones(features.shape[0]), features]


def activation_by_name(values: np.ndarray, activation_name: str) -> np.ndarray:
    activation_name = activation_name.lower()
    if activation_name == "step":
        return step_activation(values)
    if activation_name == "bipolar_step":
        return bipolar_step_activation(values)
    if activation_name == "sigmoid":
        return sigmoid_activation(values)
    if activation_name == "tanh":
        return tanh_activation(values)
    if activation_name == "relu":
        return relu_activation(values)
    if activation_name == "leaky_relu":
        return leaky_relu_activation(values)
    raise ValueError(f"Unknown activation: {activation_name}")


def binary_predictions(outputs: np.ndarray, activation_name: str) -> np.ndarray:
    if activation_name == "bipolar_step":
        return np.where(outputs >= 0, 1, 0)
    if activation_name == "sigmoid":
        return (outputs >= 0.5).astype(int)
    if activation_name in {"relu", "leaky_relu"}:
        return (outputs >= 0.5).astype(int)
    return outputs.astype(int)


def train_single_layer_perceptron(
    features: np.ndarray,
    targets: np.ndarray,
    initial_weights: np.ndarray,
    learning_rate: float,
    activation_name: str,
    max_epochs: int = 1000,
    convergence_error: float = 0.002,
) -> TrainingResult:
    features_with_bias = add_bias_column(features)
    weights = np.asarray(initial_weights, dtype=float).copy()
    targets_for_training = np.asarray(targets, dtype=float)
    if activation_name == "bipolar_step":
        targets_for_training = np.where(targets_for_training == 1, 1.0, -1.0)

    epoch_errors: list[float] = []

    for epoch in range(1, max_epochs + 1):
        for row, target in zip(features_with_bias, targets_for_training):
            output = activation_by_name(np.array([summation_unit(row, weights)]), activation_name)[0]
            error = target - output
            weights += learning_rate * error * row

        all_outputs = activation_by_name(summation_unit(features_with_bias, weights), activation_name)
        sum_square_error = float(np.sum((targets_for_training - all_outputs) ** 2))
        epoch_errors.append(sum_square_error)
        if sum_square_error <= convergence_error:
            break

    final_outputs = activation_by_name(summation_unit(features_with_bias, weights), activation_name)
    predictions = binary_predictions(final_outputs, activation_name)
    return TrainingResult(weights, epoch, epoch_errors, predictions)


def plot_epoch_errors(errors: list[float], title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(errors) + 1), errors, marker="o", markersize=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Sum Square Error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_learning_rate_iterations(results: pd.DataFrame, title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for activation_name, group in results.groupby("activation"):
        plt.plot(group["learning_rate"], group["epochs"], marker="o", label=activation_name)
    plt.title(title)
    plt.xlabel("Learning Rate")
    plt.ylabel("Epochs to Converge")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def customer_transaction_data() -> tuple[np.ndarray, np.ndarray]:
    features = np.array(
        [
            [20, 6, 2, 386],
            [16, 3, 6, 289],
            [27, 6, 2, 393],
            [19, 1, 2, 110],
            [24, 4, 2, 280],
            [22, 1, 5, 167],
            [15, 4, 2, 271],
            [18, 4, 2, 274],
            [21, 1, 4, 148],
            [16, 2, 4, 198],
        ],
        dtype=float,
    )
    targets = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=int)
    return features, targets


def minmax_scale(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum = features.min(axis=0)
    maximum = features.max(axis=0)
    denominator = np.where(maximum - minimum == 0, 1, maximum - minimum)
    return (features - minimum) / denominator, minimum, denominator


def pseudo_inverse_solution(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features_with_bias = add_bias_column(features)
    weights = np.linalg.pinv(features_with_bias) @ targets
    predictions = (features_with_bias @ weights >= 0.5).astype(int)
    return weights, predictions


def sigmoid_derivative(output: np.ndarray) -> np.ndarray:
    return output * (1.0 - output)


def train_two_layer_network(
    features: np.ndarray,
    targets: np.ndarray,
    hidden_nodes: int = 2,
    output_nodes: int = 1,
    learning_rate: float = 0.05,
    max_epochs: int = 1000,
    convergence_error: float = 0.002,
    seed: int = 7,
    initial_weight_scale: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, int, list[float], np.ndarray]:
    rng = np.random.default_rng(seed)
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    hidden_weights = rng.normal(0, initial_weight_scale, size=(features.shape[1] + 1, hidden_nodes))
    output_weights = rng.normal(0, initial_weight_scale, size=(hidden_nodes + 1, output_nodes))
    errors: list[float] = []

    for epoch in range(1, max_epochs + 1):
        for row, target in zip(features, targets):
            row_with_bias = np.r_[1.0, row]
            hidden_output = sigmoid_activation(row_with_bias @ hidden_weights)
            hidden_with_bias = np.r_[1.0, hidden_output]
            final_output = sigmoid_activation(hidden_with_bias @ output_weights)

            output_error = target - final_output
            output_delta = output_error * sigmoid_derivative(final_output)
            hidden_error = output_weights[1:] @ output_delta
            hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

            output_weights += learning_rate * np.outer(hidden_with_bias, output_delta)
            hidden_weights += learning_rate * np.outer(row_with_bias, hidden_delta)

        hidden_all = sigmoid_activation(add_bias_column(features) @ hidden_weights)
        final_all = sigmoid_activation(add_bias_column(hidden_all) @ output_weights)
        sum_square_error = float(np.sum((targets - final_all) ** 2))
        errors.append(sum_square_error)
        if sum_square_error <= convergence_error:
            break

    if output_nodes == 1:
        predictions = (final_all.ravel() >= 0.5).astype(int)
    else:
        predictions = np.argmax(final_all, axis=1)
    return hidden_weights, output_weights, epoch, errors, predictions


def one_hot_logic_targets(targets: np.ndarray) -> np.ndarray:
    return np.array([[1, 0] if target == 0 else [0, 1] for target in targets], dtype=float)


def evaluate_project_mlp() -> tuple[float, str]:
    df = pd.read_csv(DATASET_PATH)
    features = df.drop(columns=["label"])
    targets = df["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.25, random_state=42, stratify=targets
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(8, 4),
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, target_names=["Safe", "Unsafe"])
    return accuracy_score(y_test, predictions), report


def run_gate_experiments(gate_name: str, features: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    initial_weights = np.array([10.0, 0.2, -0.75])
    activation_names = ["step", "bipolar_step", "sigmoid", "relu"]
    rows = []
    for activation_name in activation_names:
        result = train_single_layer_perceptron(
            features, targets, initial_weights, 0.05, activation_name
        )
        rows.append(
            {
                "gate": gate_name,
                "activation": activation_name,
                "epochs": result.epochs,
                "final_error": result.errors[-1],
                "predictions": result.predictions.tolist(),
            }
        )
        plot_epoch_errors(
            result.errors,
            f"{gate_name} Gate - {activation_name}",
            OUTPUT_DIR / f"{gate_name.lower()}_{activation_name}_errors.png",
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    and_features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    and_targets = np.array([0, 0, 0, 1], dtype=int)
    xor_features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    xor_targets = np.array([0, 1, 1, 0], dtype=int)

    print("\nA1 activation units demo")
    demo_values = np.array([-2.0, 0.0, 2.0])
    print("Step:", step_activation(demo_values).tolist())
    print("Bipolar Step:", bipolar_step_activation(demo_values).tolist())
    print("Sigmoid:", np.round(sigmoid_activation(demo_values), 4).tolist())
    print("TanH:", np.round(tanh_activation(demo_values), 4).tolist())
    print("ReLU:", relu_activation(demo_values).tolist())
    print("Leaky ReLU:", leaky_relu_activation(demo_values).tolist())

    print("\nA2-A3 AND gate perceptron comparison")
    and_results = run_gate_experiments("AND", and_features, and_targets)
    print(and_results.to_string(index=False))

    print("\nA4 learning rate variation for AND gate")
    lr_rows = []
    for learning_rate in np.arange(0.1, 1.01, 0.1):
        result = train_single_layer_perceptron(
            and_features,
            and_targets,
            np.array([10.0, 0.2, -0.75]),
            float(round(learning_rate, 1)),
            "step",
        )
        lr_rows.append(
            {
                "activation": "step",
                "learning_rate": float(round(learning_rate, 1)),
                "epochs": result.epochs,
                "final_error": result.errors[-1],
            }
        )
    lr_results = pd.DataFrame(lr_rows)
    print(lr_results.to_string(index=False))
    plot_learning_rate_iterations(
        lr_results,
        "AND Gate Learning Rate vs Epochs",
        OUTPUT_DIR / "and_learning_rate_vs_epochs.png",
    )

    print("\nA5 XOR gate perceptron comparison")
    xor_results = run_gate_experiments("XOR", xor_features, xor_targets)
    print(xor_results.to_string(index=False))

    print("\nA6 customer transaction perceptron with sigmoid")
    customer_features, customer_targets = customer_transaction_data()
    scaled_customer_features, _, _ = minmax_scale(customer_features)
    customer_result = train_single_layer_perceptron(
        scaled_customer_features,
        customer_targets,
        np.zeros(scaled_customer_features.shape[1] + 1),
        learning_rate=0.25,
        activation_name="sigmoid",
    )
    print("Epochs:", customer_result.epochs)
    print("Predictions:", customer_result.predictions.tolist())
    print("Accuracy:", accuracy_score(customer_targets, customer_result.predictions))

    print("\nA7 pseudo-inverse comparison on customer data")
    pinv_weights, pinv_predictions = pseudo_inverse_solution(scaled_customer_features, customer_targets)
    print("Pseudo-inverse predictions:", pinv_predictions.tolist())
    print("Pseudo-inverse accuracy:", accuracy_score(customer_targets, pinv_predictions))
    print("Pseudo-inverse weights:", np.round(pinv_weights, 4).tolist())

    print("\nA8-A9 back-propagation neural network")
    _, _, and_nn_epochs, and_nn_errors, and_nn_predictions = train_two_layer_network(
        and_features, and_targets, learning_rate=0.05, seed=2, initial_weight_scale=2.0
    )
    _, _, xor_nn_epochs, xor_nn_errors, xor_nn_predictions = train_two_layer_network(
        xor_features, xor_targets, learning_rate=0.05, seed=25, initial_weight_scale=2.0
    )
    print("AND epochs:", and_nn_epochs, "predictions:", and_nn_predictions.tolist())
    print("XOR epochs:", xor_nn_epochs, "predictions:", xor_nn_predictions.tolist())
    plot_epoch_errors(and_nn_errors, "AND Gate Backpropagation Error", OUTPUT_DIR / "and_backprop_errors.png")
    plot_epoch_errors(xor_nn_errors, "XOR Gate Backpropagation Error", OUTPUT_DIR / "xor_backprop_errors.png")

    print("\nA10 two-output neural network for AND gate")
    _, _, two_output_epochs, _, two_output_predictions = train_two_layer_network(
        and_features,
        one_hot_logic_targets(and_targets),
        hidden_nodes=2,
        output_nodes=2,
        learning_rate=0.05,
        seed=9,
        initial_weight_scale=1.0,
    )
    print("Epochs:", two_output_epochs)
    print("Predictions:", two_output_predictions.tolist())

    print("\nA11 sklearn MLPClassifier on AND and XOR gates")
    for gate_name, features, targets in [
        ("AND", and_features, and_targets),
        ("XOR", xor_features, xor_targets),
    ]:
        clf = MLPClassifier(
            hidden_layer_sizes=(4,),
            activation="logistic",
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
        )
        clf.fit(features, targets)
        predictions = clf.predict(features)
        print(gate_name, "predictions:", predictions.tolist(), "accuracy:", accuracy_score(targets, predictions))

    print("\nA12 sklearn MLPClassifier on project dataset")
    project_accuracy, project_report = evaluate_project_mlp()
    print("Project dataset accuracy:", round(project_accuracy, 4))
    print(project_report)

    combined_results = pd.concat([and_results, xor_results], ignore_index=True)
    combined_results.to_csv(OUTPUT_DIR / "gate_activation_results.csv", index=False)
    lr_results.to_csv(OUTPUT_DIR / "learning_rate_results.csv", index=False)
    print(f"\nSaved Lab 8 plots and tables to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

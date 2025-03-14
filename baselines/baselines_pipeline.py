from pyod.models import (
    IForest, LOF, OCSVM, KNN, ECOD, AutoEncoder, VAE, PCA, HBOS, CBLOF, COPOD, LODA
)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from typing import List
import numpy as np
import csv
import os


def inline_report(y_test: List[np.ndarray], preds: List[np.ndarray], model_name: str, verbose: int = 1, fname: str = 'results.csv'):
    """
    Generate and save a performance report for the given predictions and ground truth labels.

    Args:
        y_test (List[np.ndarray]): List of true labels for each test set.
        preds (List[np.ndarray]): List of predicted labels for each model.
        model_name (str): Name of the model being evaluated.
        verbose (int, optional): Verbosity level for printing results. Defaults to 1.
        fname (str, optional): Filename for saving the results. Defaults to 'results.csv'.
    """
    accuracies, sensitivities, specificities, f1s = [], [], [], []

    # Compute performance metrics for each set of predictions
    for true_labels, predicted in zip(y_test, preds):
        accuracies.append(accuracy_score(true_labels, predicted))
        tn, fp, fn, tp = confusion_matrix(true_labels, predicted).ravel()
        sensitivities.append(tp / (tp + fn))
        specificities.append(tn / (tn + fp))
        f1s.append(f1_score(true_labels, predicted))

    # Print results if verbose is enabled
    if verbose > 0:
        print(f"Model: {model_name}")
        print(f"Accuracy: {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
        print(f"Sensitivity: {np.mean(sensitivities):.2f} ± {np.std(sensitivities):.2f}")
        print(f"Specificity: {np.mean(specificities):.2f} ± {np.std(specificities):.2f}")

    # Save results to CSV
    _save_results_to_csv(fname, model_name, accuracies, sensitivities, specificities, f1s)


def _save_results_to_csv(fname: str, model_name: str, accuracies: List[float], sensitivities: List[float], specificities: List[float], f1s: List[float]):
    """
    Save the computed performance metrics to a CSV file.

    Args:
        fname (str): Filename for saving the results.
        model_name (str): Name of the model being evaluated.
        accuracies (List[float]): List of accuracy values.
        sensitivities (List[float]): List of sensitivity values.
        specificities (List[float]): List of specificity values.
        f1s (List[float]): List of F1-score values.
    """
    # Check if file exists and create header if not
    file_exists = os.path.exists(fname)
    with open(fname, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['model', 'accuracy', 'sensitivity', 'specificity', 'f1', 'f1_std'])

        # Format and save the results
        results = [
            model_name,
            f"{np.mean(accuracies) * 100:.2f} ± {np.std(accuracies) * 100:.2f}",
            f"{np.mean(sensitivities) * 100:.2f} ± {np.std(sensitivities) * 100:.2f}",
            f"{np.mean(specificities) * 100:.2f} ± {np.std(specificities) * 100:.2f}",
            f"{np.mean(f1s) * 100:.2f} ± {np.std(f1s) * 100:.2f}"
        ]
        writer.writerow(results)


baselines = {
    'IForest': IForest,
    'LOF': LOF,
    'OCSVM': OCSVM,
    'KNN': KNN,
    'ECOD': ECOD,
    'AutoEncoder': AutoEncoder,
    'VAE': VAE,
    'PCA': PCA,
    'HBOS': HBOS,
    'CBLOF': CBLOF,
    'COPOD': COPOD,
    'LODA': LODA
}


def run_pipeline(model_name: str, X: List[np.ndarray], y: List[np.ndarray], X_test: List[np.ndarray], y_test: List[np.ndarray], fname: str = 'res.csv', unsupervised: bool = True):
    """
    Run the anomaly detection pipeline for a given model name.

    Args:
        model_name (str): Name of the model to run.
        X (List[np.ndarray]): Training data for each fold.
        y (List[np.ndarray]): Training labels for each fold.
        X_test (List[np.ndarray]): Test data for each fold.
        y_test (List[np.ndarray]): Test labels for each fold.
        fname (str, optional): Filename for saving the results. Defaults to 'res.csv'.
        unsupervised (bool, optional): Whether to use unsupervised mode. Defaults to True.
    """
    # Initialize models
    models = [baselines[model_name]() for _ in range(4)]

    print(f'Running {model_name}...')

    # Train models
    for i, model in enumerate(models):
        if unsupervised:
            model.fit(X[i])
        else:
            model.fit(X[i], y[i])

    # Predict and generate results
    results = [model.predict(X_test[i]) for i, model in enumerate(models)]
    print(f'Writing results to {fname}')
    inline_report(y_test, results, model_name=model_name, fname=fname)

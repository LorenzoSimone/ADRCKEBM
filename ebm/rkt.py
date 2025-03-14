import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sktime.transformations.panel.rocket import MiniRocketMultivariate
import torch
import os

def create_dataframe_from_series(x_data, indices, n_features):
    """
    Converts multivariate time series data into a DataFrame format.

    Args:
        x_data (np.ndarray): Multivariate time series data.
        indices (np.ndarray): Indices to split the data into train or test sets.
        n_features (int): Number of features in the time series data.

    Returns:
        pd.DataFrame: A DataFrame where each column represents a feature and each cell contains a pd.Series of time series data.
    """
    data_dict = {
        f"feature_{i+1}": [pd.Series(s[:, i]) for s in x_data[indices]]
        for i in range(n_features)
    }
    return pd.DataFrame(data_dict)

def main(tr_x, ts_x, tr_y, ts_y, n_features, n_pca_components=None, output_path="output_tensors.pt"):
    """
    Main function to process data through MiniRocket and PCA (if requested).

    Args:
        tr_x (np.ndarray): Training feature data.
        ts_x (np.ndarray): Testing feature data.
        tr_y (np.ndarray): Training labels.
        ts_y (np.ndarray): Testing labels.
        n_features (int): Number of features in the time series.
        n_pca_components (int, optional): Number of PCA components to reduce to. Defaults to None.
        output_path (str): Path to save the output tensors.
    """
    # Convert the input data to DataFrame format
    tr_x_df = create_dataframe_from_series(tr_x, np.arange(len(tr_x)), n_features)
    ts_x_df = create_dataframe_from_series(ts_x, np.arange(len(ts_x)), n_features)

    # Fit and transform MiniRocket
    minirocket_multi = MiniRocketMultivariate()
    minirocket_multi.fit(tr_x_df)

    tr_x_rocket = minirocket_multi.transform(tr_x_df)
    ts_x_rocket = minirocket_multi.transform(ts_x_df)

    # Standardize the data
    scaler = StandardScaler(with_mean=False)
    tr_x_rocket = scaler.fit_transform(tr_x_rocket)
    ts_x_rocket = scaler.transform(ts_x_rocket)

    # Apply PCA if the number of components is specified
    if n_pca_components:
        pca = PCA(n_components=n_pca_components)
        tr_x_rocket = pca.fit_transform(tr_x_rocket)
        ts_x_rocket = pca.transform(ts_x_rocket)

        # Re-standardize after PCA
        tr_x_rocket = scaler.fit_transform(tr_x_rocket)
        ts_x_rocket = scaler.transform(ts_x_rocket)

    # Convert to PyTorch tensors
    tensor_train = torch.tensor(tr_x_rocket, dtype=torch.float32)
    tensor_test = torch.tensor(ts_x_rocket, dtype=torch.float32)

    # Concatenate labels and save output tensors
    full_x = torch.cat((tensor_train, tensor_test), dim=0)
    full_y = torch.cat((torch.tensor(tr_y), dtype=torch.float32), torch.tensor(ts_y, dtype=torch.float32), dim=0)

    # Save tensors to specified output path
    torch.save({'features': full_x, 'labels': full_y}, output_path)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process time series data using MiniRocket and optionally PCA.")
    parser.add_argument('--n_features', type=int, required=True, help="Number of features in the time series data.")
    parser.add_argument('--n_pca_components', type=int, default=None, help="Number of PCA components to reduce to.")
    parser.add_argument('--output_path', type=str, default="output_tensors.pt", help="Path to save the output tensors.")

    args = parser.parse_args()

    # Dummy data example for script testing; replace these with actual data input
    tr_x, ts_x = np.random.randn(100, 50, 5), np.random.randn(50, 50, 5)  # Example training and test data
    tr_y, ts_y = np.random.randint(0, 2, 100), np.random.randint(0, 2, 50)  # Example training and test labels

    # Run main function
    main(tr_x, ts_x, tr_y, ts_y, args.n_features, args.n_pca_components, args.output_path)

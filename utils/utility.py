import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay, 
                             accuracy_score, f1_score, recall_score, precision_score)
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def evaluate(labels, predictions):
    """
    Evaluate classification performance by calculating key metrics and visualizing the confusion matrix.

    Args:
        labels (array-like): True labels.
        predictions (array-like): Predicted labels.

    Returns:
        None
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Extract TN, FP, FN, TP from the confusion matrix
    tn, fp, fn, tp = cm.ravel()

    # Calculate classification metrics
    accuracy = accuracy_score(labels, predictions)
    sensitivity = recall_score(labels, predictions)  # Sensitivity is the same as recall for the positive class
    specificity = tn / (tn + fp)
    f1 = f1_score(labels, predictions)

    # Print classification report
    print("Classification Report:\n", classification_report(labels, predictions))

    # Print individual metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall for positive class): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


def evaluate_classifier(classifier, X_test, y_test):
    """
    Evaluate a classifier's performance on test data by calculating key metrics 
    and plotting the confusion matrix.

    Args:
        classifier (object): The trained classifier to evaluate.
        X_test (array-like): The test feature data.
        y_test (array-like): The true labels for the test set.

    Returns:
        None
    """
    # Predict the labels for the test set
    y_pred = classifier.predict(X_test)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Extract TN, FP, FN, TP from the confusion matrix
    tn, fp, fn, tp = cm.ravel()

    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_test, y_pred)

    # Print classification report
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Print individual metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall for positive class): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


def plot_energyND(model, x_interval, y_interval, pos=None, neg=None, N=500, grid_size=100):
    """
    Plot the model's output Z over a 2D grid in a reduced dimensionality space using PCA.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        x_interval (tuple): Tuple (x_min, x_max) defining the interval for the first principal component.
        y_interval (tuple): Tuple (y_min, y_max) defining the interval for the second principal component.
        pos (torch.Tensor, optional): Positive samples, a tensor of shape (n_samples, 100).
        neg (torch.Tensor, optional): Negative samples, a tensor of shape (n_samples, 100).
        N (int, optional): Number of random points to sample from pos and neg for scatter plotting. Default is 500.
        grid_size (int, optional): Number of points along each axis for the grid. Default is 100.

    Returns:
        fig (plt.Figure): The matplotlib figure object containing the plots.
    """
    # Create 2D grid in the reduced (PCA) space
    x = np.linspace(x_interval[0], x_interval[1], grid_size)
    y = np.linspace(y_interval[0], y_interval[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Flatten the grid for model input
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]

    # Use PCA to transform 2D grid points back to the original 100D space
    pca = PCA(n_components=2)
    pca.fit(np.vstack([pos, neg]))  # Fit PCA on combined pos and neg data

    # Transform grid points to the original 100D space
    grid_points_100d = pca.inverse_transform(grid_points_2d)
    grid_tensor = torch.tensor(grid_points_100d, dtype=torch.float32)

    # Evaluate the model on the 100D grid points
    with torch.no_grad():
        zz = model(grid_tensor).numpy()
    
    # Reshape the output to match the 2D grid
    zz = zz.reshape(xx.shape)
    
    # Create a figure with two subplots
    fig = plt.figure(figsize=(15, 6))
    
    # 2D Contour plot
    ax1 = fig.add_subplot(121)
    cp = ax1.contourf(xx, yy, zz, levels=50, cmap='viridis')
    if pos is not None and neg is not None:
        # Transform pos and neg samples to 2D using PCA
        pos_2d = pca.transform(pos)
        neg_2d = pca.transform(neg)
        # Normalize to range -1 and 1
        pos_2d = (pos_2d - pos_2d.min()) / (pos_2d.max() - pos_2d.min()) * 2 - 1
        neg_2d = (neg_2d - neg_2d.min()) / (neg_2d.max() - neg_2d.min()) * 2 - 1
        
        # Plot the points
        ax1.scatter(pos_2d[:, 0], pos_2d[:, 1], c='white', alpha=0.6, s=10, label='Positive Samples')
        ax1.scatter(neg_2d[:, 0], neg_2d[:, 1], c='red', alpha=0.6, s=10, label='Negative Samples')

    fig.colorbar(cp, ax=ax1)
    ax1.set_title('2D Contour Plot of Model Output')
    ax1.set_xlabel('First Principal Component')
    ax1.set_ylabel('Second Principal Component')
    ax1.legend()

    # 3D Surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(xx, yy, zz, cmap='coolwarm', edgecolor='none')
    ax2.set_title('3D Surface Plot of Model Output')
    ax2.set_xlabel('First Principal Component')
    ax2.set_ylabel('Second Principal Component')
    ax2.set_zlabel('Model Output (Z)')

    ax2.xaxis._axinfo['grid'].update(alpha=0.1, linestyle='--')
    ax2.yaxis._axinfo['grid'].update(alpha=0.1, linestyle='--')
    ax2.zaxis._axinfo['grid'].update(alpha=0.1, linestyle='--')
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False

    return fig

def plot_energy(model, x_interval, y_interval, pos=None, neg=None, N=500, grid_size=100):
    """
    Plot the output of a feedforward model over a 2D grid in the space defined by `x_interval` and `y_interval`.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        x_interval (tuple): Tuple (x_min, x_max) defining the interval for the x-axis.
        y_interval (tuple): Tuple (y_min, y_max) defining the interval for the y-axis.
        pos (torch.Tensor, optional): Tensor of positive samples for scatter plotting.
        neg (torch.Tensor, optional): Tensor of negative samples for scatter plotting.
        N (int, optional): Number of points to sample for scatter plotting. Default is 500.
        grid_size (int, optional): Number of points along each axis for the grid. Default is 100.

    Returns:
        plt.Figure: The matplotlib figure object containing the plots.
    """
    # Create a 2D grid of x and y values
    x = np.linspace(x_interval[0], x_interval[1], grid_size)
    y = np.linspace(y_interval[0], y_interval[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Flatten the grid for model input
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    
    # Evaluate the model on the grid
    with torch.no_grad():
        zz = model(grid_tensor).numpy()
    
    # Reshape the output to match the grid
    zz = zz.reshape(xx.shape)

    # Create a figure with two subplots
    fig = plt.figure(figsize=(15, 6))
    
    # First subplot: 2D contour plot
    ax1 = fig.add_subplot(121)
    cp = ax1.contourf(xx, yy, zz, levels=50, cmap='viridis')
    if pos is not None and neg is not None:
        # Sample points for visualization
        true_points = pos[torch.randperm(pos.shape[0]), :]
        gen_points = neg[torch.randperm(neg.shape[0]), :]
        ax1.scatter(true_points[:, 0], true_points[:, 1], c='white', alpha=0.6, s=10, label='Positive Samples')
        ax1.scatter(gen_points[:, 0], gen_points[:, 1], c='red', alpha=0.6, s=10, label='Negative Samples')

    fig.colorbar(cp, ax=ax1)
    ax1.set_title('2D Contour Plot of Model Output')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()

    # Second subplot: 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(xx, yy, zz, cmap='coolwarm', edgecolor='none')
    ax2.set_title('3D Surface Plot of Model Output')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    ax2.set_zlabel('Model Output (Z)')
    
    # Customize the 3D plot's appearance
    ax2.xaxis._axinfo['grid'].update(alpha=0.1, linestyle='--')
    ax2.yaxis._axinfo['grid'].update(alpha=0.1, linestyle='--')
    ax2.zaxis._axinfo['grid'].update(alpha=0.1, linestyle='--')
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False

    return fig


def generate_spiral_points(num_points, turns, noise, x_interval, y_interval, angle_increment=None, radius_increment=None):
    """
    Generate 2D points arranged in a spiral pattern and filter them to be within a specified interval.

    Args:
        num_points (int): Number of points to generate.
        turns (float): Number of spiral turns.
        noise (float): Standard deviation of Gaussian noise added to the points.
        x_interval (tuple): The (min, max) interval for the x-axis.
        y_interval (tuple): The (min, max) interval for the y-axis.
        angle_increment (float, optional): Increment for the angle (default is calculated based on turns and num_points).
        radius_increment (float, optional): Increment for the radius (default is linear growth with angle).

    Returns:
        np.ndarray: A (num_points_filtered, 2) array of 2D points within the specified interval.

    Example:
        turns = 3
        noise = 0.01
        x_interval = (-4, 4)
        y_interval = (-4, 4)
        samples_s = generate_spiral_points(1000, 3, 0.01, x_interval, y_interval, 0.1, 0.05)
    """
    # Generate angles
    if angle_increment is not None:
        angles = np.arange(0, angle_increment * num_points, angle_increment)
    else:
        angles = np.linspace(0, 2 * np.pi * turns, num_points)

    # Define radius
    if radius_increment is not None:
        radius = radius_increment * np.arange(num_points)
    else:
        radius = angles  # Default to linear radius

    # Generate points in polar coordinates and convert to Cartesian
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    points = np.vstack((x, y)).T

    # Add Gaussian noise
    points += np.random.normal(0, noise, points.shape)

    # Filter points within the specified intervals
    filtered_points = points[
        (points[:, 0] >= x_interval[0]) & (points[:, 0] <= x_interval[1]) &
        (points[:, 1] >= y_interval[0]) & (points[:, 1] <= y_interval[1])
    ]
    
    return filtered_points


def generate_circle_points(num_points, radius, noise):
    """
    Generate 2D points arranged in a circle with added Gaussian noise.

    Args:
        num_points (int): Number of points to generate.
        radius (float): Radius of the circle.
        noise (float): Standard deviation of Gaussian noise added to the points.

    Returns:
        np.ndarray: A (num_points, 2) array of 2D points arranged in a circle.
    """
    # Generate random angles
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    # Generate points
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    points = np.vstack((x, y)).T
    # Add Gaussian noise
    points += np.random.normal(0, noise, points.shape)
    
    return points


def generate_circ_gmm(num_points_per_mixture, num_mixtures, radius, center, std_dev):
    """
    Generate 2D points from multiple Gaussian mixtures arranged in a circle.

    Args:
        num_points_per_mixture (int): Number of points to generate for each Gaussian mixture.
        num_mixtures (int): Number of Gaussian mixtures.
        radius (float): Radius of the circle on which the Gaussian centers lie.
        center (tuple): The (x, y) center of the circle.
        std_dev (float): Standard deviation of the Gaussian distributions.

    Returns:
        np.ndarray: A (num_points_per_mixture * num_mixtures, 2) array of 2D points.
    """
    points = []

    # Calculate the angle between each Gaussian center
    angles = np.linspace(0, 2 * np.pi, num_mixtures, endpoint=False)

    for angle in angles:
        # Calculate Gaussian center position
        gaussian_center_x = center[0] + radius * np.cos(angle)
        gaussian_center_y = center[1] + radius * np.sin(angle)

        # Generate points around the Gaussian center
        points_x = np.random.normal(gaussian_center_x, std_dev, num_points_per_mixture)
        points_y = np.random.normal(gaussian_center_y, std_dev, num_points_per_mixture)

        # Combine x and y coordinates
        points.append(np.vstack((points_x, points_y)).T)

    # Combine all points into one array
    points = np.vstack(points)

    return points

def plot_scatter_density(dataframes, colormaps, figsize=(6, 6), titles=None):
    """
    Plot density and scatter plots for multiple DataFrames.

    Args:
        dataframes (list of pd.DataFrame): List of DataFrames, each containing columns 'x' and 'y'.
        colormaps (list of str): List of colormaps for the density plots, corresponding to each DataFrame.
        figsize (tuple, optional): Tuple specifying the size of the figure. Default is (6, 6).
        titles (list of str, optional): List of titles for each plot. Must be the same length as dataframes and colormaps.

    Raises:
        ValueError: If the number of colormaps or titles does not match the number of DataFrames.

    Example:
        plot_scatter_density([df1, df2], ['viridis', 'plasma'], titles=['Plot 1', 'Plot 2'])
    """
    if titles is None:
        titles = ['Density Plot'] * len(dataframes)

    # Check for consistency in the number of colormaps and titles
    if len(dataframes) != len(colormaps):
        raise ValueError("The number of colormaps must match the number of dataframes.")
    if len(titles) != len(dataframes):
        raise ValueError("The number of titles must match the number of dataframes.")
    
    plt.figure(figsize=figsize)
    
    for i, (df, cmap, title) in enumerate(zip(dataframes, colormaps, titles)):
        # Plot density and scatter for each DataFrame
        sns.kdeplot(data=df, x='x', y='y', fill=True, cmap=cmap, alpha=0.2, legend=False)
        sns.scatterplot(data=df, x='x', y='y', s=10, color='k', alpha=0.5, legend=False)
        
    # Add title and labels
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True, alpha=0.1, linestyle='--')
    
    plt.show()


def Brier(predicted, labels):
    """
    Calculate the Brier score for predicted probabilities.

    Args:
        predicted (array-like): Array of predicted probabilities.
        labels (array-like): Array of true binary labels.

    Returns:
        float: Brier score loss.
    """
    return brier_score_loss(labels, predicted)


def MCE(predicted, labels, n_bins=10):
    """
    Compute the Maximum Calibration Error (MCE).

    Args:
        predicted (array-like): Array of predicted probabilities.
        labels (array-like): Array of true binary labels.
        n_bins (int, optional): Number of bins for calibration curve. Default is 10.

    Returns:
        float: Maximum Calibration Error.
    """
    prob_true, prob_pred = calibration_curve(labels, predicted, n_bins=n_bins)
    mce = np.max(np.abs(prob_pred - prob_true))
    return mce


def ECE(pred_prob, true_labels, n_bins=5):
    """
    Calculate the Expected Calibration Error (ECE) using uniform binning.

    Args:
        pred_prob (array-like): Array of predicted probabilities.
        true_labels (array-like): Array of true binary labels.
        n_bins (int, optional): Number of bins for calibration. Default is 5.

    Returns:
        float: Expected Calibration Error.
    """
    # Define bin boundaries
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Convert predictions to a numpy array
    confidences = np.array(pred_prob)
    predicted_label = np.around(pred_prob)  # Convert probabilities to binary labels

    # Determine correctness of predictions
    accuracies = predicted_label == true_labels

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine which samples fall into the current bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prob_in_bin = in_bin.mean()

        if prob_in_bin > 0:
            # Calculate accuracy and average confidence for the bin
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            # Update ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

    return ece

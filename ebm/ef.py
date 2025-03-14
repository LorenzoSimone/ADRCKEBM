import torch
import torch.nn as nn
import torch.optim as optim

class EffortFunctionalOOD(nn.Module):
    """
    A neural network module that computes out-of-distribution (OOD) detection 
    based on energy functions and effort functionals of two energy-based models.

    Args:
        model1 (nn.Module): The first energy-based model.
        model2 (nn.Module): The second energy-based model.
        alpha (float, optional): Weight for the first energy model's gradient. Defaults to 1.0.
        beta (float, optional): Weight for the second energy model's gradient. Defaults to 1.0.
        noise_scale (float, optional): Scale of noise added during Langevin dynamics. Defaults to 0.01.
        step_size (float, optional): Step size for Langevin dynamics. Defaults to 0.1.
    """
    def __init__(self, model1, model2, alpha=1.0, beta=1.0, noise_scale=0.01, step_size=0.1):
        super(EffortFunctionalOOD, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.alpha = alpha
        self.beta = beta
        self.noise_scale = noise_scale
        self.step_size = step_size

        #4 features: energy1, energy2, effort1, effort2
        self.classifier = nn.Sequential(
            nn.Linear(4, 100),  
            nn.Linear(100, 1),
            nn.Sigmoid() 
        )

    def langevin_dynamics(self, model, x_init, num_steps=100, step_size=0.1, noise_scale=0.01):
        """
        Perform Langevin dynamics to compute the effort functional.

        Args:
            model (nn.Module): Energy-based model for gradient computation.
            x_init (torch.Tensor): Initial points to start Langevin dynamics.
            num_steps (int, optional): Number of Langevin steps. Defaults to 100.
            step_size (float, optional): Step size for Langevin dynamics. Defaults to 0.1.
            noise_scale (float, optional): Scale of noise to add at each step. Defaults to 0.01.

        Returns:
            torch.Tensor: Effort computed over the Langevin path.
        """
        x1 = x_init.clone().detach().requires_grad_(True)
        x2 = x_init.clone().detach().requires_grad_(True)
        effort = 0.0
        x_new = x1.clone().detach().requires_grad_(True)

        for _ in range(num_steps):
            energy1 = model(x1)
            energy2 = self.model2(x2)

            energy1.backward(torch.ones_like(energy1))
            energy2.backward(torch.ones_like(energy2))

            # Gradient descent on energy
            with torch.no_grad():
                x_new -= step_size * x1.grad
                # Adding Gaussian noise
                x_new += noise_scale * torch.randn_like(x1)
                vel = torch.norm(x1 - x_new, dim=1)
                effort += vel * torch.norm(x1.grad)
                x1 = x_new

            if x1.grad is not None:
                x1.grad.zero_()

            if x2.grad is not None:
                x2.grad.zero_()

        return effort

    def compute_effort(self, x, num_steps=100):
        """
        Compute the effort functional for traversing from an initial point using SGLD for both models.

        Args:
            x (torch.Tensor): Input data (starting configuration).
            num_steps (int, optional): Number of steps in SGLD. Defaults to 100.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Total efforts for trajectories following both models' gradients.
        """
        # Perform Langevin Dynamics for both models and compute efforts
        total_effort1 = self.langevin_dynamics(self.model1, x, num_steps)
        total_effort2 = self.langevin_dynamics(self.model2, x, num_steps)

        return total_effort1, total_effort2

    def forward(self, x):
        """
        Forward pass through the classifier.

        Args:
            x (torch.Tensor): Input data (batch_size x features).

        Returns:
            torch.Tensor: Output probabilities between 0 and 1.
        """
        energy1 = self.model1(x).detach()
        energy2 = self.model2(x).detach()

        effort1, effort2 = self.compute_effort(x)
        effort1, effort2 = effort1.unsqueeze(1).detach(), effort2.unsqueeze(1).detach()

        # Combine energies and efforts into input for the classifier
        new_x = torch.cat((energy1, energy2, effort1, effort2), dim=1)
        return self.classifier(new_x)

    def fit(self, x, y, optimizer=None, n_epochs=100):
        """
        Fit the classifier to the input data.

        Args:
            x (torch.Tensor): Input data (batch_size x features).
            y (torch.Tensor): Ground truth labels.
            optimizer (torch.optim.Optimizer, optional): Optimizer for training. Defaults to Adam optimizer.
            n_epochs (int, optional): Number of training epochs. Defaults to 100.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Computed energy and effort functionals.
        """
        criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        if optimizer is None:
            optimizer = optim.Adam(self.classifier.parameters(), lr=0.1)

        # Pre-compute static energy and effort functionals
        energy1 = self.model1(x).detach()
        energy2 = self.model2(x).detach()

        effort1, effort2 = self.compute_effort(x)
        effort1, effort2 = effort1.unsqueeze(1).detach(), effort2.unsqueeze(1).detach()

        tr_x = torch.cat((energy1, energy2, effort1, effort2), dim=1)
        tr_y = y.float()

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            outputs = self.classifier(tr_x).squeeze()
            loss = criterion(outputs, tr_y)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")

        return energy1, energy2, effort1, effort2


class OneClass(nn.Module):
    """
    A one-class neural network model for anomaly detection based on a single energy-based model.

    Args:
        model1 (nn.Module): The energy-based model to compute the energy score.
    """
    def __init__(self, model1):
        super(OneClass, self).__init__()
        self.model1 = model1

        # Define a simple classifier head
        self.classifier = nn.Sequential(
            nn.Linear(1, 100),
            nn.Linear(100, 1),
            nn.Sigmoid()  # Output probabilities between 0 and 1
        )

    def forward(self, x):
        """
        Forward pass through the classifier.

        Args:
            x (torch.Tensor): Input data (batch_size x features).

        Returns:
            torch.Tensor: Output probabilities between 0 and 1.
        """
        energy1 = self.model1(x).detach()
        return self.classifier(energy1)

    def fit(self, x, y, optimizer=None, n_epochs=100):
        """
        Fit the classifier to the input data.

        Args:
            x (torch.Tensor): Input data (batch_size x features).
            y (torch.Tensor): Ground truth labels.
            optimizer (torch.optim.Optimizer, optional): Optimizer for training. Defaults to Adam optimizer.
            n_epochs (int, optional): Number of training epochs. Defaults to 100.

        Returns:
            torch.Tensor: Computed energy functionals.
        """
        criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        if optimizer is None:
            optimizer = optim.Adam(self.classifier.parameters(), lr=0.1)

        # Pre-compute static energy functionals
        energy1 = self.model1(x).detach()

        tr_x = energy1
        tr_y = y.float()

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            outputs = self.classifier(tr_x).squeeze()
            loss = criterion(outputs, tr_y)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")

        return energy1


class NoEF(nn.Module):
    """
    A neural network model for OOD detection without effort functionals, using only the energies of two energy-based models.

    Args:
        model1 (nn.Module): The first energy-based model.
        model2 (nn.Module): The second energy-based model.
    """
    def __init__(self, model1, model2):
        super(NoEF, self).__init__()
        self.model1 = model1
        self.model2 = model2

        # Define a simple classifier head
        self.classifier = nn.Sequential(
            nn.Linear(2, 100),
            nn.Linear(100, 1),
            nn.Sigmoid()  # Output probabilities between 0 and 1
        )

    def forward(self, x):
        """
        Forward pass through the classifier.

        Args:
            x (torch.Tensor): Input data (batch_size x features).

        Returns:
            torch.Tensor: Output probabilities between 0 and 1.
        """
        energy1 = self.model1(x).detach()
        energy2 = self.model2(x).detach()
        tot = torch.cat((energy1, energy2), dim=1)
        return self.classifier(tot)

    def fit(self, x, y, optimizer=None, n_epochs=100):
        """
        Fit the classifier to the input data.

        Args:
            x (torch.Tensor): Input data (batch_size x features).
            y (torch.Tensor): Ground truth labels.
            optimizer (torch.optim.Optimizer, optional): Optimizer for training. Defaults to Adam optimizer.
            n_epochs (int, optional): Number of training epochs. Defaults to 100.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Computed energy functionals.
        """
        criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        if optimizer is None:
            optimizer = optim.Adam(self.classifier.parameters(), lr=0.1)

        # Pre-compute static energy functionals
        energy1 = self.model1(x).detach()
        energy2 = self.model2(x).detach()

        tr_x = torch.cat((energy1, energy2), dim=1)
        tr_y = y.float()

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            outputs = self.classifier(tr_x).squeeze()
            loss = criterion(outputs, tr_y)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")

        return energy1, energy2

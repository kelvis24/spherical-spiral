import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from pn_autoencoder import PointNetAutoencoder


class Train:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # self.device = torch.device('cpu')

    def multiple_point_clouds(self, point_clouds, autoencoder=None,
                              num_points=500, latent_size=218, epochs=100, lr=0.001,
                              visualize_every_n_epochs=20, condition=False):
        criterion = nn.MSELoss()
        optimizer = Adam(autoencoder.parameters(), lr=lr)
        if autoencoder is None:
            autoencoder = PointNetAutoencoder(num_points, latent_size).to(self.device)

        for pc_index, point_cloud in enumerate(point_clouds):
            # autoencoder = PointNetAutoencoder(num_points, latent_size).to(self.device)
            # optimizer = Adam(autoencoder.parameters(), lr=lr)
            point_cloud_tensor = point_cloud.unsqueeze(0).to(self.device)  # Add batch dimension

            print(f"Training Point Cloud {pc_index + 1}/{len(point_clouds)}")

            for epoch in range(epochs):
                # Forward pass
                reconstructed, _ = autoencoder(point_cloud_tensor)
                loss = criterion(reconstructed, point_cloud_tensor)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % visualize_every_n_epochs == 0 or epoch == epochs - 1:
                    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
                    with torch.no_grad():
                        reconstructed, _ = autoencoder(point_cloud_tensor)
                        visualize_reconstruction(point_cloud_tensor.squeeze().cpu().numpy(),
                                                 reconstructed.squeeze().cpu().numpy(),
                                                 "Original", "Reconstructed", condition)

        return autoencoder

    def encode_point_cloud(self, autoencoder, point_cloud):
        # Reshape selected_point_cloud to [1, 3, num_points] for the model
        if point_cloud.dim() == 2 and point_cloud.size(1) == 3:
            model_input = point_cloud.transpose(0, 1).unsqueeze(0)
        else:
            print("Unexpected point cloud shape. Ensure it's [num_points, 3].")

        with torch.no_grad():
            autoencoder.eval()
            point_cloud = point_cloud.to(self.device).unsqueeze(0)
            _, latent_representation = autoencoder(point_cloud)
            return latent_representation.squeeze().cpu().numpy()

    def print_encode_point_cloud(self, autoencoder, point_cloud):
        print(self.encode_point_cloud(autoencoder, point_cloud))


def visualize_reconstruction(original, reconstructed, title1="Original", title2="Reconstructed",
                             condition=True):
    if condition:
        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(original[:, 0], original[:, 1], original[:, 2], s=1)
        ax1.title.set_text(title1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], s=1)
        ax2.title.set_text(title2)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        plt.show()
    else:
        pass


if __name__ == '__main__':
    print('Error: not main file')

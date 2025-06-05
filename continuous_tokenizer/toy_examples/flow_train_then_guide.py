import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import os
import argparse
from torchdiffeq import odeint
import functools

# Import the SimpleMLPAdaLN from the flowloss module
import sys
sys.path.append("..")
from modelling.flowloss import SimpleMLPAdaLN
from flow_loss_variants import flow_loss1 as flow_loss

def normalize_samples(sample_fn):
    """
    Decorator to normalize samples based on std computed from initial large sample.
    """
    @functools.wraps(sample_fn)
    def wrapper(self, n_samples):
        # Check if std is already computed
        if not hasattr(self, '_std'):
            print("Computing normalization statistics from large sample...")
            # Get large sample for computing statistics
            with torch.no_grad():
                large_samples = sample_fn(self, 10000000)
                self._std = torch.std(large_samples)
                print(f"Computed std: {self._std}")
        
        # Get samples and normalize
        samples = sample_fn(self, n_samples)
        return samples / self._std
    
    return wrapper

class TrainableFlowModel(nn.Module):
    """
    Trainable flow model based on SimpleMLPAdaLN architecture.
    This model is trained to transform noise to the target distribution.
    """
    def __init__(
        self, 
        in_channels=2,  # 2D data
        model_channels=128, 
        num_res_blocks=4,
        num_sampling_steps=100
    ):
        super(TrainableFlowModel, self).__init__()
        self.in_channels = in_channels
        self.num_sampling_steps = num_sampling_steps
        
        # Use SimpleMLPAdaLN for the vector field
        self.net = SimpleMLPAdaLN(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
        )
        
    def forward(self, target, reduction='mean'):
        """
        Compute flow loss for target points.
        """
        batch_size = target.shape[0]
        
        # Pick random timesteps
        t = torch.rand(batch_size, device=target.device)
        
        # Generate noise
        noise = torch.randn_like(target)
        
        # Compute x_t (points along the flow trajectory)
        x_t = target * t[:, None] + noise * (1 - t[:, None])
        
        # Predict velocity field
        v = self.net(x_t, t)
        
        # Ideal velocity field
        ideal_v = target - noise
        
        # Loss is MSE between predicted and ideal velocity
        loss = torch.nn.functional.mse_loss(v, ideal_v, reduction=reduction)
        
        return loss
    
    @torch.no_grad()
    def sample(self, batch_size=1000, steps=None, device='cuda'):
        """
        Sample from the flow model by running the reverse process
        """
        steps = steps or self.num_sampling_steps
        
        # Start with noise
        x = torch.randn(batch_size, self.in_channels, device=device)
        
        # Euler steps for sampling
        dt = 1.0 / steps
        for t in torch.linspace(1.0, dt, steps, device=device):
            # Predict velocity at this timestep
            t_batch = t.expand(batch_size)
            v = self.net(x, t_batch)
            
            # Euler step
            x = x + v * dt
            
        return x    
    
    @torch.no_grad()
    def sample(self, batch_size=1000, steps=None, device='cuda'):
        steps = steps or self.num_sampling_steps
        
        # Start with noise
        x = torch.randn(batch_size, self.in_channels, device=device)

        # Define ODE function
        def ode_fn(t, x):
            if isinstance(t, float):
                t = torch.tensor([t], device=x.device)
            t_batch = t.expand(x.shape[0])
            v = self.net(x, t_batch)
            return v

        # Time span from 1.0 to 0.0
        t = torch.linspace(0.0, 1.0, steps, device=device)

        # Solve ODE using Dormand-Prince RK45
        x = odeint(
            ode_fn,
            x,
            t,
            method='euler',
            rtol=1e-5,
            atol=1e-5,
        )[-1]  # Get the final state at t=0.0

        return x
    
    @torch.no_grad()
    def vector_field(self, x, t):
        """
        Get the vector field at point x, time t
        """
        return self.net(x, t)


def train_flow_model(target_sampler, n_steps=10000, batch_size=256, lr=2e-4, device='cuda',
                    model_channels=128, num_res_blocks=4, save_dir="flow_model_results"):
    """
    Train a flow model on target distribution
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create flow model
    flow_model = TrainableFlowModel(
        in_channels=2,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks
    ).to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(flow_model.parameters(), lr=lr)
    
    # For visualization and tracking
    loss_history = []
    sample_history = []
    
    # Training loop
    pbar = tqdm(range(n_steps))
    
    for step in pbar:
        optimizer.zero_grad()
        
        # Get batch of target samples
        target = target_sampler.sample(batch_size).to(device)
        
        # Compute loss
        loss = flow_model(target)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Log
        loss_val = loss.item()
        loss_history.append(loss_val)
        pbar.set_description(f"Flow Loss: {loss_val:.6f}")
        
        # Save samples periodically for visualization
        if step % (n_steps // 20) == 0 or step == n_steps - 1:
            samples = flow_model.sample(batch_size=1000).cpu()
            sample_history.append(samples.clone())
            
            # Visualize current state
            if step % (n_steps // 5) == 0 or step == n_steps - 1:
                with torch.no_grad():
                    target_samples = target_sampler.sample(2000).cpu().numpy()
                    curr_samples = samples.numpy()
                    
                    plt.figure(figsize=(10, 10))
                    plt.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.3, color='blue', label='Target')
                    plt.scatter(curr_samples[:, 0], curr_samples[:, 1], alpha=0.3, color='red', label='Flow Samples')
                    plt.legend()
                    plt.title(f"Training Progress - Step {step}")
                    plt.savefig(f"{save_dir}/flow_samples_step_{step}.png")
                    plt.close()
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel("Training Step")
    plt.ylabel("Flow Loss")
    plt.title("Flow Loss During Training")
    plt.grid(True)
    plt.savefig(f"{save_dir}/flow_loss_curve.png")
    plt.close()
    
    # Create animation of training progress
    fig, ax = plt.subplots(figsize=(10, 10))
    target_samples = target_sampler.sample(2000).cpu().numpy()
    
    def update(frame):
        ax.clear()
        ax.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.3, color='blue', label='Target')
        
        samples_frame = sample_history[frame].numpy()
        ax.scatter(samples_frame[:, 0], samples_frame[:, 1], alpha=0.3, color='red', label='Flow Samples')
        
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.legend()
        ax.set_title(f"Training Progress - Step {frame * (n_steps // 20)}")
        return ax,
    
    ani = FuncAnimation(fig, update, frames=len(sample_history), interval=200)
    ani.save(f"{save_dir}/flow_training_animation.gif", writer='pillow', dpi=100)
    plt.close()
    
    # Save the model weights
    model_save_path = os.path.join(save_dir, "flow_model.pt")
    torch.save(flow_model.state_dict(), model_save_path)
    print(f"Flow model training completed. Model saved to {model_save_path}")
    
    return flow_model, loss_history, sample_history


def load_flow_model(model_path, device='cuda', model_channels=128, num_res_blocks=4):
    """
    Load a pre-trained flow model from disk
    """
    flow_model = TrainableFlowModel(
        in_channels=2,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks
    ).to(device)
    
    flow_model.load_state_dict(torch.load(model_path, map_location=device))
    flow_model.eval()
    
    print(f"Loaded pre-trained flow model from {model_path}")
    return flow_model


def train_latents_with_flow(flow_model, n_latents=1000, n_steps=2000, lr=1e-4, device='cuda', save_dir="flow_guided_latents", target_sampler=None):
    """Train learnable latents with a trained flow model"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Freeze flow model
    flow_model.eval()
    for param in flow_model.parameters():
        param.requires_grad = False
    
    # Initialize random latents
    latents = torch.randn(n_latents, 2, device=device, requires_grad=True)
    # latents = torch.tensor(
    #     torch.randn(n_latents, 2, device=device) * 5.0, 
    #     requires_grad=True
    # )
    
    # Setup optimizer
    optimizer = optim.Adam([latents], lr=lr)
    
    # For visualization
    history = []
    loss_history = []
    nll_history = []  # Track NLL if using MixtureOfGaussians
    
    # Training loop
    pbar = tqdm(range(n_steps))
    for step in pbar:
        optimizer.zero_grad()
        
        # Compute loss
        loss = flow_loss(latents, flow_model)
        
        # # Compute NLL if using MixtureOfGaussians
        # if isinstance(target_sampler, MixtureOfGaussians):
        with torch.no_grad():
            nll = -target_sampler.log_likelihood(latents).mean().item()
            nll_history.append(nll)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Log
        loss_val = loss.item()
        loss_history.append(loss_val)
        desc = f"Latent Flow Loss: {loss_val:.6f}"
        # if isinstance(target_sampler, MixtureOfGaussians):
        desc += f", NLL: {nll:.6f}"
        pbar.set_description(desc)
        
        # Save state periodically for visualization
        if step % (n_steps // 20) == 0 or step == n_steps - 1:
            history.append(latents.detach().cpu().clone())
    
    # Generate flow samples for comparison
    flow_samples = flow_model.sample(batch_size=n_latents, device=device)
    
    # Save the trained latents
    latents_save_path = os.path.join(save_dir, "trained_latents.pt")
    torch.save(latents.detach().cpu(), latents_save_path)
    print(f"Trained latents saved to {latents_save_path}")
    
    return latents.detach(), history, loss_history, flow_samples, nll_history


def visualize_vector_field(flow_model, target_sampler=None, t=0.5, resolution=20, save_path="vector_field.png"):
    """
    Visualize the vector field of the flow model at a specific time t.
    
    Args:
        flow_model: Trained flow model
        target_sampler: Optional target distribution sampler for background points
        t: Time at which to visualize the vector field (0=target, 1=noise)
        resolution: Resolution of the grid for vector field
        save_path: Path to save the visualization
    """
    # Set up the grid
    x = np.linspace(-6, 6, resolution)
    y = np.linspace(-6, 6, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Get the device from the model (more reliable way)
    device = next(flow_model.parameters()).device
    
    # Reshape the grid points to batch format
    grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), 
                              dtype=torch.float32, device=device)
    
    # Get time tensor (same time t for all points)
    t_batch = torch.ones(grid_points.shape[0], device=device) * t
    
    # Compute vector field at these points
    with torch.no_grad():
        vectors = flow_model.vector_field(grid_points, t_batch).cpu().numpy()
    
    # Reshape vectors back to grid
    U = vectors[:, 0].reshape(resolution, resolution)
    V = vectors[:, 1].reshape(resolution, resolution)
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Generate points from distributions
    n_samples = 2000
    
    # Plot target distribution as background if provided
    if target_sampler is not None:
        with torch.no_grad():
            target_samples = target_sampler.sample(n_samples).cpu().numpy()
        plt.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.2, color='blue', label='Target Distribution')
    
    # Plot source distribution (noise)
    with torch.no_grad():
        noise_samples = torch.randn(n_samples, 2, device=device).cpu().numpy()
    plt.scatter(noise_samples[:, 0], noise_samples[:, 1], alpha=0.2, color='green', label='Source (Noise) Distribution')
    
    # Plot interpolated distribution at time t
    if target_sampler is not None:
        with torch.no_grad():
            target_points = target_sampler.sample(n_samples).to(device)
            noise_points = torch.randn_like(target_points)
            # Compute x_t (points along the flow trajectory)
            # For t=0, this is all target, for t=1, this is all noise
            interpolated_points = target_points * (1-t) + noise_points * t
            interpolated_samples = interpolated_points.cpu().numpy()
        plt.scatter(interpolated_samples[:, 0], interpolated_samples[:, 1], alpha=0.3, color='purple', 
                   label=f'Interpolated Distribution (t={t:.2f})')
    
    # Normalize the vectors for better visualization
    magnitude = np.sqrt(U**2 + V**2)
    max_magnitude = np.max(magnitude)
    if max_magnitude > 0:  # Avoid division by zero
        U = U / max_magnitude
        V = V / max_magnitude
    
    # Plot the vector field
    plt.quiver(X, Y, U, V, magnitude, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Vector Magnitude (Normalized)')
    
    # Set axis limits and labels
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Flow Vector Field at t={t}')
    plt.legend()
    
    # Save the figure
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_vector_fields_at_times(flow_model, target_sampler, save_dir="vector_fields"):
    """
    Visualize vector fields at different timesteps to show progression
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize at different timesteps
    time_points = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for t in time_points:
        save_path = os.path.join(save_dir, f"vector_field_t_{t:.2f}.png")
        visualize_vector_field(
            flow_model=flow_model,
            target_sampler=target_sampler,
            t=t,
            resolution=25,
            save_path=save_path
        )
    
    print(f"Vector field visualizations saved to {save_dir}")


def visualize_flow_trajectories(flow_model, n_points=10, n_steps=50, save_path="flow_trajectories.png", device="cuda"):
    """
    Visualize how points evolve from noise to the target distribution.
    
    Args:
        flow_model: Trained flow model
        n_points: Number of points to visualize
        n_steps: Number of steps to visualize in the trajectory
        save_path: Path to save the visualization
    """
    # Generate random noise points
    torch.manual_seed(42)  # For reproducibility
    points = torch.randn(n_points, 2, device=device)
    
    # Track trajectories
    trajectories = []
    trajectories.append(points.clone().cpu().numpy())
    
    # Define ODE function
    def ode_fn(t, x):
        if isinstance(t, float):
            t = torch.tensor([t], device=x.device)
        t_batch = t.expand(x.shape[0])
        v = flow_model.vector_field(x, t_batch)
        return v
    
    # Generate trajectory steps using odeint
    times = torch.linspace(0.0, 1.0, n_steps, device=device)
    
    # Store initial points as they'll be modified by odeint
    initial_points = points.clone()
    
    with torch.no_grad():
        # Solve ODE to get full trajectory
        full_trajectory = odeint(
            ode_fn,
            initial_points,
            times,
            method='euler',
            rtol=1e-5,
            atol=1e-5,
        )
        
        # Record all points in the trajectory
        for step_points in full_trajectory:
            trajectories.append(step_points.cpu().numpy())
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    
    # Plot all trajectories
    colors = plt.cm.rainbow(np.linspace(0, 1, n_points))
    
    for i in range(n_points):
        traj_x = [traj[i, 0] for traj in trajectories]
        traj_y = [traj[i, 1] for traj in trajectories]
        
        # Plot trajectory as a line with gradient color
        for j in range(len(traj_x) - 1):
            alpha = 0.3 + 0.7 * j / (len(traj_x) - 2)  # Increase opacity as we get closer to target
            plt.plot(traj_x[j:j+2], traj_y[j:j+2], color=colors[i], alpha=alpha, linewidth=1.5)
        
        # Mark start and end points
        plt.scatter(traj_x[0], traj_y[0], color=colors[i], marker='o', s=50, label=f"Noise {i+1}" if i == 0 else "")
        plt.scatter(traj_x[-1], traj_y[-1], color=colors[i], marker='*', s=100, label=f"Target {i+1}" if i == 0 else "")
    
    # Add legend for start/end markers only (not for each point)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # Set plot properties
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Flow Trajectories: Noise to Target Distribution')
    plt.grid(True, alpha=0.3)
    
    # Save the visualization
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_loss_heatmap(flow_model, target_sampler, resolution=50, n_samples_t=20, save_path="loss_heatmap.png", device="cuda"):
    """
    Visualize a heatmap of the flow_loss6 values across the vector field, with t uniformly sampled and averaged.
    Also plot the target distribution to visualize if they match.
    
    Args:
        flow_model: Trained flow model
        target_sampler: Target distribution sampler
        resolution: Resolution of the grid for heatmap
        n_samples_t: Number of time samples to average over
        save_path: Path to save the visualization
        device: Device to use for computation
    """
    # Set up the grid
    x = np.linspace(-6, 6, resolution)
    y = np.linspace(-6, 6, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Reshape the grid points to batch format
    grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), 
                              dtype=torch.float32, device=device)
    
    # Compute average loss over multiple t values
    total_loss = torch.zeros(grid_points.shape[0], device=device)
    
    # Sample t uniformly
    t_values = torch.linspace(0.1, 0.9, n_samples_t, device=device)
    
    # Batch computation of losses
    for t_val in t_values:
        t_batch = torch.ones(grid_points.shape[0], device=device) * t_val
        # Compute loss using flow_loss6
        with torch.no_grad():
            loss = flow_loss(grid_points, flow_model, reduction='none', t=t_batch)
            total_loss += loss.sum(dim=1)  # Sum over dimensions (x, y)
    
    # Average loss
    avg_loss = total_loss / n_samples_t
    
    # Reshape loss to grid
    loss_map = avg_loss.cpu().numpy().reshape(resolution, resolution)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot loss heatmap
    heatmap = plt.pcolormesh(X, Y, loss_map, cmap='viridis', alpha=0.7, shading='auto')
    plt.colorbar(heatmap, label='Average Flow Loss')
    
    # Generate points from target distribution
    n_points = 3000
    with torch.no_grad():
        target_samples = target_sampler.sample(n_points).cpu().numpy()
    
    # Plot target distribution
    plt.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.5, color='red', 
                s=5, label='Target Distribution')
    
    # Set plot properties
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Flow Loss Heatmap vs Target Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the visualization
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Loss heatmap visualization saved to {save_path}")


def visualize_nll_heatmap(target_sampler, trained_latents=None, resolution=50, save_path="nll_heatmap.png", device="cuda"):
    """
    Visualize a heatmap of the negative log likelihood values for MixtureOfGaussians.
    Optionally overlay trained latents on the heatmap.
    
    Args:
        target_sampler: MixtureOfGaussians instance
        trained_latents: Optional tensor of trained latents to overlay
        resolution: Resolution of the grid for heatmap
        save_path: Path to save the visualization
        device: Device to use for computation
    """
    # if not isinstance(target_sampler, MixtureOfGaussians):
    #     print("NLL heatmap visualization only supported for MixtureOfGaussians")
    #     return
        
    # Set up the grid
    x = np.linspace(-6, 6, resolution)
    y = np.linspace(-6, 6, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Reshape the grid points to batch format
    grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), 
                             dtype=torch.float32, device=device)
    
    # Compute NLL for all points
    with torch.no_grad():
        nll = -target_sampler.log_likelihood(grid_points)
    
    # Reshape NLL to grid
    nll_map = nll.cpu().numpy().reshape(resolution, resolution)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot NLL heatmap
    heatmap = plt.pcolormesh(X, Y, nll_map, cmap='viridis', alpha=0.7, shading='auto')
    plt.colorbar(heatmap, label='Negative Log Likelihood')
    
    # If trained latents are provided, overlay them
    if trained_latents is not None:
        trained_np = trained_latents.cpu().numpy()
        plt.scatter(trained_np[:, 0], trained_np[:, 1], alpha=0.5, color='red', 
                   s=5, label='Trained Latents')
    
    # Set plot properties
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Negative Log Likelihood Heatmap')
    if trained_latents is not None:
        plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the visualization
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"NLL heatmap visualization saved to {save_path}")


def visualize_results(target_sampler, trained_latents, history, loss_history, flow_samples, flow_model=None, save_dir="flow_guided_results", nll_history=None):
    """Visualize training results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Sample target distribution for visualization
    with torch.no_grad():
        target_samples = target_sampler.sample(5000).cpu().numpy()
    
    # Plot final result
    plt.figure(figsize=(10, 10))
    plt.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.3, color='blue', label='Target')
    
    trained_np = trained_latents.cpu().numpy()
    plt.scatter(trained_np[:, 0], trained_np[:, 1], alpha=0.3, color='red', label='Trained Latents')
    
    plt.legend()
    plt.title("Target Distribution vs Trained Latents")
    plt.savefig(f"{save_dir}/final_comparison.png")
    plt.close()
    
    # Plot flow samples vs trained latents
    plt.figure(figsize=(10, 10))
    plt.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.3, color='blue', label='Target')
    
    flow_samples_np = flow_samples.cpu().numpy()
    plt.scatter(flow_samples_np[:, 0], flow_samples_np[:, 1], alpha=0.3, color='green', label='Flow Samples')
    plt.scatter(trained_np[:, 0], trained_np[:, 1], alpha=0.3, color='red', label='Trained Latents')
    
    plt.legend()
    plt.title("Target vs Flow Samples vs Trained Latents")
    plt.savefig(f"{save_dir}/three_way_comparison.png")
    plt.close()
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Flow Loss')
    if nll_history is not None:
        # Plot NLL on the same figure but with a different y-axis
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(nll_history, 'r--', label='NLL')
        ax2.set_ylabel('Negative Log Likelihood', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
    plt.xlabel("Training Step")
    plt.ylabel("Flow Loss")
    plt.title("Training Metrics Over Time")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_dir}/training_metrics.png")
    plt.close()
    
    # Create animation of training progress
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(frame):
        ax.clear()
        ax.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.3, color='blue', label='Target')
        
        latents_frame = history[frame].numpy()
        ax.scatter(latents_frame[:, 0], latents_frame[:, 1], alpha=0.3, color='red', label='Trained Latents')
        
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.legend()
        frame_step = frame * (len(loss_history) // len(history))
        title = f"Training Progress - Step {frame_step}"
        if nll_history is not None:
            title += f"\nNLL: {nll_history[frame_step]:.4f}"
        ax.set_title(title)
        return ax,
    
    ani = FuncAnimation(fig, update, frames=len(history), interval=200)
    ani.save(f"{save_dir}/latent_training_animation.gif", writer='pillow', dpi=100)
    plt.close()
    
    # Visualize vector fields if flow model is provided
    if flow_model is not None:
        vector_fields_dir = os.path.join(save_dir, "vector_fields")
        visualize_vector_fields_at_times(
            flow_model=flow_model,
            target_sampler=target_sampler,
            save_dir=vector_fields_dir
        )
        
        # Also visualize flow trajectories
        trajectories_path = os.path.join(save_dir, "flow_trajectories.png")
        try:
            device = next(flow_model.parameters()).device
            visualize_flow_trajectories(
                flow_model=flow_model,
                n_points=15,
                n_steps=100,
                save_path=trajectories_path,
                device=device
            )
        except Exception as e:
            print(f"Could not generate flow trajectories: {e}")
            
        # Add the loss heatmap visualization
        heatmap_path = os.path.join(save_dir, "loss_heatmap.png")
        try:
            device = next(flow_model.parameters()).device
            visualize_loss_heatmap(
                flow_model=flow_model,
                target_sampler=target_sampler,
                resolution=50,
                n_samples_t=20,
                save_path=heatmap_path,
                device=device
            )
        except Exception as e:
            print(f"Could not generate loss heatmap: {e}")
            
        # # Add NLL heatmap visualization for MixtureOfGaussians
        # if isinstance(target_sampler, MixtureOfGaussians):
        nll_heatmap_path = os.path.join(save_dir, "nll_heatmap.png")
        try:
            device = next(flow_model.parameters()).device
            visualize_nll_heatmap(
                target_sampler=target_sampler,
                trained_latents=trained_latents,
                resolution=50,
                save_path=nll_heatmap_path,
                device=device
            )
        except Exception as e:
            print(f"Could not generate NLL heatmap: {e}")


class KDELogLikelihood:
    """Estimate log likelihood using Kernel Density Estimation"""
    def __init__(self, target_sampler, n_samples=100000, bandwidth=0.1, device='cuda'):
        self.device = device
        self.bandwidth = bandwidth
        
        # Generate reference samples from target distribution
        with torch.no_grad():
            self.reference_samples = target_sampler.sample(n_samples).to(device)
    
    def log_likelihood(self, samples):
        """
        Compute log likelihood estimate using KDE
        Args:
            samples: Tensor of shape [batch_size, 2]
        Returns:
            Tensor of shape [batch_size] containing log likelihood estimates
        """
        # Compute pairwise distances between samples and reference points
        # Using broadcasting to compute differences
        diff = samples[:, None, :] - self.reference_samples[None, :, :]  # [batch_size, n_ref, 2]
        sq_distances = torch.sum(diff * diff, dim=-1)  # [batch_size, n_ref]
        
        # Apply Gaussian kernel
        kernel_values = torch.exp(-0.5 * sq_distances / (self.bandwidth ** 2))
        
        # Average over reference points and take log
        # Add small constant for numerical stability
        log_likelihood = torch.log(torch.mean(kernel_values, dim=1) + 1e-10)
        
        # Add normalization constant
        log_likelihood = log_likelihood - torch.log(torch.tensor(2 * np.pi * self.bandwidth ** 2, device=self.device))
        
        return log_likelihood


def run_experiment(target_sampler, name, train_flow=True, model_path=None, device='cuda'):
    """Run an experiment with a specific target distribution"""
    flow_model_save_dir = f"flow_model_{name}"
    latents_save_dir = f"flow_guided_latents_{name}"
    n_training_steps = 100000
    lr = 1e-4
    
    # Initialize KDE log likelihood estimator if not using MixtureOfGaussians
    if not isinstance(target_sampler, (MixtureOfGaussians, GridOfGaussiansDistribution)):
        kde_estimator = KDELogLikelihood(target_sampler, n_samples=100000, bandwidth=0.1, device=device)
        target_sampler.log_likelihood = kde_estimator.log_likelihood
    
    # Step 1: Train or load flow model
    if train_flow:
        print(f"Training flow model for {name} distribution...")
        flow_model, flow_loss_history, sample_history = train_flow_model(
            target_sampler=target_sampler,
            n_steps=n_training_steps,  # More steps for better flow model training
            batch_size=256,
            lr=lr,
            device=device,
            model_channels=512,
            num_res_blocks=4,
            save_dir=flow_model_save_dir
        )
    else:
        if model_path is None:
            # Use default path if not specified
            model_path = os.path.join(flow_model_save_dir, "flow_model.pt")
        
        # Check if the model file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print(f"Please use --train_flow flag to train a new flow model first, or specify a valid model path.")
            print(f"Falling back to training a new flow model...")
            
            # Fall back to training a new model
            flow_model, flow_loss_history, sample_history = train_flow_model(
                target_sampler=target_sampler,
                n_steps=n_training_steps,
                batch_size=256,
                lr=lr,
                device=device,
                model_channels=512,
                num_res_blocks=4,
                save_dir=flow_model_save_dir
            )
        else:
            print(f"Loading pre-trained flow model for {name} distribution from {model_path}")
            flow_model = load_flow_model(
                model_path=model_path,
                device=device,
                model_channels=512,
                num_res_blocks=4
            )
            
            # Ensure target_sampler is normalized by sampling once
            print("Computing normalization statistics for target sampler...")
            with torch.no_grad():
                _ = target_sampler.sample(1000)
    
    # Step 2: Train latents using the flow model
    print(f"Training latents for {name} distribution using flow model...")
    trained_latents, latent_history, latent_loss_history, flow_samples, nll_history = train_latents_with_flow(
        flow_model=flow_model,
        n_latents=1000,
        n_steps=5000,
        lr=1e-2,
        device=device,
        save_dir=latents_save_dir,
        target_sampler=target_sampler
    )

    # Save everything for visualization
    results = {
        'name': name,
        'flow_model': flow_model,
        'trained_latents': trained_latents,
        'latent_history': latent_history,
        'latent_loss_history': latent_loss_history,
        'flow_samples': flow_samples,
        'target_sampler': target_sampler,
        'nll_history': nll_history
    }

    # Create results directory if it doesn't exist
    results_dir = os.path.join(latents_save_dir, 'full_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results
    results_path = os.path.join(results_dir, f'{name}_results.pt')
    torch.save(results, results_path)
    print(f"Saved full results to {results_path}")
    
    # Step 3: Visualize results
    print(f"Visualizing results for {name} distribution...")
    visualize_results(
        target_sampler=target_sampler,
        trained_latents=trained_latents,
        history=latent_history,
        loss_history=latent_loss_history,
        flow_samples=flow_samples,
        flow_model=flow_model,  # Pass flow model to visualize vector fields
        save_dir=latents_save_dir,
        nll_history=nll_history
    )
    
    return flow_model, trained_latents, flow_samples


# Define target distributions
class MixtureOfGaussians:
    """Mixture of Gaussians target distribution"""
    def __init__(self, n_components=5, scale=1.0, radius=3.0, device='cuda'):
        self.n_components = n_components
        self.scale = scale
        self.radius = radius
        self.device = device
        
        # Create mixture component means
        angles = torch.linspace(0, 2*np.pi, n_components+1)[:-1]
        self.means = torch.stack([
            radius * torch.cos(angles),
            radius * torch.sin(angles)
        ], dim=1).to(device)
        
    @normalize_samples
    def sample(self, n_samples):
        # Choose components for each sample
        component_indices = torch.randint(0, self.n_components, (n_samples,), device=self.device)
        selected_means = self.means[component_indices]
        
        # Add Gaussian noise around the means
        samples = selected_means + self.scale * torch.randn_like(selected_means)
        return samples

    def log_likelihood(self, samples):
        """
        Compute the log likelihood of samples under the mixture of Gaussians distribution.
        Takes into account the normalization from the normalize_samples decorator.
        
        Args:
            samples: Tensor of shape [batch_size, 2] containing the samples to evaluate
            
        Returns:
            Tensor of shape [batch_size] containing the log likelihood of each sample
        """
        # First, ensure we have computed the normalization std
        if not hasattr(self, '_std'):
            print("Computing normalization statistics from large sample...")
            # Get large sample for computing statistics
            with torch.no_grad():
                large_samples = self.sample(10000000)
                self._std = torch.std(large_samples)
                print(f"Computed std: {self._std}")
        
        # Unnormalize the samples since they are assumed to be normalized
        unnormalized_samples = samples * self._std
        
        # Compute log likelihood for each component
        log_probs = []
        for mean in self.means:
            # For each component, compute Gaussian log probability
            diff = unnormalized_samples - mean[None, :]  # [batch_size, 2]
            log_prob = -0.5 * torch.sum(diff * diff, dim=1) / (self.scale ** 2) \
                      - torch.log(torch.tensor(2 * np.pi * self.scale ** 2, device=self.device))
            log_probs.append(log_prob)
        
        # Stack component log probabilities [n_components, batch_size]
        log_probs = torch.stack(log_probs)
        
        # Log sum exp trick for numerical stability
        max_log_prob = torch.max(log_probs, dim=0)[0]
        log_likelihood = max_log_prob + torch.log(
            torch.sum(
                torch.exp(log_probs - max_log_prob[None, :]), 
                dim=0
            )
        )
        
        # Add log(1/n_components) since components are equally weighted
        log_likelihood = log_likelihood - torch.log(torch.tensor(self.n_components, device=self.device))
        
        return log_likelihood


class SpiralDistribution:
    """Simplified spiral distribution with options to make it easier to fit"""
    def __init__(self, noise=0.15, device='cuda', turns=3, single_arm=False, max_radius=4.0, growth_rate=0.5):
        self.noise = noise
        self.device = device
        self.turns = turns  # Number of spiral turns (lower is easier to fit)
        self.single_arm = single_arm  # Whether to use a single arm (easier) or two arms
        self.max_radius = max_radius  # Maximum radius to keep points from spreading too far
        self.growth_rate = growth_rate  # Rate at which spiral grows (higher = less tightly wound)
        
    @normalize_samples
    def sample(self, n_samples):
        # Generate random angles
        theta = self.turns * torch.rand(n_samples, device=self.device) * torch.pi
        
        # Compute radius (increasing with angle)
        r = 0.5 + theta * self.growth_rate / (2 * torch.pi)
        
        # Clamp radius to maximum value
        r = torch.clamp(r, max=self.max_radius)
        
        # Convert to Cartesian coordinates
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        
        # For two-arm spiral, flip half of the points
        if not self.single_arm:
            arm = torch.randint(0, 2, (n_samples,), device=self.device)
            y = y * (1 - 2 * arm)  # Flip half of points to second arm
        
        # Add noise
        x = x + self.noise * torch.randn_like(x)
        y = y + self.noise * torch.randn_like(y)
        
        # Combine
        samples = torch.stack([x, y], dim=1)
        return samples


class CheckerboardDistribution:
    """2D Checkerboard distribution with adjustable grid size"""
    def __init__(self, grid_size=4, scale=3.0, noise=0.05, device='cuda'):
        self.grid_size = grid_size
        self.scale = scale
        self.noise = noise
        self.device = device
        
    @normalize_samples
    def sample(self, n_samples):
        # Create uniform samples between -scale and scale
        samples = torch.rand(n_samples, 2, device=self.device) * 2 * self.scale - self.scale
        print(f"Initial samples shape: {samples.shape}, range: [{samples.min():.2f}, {samples.max():.2f}]")
        
        # Create grid pattern
        grid = torch.floor(samples * self.grid_size / (2 * self.scale))
        is_black = ((grid[:, 0] + grid[:, 1]) % 2 == 0)
        print(f"Grid shape: {grid.shape}, Black cells: {is_black.sum()}/{len(is_black)}")
        
        # Reject samples from "white" cells
        accepted = torch.zeros(n_samples, dtype=torch.bool, device=self.device)
        accepted[is_black] = (torch.rand(is_black.sum(), device=self.device) < 0.5)
        print(f"Initially accepted samples: {accepted.sum()}/{n_samples}")
        
        # Replace rejected samples until we have enough
        iteration = 0
        while accepted.sum() < n_samples:
            iteration += 1
            # Get number of samples still needed
            needed = n_samples - accepted.sum()
            print(f"\nIteration {iteration}: Generating {needed} more samples")
            
            # Generate new candidates
            new_samples = torch.rand(needed, 2, device=self.device) * 2 * self.scale - self.scale
            new_grid = torch.floor(new_samples * self.grid_size / (2 * self.scale))
            new_is_black = ((new_grid[:, 0] + new_grid[:, 1]) % 2 == 0)
            print(f"New black cells: {new_is_black.sum()}/{len(new_is_black)}")
            
            # Accept with 0.5 probability from black cells
            new_accepted = torch.zeros(needed, dtype=torch.bool, device=self.device)
            new_accepted[new_is_black] = (torch.rand(new_is_black.sum(), device=self.device) < 0.5)
            print(f"Newly accepted samples: {new_accepted.sum()}")
            
            # Add to samples
            samples[~accepted][:new_accepted.sum()] = new_samples[new_accepted]
            accepted[~accepted][:new_accepted.sum()] = True
            print(f"Total accepted so far: {accepted.sum()}/{n_samples}")
        
        # Add noise
        samples = samples + self.noise * torch.randn_like(samples)
        print(f"\nFinal samples shape: {samples.shape}, range: [{samples.min():.2f}, {samples.max():.2f}]")
        
        return samples


class MoonsDistribution:
    """Two interleaving half-moons distribution"""
    def __init__(self, radius=4.0, noise=0.15, device='cuda'):
        self.radius = radius
        self.noise = noise
        self.device = device
        
    @normalize_samples
    def sample(self, n_samples):
        # Ensure even number of samples
        n_per_moon = n_samples // 2
        remainder = n_samples - 2 * n_per_moon
        
        # Generate angles for the first moon (upper half)
        theta1 = torch.rand(n_per_moon, device=self.device) * torch.pi
        
        # Generate coordinates for first moon
        x1 = self.radius * torch.cos(theta1)
        y1 = self.radius * torch.sin(theta1)
        
        # Generate angles for the second moon (lower half, shifted)
        theta2 = torch.rand(n_per_moon + remainder, device=self.device) * torch.pi + torch.pi
        
        # Generate coordinates for second moon (offset to interleave)
        x2 = self.radius * torch.cos(theta2) + self.radius
        y2 = self.radius * torch.sin(theta2) - 0.5
        
        # Combine moons
        x = torch.cat([x1, x2])
        y = torch.cat([y1, y2])
        
        # Add noise
        x = x + self.noise * torch.randn_like(x)
        y = y + self.noise * torch.randn_like(y)
        
        # Combine coordinates
        samples = torch.stack([x, y], dim=1)
        return samples


class ConcentricRingsDistribution:
    """Concentric rings distribution with specified number of rings"""
    def __init__(self, n_rings=3, max_radius=4.0, noise=0.1, device='cuda'):
        self.n_rings = n_rings
        self.max_radius = max_radius
        self.noise = noise
        self.device = device
        
        # Calculate radii for the rings
        self.radii = torch.linspace(max_radius / n_rings, max_radius, n_rings)
        
    @normalize_samples
    def sample(self, n_samples):
        # Determine samples per ring (roughly equal)
        n_per_ring = [n_samples // self.n_rings] * self.n_rings
        # Add remainder to last ring
        n_per_ring[-1] += n_samples - sum(n_per_ring)
        
        samples_list = []
        
        # Generate samples for each ring
        for i, radius in enumerate(self.radii):
            # Generate random angles
            theta = torch.rand(n_per_ring[i], device=self.device) * 2 * torch.pi
            
            # Convert to cartesian coordinates
            x = radius * torch.cos(theta)
            y = radius * torch.sin(theta)
            
            # Add noise (radial and angular)
            radial_noise = self.noise * torch.randn(n_per_ring[i], device=self.device)
            x = x + radial_noise * torch.cos(theta)
            y = y + radial_noise * torch.sin(theta)
            
            # Stack and add to list
            ring_samples = torch.stack([x, y], dim=1)
            samples_list.append(ring_samples)
        
        # Combine all samples
        samples = torch.cat(samples_list, dim=0)
        
        return samples


class SwissRollDistribution:
    """Swiss roll distribution - a 2D manifold that resembles a rolled-up sheet"""
    def __init__(self, turns=3, noise=0.1, device='cuda'):
        self.turns = turns
        self.noise = noise
        self.device = device
        
    @normalize_samples
    def sample(self, n_samples):
        # Generate random parameters
        t = torch.rand(n_samples, device=self.device) * self.turns * 2 * torch.pi
        y = torch.rand(n_samples, device=self.device) * 2 - 1
        
        # Generate spiral
        x = t * torch.cos(t)
        z = t * torch.sin(t)
        
        # Scale to make it more visually appealing
        x = x / (2 * torch.pi) * 2
        z = z / (2 * torch.pi) * 2
        
        # Add noise
        x = x + self.noise * torch.randn_like(x)
        z = z + self.noise * torch.randn_like(z)
        
        # For 2D visualization, use x and z as coordinates
        samples = torch.stack([x, z], dim=1)
        return samples


class GridOfGaussiansDistribution:
    """A grid of Gaussian distributions"""
    def __init__(self, grid_size=4, cell_scale=1.0, gaussian_scale=0.1, device='cuda'):
        self.grid_size = grid_size
        self.cell_scale = cell_scale
        self.gaussian_scale = gaussian_scale
        self.device = device
        
        # Create grid of means
        x_means = torch.linspace(-cell_scale * (grid_size-1)/2, cell_scale * (grid_size-1)/2, grid_size)
        y_means = torch.linspace(-cell_scale * (grid_size-1)/2, cell_scale * (grid_size-1)/2, grid_size)
        
        # Create all combinations of x, y means
        self.means = torch.stack(torch.meshgrid(x_means, y_means, indexing='ij'), dim=-1).reshape(-1, 2).to(device)
        self.n_components = len(self.means)
        
    @normalize_samples
    def sample(self, n_samples):
        # Choose components for each sample
        component_indices = torch.randint(0, self.n_components, (n_samples,), device=self.device)
        selected_means = self.means[component_indices]
        
        # Add Gaussian noise around the means
        samples = selected_means + self.gaussian_scale * torch.randn_like(selected_means)
        return samples
    
    def log_likelihood(self, samples):
        if not hasattr(self, '_std'):
            print("Computing normalization statistics from large sample...")
            # Get large sample for computing statistics
            with torch.no_grad():
                large_samples = self.sample(10000000)
                self._std = torch.std(large_samples)
                print(f"Computed std: {self._std}")

        unnormalized_samples = samples * self._std

        log_probs = []
        for mean in self.means:
            diff = unnormalized_samples - mean
            log_prob =  -0.5 * torch.sum(diff * diff, dim=1) / (self.gaussian_scale ** 2) \
                      - torch.log(torch.tensor(2 * np.pi * self.gaussian_scale ** 2, device=self.device))
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs)
        
        # Log sum exp trick for numerical stability
        max_log_prob = torch.max(log_probs, dim=0)[0]
        log_likelihood = max_log_prob + torch.log(
            torch.sum(
                torch.exp(log_probs - max_log_prob[None, :]), 
                dim=0
            )
        )
        
        # Add log(1/n_components) since components are equally weighted
        log_likelihood = log_likelihood - torch.log(torch.tensor(self.n_components, device=self.device))
        
        return log_likelihood


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Flow-based distribution learning")
    parser.add_argument("--train_flow", action="store_true", help="Train the flow model (instead of loading from disk)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pre-trained flow model (if not training)")
    parser.add_argument("--distribution", type=str, default="gaussian", 
                       choices=["gaussian", "spiral", "checkerboard", "moons", "rings", "swissroll", "grid", "gaussian_large"], 
                       help="Target distribution to use")
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define target distribution based on args
    if args.distribution == "gaussian":
        target_distribution = MixtureOfGaussians(n_components=5, scale=0.3, device=device)
        # target_distribution = MixtureOfGaussians(n_components=5, scale=0.3, device=device)
        name = "mixture_of_gaussians"
    elif args.distribution == "spiral":
        # Create an easier-to-fit spiral with more noise, fewer turns, and simpler structure
        target_distribution = SpiralDistribution(
            noise=0.10,          # More noise makes it easier to fit
            device='cpu',
            turns=3,             # Fewer turns (was 8 originally)
            single_arm=True,     # Single arm is simpler than two arms
            max_radius=3.0,      # Limit the maximum radius
            growth_rate=1.0      # Less tightly wound spiral
        )
        name = "spiral_easy"
    elif args.distribution == "moons":
        target_distribution = MoonsDistribution(radius=2.0, noise=0.10, device=device)
        name = "moons"
    elif args.distribution == "rings":
        target_distribution = ConcentricRingsDistribution(n_rings=3, max_radius=4.0, noise=0.1, device=device)
        name = "rings"
    elif args.distribution == "swissroll":
        target_distribution = SwissRollDistribution(turns=3, noise=0.1, device=device)
        name = "swissroll"
    elif args.distribution == "grid":
        target_distribution = GridOfGaussiansDistribution(grid_size=4, cell_scale=1.0, gaussian_scale=0.1, device=device)
        name = "grid_of_gaussians"
    else:
        print(f"Error: Unknown distribution {args.distribution}")
        return
    
    # Run experiment
    run_experiment(
        target_sampler=target_distribution,
        name=name,
        train_flow=args.train_flow,
        model_path=args.model_path,
        device=device
    )


if __name__ == "__main__":
    main() 

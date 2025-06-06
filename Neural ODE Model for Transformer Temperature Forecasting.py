"""
Neural ODE Model for Transformer Temperature Forecasting
Author: Berk Hadzhamolla
Date: June 2025

This script demonstrates how to train a Neural Ordinary Differential Equation (Neural ODE) model
on synthetic time-series data representing power transformer thermal behavior.

Key Features:
- Neural ODE implementation for continuous-time modeling
- Time-dependent heat transfer simulation
- Comprehensive evaluation metrics and visualization
- GPU acceleration support
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def generate_dummy_data(n=3000, seed=0):
    """
    Generate synthetic transformer sensor data.
    
    Args:
        n (int): Number of data points
        seed (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Synthetic transformer data
    """
    np.random.seed(seed)
    return pd.DataFrame({
        "highvoltpower": np.random.normal(2000, 300, n),
        "lowvoltpower": np.random.normal(1000, 150, n),
        "ambtemperatures": np.random.normal(15, 5, n),
        "oiltemp": np.random.normal(60, 5, n),
        "highvoltwinding": np.random.normal(70, 4, n),
        "lowvoltwinding": np.random.normal(65, 4, n),
        "highvolthotspot": np.random.normal(90, 6, n),
        "lowvolthotspot": np.random.normal(85, 6, n),
    })

class HeatODEFunc(nn.Module):
    """
    Neural ODE function for modeling heat transfer dynamics in transformers.
    
    This module represents the right-hand side of the ODE: dy/dt = f(t, y, x(t))
    where y represents temperatures and x(t) represents external inputs.
    """
    
    def __init__(self, X_tensor, input_dim, hidden_dim, output_dim, delta_t=900.0, scale=0.01):
        super().__init__()
        self.X_tensor = X_tensor
        self.delta_t = delta_t  # Time step in seconds (15 minutes)
        self.scale = scale      # Scaling factor for stability
        
        # Neural network to approximate the ODE function
        self.net = nn.Sequential(
            nn.Linear(input_dim + output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, t, y, base_idx):
        """
        Compute dy/dt at time t.
        
        Args:
            t (torch.Tensor): Current time
            y (torch.Tensor): Current state (temperatures)
            base_idx (torch.Tensor): Base indices for input lookup
        
        Returns:
            torch.Tensor: Time derivative dy/dt
        """
        offset = int(t.item() // self.delta_t)
        global_indices = base_idx + offset
        global_indices = torch.clamp(global_indices, 0, len(self.X_tensor) - 1)
        
        # Get input features at current time
        x_t = self.X_tensor[global_indices]
        
        # Combine current state with input features
        input_combined = torch.cat([y, x_t], dim=1)
        
        # Return scaled derivative
        return self.net(input_combined) * self.scale

class NeuralODE(nn.Module):
    """
    Neural ODE wrapper for solving differential equations.
    """
    
    def __init__(self, ode_func):
        super().__init__()
        self.ode_func = ode_func

    def forward(self, y0, t, base_idx):
        """
        Solve the ODE from initial condition y0 over time points t.
        
        Args:
            y0 (torch.Tensor): Initial conditions
            t (torch.Tensor): Time points
            base_idx (torch.Tensor): Base indices for input lookup
        
        Returns:
            torch.Tensor: Solution trajectories
        """
        return odeint(
            lambda time, y: self.ode_func(time, y, base_idx), 
            y0, t, method='dopri5'
        )

def choose_time_unit_and_factor(t_values):
    """
    Choose appropriate time unit for plotting based on time scale.
    
    Args:
        t_values (np.ndarray): Time values in seconds
    
    Returns:
        tuple: (unit_name, conversion_factor)
    """
    max_time = t_values[-1]
    if max_time > 86400:  # More than 1 day
        return "Days", 1 / 86400
    elif max_time > 3600:  # More than 1 hour
        return "Hours", 1 / 3600
    elif max_time > 60:    # More than 1 minute
        return "Minutes", 1 / 60
    else:
        return "Seconds", 1.0

def evaluate_model(model, X_tensor, Y_tensor, scaler_y, device, T_eval=50000, delta_t=900.0, title_prefix=""):
    """
    Evaluate model performance and generate plots.
    
    Args:
        model: Trained Neural ODE model
        X_tensor: Input features tensor
        Y_tensor: Target tensor
        scaler_y: Target scaler for inverse transformation
        device: Computing device
        T_eval: Number of time steps to evaluate
        delta_t: Time step size
        title_prefix: Prefix for plot titles
    
    Returns:
        dict: Evaluation metrics
    """
    # Ensure evaluation doesn't exceed data length
    T_eval = min(T_eval, len(Y_tensor))
    
    # Time vector for evaluation
    t_eval = torch.linspace(0, (T_eval - 1) * delta_t, T_eval).to(device)
    
    # Set input tensor for model
    model.ode_func.X_tensor = X_tensor
    
    # Use mean of first 10 states as starting condition
    y0 = Y_tensor[:10].mean(dim=0, keepdim=True)
    
    # Generate predictions
    with torch.no_grad():
        pred = odeint(
            lambda t, y: model.ode_func(t, y, base_idx=torch.tensor([0], device=device)),
            y0, t_eval, method='rk4'
        ).squeeze(1)
    
    # Inverse transform predictions and true values
    pred_inv = scaler_y.inverse_transform(pred.cpu().numpy())
    Y_true_inv = scaler_y.inverse_transform(Y_tensor.cpu().numpy()[:T_eval])
    
    # Time scaling for plots
    t_values = t_eval.cpu().numpy()
    time_unit, factor = choose_time_unit_and_factor(t_values)
    t_scaled = t_values * factor
    
    # Variable names
    variables = ['Oil Temp', 'High Voltage Winding', 'Low Voltage Winding', 
                'High Voltage Hotspot', 'Low Voltage Hotspot']
    
    # Calculate metrics
    mae_values, mse_values, r2_values = [], [], []
    for i in range(len(variables)):
        y_true, y_pred = Y_true_inv[:, i], pred_inv[:, i]
        mae_values.append(mean_absolute_error(y_true, y_pred))
        mse_values.append(mean_squared_error(y_true, y_pred))
        r2_values.append(r2_score(y_true, y_pred))
    
    # Overall metrics
    overall_mae = np.mean(mae_values)
    overall_mse = np.mean(mse_values)
    overall_r2 = np.mean(r2_values)
    
    # Print metrics
    print(f"\n{title_prefix}Performance Metrics:")
    print("-" * 50)
    for i, var in enumerate(variables):
        print(f"{var:20} -> MAE: {mae_values[i]:.3f}, MSE: {mse_values[i]:.3f}, R²: {r2_values[i]:.3f}")
    print(f"\n📊 Overall Metrics -> MAE: {overall_mae:.3f}, MSE: {overall_mse:.3f}, R²: {overall_r2:.3f}")
    
    # Generate plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(variables):
        ax = axes[i]
        ax.plot(t_scaled, Y_true_inv[:, i], label='Actual', color='blue', linewidth=1.5)
        ax.plot(t_scaled, pred_inv[:, i], label='Predicted', color='red', 
                linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_title(f"{title_prefix}{var}")
        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel(f"{var} (°C)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide the last subplot if not needed
    if len(variables) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mae': overall_mae,
        'mse': overall_mse,
        'r2': overall_r2,
        'individual_metrics': {
            'mae': mae_values,
            'mse': mse_values,
            'r2': r2_values
        }
    }

def main():
    """
    Main training and evaluation pipeline.
    """
    print("🔥 Neural ODE Transformer Temperature Forecasting")
    print("=" * 60)
    
    # Generate synthetic data
    print("📊 Generating synthetic transformer data...")
    df = generate_dummy_data(n=3000, seed=42)  # Changed seed for reproducibility
    
    # Define features and targets
    features = ["highvoltpower", "lowvoltpower", "ambtemperatures"]
    targets = ["oiltemp", "highvoltwinding", "lowvoltwinding", "highvolthotspot", "lowvolthotspot"]
    
    # Split data
    train_ratio = 0.6
    val_ratio = 0.2  # Added validation set
    num_train = int(len(df) * train_ratio)
    num_val = int(len(df) * val_ratio)
    
    train_df = df.iloc[:num_train].copy()
    val_df = df.iloc[num_train:num_train+num_val].copy()
    test_df = df.iloc[num_train+num_val:].copy()
    
    print(f"📈 Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Normalize data
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_np = scaler_x.fit_transform(train_df[features])
    X_val_np = scaler_x.transform(val_df[features])
    X_test_np = scaler_x.transform(test_df[features])
    
    Y_train_np = scaler_y.fit_transform(train_df[targets])
    Y_val_np = scaler_y.transform(val_df[targets])
    Y_test_np = scaler_y.transform(test_df[targets])
    
    # Convert to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val_np, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    
    Y_train = torch.tensor(Y_train_np, dtype=torch.float32).to(device)
    Y_val = torch.tensor(Y_val_np, dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y_test_np, dtype=torch.float32).to(device)
    
    # Initialize model
    delta_t = 900.0  # 15 minutes
    ode_func = HeatODEFunc(
        X_train, 
        input_dim=len(features), 
        hidden_dim=256, 
        output_dim=len(targets), 
        delta_t=delta_t,
        scale=0.01
    )
    model = NeuralODE(ode_func).to(device)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    # Training parameters
    epochs = 60
    batch_size = 512  # Reduced for better memory usage
    base_time_steps = 5
    time_step_increment = 1
    stride = 1
    
    print(f"🚀 Starting training for {epochs} epochs...")
    
    # Training loop
    for epoch in range(epochs):
        time_steps = base_time_steps + epoch * time_step_increment
        t_window = torch.linspace(0, delta_t * (time_steps - 1), time_steps).to(device)
        # Add small noise to prevent numerical issues
        t_window += torch.arange(len(t_window)).float().to(device) * 1e-6
        
        max_start_idx = len(Y_train) - time_steps
        all_start_indices = list(range(0, max_start_idx, stride))
        
        model.train()
        epoch_loss = 0.0
        num_batches = len(all_start_indices) // batch_size
        
        if epoch % 10 == 0:  # Print less frequently
            print(f"\n=== Epoch {epoch+1}/{epochs} | Time Steps: {time_steps} ===")
        
        for b in range(num_batches):
            batch_start = b * batch_size
            batch_indices = all_start_indices[batch_start:batch_start + batch_size]
            if len(batch_indices) < batch_size:
                break
            
            base_idx = torch.tensor(batch_indices, device=device)
            y0 = torch.stack([Y_train[idx] for idx in base_idx], dim=0)
            
            targets = torch.cat([
                Y_train[idx:idx + time_steps].unsqueeze(0)
                for idx in base_idx
            ], dim=0)
            
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = odeint(
                lambda time, y: model.ode_func(time, y, base_idx),
                y0, t_window, method='rk4'
            ).permute(1, 0, 2)
            
            loss = criterion(y_pred, targets)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 10 == 0 and num_batches > 0:
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.6f}")
        
        scheduler.step()
    
    print("\n✅ Training completed!")
    
    # Evaluation
    print("\n🔍 Evaluating model performance...")
    
    # Training set evaluation
    train_metrics = evaluate_model(
        model, X_train, Y_train, scaler_y, device, 
        T_eval=1000, delta_t=delta_t, title_prefix="Training - "
    )
    
    # Validation set evaluation
    val_metrics = evaluate_model(
        model, X_val, Y_val, scaler_y, device, 
        T_eval=len(Y_val), delta_t=delta_t, title_prefix="Validation - "
    )
    
    # Test set evaluation
    test_metrics = evaluate_model(
        model, X_test, Y_test, scaler_y, device, 
        T_eval=len(Y_test), delta_t=delta_t, title_prefix="Test - "
    )
    
    # Summary
    print("\n📋 FINAL SUMMARY")
    print("=" * 60)
    print(f"Training   R²: {train_metrics['r2']:.3f}")
    print(f"Validation R²: {val_metrics['r2']:.3f}")
    print(f"Test       R²: {test_metrics['r2']:.3f}")
    
    return model, scaler_x, scaler_y

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    model, scaler_x, scaler_y = main()
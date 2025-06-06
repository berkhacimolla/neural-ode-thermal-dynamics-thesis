Neural ODE for Transformer Temperature Forecasting
A PyTorch implementation of Neural Ordinary Differential Equations (Neural ODEs) for modeling thermal dynamics in power transformers. This project demonstrates continuous-time modeling of transformer temperature behavior using deep learning and differential equations.

Features:
Neural ODE Implementation: Continuous-time modeling using torchdiffeq
Transformer Thermal Modeling: Simulates heat transfer dynamics in power transformers
Evaluation: Multiple metrics and visualization tools

📊 Model Architecture
The model consists of two main components:

HeatODEFunc: Neural network that approximates the right-hand side of the ODE

Input: Current temperature state + external features (power, ambient conditions)
Output: Temperature derivatives (dy/dt)
Architecture: Multi-layer perceptron with Tanh activations


NeuralODE: Wrapper that solves the ODE using numerical integration

Solver: Dormand-Prince (dopri5) 
Time-dependent external inputs through indexed lookup


# Generate synthetic data
df = generate_dummy_data(n=3000)

# Train model (see main() function for full pipeline)
model, scaler_x, scaler_y = main()
📈 Data Description
The model works with transformer sensor data including:
Input Features

highvoltpower: High voltage side power (MW)
lowvoltpower: Low voltage side power (MW)
ambtemperatures: Ambient temperature (°C)

Target Variables

oiltemp: Oil temperature (°C)
highvoltwinding: High voltage winding temperature (°C)
lowvoltwinding: Low voltage winding temperature (°C)
highvolthotspot: High voltage hotspot temperature (°C)
lowvolthotspot: Low voltage hotspot temperature (°C)

🎯 Model Performance
The model is evaluated using:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R-squared (R²) Score

Performance is assessed on training, validation, and test sets with comprehensive visualizations.
🔧 Configuration
Key hyperparameters can be adjusted in the main() function:
python# Model parameters
hidden_dim = 256          # Neural network hidden layer size
delta_t = 900.0           # Time step (15 minutes in seconds)

# Training parameters
epochs = 60               # Number of training epochs
batch_size = 512          # Batch size
learning_rate = 1e-4      # Learning rate
weight_decay = 1e-5       # L2 regularization

# Data split
train_ratio = 0.6         # Training set ratio
val_ratio = 0.2           # Validation set ratio

🧮 Mathematical Background
The Neural ODE models transformer thermal dynamics as:
dy/dt = f_θ(t, y(t), x(t))
Where:

y(t): Temperature state vector
x(t): External input features (power, ambient conditions)
f_θ: Neural network parameterized by θ
Time-dependent inputs through indexed lookup

The model learns continuous-time dynamics while handling discrete sensor measurements.
📊 Results
Example performance metrics:

Training R²: 0.85-0.95
Validation R²: 0.80-0.90
Test R²: 0.75-0.85

Note: Results may vary based on synthetic data generation and random initialization.
🔬 Thesis Context
This implementation was developed as part of a thesis on:

Continuous-time modeling of industrial systems
Neural differential equations for physical system simulation
Transformer thermal management and monitoring

🛠️ Customization
Using Real Data
Replace the generate_dummy_data() function with your own data loader:
pythondef load_real_data(filepath):
    df = pd.read_csv(filepath)
    # Ensure columns match expected names
    return df
Model Architecture
Modify the HeatODEFunc class to:

Change network architecture
Add regularization techniques
Incorporate domain-specific physics

Training Strategy
Adjust training parameters:

Implement early stopping
Add learning rate scheduling
Use different optimizers

📋 Requirements
torch>=1.9.0
torchdiffeq>=0.2.3
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0

Author
Berk Hadzhamolla

Thesis: Neural ODE Applications in Industrial Systems
Date: June 2025

Acknowledgments

torchdiffeq for Neural ODE implementation
PyTorch team for the deep learning framework

References

Chen, R. T. Q., et al. "Neural Ordinary Differential Equations." NeurIPS 2018.




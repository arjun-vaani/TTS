import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import StudentT, Normal, InverseGamma

class EvidentialUSMProjector(nn.Module):
    def __init__(self, mimi_dim=512, usm_dim=1536):
        super().__init__()
        self.mimi_dim = mimi_dim
        self.usm_dim = usm_dim
        
        # Projects from Mimi hidden state to 4 parameters for EACH USM dimension
        # 1536 * 4 = 6144 output channels
        self.projector = nn.Linear(mimi_dim, usm_dim * 4)

    def forward(self, x):
        """
        x: Tensor of shape (Batch, Time, mimi_dim)
        Returns: (mu, v, alpha, beta) - each of shape (Batch, Time, usm_dim)
        """
        # 1. Linear Projection
        # Shape: (B, T, 512) -> (B, T, 6144)
        raw_output = self.projector(x)
        
        # 2. Reshape to separate the 4 parameters
        # Shape: (B, T, 1536, 4)
        batch, time, _ = raw_output.shape
        reshaped = raw_output.view(batch, time, self.usm_dim, 4)
        
        # 3. Split the parameters
        mu, v_raw, alpha_raw, beta_raw = torch.split(reshaped, 1, dim=-1)
        
        # 4. Apply Activations (Constraints)
        # mu: (-inf, inf) -> Linear (no activation)
        # v (lambda), alpha, beta: (0, inf) -> Softplus
        
        mu = mu.squeeze(-1)
        
        # We add epsilon (1e-6) to ensure stability and prevent division by zero
        # We add 1.0 to alpha to ensure "degrees of freedom" > 2 for well-defined variance
        v = F.softplus(v_raw.squeeze(-1)) + 1e-6
        alpha = F.softplus(alpha_raw.squeeze(-1)) + 1.0 
        beta = F.softplus(beta_raw.squeeze(-1)) + 1e-6
        
        return mu, v, alpha, beta

    def compute_loss(self, mu, v, alpha, beta, target_usm):
        """
        Training Step: Calculates the Negative Log Likelihood (NLL)
        of the Student's t-distribution.
        """
        # 1. Define Student's t-distribution parameters derived from NIG
        # df = 2 * alpha
        # loc = mu
        # scale = sqrt(beta * (1 + v) / (v * alpha))
        
        df = 2 * alpha
        scale = torch.sqrt(beta * (1 + v) / (v * alpha))

        # 2. Create Distribution
        student_dist = StudentT(df=df, loc=mu, scale=scale)
        
        # 3. Calculate Log Probability of the Target
        # "How likely is the real USM latent under this distribution?"
        log_prob = student_dist.log_prob(target_usm)
        
        # 4. Negative Log Likelihood (we want to minimize this)
        nll_loss = -torch.mean(log_prob)
        
        return nll_loss

    def sample(self, mu, v, alpha, beta):
        """
        Inference Step: Hierarchical Sampling
        """
        # 1. Sample Variance (sigma^2) from Inverse Gamma
        # alpha and beta define the distribution of the variance
        ig_dist = InverseGamma(concentration=alpha, rate=beta)
        sigma_sq = ig_dist.rsample() # rsample allows gradients if needed later
        
        # 2. Sample Latent (z) from Normal
        # Mean is mu, Variance is sigma^2 / v
        # Therefore standard deviation is sqrt(sigma^2 / v)
        std_dev = torch.sqrt(sigma_sq / v)
        
        normal_dist = Normal(loc=mu, scale=std_dev)
        z_sampled = normal_dist.rsample()
        
        return z_sampled

# ==========================================
#  Example / Debug Usage (protected)
# ==========================================

if __name__ == "__main__":
    # Hyperparameters
    B, T = 4, 100 # Batch 4, Sequence Length 100

    # Instantiate the model
    model = EvidentialUSMProjector(mimi_dim=512, usm_dim=1536)

    # --- 1. Simulate Training Loop ---
    mimi_output = torch.randn(B, T, 512)     # From your Mimi Codec
    real_usm_target = torch.randn(B, T, 1536) # From Google USM Encoder

    # Forward pass
    mu, v, alpha, beta = model(mimi_output)

    print(mu.shape, v.shape, alpha.shape, beta.shape)  # Each of shape (4, 100, 1536)

    # Calculate Loss (Evidential Regression Loss)
    loss = model.compute_loss(mu, v, alpha, beta, real_usm_target)
    print(f"Training Loss: {loss.item()}")

    # (You would do loss.backward() here)
    loss.backward()

    # --- 2. Simulate Inference Loop ---
    # Now we just want to generate a latent vector
    sampled_z = model.sample(mu, v, alpha, beta)

    print(f"Generated USM Latent Shape: {sampled_z.shape}") 
    # Output: (4, 100, 1536)



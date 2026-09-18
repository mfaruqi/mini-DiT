import torch
class LinearNoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas =  1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)




    def add_noise(self, x0, noise, t):
        """x0: (B, C, H, W), noise: (B, C, H, W), t: (B, ) int64
        returns x_t = sqrt(a_t)*x0 + sqrt(1-a_t)*noise"""

        x_t = self.unsqueeze_to(self.sqrt_alpha_cumprod[t], x0.ndim)*x0 + \
            self.unsqueeze_to(self.sqrt_one_minus_alpha_cumprod[t], noise.ndim)*noise


    def unsqueeze_to(self, x, ndim):
        """Helper function to convert somthing like [B] to [B, 1, 1, 1] 
        to broadcast a[t] to C, H, W channels for example"""
        while x.ndim < ndim:
            x = x.unsqueeze(-1)
        return x

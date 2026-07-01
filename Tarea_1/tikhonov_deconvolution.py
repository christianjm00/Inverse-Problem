#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of Tikhonov regularization with Morozov discrepancy principle
for the deconvolution problem from Chapter 1 of:

Vogel, Curtis R. Computational methods for inverse problems. Vol. 23. Siam, 2002.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import os


class TikhonovDeconvolution:
    """
    Solves the deconvolution problem using classical Tikhonov regularization
    with the Morozov discrepancy principle for parameter selection.
    """
    
    def __init__(self, n=200, gamma=0.025, SNR=100.0):
        """
        Initialize the deconvolution problem parameters.
        
        Parameters:
        -----------
        n : int
            Grid size for discretization
        gamma : float
            Standard deviation of the Gaussian kernel
        SNR : float
            Signal-to-noise ratio
        """
        self.n = n
        self.h = 1.0 / n
        self.gamma = gamma
        self.SNR = SNR
        self.C = 1.0 / (self.gamma * np.sqrt(2.0 * np.pi))
        
        # Create spatial grid
        x = np.linspace(1.0, float(self.n), self.n)
        self.x = (x - 0.5) * self.h
        
        # Build convolution matrix A (Gaussian kernel)
        xx, yy = np.meshgrid(self.x, self.x)
        self.A = self.h * self.C * np.exp(-(xx - yy)**2 / (2.0 * self.gamma**2))
    
    def f_true(self, x):
        """
        True function from Exercise 1.14 of Vogel's book.
        
        Parameters:
        -----------
        x : array
            Spatial coordinates
            
        Returns:
        --------
        fx : array
            Function values
        """
        fx = np.zeros(self.n)
        for j in range(self.n):
            if 0.1 < x[j] < 0.25:
                fx[j] = 0.75
            elif 0.3 < x[j] < 0.32:
                fx[j] = 0.25
            elif 0.5 < x[j] < 1.0:
                fx[j] = np.sin(2.0 * np.pi * x[j])**4
            else:
                fx[j] = 0.0
        return fx
    
    def generate_data(self):
        """
        Generate synthetic data: f, Af, and noisy observations.
        """
        # True function
        self.f_exact = self.f_true(self.x)
        
        # Forward model (convolution)
        self.b_exact = np.dot(self.A, self.f_exact)
        
        # Add noise
        self.noise_std = np.max(self.b_exact) / self.SNR
        np.random.seed(0)  # For reproducibility
        self.b_noisy = self.b_exact + self.noise_std * np.random.normal(size=self.n)
        
        # Calculate actual noise level
        self.noise_level = np.linalg.norm(self.b_noisy - self.b_exact)
    
    def solve_tikhonov(self, alpha):
        """
        Solve the Tikhonov regularized problem:
        min ||Af - b||2 + ||f||2
        
        Parameters:
        -----------
        alpha : float
            Regularization parameter
            
        Returns:
        --------
        f_reg : array
            Regularized solution
        """
        # Form regularized normal equations: (A^T A +  I) f = A^T b
        ATA = self.A.T @ self.A
        ATb = self.A.T @ self.b_noisy
        regularized_matrix = ATA + alpha * np.eye(self.n)
        
        # Solve the system
        f_reg = linalg.solve(regularized_matrix, ATb)
        return f_reg
    
    def residual_function(self, alpha):
        """
        Residual function for the discrepancy principle:
        ||A f - b|| = eta
        
        Parameters:
        -----------
        alpha : float
            Regularization parameter
            
        Returns:
        --------
        residual : float
            Residual value
        """
        f_reg = self.solve_tikhonov(alpha)
        data_residual = np.linalg.norm(self.A @ f_reg - self.b_noisy)
        return data_residual - self.noise_level
    
    def find_alpha_discrepancy(self, alpha_min=1e-10, alpha_max=1.0, tol=1e-6):
        """
        Find optimal regularization parameter using Morozov discrepancy principle.
        
        Parameters:
        -----------
        alpha_min : float
            Lower bound for alpha search
        alpha_max : float
            Upper bound for alpha search
        tol : float
            Tolerance for bisection method
            
        Returns:
        --------
        alpha : float
            Optimal regularization parameter
        """
        def same_sign(a, b):
            return a * b > 0
        
        # Check if root exists in the interval
        f_min = self.residual_function(alpha_min)
        f_max = self.residual_function(alpha_max)
        
        if same_sign(f_min, f_max):
            print(f"Warning: No sign change in interval [{alpha_min}, {alpha_max}]")
            print(f"f({alpha_min}) = {f_min}, f({alpha_max}) = {f_max}")
            return alpha_max  # Return conservative choice
        
        # Bisection method
        low, high = alpha_min, alpha_max
        for i in range(50):  # Maximum iterations
            mid = (low + high) / 2.0
            
            if abs(high - low) < tol:
                break
                
            f_mid = self.residual_function(mid)
            
            if same_sign(self.residual_function(low), f_mid):
                low = mid
            else:
                high = mid
        
        return (low + high) / 2.0
    
    def solve_and_plot(self):
        """
        Complete solution pipeline: generate data, find alpha, solve, and plot.
        """
        # Generate synthetic data
        self.generate_data()
        
        # Find optimal regularization parameter
        alpha_opt = self.find_alpha_discrepancy()
        print(f'Optimal alpha (discrepancy principle): {alpha_opt:.6e}')
        print(f'Noise level: {self.noise_level:.6e}')
        
        # Solve with optimal parameter
        f_tikhonov = self.solve_tikhonov(alpha_opt)
        
        # Compute condition-related quantities for second plot
        ATA_reg = self.A.T @ self.A + alpha_opt * np.eye(self.n)
        singular_values = linalg.svdvals(ATA_reg)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Plot 1: Reconstruction
        ax1.plot(self.x, self.f_exact, 'b.-', label=r'$x_{true}$)')
        ax1.plot(self.x, f_tikhonov, 'k.-', label=r'$(A^{T}A+\lambda I)A^{T}y_{obs}$',alpha=0.25)
        ax1.set_xlabel('t')
        ax1.set_ylabel('x')
        ax1.set_title('Regularizacion de Tikhonov')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Singular values and noise level
        ax2.loglog(singular_values, 'go', markersize=6, label='Valores singulares de $(A^TA + \\lambda I)$')
        ax2.axhline(y=self.noise_std, color='r', linestyle='--', linewidth=2, 
                   label=r'$\gamma_{noise}$')
        ax2.set_xlabel('Indice')
        ax2.set_ylabel('Valor')
        ax2.set_title('Valores singulares vs Nivel de ruido')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs("figs", exist_ok=True)
        plt.savefig('figs/tikhonov_clean.png', dpi=300, bbox_inches='tight')
        print("Figure saved as 'figs/tikhonov_clean.png'")        
        
        return f_tikhonov, alpha_opt


if __name__ == "__main__":
    # Create and solve the deconvolution problem
    deconv = TikhonovDeconvolution(n=100, gamma=0.025, SNR=100.0)
    f_solution, alpha = deconv.solve_and_plot()
    
    # Print some statistics
    error = np.linalg.norm(f_solution - deconv.f_exact)
    print(f'Reconstruction error: {error:.6e}')
    print(f'Relative error: {error/np.linalg.norm(deconv.f_exact):.6e}')

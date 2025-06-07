"""
Stochastic Optimization Module for Chain of Solution System
확률적 최적화부 (Stochastic Optimization Unit) - Patent Component 200

This module implements probabilistic optimization techniques for handling uncertainty
and finding optimal solutions in complex problem spaces.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import json
import logging
from abc import ABC, abstractmethod
from scipy import stats
from scipy.optimize import differential_evolution, basinhopping
import random

logger = logging.getLogger(__name__)


@dataclass
class OptimizationProblem:
    """Defines an optimization problem"""
    objective_function: Callable[[np.ndarray], float]
    bounds: List[Tuple[float, float]]
    constraints: Optional[List[Callable]] = None
    is_stochastic: bool = True
    noise_level: float = 0.1
    

@dataclass
class OptimizationResult:
    """Result of optimization process"""
    best_solution: np.ndarray
    best_value: float
    convergence_history: List[float]
    uncertainty_estimate: float
    confidence_interval: Tuple[float, float]
    iterations: int
    

class StochasticOptimizer(ABC):
    """Abstract base class for stochastic optimization algorithms"""
    
    @abstractmethod
    def optimize(self, problem: OptimizationProblem, max_iterations: int) -> OptimizationResult:
        """Perform optimization on the given problem"""
        pass


class ParticleSwarmOptimizer(StochasticOptimizer):
    """Particle Swarm Optimization for stochastic problems"""
    
    def __init__(self, n_particles: int = 30, w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        self.n_particles = n_particles
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter
        
    def optimize(self, problem: OptimizationProblem, max_iterations: int = 100) -> OptimizationResult:
        """Perform PSO optimization"""
        dim = len(problem.bounds)
        
        # Initialize particles
        particles = np.random.uniform(
            [b[0] for b in problem.bounds],
            [b[1] for b in problem.bounds],
            (self.n_particles, dim)
        )
        velocities = np.zeros((self.n_particles, dim))
        
        # Initialize personal and global bests
        personal_best = particles.copy()
        personal_best_values = np.array([problem.objective_function(p) for p in particles])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx].copy()
        global_best_value = personal_best_values[global_best_idx]
        
        convergence_history = [global_best_value]
        
        # Main optimization loop
        for iteration in range(max_iterations):
            for i in range(self.n_particles):
                # Update velocity
                r1, r2 = np.random.random(dim), np.random.random(dim)
                velocities[i] = (
                    self.w * velocities[i] +
                    self.c1 * r1 * (personal_best[i] - particles[i]) +
                    self.c2 * r2 * (global_best - particles[i])
                )
                
                # Update position
                particles[i] += velocities[i]
                
                # Apply bounds
                particles[i] = np.clip(
                    particles[i],
                    [b[0] for b in problem.bounds],
                    [b[1] for b in problem.bounds]
                )
                
                # Evaluate with noise if stochastic
                if problem.is_stochastic:
                    # Multiple evaluations for robustness
                    values = [problem.objective_function(particles[i]) for _ in range(5)]
                    value = np.mean(values)
                else:
                    value = problem.objective_function(particles[i])
                
                # Update personal best
                if value < personal_best_values[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_values[i] = value
                    
                    # Update global best
                    if value < global_best_value:
                        global_best = particles[i].copy()
                        global_best_value = value
                        
            convergence_history.append(global_best_value)
            
        # Estimate uncertainty
        final_evaluations = [problem.objective_function(global_best) for _ in range(30)]
        uncertainty = np.std(final_evaluations)
        confidence_interval = stats.t.interval(
            0.95, len(final_evaluations)-1,
            loc=np.mean(final_evaluations),
            scale=stats.sem(final_evaluations)
        )
        
        return OptimizationResult(
            best_solution=global_best,
            best_value=global_best_value,
            convergence_history=convergence_history,
            uncertainty_estimate=uncertainty,
            confidence_interval=confidence_interval,
            iterations=max_iterations
        )


class BayesianOptimizer(StochasticOptimizer):
    """Bayesian Optimization for expensive stochastic functions"""
    
    def __init__(self, acquisition_function: str = "ei"):
        self.acquisition_function = acquisition_function
        self.observations = []
        
    def _gaussian_process_predict(self, X_train, y_train, X_test):
        """Simple Gaussian Process prediction"""
        # Simplified GP - in practice would use sklearn GaussianProcessRegressor
        # Calculate RBF kernel
        def rbf_kernel(x1, x2, length_scale=1.0):
            return np.exp(-0.5 * np.sum((x1 - x2)**2) / length_scale**2)
        
        n_train = len(X_train)
        K = np.zeros((n_train, n_train))
        for i in range(n_train):
            for j in range(n_train):
                K[i, j] = rbf_kernel(X_train[i], X_train[j])
        K += 1e-6 * np.eye(n_train)  # Add noise term
        
        # Predict mean and variance
        k_star = np.array([rbf_kernel(X_test, X_train[i]) for i in range(n_train)])
        K_inv = np.linalg.inv(K)
        
        mean = k_star.dot(K_inv).dot(y_train)
        variance = 1.0 - k_star.dot(K_inv).dot(k_star)
        
        return mean, np.sqrt(max(0, variance))
    
    def _expected_improvement(self, X, X_train, y_train, best_y):
        """Calculate expected improvement acquisition function"""
        mean, std = self._gaussian_process_predict(X_train, y_train, X)
        
        if std == 0:
            return 0
            
        z = (best_y - mean) / std
        ei = (best_y - mean) * stats.norm.cdf(z) + std * stats.norm.pdf(z)
        
        return ei
    
    def optimize(self, problem: OptimizationProblem, max_iterations: int = 50) -> OptimizationResult:
        """Perform Bayesian optimization"""
        dim = len(problem.bounds)
        
        # Initial random samples
        n_initial = min(10, max_iterations // 2)
        X_train = np.random.uniform(
            [b[0] for b in problem.bounds],
            [b[1] for b in problem.bounds],
            (n_initial, dim)
        )
        y_train = np.array([problem.objective_function(x) for x in X_train])
        
        best_idx = np.argmin(y_train)
        best_x = X_train[best_idx]
        best_y = y_train[best_idx]
        
        convergence_history = [best_y]
        
        # Bayesian optimization loop
        for iteration in range(n_initial, max_iterations):
            # Define acquisition function to maximize
            def neg_acquisition(x):
                return -self._expected_improvement(x, X_train, y_train, best_y)
            
            # Find next point to evaluate
            result = differential_evolution(
                neg_acquisition,
                problem.bounds,
                maxiter=50,
                popsize=15
            )
            
            next_x = result.x
            
            # Evaluate objective function
            if problem.is_stochastic:
                # Multiple evaluations for robustness
                values = [problem.objective_function(next_x) for _ in range(3)]
                next_y = np.mean(values)
            else:
                next_y = problem.objective_function(next_x)
            
            # Update dataset
            X_train = np.vstack([X_train, next_x])
            y_train = np.append(y_train, next_y)
            
            # Update best
            if next_y < best_y:
                best_x = next_x
                best_y = next_y
                
            convergence_history.append(best_y)
            
        # Estimate uncertainty
        final_evaluations = [problem.objective_function(best_x) for _ in range(20)]
        uncertainty = np.std(final_evaluations)
        confidence_interval = stats.t.interval(
            0.95, len(final_evaluations)-1,
            loc=np.mean(final_evaluations),
            scale=stats.sem(final_evaluations)
        )
        
        return OptimizationResult(
            best_solution=best_x,
            best_value=best_y,
            convergence_history=convergence_history,
            uncertainty_estimate=uncertainty,
            confidence_interval=confidence_interval,
            iterations=max_iterations
        )


class StochasticOptimization:
    """Main stochastic optimization component"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.optimizers = {
            "pso": ParticleSwarmOptimizer(),
            "bayesian": BayesianOptimizer()
        }
        self.optimization_history = []
        
    def add_optimizer(self, name: str, optimizer: StochasticOptimizer):
        """Add a custom optimizer"""
        self.optimizers[name] = optimizer
        
    def optimize(self, problem: OptimizationProblem, 
                method: str = "pso", 
                max_iterations: int = 100) -> OptimizationResult:
        """Perform stochastic optimization"""
        if method not in self.optimizers:
            raise ValueError(f"Unknown optimization method: {method}")
            
        logger.info(f"Starting {method} optimization")
        
        optimizer = self.optimizers[method]
        result = optimizer.optimize(problem, max_iterations)
        
        # Store in history
        self.optimization_history.append({
            "method": method,
            "result": result,
            "problem_info": {
                "dimensions": len(problem.bounds),
                "is_stochastic": problem.is_stochastic,
                "noise_level": problem.noise_level
            }
        })
        
        logger.info(f"Optimization completed. Best value: {result.best_value:.6f}")
        
        return result
    
    def multi_objective_optimize(self, 
                               objectives: List[Callable],
                               bounds: List[Tuple[float, float]],
                               weights: Optional[List[float]] = None) -> OptimizationResult:
        """Perform multi-objective optimization"""
        if weights is None:
            weights = [1.0] * len(objectives)
            
        # Combine objectives with weights
        def combined_objective(x):
            values = [w * obj(x) for w, obj in zip(weights, objectives)]
            return sum(values)
            
        problem = OptimizationProblem(
            objective_function=combined_objective,
            bounds=bounds
        )
        
        # Use Bayesian optimization for multi-objective problems
        return self.optimize(problem, method="bayesian")
    
    def robust_optimize(self, problem: OptimizationProblem, 
                       n_scenarios: int = 10) -> OptimizationResult:
        """Perform robust optimization considering multiple scenarios"""
        logger.info(f"Starting robust optimization with {n_scenarios} scenarios")
        
        # Generate scenarios with different noise realizations
        def robust_objective(x):
            values = []
            for _ in range(n_scenarios):
                # Add random perturbation to simulate different scenarios
                perturbed_x = x + np.random.normal(0, problem.noise_level, len(x))
                perturbed_x = np.clip(
                    perturbed_x,
                    [b[0] for b in problem.bounds],
                    [b[1] for b in problem.bounds]
                )
                values.append(problem.objective_function(perturbed_x))
            
            # Return worst-case or average performance
            return np.mean(values) + 0.5 * np.std(values)  # Risk-averse objective
            
        robust_problem = OptimizationProblem(
            objective_function=robust_objective,
            bounds=problem.bounds,
            is_stochastic=True,
            noise_level=problem.noise_level
        )
        
        return self.optimize(robust_problem, method="pso")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization history"""
        if not self.optimization_history:
            return {"message": "No optimization history available"}
            
        summary = {
            "total_optimizations": len(self.optimization_history),
            "methods_used": {},
            "average_performance": {},
            "best_results": []
        }
        
        for entry in self.optimization_history:
            method = entry["method"]
            result = entry["result"]
            
            if method not in summary["methods_used"]:
                summary["methods_used"][method] = 0
            summary["methods_used"][method] += 1
            
            if method not in summary["average_performance"]:
                summary["average_performance"][method] = []
            summary["average_performance"][method].append(result.best_value)
            
        # Calculate averages
        for method, values in summary["average_performance"].items():
            summary["average_performance"][method] = np.mean(values)
            
        # Get best results
        summary["best_results"] = sorted(
            self.optimization_history,
            key=lambda x: x["result"].best_value
        )[:5]
        
        return summary

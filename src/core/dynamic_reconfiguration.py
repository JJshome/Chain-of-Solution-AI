"""
Dynamic Reconfiguration Module for Chain of Solution System
동적 재구성부 (Dynamic Reconfiguration Unit) - Patent Component 160

This module implements dynamic system reconfiguration based on real-time performance
analysis and changing problem requirements.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Represents the current state of the system"""
    performance_metrics: Dict[str, float]
    active_components: List[str]
    resource_utilization: Dict[str, float]
    timestamp: float
    

class ReconfigurationStrategy(ABC):
    """Abstract base class for reconfiguration strategies"""
    
    @abstractmethod
    def evaluate(self, current_state: SystemState, target_metrics: Dict[str, float]) -> float:
        """Evaluate the need for reconfiguration (0.0 = no need, 1.0 = urgent need)"""
        pass
    
    @abstractmethod
    def propose_configuration(self, current_state: SystemState) -> Dict[str, Any]:
        """Propose a new configuration based on current state"""
        pass


class PerformanceBasedStrategy(ReconfigurationStrategy):
    """Reconfiguration strategy based on performance metrics"""
    
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
    
    def evaluate(self, current_state: SystemState, target_metrics: Dict[str, float]) -> float:
        """Evaluate based on deviation from target metrics"""
        deviations = []
        for metric, target in target_metrics.items():
            if metric in current_state.performance_metrics:
                current = current_state.performance_metrics[metric]
                deviation = abs(current - target) / target
                deviations.append(deviation)
        
        return min(1.0, np.mean(deviations) if deviations else 0.0)
    
    def propose_configuration(self, current_state: SystemState) -> Dict[str, Any]:
        """Propose configuration changes to improve performance"""
        config_changes = {}
        
        # Analyze underperforming metrics
        for metric, value in current_state.performance_metrics.items():
            if metric in self.thresholds and value < self.thresholds[metric]:
                # Propose component activation/deactivation
                if metric == "accuracy" and value < self.thresholds[metric]:
                    config_changes["enable_advanced_analysis"] = True
                elif metric == "speed" and value < self.thresholds[metric]:
                    config_changes["enable_parallel_processing"] = True
                elif metric == "efficiency" and value < self.thresholds[metric]:
                    config_changes["optimize_resource_allocation"] = True
        
        return config_changes


class DynamicReconfiguration:
    """Main dynamic reconfiguration component"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.strategies: List[ReconfigurationStrategy] = []
        self.reconfiguration_history: List[Dict[str, Any]] = []
        self.current_configuration: Dict[str, Any] = {}
        
        # Initialize default strategies
        self._initialize_strategies()
        
    def _initialize_strategies(self):
        """Initialize default reconfiguration strategies"""
        # Performance-based strategy
        perf_thresholds = self.config.get("performance_thresholds", {
            "accuracy": 0.85,
            "speed": 0.7,
            "efficiency": 0.8
        })
        self.strategies.append(PerformanceBasedStrategy(perf_thresholds))
        
    def add_strategy(self, strategy: ReconfigurationStrategy):
        """Add a new reconfiguration strategy"""
        self.strategies.append(strategy)
        
    def evaluate_reconfiguration_need(self, current_state: SystemState, 
                                    target_metrics: Dict[str, float]) -> float:
        """Evaluate the need for system reconfiguration"""
        if not self.strategies:
            return 0.0
            
        # Get evaluation scores from all strategies
        scores = [strategy.evaluate(current_state, target_metrics) 
                 for strategy in self.strategies]
        
        # Return the maximum score (most urgent need)
        return max(scores)
    
    def reconfigure_system(self, current_state: SystemState) -> Dict[str, Any]:
        """Perform system reconfiguration based on current state"""
        logger.info("Starting system reconfiguration")
        
        # Collect configuration proposals from all strategies
        all_proposals = {}
        for strategy in self.strategies:
            proposal = strategy.propose_configuration(current_state)
            all_proposals.update(proposal)
        
        # Apply configuration changes
        self.current_configuration.update(all_proposals)
        
        # Log reconfiguration
        self.reconfiguration_history.append({
            "timestamp": current_state.timestamp,
            "previous_state": current_state.__dict__,
            "changes": all_proposals,
            "new_configuration": self.current_configuration.copy()
        })
        
        logger.info(f"System reconfigured with changes: {all_proposals}")
        
        return self.current_configuration
    
    def get_configuration_history(self) -> List[Dict[str, Any]]:
        """Get the history of system reconfigurations"""
        return self.reconfiguration_history
    
    def optimize_configuration(self, performance_data: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
        """Optimize configuration based on historical performance data"""
        if not performance_data:
            return self.current_configuration
            
        # Find configuration with best performance
        best_config = max(performance_data, key=lambda x: x[1])[0]
        
        # Apply machine learning to predict optimal configuration
        # (Simplified version - in practice, would use more sophisticated ML)
        optimal_config = self.current_configuration.copy()
        optimal_config.update(best_config)
        
        return optimal_config
    
    def adapt_to_problem_type(self, problem_type: str) -> Dict[str, Any]:
        """Adapt configuration to specific problem type"""
        problem_configs = {
            "optimization": {
                "enable_genetic_algorithms": True,
                "enable_simulated_annealing": True,
                "parallel_evaluations": True
            },
            "classification": {
                "enable_ensemble_methods": True,
                "enable_feature_engineering": True,
                "cross_validation": True
            },
            "design": {
                "enable_triz_analysis": True,
                "enable_morphological_analysis": True,
                "enable_constraint_solver": True
            }
        }
        
        if problem_type in problem_configs:
            self.current_configuration.update(problem_configs[problem_type])
            
        return self.current_configuration
    
    def save_configuration(self, filepath: str):
        """Save current configuration to file"""
        with open(filepath, 'w') as f:
            json.dump({
                "current_configuration": self.current_configuration,
                "history": self.reconfiguration_history
            }, f, indent=2)
            
    def load_configuration(self, filepath: str):
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.current_configuration = data.get("current_configuration", {})
            self.reconfiguration_history = data.get("history", [])

"""TRIZ60 module implementation for Chain of Solution framework."""

import json
import os
from typing import Dict, List, Any, Optional
import logging


class TRIZ60:
    """Implementation of the extended TRIZ60 methodology.
    
    This class provides access to the extended 60 TRIZ principles and
    contradiction matrix, as well as methods for applying TRIZ principles
    to solve problems.
    """
    
    def __init__(self, config):
        """Initialize the TRIZ60 module.
        
        Args:
            config: Configuration object or dictionary containing TRIZ settings
        """
        self.logger = logging.getLogger('cos_framework.triz60')
        
        # Get configuration
        if hasattr(config, 'get'):
            self.config = config
        else:
            from ..core.config import CoSConfig
            self.config = CoSConfig()
            if isinstance(config, dict):
                self.config._update_dict(self.config.config, config)
        
        # Load TRIZ principles
        self.principles = {}
        self.load_principles()
        
        # Load contradiction matrix
        self.contradiction_matrix = {}
        self.load_contradiction_matrix()
        
        self.logger.info("TRIZ60 module initialized")
    
    def load_principles(self) -> None:
        """Load TRIZ principles from the configured path."""
        principles_path = self.config.get('triz.principles_path')
        
        if principles_path and os.path.exists(principles_path):
            try:
                with open(principles_path, 'r') as f:
                    self.principles = json.load(f)
                self.logger.info(f"Loaded {len(self.principles)} TRIZ principles from {principles_path}")
            except Exception as e:
                self.logger.error(f"Failed to load TRIZ principles from {principles_path}: {e}")
                self._load_default_principles()
        else:
            self.logger.warning(f"TRIZ principles file not found at {principles_path}. Using default principles.")
            self._load_default_principles()
    
    def _load_default_principles(self) -> None:
        """Load default TRIZ principles when the configured file is not available."""
        # Create a basic set of TRIZ60 principles for demonstration
        self.principles = {
            # Original 40 TRIZ principles
            "1": {"name": "Segmentation", "description": "Divide an object into independent parts."},
            "2": {"name": "Taking out", "description": "Extract the disturbing part or property from an object."},
            "3": {"name": "Local quality", "description": "Make each part of an object fulfill a different function."},
            "4": {"name": "Asymmetry", "description": "Change the shape of an object from symmetrical to asymmetrical."},
            "5": {"name": "Merging", "description": "Bring closer together or merge identical or related objects."},
            # ... and so on for principles 6-40
            
            # Extended TRIZ60 principles (41-60)
            "41": {"name": "Adjustment", "description": "Provide elements for fine-tuning and self-adjustment of a system."},
            "42": {"name": "Dynamic Reconfiguration", "description": "Allow system components to rearrange themselves in response to changing conditions."},
            "43": {"name": "Energy Redistribution", "description": "Redirect excess energy to areas that need it within a system."},
            "44": {"name": "Multi-state Operation", "description": "Design a system to operate in multiple distinct states or configurations."},
            "45": {"name": "Non-linear Interactions", "description": "Utilize non-linear relationships between system components to achieve desired effects."},
            "46": {"name": "Self-diagnosis", "description": "Incorporate mechanisms for a system to detect its own problems or failures."},
            "47": {"name": "Information Compression", "description": "Represent information or functionality in a more compact form without loss of utility."},
            "48": {"name": "Resource Recycling", "description": "Reuse waste materials, energy, or byproducts from the system."},
            "49": {"name": "Scalability", "description": "Design a system that can maintain its function while changing in size or capacity."},
            "50": {"name": "Boundary Condition Optimization", "description": "Focus on optimizing the conditions at the boundaries of a system."},
            "51": {"name": "Interoperability", "description": "Design components to work effectively across different systems and platforms."},
            "52": {"name": "Emotion Recognition", "description": "Incorporate systems that can detect and respond to human emotional states."},
            "53": {"name": "Stochastic Optimization", "description": "Use probabilistic methods to find optimal solutions in uncertain environments."},
            "54": {"name": "Energy Field Manipulation", "description": "Control and direct energy fields to achieve desired effects."},
            "55": {"name": "Wave Control", "description": "Manipulate wave propagation, interference patterns, or resonances."},
            "56": {"name": "Time Management", "description": "Optimize processes with respect to time utilization or scheduling."},
            "57": {"name": "Flexibility", "description": "Design systems with the ability to adapt to varying conditions or requirements."},
            "58": {"name": "Molecular Level Control", "description": "Manipulate materials or processes at the molecular or atomic level."},
            "59": {"name": "Feedback Loop Enhancement", "description": "Improve system performance by enhancing feedback mechanisms."},
            "60": {"name": "Variable Connectivity", "description": "Allow dynamic changes in how system components connect or interact."}
        }
        self.logger.info("Loaded default TRIZ60 principles")
    
    def load_contradiction_matrix(self) -> None:
        """Load the TRIZ contradiction matrix from the configured path."""
        matrix_path = self.config.get('triz.contradiction_matrix_path')
        
        if matrix_path and os.path.exists(matrix_path):
            try:
                with open(matrix_path, 'r') as f:
                    self.contradiction_matrix = json.load(f)
                self.logger.info(f"Loaded contradiction matrix from {matrix_path}")
            except Exception as e:
                self.logger.error(f"Failed to load contradiction matrix from {matrix_path}: {e}")
                self._create_default_matrix()
        else:
            self.logger.warning(f"Contradiction matrix file not found at {matrix_path}. Using a default matrix.")
            self._create_default_matrix()
    
    def _create_default_matrix(self) -> None:
        """Create a default contradiction matrix when the configured file is not available."""
        # Create a minimal contradiction matrix for demonstration
        # In a real implementation, this would be a 39x39 matrix with principles for each contradiction
        self.contradiction_matrix = {
            "parameters": {
                "1": "Weight of moving object",
                "2": "Weight of stationary object",
                "3": "Length of moving object",
                # ... and so on for all 39 parameters
                "39": "Productivity"
            },
            "matrix": {
                # Format: "improving_param,worsening_param": [principle1, principle2, ...]
                "1,2": [1, 8, 15, 34],
                "1,3": [8, 10, 29, 40],
                # ... and so on for all combinations
                "39,38": [35, 38, 42, 53]
            }
        }
        self.logger.info("Created default contradiction matrix")
    
    def get_principle(self, principle_id: int) -> Dict[str, str]:
        """Get details of a specific TRIZ principle.
        
        Args:
            principle_id: ID of the TRIZ principle (1-60)
            
        Returns:
            Dictionary containing the principle details
        """
        principle_key = str(principle_id)
        
        if principle_key in self.principles:
            return self.principles[principle_key]
        else:
            self.logger.warning(f"Principle {principle_id} not found")
            return {"name": f"Unknown Principle {principle_id}", "description": "No description available"}
    
    def get_contradiction_matrix(self) -> Dict[str, Any]:
        """Get the full contradiction matrix.
        
        Returns:
            Dictionary containing the contradiction matrix
        """
        return self.contradiction_matrix
    
    def recommend_principles(self, improving_param: int, worsening_param: int) -> List[int]:
        """Recommend TRIZ principles based on contradiction parameters.
        
        Args:
            improving_param: Parameter to improve (1-39)
            worsening_param: Parameter that worsens (1-39)
            
        Returns:
            List of recommended principle IDs
        """
        matrix_key = f"{improving_param},{worsening_param}"
        
        if "matrix" in self.contradiction_matrix and matrix_key in self.contradiction_matrix["matrix"]:
            return self.contradiction_matrix["matrix"][matrix_key]
        else:
            self.logger.warning(f"No principles found for contradiction {matrix_key}")
            # Return some default principles when no specific recommendation is available
            return [1, 2, 13, 42, 53]  # Mix of traditional and extended principles
    
    def identify_contradictions(self, patterns: List[Any]) -> List[Dict[str, Any]]:
        """Identify contradictions in the system using detected patterns.
        
        Args:
            patterns: List of cross-modal patterns detected by the analyzer
            
        Returns:
            List of identified contradictions
        """
        contradictions = []
        
        # In a real implementation, this would analyze the patterns to find contradictions
        # For demonstration, we'll create sample contradictions based on pattern count
        for i, pattern in enumerate(patterns):
            # Create a sample contradiction
            contradiction = {
                'id': i,
                'description': f"Contradiction in pattern {i}",
                'improving_parameter': i % 39 + 1,  # Cycle through parameters 1-39
                'worsening_parameter': (i + 10) % 39 + 1,  # Offset to avoid same params
                'pattern_id': i,
                'severity': 0.5 + (i % 5) / 10.0  # Values between 0.5 and 0.9
            }
            contradictions.append(contradiction)
        
        return contradictions
    
    def apply_principles(self, contradiction: Dict[str, Any], principles: List[int]) -> List[Dict[str, Any]]:
        """Apply TRIZ principles to resolve a contradiction.
        
        Args:
            contradiction: The contradiction to resolve
            principles: List of TRIZ principle IDs to apply
            
        Returns:
            List of potential solutions
        """
        solutions = []
        
        for principle_id in principles:
            principle = self.get_principle(principle_id)
            
            # Create a solution based on the principle
            solution = {
                'principle_id': principle_id,
                'principle_name': principle['name'],
                'principle_description': principle['description'],
                'contradiction_id': contradiction['id'],
                'solution_description': f"Apply {principle['name']} to resolve {contradiction['description']}",
                'confidence': 0.7 + (principle_id % 3) / 10.0  # Values between 0.7 and 0.9
            }
            
            solutions.append(solution)
        
        return solutions
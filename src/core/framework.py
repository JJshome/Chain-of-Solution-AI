"""Main framework module for Chain of Solution."""

from typing import Dict, List, Any, Optional, Union
import logging

from .config import CoSConfig
from ..triz.triz60 import TRIZ60
from ..triz.su_field import SuField100
from ..multimodal.analyzer import MultimodalAnalyzer
from ..models.llm_integration import CoSLLM


class CoSFramework:
    """Main Chain of Solution framework class.
    
    This class integrates all components of the Chain of Solution framework,
    providing a unified interface for problem-solving using the CoS approach.
    """
    
    def __init__(self, config: Optional[Union[str, Dict, CoSConfig]] = None):
        """Initialize the Chain of Solution framework.
        
        Args:
            config: Configuration for the framework. Can be a path to a JSON file,
                   a dictionary, or a CoSConfig object. If None, default configuration
                   is used.
        """
        # Initialize configuration
        if isinstance(config, str):
            self.config = CoSConfig(config)
        elif isinstance(config, dict):
            self.config = CoSConfig()
            self.config._update_dict(self.config.config, config)
        elif isinstance(config, CoSConfig):
            self.config = config
        else:
            self.config = CoSConfig()
        
        # Setup logging
        logging_config = self.config.get('logging', {})
        logging.basicConfig(
            level=getattr(logging, logging_config.get('level', 'INFO')),
            format=logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            filename=logging_config.get('file')
        )
        self.logger = logging.getLogger('cos_framework')
        
        # Initialize components
        self.triz = TRIZ60(self.config)
        self.su_field = SuField100(self.config)
        self.multimodal_analyzer = MultimodalAnalyzer(self.config)
        self.llm = CoSLLM(self.config)
        
        # Data sources
        self.data_sources = {}
        
        # Analysis results
        self.patterns = []
        self.contradictions = []
        self.solutions = []
        
        self.logger.info("Chain of Solution framework initialized")
    
    def add_data_source(self, source_name: str, data: Any, modality_type: str) -> bool:
        """Add a data source to the framework.
        
        Args:
            source_name: Name of the data source
            data: The data to add
            modality_type: Type of the data modality ('text', 'image', 'audio', etc.)
            
        Returns:
            Success status
        """
        try:
            self.data_sources[source_name] = {
                'data': data,
                'modality_type': modality_type,
                'features': None  # Will be filled during analysis
            }
            self.logger.info(f"Added data source '{source_name}' of type '{modality_type}'")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add data source '{source_name}': {e}")
            return False
    
    def extract_features(self) -> bool:
        """Extract features from all data sources.
        
        Returns:
            Success status
        """
        try:
            for source_name, source_info in self.data_sources.items():
                self.logger.info(f"Extracting features from '{source_name}'")
                features = self.multimodal_analyzer.extract_features(
                    source_info['data'], 
                    source_info['modality_type']
                )
                self.data_sources[source_name]['features'] = features
            
            self.logger.info("Feature extraction completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return False
    
    def analyze_cross_modal_patterns(self) -> Dict[str, Any]:
        """Analyze patterns across different data modalities.
        
        Returns:
            Dictionary of detected patterns
        """
        try:
            # First ensure features are extracted
            if any(source_info['features'] is None for source_info in self.data_sources.values()):
                self.extract_features()
            
            # Prepare feature sets for cross-modal analysis
            feature_sets = {}
            for source_name, source_info in self.data_sources.items():
                feature_sets[source_name] = source_info['features']
            
            # Detect cross-modal patterns
            self.logger.info("Detecting cross-modal patterns")
            self.patterns = self.multimodal_analyzer.detect_cross_modal_patterns(feature_sets)
            
            # Analyze interactions within detected patterns
            pattern_details = {}
            for i, pattern in enumerate(self.patterns):
                pattern_id = f"pattern_{i}"
                pattern_details[pattern_id] = self.multimodal_analyzer.analyze_interactions(pattern)
            
            self.logger.info(f"Detected {len(self.patterns)} cross-modal patterns")
            return pattern_details
        except Exception as e:
            self.logger.error(f"Cross-modal pattern analysis failed: {e}")
            return {}
    
    def identify_contradictions(self) -> List[Dict[str, Any]]:
        """Identify contradictions in the system using TRIZ principles.
        
        Returns:
            List of identified contradictions
        """
        try:
            # Make sure patterns are analyzed first
            if not self.patterns:
                self.analyze_cross_modal_patterns()
            
            self.logger.info("Identifying contradictions using TRIZ principles")
            self.contradictions = self.triz.identify_contradictions(self.patterns)
            
            self.logger.info(f"Identified {len(self.contradictions)} contradictions")
            return self.contradictions
        except Exception as e:
            self.logger.error(f"Contradiction identification failed: {e}")
            return []
    
    def apply_triz_principles(self, contradiction_id: int, principles: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Apply TRIZ principles to resolve a contradiction.
        
        Args:
            contradiction_id: ID of the contradiction to resolve
            principles: List of TRIZ principle IDs to apply (if None, recommended principles are used)
            
        Returns:
            List of potential solutions
        """
        try:
            if contradiction_id >= len(self.contradictions):
                raise ValueError(f"Invalid contradiction ID: {contradiction_id}")
            
            contradiction = self.contradictions[contradiction_id]
            
            # If principles are not specified, get recommendations
            if principles is None:
                principles = self.triz.recommend_principles(
                    contradiction['improving_parameter'],
                    contradiction['worsening_parameter']
                )
            
            self.logger.info(f"Applying TRIZ principles {principles} to contradiction {contradiction_id}")
            solutions = self.triz.apply_principles(contradiction, principles)
            
            # Store solutions for later use
            for solution in solutions:
                solution['contradiction_id'] = contradiction_id
                self.solutions.append(solution)
            
            return solutions
        except Exception as e:
            self.logger.error(f"Failed to apply TRIZ principles: {e}")
            return []
    
    def create_su_field_model(self, substances: List[Dict], fields: List[Dict], interactions: List[Dict]) -> Dict[str, Any]:
        """Create a Su-Field model for the system.
        
        Args:
            substances: List of substances in the system
            fields: List of fields in the system
            interactions: List of interactions between substances and fields
            
        Returns:
            Su-Field model
        """
        try:
            self.logger.info("Creating Su-Field model")
            model = self.su_field.create_model(substances, fields, interactions)
            
            self.logger.info("Su-Field model created successfully")
            return model
        except Exception as e:
            self.logger.error(f"Failed to create Su-Field model: {e}")
            return {}
    
    def recommend_su_field_solutions(self, model: Dict[str, Any], problem_type: str) -> List[Dict[str, Any]]:
        """Recommend standard solutions for a Su-Field model.
        
        Args:
            model: Su-Field model to improve
            problem_type: Type of problem to solve
            
        Returns:
            List of recommended standard solutions
        """
        try:
            self.logger.info(f"Recommending Su-Field solutions for problem type '{problem_type}'")
            solutions = self.su_field.recommend_solutions(model, problem_type)
            
            # Store solutions
            for solution in solutions:
                solution['type'] = 'su_field'
                self.solutions.append(solution)
            
            self.logger.info(f"Recommended {len(solutions)} Su-Field solutions")
            return solutions
        except Exception as e:
            self.logger.error(f"Failed to recommend Su-Field solutions: {e}")
            return []
    
    def generate_llm_solution(self, problem_description: str) -> Dict[str, Any]:
        """Generate a solution using the Chain of Solution LLM integration.
        
        Args:
            problem_description: Description of the problem
            
        Returns:
            Generated solution
        """
        try:
            # Prepare data sources for the LLM
            data_sources = {}
            for name, source_info in self.data_sources.items():
                data_sources[name] = {
                    'modality_type': source_info['modality_type'],
                    'summary': str(source_info['data'])[:100] + '...' if len(str(source_info['data'])) > 100 else str(source_info['data'])
                }
            
            # Prepare pattern information
            patterns_info = [{'id': f"pattern_{i}", 'summary': str(p)[:100] + '...'} for i, p in enumerate(self.patterns)]
            
            # Generate solution
            self.logger.info("Generating solution using LLM integration")
            solution = self.llm.generate_solution(problem_description, data_sources, patterns_info, self.contradictions)
            
            self.logger.info("LLM solution generated successfully")
            return solution
        except Exception as e:
            self.logger.error(f"Failed to generate LLM solution: {e}")
            return {'error': str(e)}
    
    def generate_solution(self, problem_description: str = "") -> Dict[str, Any]:
        """Generate a comprehensive solution using the Chain of Solution approach.
        
        Args:
            problem_description: Optional description of the problem
            
        Returns:
            Dictionary containing the complete solution and supporting data
        """
        try:
            # Make sure we have analyzed patterns and identified contradictions
            if not self.patterns:
                self.analyze_cross_modal_patterns()
            
            if not self.contradictions:
                self.identify_contradictions()
            
            # Generate solutions for each contradiction if we haven't already
            for i, contradiction in enumerate(self.contradictions):
                if not any(s.get('contradiction_id') == i for s in self.solutions):
                    self.apply_triz_principles(i)
            
            # Use LLM to generate a comprehensive solution
            llm_solution = self.generate_llm_solution(problem_description)
            
            # Compile the final solution
            solution = {
                'problem_description': problem_description,
                'cross_modal_patterns': self.patterns,
                'contradictions': self.contradictions,
                'triz_solutions': [s for s in self.solutions if s.get('type') != 'su_field'],
                'su_field_solutions': [s for s in self.solutions if s.get('type') == 'su_field'],
                'llm_solution': llm_solution,
                'integrated_solution': llm_solution.get('solution', 'No solution generated')
            }
            
            self.logger.info("Comprehensive solution generated successfully")
            return solution
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive solution: {e}")
            return {'error': str(e)}

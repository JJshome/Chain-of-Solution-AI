"""LLM integration module for Chain of Solution framework."""

import logging
import os
from typing import Dict, List, Any, Optional


class CoSLLM:
    """Chain of Solution LLM integration.
    
    This class integrates Large Language Models with the Chain of Solution
    framework, enabling structured problem-solving through natural language
    reasoning.
    """
    
    def __init__(self, config):
        """Initialize the CoS-LLM integration module.
        
        Args:
            config: Configuration object or dictionary
        """
        self.logger = logging.getLogger('cos_framework.llm')
        
        # Get configuration
        if hasattr(config, 'get'):
            self.config = config
        else:
            from ..core.config import CoSConfig
            self.config = CoSConfig()
            if isinstance(config, dict):
                self.config._update_dict(self.config.config, config)
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template()
        
        # LLM configuration
        self.model_name = self.config.get('llm.model', 'llama3-70b')
        self.temperature = self.config.get('llm.temperature', 0.7)
        self.max_tokens = self.config.get('llm.max_tokens', 1024)
        
        self.logger.info(f"CoS-LLM integration initialized with model {self.model_name}")
    
    def _load_prompt_template(self) -> str:
        """Load the prompt template for the CoS framework.
        
        Returns:
            Prompt template as a string
        """
        template_path = self.config.get('llm.prompt_template_path')
        
        if template_path and os.path.exists(template_path):
            try:
                with open(template_path, 'r') as f:
                    template = f.read()
                self.logger.info(f"Loaded prompt template from {template_path}")
                return template
            except Exception as e:
                self.logger.error(f"Failed to load prompt template from {template_path}: {e}")
        
        # Use default template if file not found or error occurred
        self.logger.warning("Using default prompt template")
        return """
# Chain of Solution Analysis

## Problem Description
{problem_description}

## Available Data Modalities
{data_sources_str}

## Cross-Modal Pattern Analysis
{patterns_str}

## Contradictions Identified
{contradictions_str}

## TRIZ Principles Applied
{triz_principles_str}

## Recommended Su-Field Solutions
{su_field_solutions_str}

## Integrated Solution
Based on the analysis above, please provide a comprehensive solution to the problem.
"""
    
    def generate_cos_prompt(self, problem_description: str, data_sources: Dict[str, Any], 
                          patterns: List[Dict[str, Any]], contradictions: List[Dict[str, Any]],
                          triz_solutions: List[Dict[str, Any]] = None, 
                          su_field_solutions: List[Dict[str, Any]] = None) -> str:
        """Generate a CoS-structured prompt for the LLM.
        
        Args:
            problem_description: Description of the problem
            data_sources: Available data sources
            patterns: Detected cross-modal patterns
            contradictions: Identified contradictions
            triz_solutions: TRIZ-based solutions (optional)
            su_field_solutions: Su-Field based solutions (optional)
            
        Returns:
            Structured prompt for the LLM
        """
        # Format data sources section
        data_sources_str = "\n".join([f"- {name} ({info.get('modality_type', 'unknown')}): {info.get('summary', 'No summary')}" 
                                 for name, info in data_sources.items()])
        
        # Format patterns section
        patterns_str = "\n".join([f"- Pattern {i+1}: {p.get('description', 'No description')} " 
                              f"(Strength: {p.get('strength', 0):.2f})" 
                              for i, p in enumerate(patterns)])
        
        # Format contradictions section
        contradictions_str = "\n".join([f"- Contradiction {i+1}: {c.get('description', 'No description')} " 
                                   f"(Severity: {c.get('severity', 0):.2f})" 
                                   for i, c in enumerate(contradictions)])
        
        # Format TRIZ solutions section
        triz_principles_str = ""
        if triz_solutions:
            triz_principles_str = "\n".join([f"- Principle {s.get('principle_id', 'unknown')}: {s.get('principle_name', 'Unknown')} - " 
                                        f"{s.get('solution_description', 'No description')}" 
                                        for s in triz_solutions])
        else:
            triz_principles_str = "No TRIZ principles applied yet."
        
        # Format Su-Field solutions section
        su_field_solutions_str = ""
        if su_field_solutions:
            su_field_solutions_str = "\n".join([f"- Solution {s.get('solution_id', 'unknown')}: {s.get('solution_name', 'Unknown')} - " 
                                          f"{s.get('solution_description', 'No description')}" 
                                          for s in su_field_solutions])
        else:
            su_field_solutions_str = "No Su-Field solutions recommended yet."
        
        # Fill in the template
        prompt = self.prompt_template.format(
            problem_description=problem_description,
            data_sources_str=data_sources_str,
            patterns_str=patterns_str,
            contradictions_str=contradictions_str,
            triz_principles_str=triz_principles_str,
            su_field_solutions_str=su_field_solutions_str
        )
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt.
        
        In a real implementation, this would make an API call to the LLM.
        For demonstration, we'll simulate a response.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLM response as a string
        """
        self.logger.info(f"Calling LLM model {self.model_name} with prompt length {len(prompt)}")
        
        # In a real implementation, this would call an API like Anthropic, OpenAI, or a local model
        # For demonstration, return a simple response that includes parts of the prompt
        if 'problem_description' in prompt and 'contradictions' in prompt:
            # Extract problem description
            try:
                problem_start = prompt.find('## Problem Description') + 22
                problem_end = prompt.find('##', problem_start)
                problem_text = prompt[problem_start:problem_end].strip()
                
                # Create a simulated response
                return f"""Based on the analysis of the cross-modal patterns and identified contradictions, I recommend the following solution:

1. The problem of {problem_text[:50]}... can be addressed by applying TRIZ principles to resolve the key contradictions.

2. From the patterns detected across different data modalities, it's clear that there's a significant relationship between the observed phenomena that wasn't evident when analyzing each modality separately.

3. The most effective approach would be to implement a system that dynamically reconfigures itself based on real-time feedback (TRIZ Principle 42), while simultaneously optimizing the boundary conditions (TRIZ Principle 50).

4. This solution addresses the contradictions by creating a balance between competing requirements and leveraging the unique insights that emerge from cross-modal analysis.

5. Implementation should proceed in phases, starting with a prototype that focuses on the most critical interactions identified in the pattern analysis, followed by gradual expansion to address all aspects of the problem.

This approach not only resolves the immediate issues but creates a framework for addressing similar problems in the future through continuous learning and adaptation."""
            except Exception as e:
                self.logger.error(f"Error generating simulated LLM response: {e}")
                return "I couldn't generate a proper solution for this problem due to an error in processing the prompt."
        else:
            return "Insufficient information to generate a comprehensive solution. Please provide a complete problem description and analysis."
    
    def process_response(self, llm_response: str) -> Dict[str, Any]:
        """Process and structure the LLM response according to CoS methodology.
        
        Args:
            llm_response: Raw response from the LLM
            
        Returns:
            Structured response
        """
        # In a real implementation, this would parse the LLM response into structured components
        # For demonstration, create a simple structured response
        structured_response = {
            'solution': llm_response,
            'key_points': [],
            'implementation_steps': []
        }
        
        # Extract key points (lines starting with numbers)
        lines = llm_response.split('\n')
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and '. ' in line:
                point = line[line.find('. ')+2:]
                structured_response['key_points'].append(point)
        
        # Identify implementation steps if present
        if 'Implementation' in llm_response or 'implementation' in llm_response:
            in_implementation = False
            implementation_steps = []
            
            for line in lines:
                if 'Implementation' in line or 'implementation' in line:
                    in_implementation = True
                    continue
                    
                if in_implementation and line.strip() and not line.startswith('#'):
                    if line[0].isdigit() and '. ' in line:
                        step = line[line.find('. ')+2:]
                        implementation_steps.append(step)
            
            structured_response['implementation_steps'] = implementation_steps
        
        return structured_response
    
    def generate_solution(self, problem_description: str, data_sources: Dict[str, Any], 
                        patterns: List[Dict[str, Any]] = None, 
                        contradictions: List[Dict[str, Any]] = None,
                        triz_solutions: List[Dict[str, Any]] = None,
                        su_field_solutions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a complete solution using the CoS-LLM integration.
        
        Args:
            problem_description: Description of the problem
            data_sources: Available data sources
            patterns: Detected cross-modal patterns (optional)
            contradictions: Identified contradictions (optional)
            triz_solutions: TRIZ-based solutions (optional)
            su_field_solutions: Su-Field based solutions (optional)
            
        Returns:
            Generated solution
        """
        # Use empty lists if any optional arguments are None
        patterns = patterns or []
        contradictions = contradictions or []
        triz_solutions = triz_solutions or []
        su_field_solutions = su_field_solutions or []
        
        # Generate prompt
        prompt = self.generate_cos_prompt(
            problem_description, 
            data_sources, 
            patterns, 
            contradictions, 
            triz_solutions, 
            su_field_solutions
        )
        
        # Call LLM
        llm_response = self._call_llm(prompt)
        
        # Process response
        solution = self.process_response(llm_response)
        
        # Add metadata
        solution['problem_description'] = problem_description
        solution['used_patterns'] = len(patterns)
        solution['used_contradictions'] = len(contradictions)
        solution['used_triz_solutions'] = len(triz_solutions)
        solution['used_su_field_solutions'] = len(su_field_solutions)
        
        self.logger.info("Generated solution using LLM integration")
        
        return solution
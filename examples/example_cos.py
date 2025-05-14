#!/usr/bin/env python3
"""
Example script for using the Chain of Solution framework.

This script demonstrates how to initialize the Chain of Solution framework
and use it to solve a sample problem.
"""

import sys
import os
import logging
from pathlib import Path

# Add the src directory to the path so we can import the Chain of Solution package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ChainOfSolution

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('example')


def main():
    """Run an example of the Chain of Solution framework."""
    logger.info("Initializing Chain of Solution framework")
    
    # Initialize Chain of Solution framework with configuration
    config = {
        'model_type': 'llama3.1',
        'model_size': '8B',
        'use_emergent_pattern_detection': True,
        'use_triz60': True,
        'use_su_field_analysis': True,
        'max_recommendations': 5
    }
    cos = ChainOfSolution(config)
    
    # Define a sample problem
    problem = "Design a non-invasive continuous glucose monitoring system for diabetes patients that can operate for extended periods without maintenance"
    
    logger.info(f"Solving problem: {problem}")
    
    # Solve the problem
    solution = cos.solve_problem(problem)
    
    # Print solution summary and details
    print("\n" + "="*80)
    print("SOLUTION SUMMARY")
    print("="*80)
    print(solution['summary'])
    print(f"Confidence: {solution['confidence']:.2f}")
    print(f"Execution Time: {solution['execution_time']:.2f} seconds")
    print(f"Number of Recommendations: {len(solution['recommendations'])}")
    
    # Print detailed recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    for i, rec in enumerate(solution['recommendations'], 1):
        print(f"\n{i}. {rec['type']} Recommendation (Confidence: {rec['confidence']:.2f})")
        print("-"*40)
        print(f"Description: {rec['description']}")
        if 'principle' in rec:
            print(f"Based on TRIZ Principle: {rec['principle']}")
    
    # Print TRIZ principles used (if any)
    if solution['triz_principles']['used']:
        print("\n" + "="*80)
        print("TRIZ PRINCIPLES APPLIED")
        print("="*80)
        
        for principle in solution['triz_principles']['principles']:
            print(f"\n- {principle['principle_name']} (ID: {principle['principle_id']})")
            print(f"  Application: {principle['application']}")
    
    # Print Su-Field model (if used)
    if solution['su_field_analysis']['used']:
        print("\n" + "="*80)
        print("SU-FIELD ANALYSIS")
        print("="*80)
        
        model = solution['su_field_analysis']['model']
        print(f"Substances: {', '.join(model['substances'])}")
        print(f"Fields: {', '.join(model['fields'])}")
        
        print("\nInteractions:")
        for interaction in model['interactions']:
            print(f"- {interaction['substance1']} interacts with {interaction['substance2']} via {interaction['field']} field")
            print(f"  Type: {interaction['type']}")
            print(f"  Description: {interaction['description']}")
    
    # Print emergent findings (if any)
    if solution['multimodal_analysis']['used'] and solution['multimodal_analysis']['emergent_findings']:
        print("\n" + "="*80)
        print("EMERGENT FINDINGS")
        print("="*80)
        
        for finding in solution['multimodal_analysis']['emergent_findings']:
            print(f"\n- {finding['pattern_name']} (Confidence: {finding['confidence']:.2f})")
            print(f"  Description: {finding['description']}")
            print(f"  Implication: {finding['implication']}")


if __name__ == "__main__":
    main()

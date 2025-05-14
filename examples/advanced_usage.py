"""Advanced usage example of the Chain of Solution framework."""

import sys
import os
import logging
import json
import numpy as np

# Add src directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import ChainOfSolution
from src.triz import TRIZEngine
from src.multimodal import MultimodalEngine

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_sample_data(data_path):
    """Load sample data from a JSON file.
    
    In a real application, this would load actual multimodal data.
    For this example, we simulate the data.
    
    Args:
        data_path: Path to data file
        
    Returns:
        Dictionary containing multimodal data
    """
    # In a real application, this would load actual data
    # For this example, we'll create simulated data
    
    # Simulate medical imaging data
    medical_image = np.random.rand(256, 256, 3)  # RGB image
    
    # Simulate audio data
    audio_data = np.random.rand(20000)
    
    # Simulate time series data
    time_series = np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
    
    # Simulate structured patient data
    patient_data = {
        'age': 62,
        'gender': 'female',
        'vital_signs': {
            'heart_rate': [72, 75, 70, 68, 73],
            'blood_pressure': [(120, 80), (125, 82), (118, 79)],
            'temperature': [98.6, 98.7, 98.5]
        },
        'lab_results': {
            'cholesterol': 195,
            'glucose': 105,
            'hba1c': 5.8
        }
    }
    
    # Combine all data
    multimodal_data = {
        'medical_image': medical_image,
        'heart_sound_recording': audio_data,
        'vitals_time_series': time_series,
        'patient_data': patient_data,
        'patient_notes': "Patient presents with occasional chest discomfort during exercise. "
                         "No significant changes in medication response. "
                         "Previous imaging showed minor calcification in coronary arteries.",
        'context': {
            'patient_id': 'SAMPLE-12345',
            'recording_type': 'heart',
            'type': 'health',
            'clinical_setting': 'follow_up'
        }
    }
    
    return multimodal_data

def custom_triz_analysis(problem_description):
    """Perform custom TRIZ analysis on a problem.
    
    Args:
        problem_description: Description of the problem
        
    Returns:
        TRIZ analysis results
    """
    print("\nPerforming custom TRIZ analysis...")
    
    # Initialize TRIZ engine
    triz_engine = TRIZEngine()
    
    # Create problem analysis structure
    problem_analysis = {
        'problem_statement': problem_description,
        'keywords': ['medical', 'monitoring', 'remote', 'continuous', 'patient', 'cardiac', 'health'],
        'domains': ['healthcare', 'monitoring systems', 'telemetry'],
        'constraints': [
            'Must be non-invasive',
            'Must be user-friendly for elderly patients',
            'Must ensure data privacy and security',
            'Must function reliably without constant professional supervision'
        ],
        'objectives': [
            'Provide continuous cardiac monitoring',
            'Reduce hospital visits',
            'Enable early detection of cardiac events',
            'Improve patient quality of life'
        ]
    }
    
    # Identify relevant TRIZ principles
    principles = triz_engine.identify_principles(problem_analysis)
    
    # Perform Su-Field analysis
    system_description = "A remote cardiac monitoring system that connects patients with healthcare providers."
    su_field_analysis = triz_engine.perform_su_field_analysis(system_description)
    
    # Generate innovative solutions
    solutions = triz_engine.generate_innovative_solutions(
        problem_analysis, principles, su_field_analysis)
    
    # Analyze system evolution trends
    evolution_analysis = triz_engine.analyze_system_evolution("Current cardiac monitoring systems")
    
    return {
        'principles': principles,
        'su_field_analysis': su_field_analysis,
        'solutions': solutions,
        'evolution_analysis': evolution_analysis
    }

def custom_multimodal_analysis(data):
    """Perform custom multimodal analysis on data.
    
    Args:
        data: Multimodal data
        
    Returns:
        Multimodal analysis results
    """
    print("\nPerforming custom multimodal analysis...")
    
    # Initialize multimodal engine
    multimodal_engine = MultimodalEngine()
    
    # Perform multimodal analysis
    results = multimodal_engine.analyze(data)
    
    return results

def main():
    """Demonstrate advanced usage of the Chain of Solution framework."""
    print("Chain of Solution (CoS) Framework - Advanced Usage Example")
    print("-" * 60)
    
    # Define a complex problem
    problem_description = """
    Develop an advanced remote cardiac monitoring system for patients with chronic heart conditions
    that can detect early warning signs of cardiac events before they become serious,
    while being minimally intrusive to the patient's daily life. The system should combine
    multiple data sources to improve diagnostic accuracy and reduce false positives.
    It must be suitable for elderly patients with limited technical skills and comply with
    medical privacy regulations. The solution should significantly reduce the need for in-person
    hospital visits while maintaining or improving the quality of care.
    """
    
    print("\nProblem:")
    print(problem_description.strip())
    
    # Load sample multimodal data
    print("\nLoading multimodal data...")
    sample_data = load_sample_data(None)  # In a real app, you'd specify a file path
    
    # Custom TRIZ analysis
    triz_results = custom_triz_analysis(problem_description)
    
    # Custom multimodal analysis
    multimodal_results = custom_multimodal_analysis(sample_data)
    
    # Initialize the full Chain of Solution framework
    print("\nInitializing Chain of Solution framework...")
    cos = ChainOfSolution()
    
    # Solve the problem using the framework
    print("\nSolving problem using Chain of Solution framework...")
    solution = cos.solve_problem(problem_description, data=sample_data)
    
    # Output TRIZ results
    print("\n" + "=" * 40)
    print("TRIZ ANALYSIS RESULTS")
    print("=" * 40)
    
    # Print top principles
    print("\nTop TRIZ Principles:")
    for principle in triz_results['principles']['principles'][:3]:
        print(f"- {principle['name']}: {principle['description']}")
    
    # Print ideal final result
    if triz_results['principles']['ideal_final_result']:
        print("\nIdeal Final Result:")
        print(triz_results['principles']['ideal_final_result']['statement'])
    
    # Print top solutions
    print("\nTop Innovative Solutions:")
    for solution in triz_results['solutions']['solutions'][:2]:
        print(f"- {solution['title']} (Confidence: {solution['confidence']:.2f})")
        print(f"  Description: {solution['description']}")
        print(f"  Steps:")
        for step in solution['steps']:
            print(f"   * {step}")
    
    # Print evolution trends
    print("\nSystem Evolution Trends:")
    print(f"Current Stage: {triz_results['evolution_analysis']['current_evolutionary_stage']}")
    print("Active Trends:")
    for trend in triz_results['evolution_analysis']['active_trends']:
        print(f"- {trend}")
    
    # Output multimodal analysis results
    print("\n" + "=" * 40)
    print("MULTIMODAL ANALYSIS RESULTS")
    print("=" * 40)
    
    print("\nDetected Modalities:")
    for modality in multimodal_results['modalities_detected']:
        print(f"- {modality}")
    
    print("\nEmergent Findings:")
    for finding in multimodal_results['emergent_findings']:
        print(f"- {finding['description']} (Relevance: {finding['relevance']:.2f})")
        print(f"  Modalities involved: {', '.join(finding['modalities_involved'])}")
        print(f"  Details: {finding['details']}")
    
    # Output final solution from Chain of Solution framework
    print("\n" + "=" * 40)
    print("CHAIN OF SOLUTION FRAMEWORK RESULTS")
    print("=" * 40)
    
    print("\nSolution Summary:")
    print(solution.get('summary', 'No summary available'))
    
    print("\nRecommendations:")
    for i, recommendation in enumerate(solution.get('recommendations', []), 1):
        print(f"{i}. {recommendation}")

if __name__ == "__main__":
    main()

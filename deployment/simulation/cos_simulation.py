"""Chain of Solution Framework Simulation.

This module provides a simulation environment for the Chain of Solution framework,
allowing users to test the framework with synthetic data and visualize results.
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Add src directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src import ChainOfSolution

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('cos_simulation')


class CoSSimulation:
    """Chain of Solution Framework Simulation environment.
    
    This class provides methods for simulating the Chain of Solution framework
    with synthetic data and visualizing the results.
    """
    
    def __init__(self, config=None):
        """Initialize the simulation environment.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.logger = logging.getLogger('cos_simulation.environment')
        self.logger.info("Initializing Chain of Solution simulation environment")
        
        # Initialize the Chain of Solution framework
        self.cos = ChainOfSolution(config)
        
        # Initialize simulation parameters
        self.config = config or {}
        self.simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = self.config.get('results_dir', 'simulation_results')
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger.info(f"Simulation environment initialized with ID: {self.simulation_id}")
    
    def generate_synthetic_data(self, data_type='healthcare', complexity=0.5, noise_level=0.2):
        """Generate synthetic multimodal data for simulation.
        
        Args:
            data_type: Type of data to generate ('healthcare', 'engineering', 'business')
            complexity: Complexity level of the data (0.0 to 1.0)
            noise_level: Amount of noise to add to the data (0.0 to 1.0)
            
        Returns:
            Dictionary containing synthetic multimodal data
        """
        self.logger.info(f"Generating synthetic {data_type} data (complexity={complexity}, noise={noise_level})")
        
        # Base data structure
        data = {}
        
        # Add random seed for reproducibility
        np.random.seed(int(complexity * 1000) + int(noise_level * 1000))
        
        # Generate different types of data based on domain
        if data_type == 'healthcare':
            # Text data - patient notes
            data['patient_notes'] = self._generate_healthcare_text(complexity)
            
            # Image data - medical scan
            data['medical_image'] = self._generate_medical_image(complexity, noise_level)
            
            # Audio data - heart sounds
            data['heart_sound_recording'] = self._generate_heart_sound(complexity, noise_level)
            
            # Time series data - vital signs
            data['vitals_time_series'] = self._generate_vital_signs(complexity, noise_level)
            
            # Context data
            data['context'] = {
                'patient_age': np.random.randint(20, 90),
                'patient_gender': np.random.choice(['male', 'female']),
                'recording_type': 'heart',
                'type': 'health'
            }
            
        elif data_type == 'engineering':
            # Text data - engineering specifications
            data['specifications'] = self._generate_engineering_text(complexity)
            
            # Image data - technical drawing or prototype
            data['technical_image'] = self._generate_technical_image(complexity, noise_level)
            
            # Time series data - sensor readings
            data['sensor_readings'] = self._generate_sensor_data(complexity, noise_level)
            
            # Context data
            data['context'] = {
                'domain': np.random.choice(['mechanical', 'electrical', 'chemical', 'software']),
                'complexity_level': complexity,
                'type': 'engineering'
            }
            
        elif data_type == 'business':
            # Text data - market analysis
            data['market_analysis'] = self._generate_business_text(complexity)
            
            # Time series data - financial data
            data['financial_data'] = self._generate_financial_data(complexity, noise_level)
            
            # Structured data - customer information
            data['customer_data'] = self._generate_customer_data(complexity, noise_level)
            
            # Context data
            data['context'] = {
                'industry': np.random.choice(['technology', 'finance', 'healthcare', 'retail']),
                'market_condition': np.random.choice(['growing', 'stable', 'declining']),
                'type': 'business'
            }
        
        self.logger.info(f"Generated synthetic data with {len(data)} components")
        return data
    
    def generate_synthetic_problem(self, domain='healthcare', complexity=0.5):
        """Generate a synthetic problem description.
        
        Args:
            domain: Problem domain ('healthcare', 'engineering', 'business')
            complexity: Complexity level of the problem (0.0 to 1.0)
            
        Returns:
            Problem description string
        """
        self.logger.info(f"Generating synthetic problem in {domain} domain (complexity={complexity})")
        
        if domain == 'healthcare':
            problems = [
                "Design a non-invasive continuous glucose monitoring system for diabetes patients",
                "Develop a remote cardiac monitoring solution for elderly patients",
                "Create a wearable device for early detection of neurological disorders",
                "Design an AI-based diagnostic tool for rare diseases",
                "Develop a portable dialysis system for kidney patients"               
            ]
            
            # Select problem based on complexity
            index = min(int(complexity * len(problems)), len(problems) - 1)
            base_problem = problems[index]
            
            # Add complexity-based details
            if complexity > 0.7:
                base_problem += " that can operate independently for extended periods without maintenance"
            if complexity > 0.4:
                base_problem += " while ensuring patient comfort and ease of use"
            if complexity > 0.2:
                base_problem += " and maintaining data privacy and security"
                
        elif domain == 'engineering':
            problems = [
                "Design a more efficient cooling system for electronic devices",
                "Develop a sustainable packaging solution that reduces plastic waste",
                "Create an energy harvesting system for IoT devices",
                "Design a noise cancellation system for urban environments",
                "Develop a water purification system for remote areas"               
            ]
            
            # Select problem based on complexity
            index = min(int(complexity * len(problems)), len(problems) - 1)
            base_problem = problems[index]
            
            # Add complexity-based details
            if complexity > 0.7:
                base_problem += " that significantly outperforms existing solutions"
            if complexity > 0.4:
                base_problem += " while minimizing manufacturing costs"
            if complexity > 0.2:
                base_problem += " and meeting regulatory requirements"
                
        elif domain == 'business':
            problems = [
                "Develop a customer retention strategy for a subscription-based service",
                "Create a market entry plan for a new technology product",
                "Design a supply chain optimization strategy for a global company",
                "Develop a digital transformation roadmap for a traditional business",
                "Create a pricing strategy for a premium service in a competitive market"               
            ]
            
            # Select problem based on complexity
            index = min(int(complexity * len(problems)), len(problems) - 1)
            base_problem = problems[index]
            
            # Add complexity-based details
            if complexity > 0.7:
                base_problem += " that can adapt to rapidly changing market conditions"
            if complexity > 0.4:
                base_problem += " while maintaining profitability"
            if complexity > 0.2:
                base_problem += " and addressing sustainability concerns"
        
        else:
            base_problem = "Design an innovative solution to improve efficiency and sustainability"
        
        self.logger.info(f"Generated problem: {base_problem}")
        return base_problem
    
    def run_simulation(self, domain='healthcare', complexity=0.5, noise_level=0.2, iterations=5):
        """Run a full simulation of the Chain of Solution framework.
        
        Args:
            domain: Problem domain ('healthcare', 'engineering', 'business')
            complexity: Complexity level of the problem (0.0 to 1.0)
            noise_level: Amount of noise in the data (0.0 to 1.0)
            iterations: Number of iterations to run
            
        Returns:
            Dictionary containing simulation results
        """
        self.logger.info(f"Running simulation in {domain} domain with {iterations} iterations")
        
        results = {
            'simulation_id': self.simulation_id,
            'domain': domain,
            'complexity': complexity,
            'noise_level': noise_level,
            'iterations': iterations,
            'timestamp': datetime.now().isoformat(),
            'iteration_results': []
        }
        
        for i in range(iterations):
            self.logger.info(f"Starting iteration {i+1}/{iterations}")
            
            # Generate problem and data for this iteration
            iteration_complexity = complexity * (0.8 + 0.4 * np.random.random())  # Add some variation
            iteration_noise = noise_level * (0.8 + 0.4 * np.random.random())  # Add some variation
            
            problem = self.generate_synthetic_problem(domain, iteration_complexity)
            data = self.generate_synthetic_data(domain, iteration_complexity, iteration_noise)
            
            # Measure execution time
            start_time = datetime.now()
            
            # Solve the problem using Chain of Solution framework
            solution = self.cos.solve_problem(problem, data=data)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Record results for this iteration
            iteration_result = {
                'iteration': i+1,
                'problem': problem,
                'execution_time': execution_time,
                'solution_summary': solution.get('summary', 'No summary available'),
                'recommendation_count': len(solution.get('recommendations', [])),
                'confidence': solution.get('confidence', 0.0),
                'triz_principles_used': len(solution.get('triz_principles', {}).get('principles', [])) if 'triz_principles' in solution else 0,
                'emergent_findings': len(solution.get('multimodal_analysis', {}).get('emergent_findings', [])) if 'multimodal_analysis' in solution else 0
            }
            
            results['iteration_results'].append(iteration_result)
            
            self.logger.info(f"Completed iteration {i+1} in {execution_time:.2f} seconds")
        
        # Calculate aggregate metrics
        results['average_execution_time'] = sum(r['execution_time'] for r in results['iteration_results']) / iterations
        results['average_recommendation_count'] = sum(r['recommendation_count'] for r in results['iteration_results']) / iterations
        results['average_confidence'] = sum(r['confidence'] for r in results['iteration_results']) / iterations
        results['average_triz_principles'] = sum(r['triz_principles_used'] for r in results['iteration_results']) / iterations
        results['average_emergent_findings'] = sum(r['emergent_findings'] for r in results['iteration_results']) / iterations
        
        # Save results
        self._save_simulation_results(results)
        
        self.logger.info(f"Simulation completed with average execution time of {results['average_execution_time']:.2f} seconds")
        return results
    
    def visualize_results(self, results):
        """Visualize simulation results.
        
        Args:
            results: Simulation results dictionary
            
        Returns:
            Path to saved visualization file
        """
        self.logger.info("Visualizing simulation results")
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot execution time per iteration
        axs[0, 0].plot([r['iteration'] for r in results['iteration_results']], 
                     [r['execution_time'] for r in results['iteration_results']], 
                     'o-', linewidth=2)
        axs[0, 0].set_title('Execution Time per Iteration')
        axs[0, 0].set_xlabel('Iteration')
        axs[0, 0].set_ylabel('Time (seconds)')
        axs[0, 0].grid(True)
        
        # Plot recommendation count per iteration
        axs[0, 1].plot([r['iteration'] for r in results['iteration_results']], 
                     [r['recommendation_count'] for r in results['iteration_results']], 
                     'o-', linewidth=2, color='green')
        axs[0, 1].set_title('Recommendations per Iteration')
        axs[0, 1].set_xlabel('Iteration')
        axs[0, 1].set_ylabel('Count')
        axs[0, 1].grid(True)
        
        # Plot confidence per iteration
        axs[1, 0].plot([r['iteration'] for r in results['iteration_results']], 
                     [r['confidence'] for r in results['iteration_results']], 
                     'o-', linewidth=2, color='purple')
        axs[1, 0].set_title('Solution Confidence per Iteration')
        axs[1, 0].set_xlabel('Iteration')
        axs[1, 0].set_ylabel('Confidence')
        axs[1, 0].grid(True)
        
        # Plot TRIZ principles and emergent findings
        x = [r['iteration'] for r in results['iteration_results']]
        y1 = [r['triz_principles_used'] for r in results['iteration_results']]
        y2 = [r['emergent_findings'] for r in results['iteration_results']]
        
        axs[1, 1].plot(x, y1, 'o-', linewidth=2, color='red', label='TRIZ Principles')
        axs[1, 1].plot(x, y2, 'o-', linewidth=2, color='orange', label='Emergent Findings')
        axs[1, 1].set_title('TRIZ Principles & Emergent Findings')
        axs[1, 1].set_xlabel('Iteration')
        axs[1, 1].set_ylabel('Count')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        
        # Add simulation metadata
        plt.figtext(0.02, 0.02, f"Domain: {results['domain']} | Complexity: {results['complexity']:.2f} | "
                           f"Noise: {results['noise_level']:.2f} | Iterations: {results['iterations']}")
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Add super title
        plt.suptitle(f"Chain of Solution Simulation Results - {results['simulation_id']}", fontsize=16)
        
        # Save figure
        output_file = os.path.join(self.results_dir, f"{results['simulation_id']}_visualization.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualization saved to {output_file}")
        return output_file
    
    def compare_simulations(self, sim_results_list, labels=None):
        """Compare multiple simulation results.
        
        Args:
            sim_results_list: List of simulation results dictionaries
            labels: List of labels for the simulations (optional)
            
        Returns:
            Path to saved comparison visualization file
        """
        self.logger.info(f"Comparing {len(sim_results_list)} simulations")
        
        if not labels:
            labels = [f"Sim {i+1}" for i in range(len(sim_results_list))]
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot average execution time
        axs[0, 0].bar(labels, [r['average_execution_time'] for r in sim_results_list])
        axs[0, 0].set_title('Average Execution Time')
        axs[0, 0].set_ylabel('Time (seconds)')
        axs[0, 0].grid(True, axis='y')
        
        # Plot average recommendation count
        axs[0, 1].bar(labels, [r['average_recommendation_count'] for r in sim_results_list], color='green')
        axs[0, 1].set_title('Average Recommendation Count')
        axs[0, 1].set_ylabel('Count')
        axs[0, 1].grid(True, axis='y')
        
        # Plot average confidence
        axs[1, 0].bar(labels, [r['average_confidence'] for r in sim_results_list], color='purple')
        axs[1, 0].set_title('Average Solution Confidence')
        axs[1, 0].set_ylabel('Confidence')
        axs[1, 0].grid(True, axis='y')
        
        # Plot average TRIZ principles and emergent findings
        x = np.arange(len(labels))
        width = 0.35
        
        axs[1, 1].bar(x - width/2, [r['average_triz_principles'] for r in sim_results_list], 
                     width, label='TRIZ Principles', color='red')
        axs[1, 1].bar(x + width/2, [r['average_emergent_findings'] for r in sim_results_list], 
                     width, label='Emergent Findings', color='orange')
        axs[1, 1].set_title('Average TRIZ Principles & Emergent Findings')
        axs[1, 1].set_ylabel('Count')
        axs[1, 1].set_xticks(x)
        axs[1, 1].set_xticklabels(labels)
        axs[1, 1].legend()
        axs[1, 1].grid(True, axis='y')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Add super title
        plt.suptitle("Chain of Solution Simulation Comparison", fontsize=16)
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.results_dir, f"simulation_comparison_{timestamp}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comparison visualization saved to {output_file}")
        return output_file
    
    def _save_simulation_results(self, results):
        """Save simulation results to a file.
        
        Args:
            results: Simulation results dictionary
            
        Returns:
            Path to saved results file
        """
        import json
        
        # Create output file path
        output_file = os.path.join(self.results_dir, f"{results['simulation_id']}_results.json")
        
        # Save results to file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Simulation results saved to {output_file}")
        return output_file
    
    def _generate_healthcare_text(self, complexity):
        """Generate synthetic healthcare text data."""
        # Simple template-based text generation
        templates = [
            "Patient reports occasional {symptom1}. Has been following medication regimen {adherence}. {additional_note}",
            "Patient experiencing {symptom1} and {symptom2}. Medication {effectiveness}. {additional_note}",
            "Follow-up examination shows {finding}. Patient reports {symptom1} {frequency}. {additional_note}"
        ]
        
        # Variables to fill in templates
        symptoms = ['shortness of breath', 'chest pain', 'fatigue', 'dizziness', 'headache', 'nausea']
        adherences = ['consistently', 'with occasional lapses', 'poorly']
        effectiveness = ['seems to be effective', 'shows limited effectiveness', 'needs adjustment']
        findings = ['improvement in condition', 'stable condition', 'slight deterioration']
        frequencies = ['occasionally', 'frequently', 'rarely', 'only during physical activity']
        additional_notes = [
            'No other complaints.', 
            'Mentioned feeling unwell twice last week.',
            'Has been monitoring blood pressure at home.',
            'Reports difficulty sleeping.'            
        ]
        
        # Select template based on complexity
        template_index = min(int(complexity * len(templates)), len(templates) - 1)
        template = templates[template_index]
        
        # Fill in template with random selections
        text = template.format(
            symptom1=np.random.choice(symptoms),
            symptom2=np.random.choice(symptoms),
            adherence=np.random.choice(adherences),
            effectiveness=np.random.choice(effectiveness),
            finding=np.random.choice(findings),
            frequency=np.random.choice(frequencies),
            additional_note=np.random.choice(additional_notes)
        )
        
        # Add complexity-based details
        if complexity > 0.6:
            family_history = ["Family history of heart disease.", "No significant family history.", "Family history of diabetes."]
            text += " " + np.random.choice(family_history)
        
        if complexity > 0.8:
            lifestyle = ["Maintains regular exercise routine.", "Sedentary lifestyle.", "Recently started exercise program."]
            text += " " + np.random.choice(lifestyle)
        
        return text
    
    def _generate_engineering_text(self, complexity):
        """Generate synthetic engineering text data."""
        # Simple template-based text generation
        templates = [
            "System requires {requirement1} while maintaining {constraint1}.",
            "Design must incorporate {requirement1} and {requirement2} without exceeding {constraint1}.",
            "New prototype shows {result1} but suffers from {issue1}. {additional_note}"
        ]
        
        # Variables to fill in templates
        requirements = ['improved efficiency', 'reduced weight', 'increased durability', 'better thermal management', 'enhanced user interface']
        constraints = ['current form factor', 'manufacturing cost limits', 'power consumption restrictions', 'regulatory requirements']
        results = ['promising performance improvements', 'significant weight reduction', 'enhanced durability in testing']
        issues = ['overheating under load', 'manufacturing complexity', 'component availability issues', 'higher than expected costs']
        additional_notes = [
            'Further testing required.', 
            'Engineering team is evaluating alternatives.',
            'Management approval needed for design changes.',
            'Customers have expressed satisfaction with initial samples.'            
        ]
        
        # Select template based on complexity
        template_index = min(int(complexity * len(templates)), len(templates) - 1)
        template = templates[template_index]
        
        # Fill in template with random selections
        text = template.format(
            requirement1=np.random.choice(requirements),
            requirement2=np.random.choice(requirements),
            constraint1=np.random.choice(constraints),
            result1=np.random.choice(results),
            issue1=np.random.choice(issues),
            additional_note=np.random.choice(additional_notes)
        )
        
        # Add complexity-based details
        if complexity > 0.6:
            timeline = ["Project timeline is tight.", "Timeline has been extended by two months.", "Deadline remains unchanged despite challenges."]
            text += " " + np.random.choice(timeline)
        
        if complexity > 0.8:
            competition = ["Competitor has announced similar product.", "We maintain technological advantage over competitors.", "Market competition is intensifying."]
            text += " " + np.random.choice(competition)
        
        return text
    
    def _generate_business_text(self, complexity):
        """Generate synthetic business text data."""
        # Simple template-based text generation
        templates = [
            "Market analysis shows {trend1}. Customer feedback indicates {feedback1}.",
            "Quarterly results indicate {trend1} with {trend2}. {additional_note}",
            "Competitive landscape is {landscape}. Our positioning is {positioning}. {additional_note}"
        ]
        
        # Variables to fill in templates
        trends = ['growing demand', 'shifting customer preferences', 'increasing competition', 'price sensitivity', 'emerging market opportunities']
        feedbacks = ['satisfaction with product features', 'concerns about pricing', 'requests for additional functionality', 'positive response to recent changes']
        landscapes = ['becoming more crowded', 'evolving rapidly', 'relatively stable', 'dominated by few key players']
        positionings = ['strong in premium segment', 'challenged in current market', 'differentiated by quality', 'competitive but not leading']
        additional_notes = [
            'Strategic review scheduled next month.', 
            'Marketing team recommends repositioning.',
            'Board has requested detailed analysis.',
            'Customer retention remains a priority.'            
        ]
        
        # Select template based on complexity
        template_index = min(int(complexity * len(templates)), len(templates) - 1)
        template = templates[template_index]
        
        # Fill in template with random selections
        text = template.format(
            trend1=np.random.choice(trends),
            trend2=np.random.choice(trends),
            feedback1=np.random.choice(feedbacks),
            landscape=np.random.choice(landscapes),
            positioning=np.random.choice(positionings),
            additional_note=np.random.choice(additional_notes)
        )
        
        # Add complexity-based details
        if complexity > 0.6:
            finance = ["Profit margins are under pressure.", "Cash flow remains strong.", "Investment in R&D has increased."]
            text += " " + np.random.choice(finance)
        
        if complexity > 0.8:
            global_factors = ["Global supply chain issues are affecting delivery.", "International expansion proceeding as planned.", "Currency fluctuations impacting overseas markets."]
            text += " " + np.random.choice(global_factors)
        
        return text
    
    def _generate_medical_image(self, complexity, noise_level):
        """Generate synthetic medical image data."""
        # For simulation purposes, just create a numpy array
        # In a real implementation, this would generate more realistic medical imagery
        
        # Image size increases with complexity
        size = int(64 + complexity * 192)  # Size ranges from 64x64 to 256x256
        
        # Create base image (grayscale)
        image = np.zeros((size, size))
        
        # Add some simple structures
        center_x, center_y = size // 2, size // 2
        radius = int(size * 0.3)
        
        # Create a circular structure (e.g., organ)
        y, x = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = dist_from_center <= radius
        image[mask] = 0.8
        
        # Add some internal structures based on complexity
        if complexity > 0.3:
            # Add smaller circular structure inside
            small_radius = int(radius * 0.6)
            small_mask = dist_from_center <= small_radius
            image[small_mask] = 0.5
        
        if complexity > 0.6:
            # Add even more detailed structures
            tiny_radius = int(radius * 0.3)
            tiny_mask = dist_from_center <= tiny_radius
            image[tiny_mask] = 0.2
            
            # Add some "vessels" (lines)
            for i in range(int(complexity * 10)):
                angle = np.random.random() * 2 * np.pi
                length = int(radius * (0.5 + 0.5 * np.random.random()))
                thickness = int(1 + complexity * 3)
                
                for r in range(radius, radius + length):
                    x = int(center_x + r * np.cos(angle))
                    y = int(center_y + r * np.sin(angle))
                    
                    if 0 <= x < size and 0 <= y < size:
                        for dx in range(-thickness, thickness+1):
                            for dy in range(-thickness, thickness+1):
                                if dx*dx + dy*dy <= thickness*thickness:
                                    px, py = x + dx, y + dy
                                    if 0 <= px < size and 0 <= py < size:
                                        image[py, px] = 0.6
        
        # Add some "abnormalities" based on complexity
        if complexity > 0.7:
            # Add small bright spot (e.g., lesion)
            spot_x = int(center_x + radius * 0.7 * np.cos(np.pi/4))
            spot_y = int(center_y + radius * 0.7 * np.sin(np.pi/4))
            spot_radius = int(size * 0.05)
            
            for y in range(size):
                for x in range(size):
                    dist = np.sqrt((x - spot_x)**2 + (y - spot_y)**2)
                    if dist <= spot_radius:
                        image[y, x] = 1.0
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, (size, size))
            image = np.clip(image + noise, 0, 1)
        
        return image
    
    def _generate_heart_sound(self, complexity, noise_level):
        """Generate synthetic heart sound data."""
        # For simulation purposes, generate a synthetic heart sound waveform
        # In a real implementation, this would be more realistic
        
        # Generate a 2-second audio sample at 1000 Hz
        sample_rate = 1000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Create a basic heart sound (simplified as two peaks per heart beat)
        heart_rate = 60 + complexity * 40  # 60-100 bpm depending on complexity
        beats_per_second = heart_rate / 60
        beat_period = 1.0 / beats_per_second
        
        # Initialize the waveform
        waveform = np.zeros_like(t)
        
        # Generate heart beats
        for beat_time in np.arange(0, duration, beat_period):
            # First sound (lub) - higher amplitude
            mask1 = (t >= beat_time) & (t < beat_time + 0.1)
            waveform[mask1] += 0.8 * np.sin(2 * np.pi * 20 * (t[mask1] - beat_time))
            
            # Second sound (dub) - lower amplitude, a bit later
            mask2 = (t >= beat_time + 0.3) & (t < beat_time + 0.4)
            waveform[mask2] += 0.5 * np.sin(2 * np.pi * 30 * (t[mask2] - (beat_time + 0.3)))
        
        # Add complexity - murmur or irregularity
        if complexity > 0.5:
            # Add a murmur (simplified as continuous low frequency noise between beats)
            for beat_time in np.arange(0, duration, beat_period):
                mask = (t >= beat_time + 0.1) & (t < beat_time + 0.3)
                waveform[mask] += 0.2 * complexity * np.sin(2 * np.pi * 15 * (t[mask] - (beat_time + 0.1)))
        
        # Add irregularity in beat timing
        if complexity > 0.7:
            irregular_beat = int(duration * beats_per_second * 0.8)  # Index of beat to modify
            if irregular_beat < len(waveform) - int(sample_rate * 0.2):
                # Make one beat appear early or be skipped
                if np.random.random() > 0.5:
                    # Early beat - appear at 80% of normal interval
                    early_time = irregular_beat * beat_period * 0.8
                    mask1 = (t >= early_time) & (t < early_time + 0.1)
                    waveform[mask1] += 0.8 * np.sin(2 * np.pi * 20 * (t[mask1] - early_time))
                else:
                    # Skip a beat - zero out a section
                    skip_time = irregular_beat * beat_period
                    mask = (t >= skip_time) & (t < skip_time + beat_period)
                    waveform[mask] = 0
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, len(waveform))
            waveform = waveform + noise
        
        # Normalize
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-6)
        
        return waveform
    
    def _generate_vital_signs(self, complexity, noise_level):
        """Generate synthetic vital signs time series data."""
        # Generate a 60-second time series at 1 Hz
        sample_rate = 1
        duration = 60.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Base heart rate: 60-100 bpm depending on complexity
        base_heart_rate = 60 + complexity * 40
        
        # Generate heart rate time series with some variation
        variation_freq = 0.05  # Slow variation over time
        variation_amp = 5 + complexity * 10  # 5-15 bpm variation
        
        heart_rate = base_heart_rate + variation_amp * np.sin(2 * np.pi * variation_freq * t)
        
        # Add complexity-based patterns
        if complexity > 0.4:
            # Add a sudden increase and recovery (e.g., response to stimulus)
            start_idx = int(len(t) * 0.3)
            peak_idx = start_idx + int(len(t) * 0.05)
            recovery_idx = peak_idx + int(len(t) * 0.15)
            
            # Create a smooth increase and recovery
            for i in range(start_idx, peak_idx):
                progress = (i - start_idx) / (peak_idx - start_idx)
                heart_rate[i] += 20 * progress  # Up to 20 bpm increase
            
            for i in range(peak_idx, min(recovery_idx, len(heart_rate))):
                progress = (i - peak_idx) / (recovery_idx - peak_idx)
                heart_rate[i] += 20 * (1 - progress)  # Gradual recovery
        
        if complexity > 0.7:
            # Add irregular pattern (e.g., arrhythmia)
            arrhythmia_start = int(len(t) * 0.6)
            arrhythmia_end = arrhythmia_start + int(len(t) * 0.1)
            
            for i in range(arrhythmia_start, min(arrhythmia_end, len(heart_rate))):
                if i % 2 == 0:  # Every other point
                    heart_rate[i] += 15  # Sudden increase
                else:
                    heart_rate[i] -= 10  # Sudden decrease
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 5, len(heart_rate))  # Scale noise by 5 for heart rate
            heart_rate = heart_rate + noise
        
        return heart_rate
    
    def _generate_technical_image(self, complexity, noise_level):
        """Generate synthetic technical image data (e.g., technical drawing)."""
        # For simulation purposes, create a simple technical drawing
        # In a real implementation, this would be more realistic
        
        # Image size increases with complexity
        size = int(64 + complexity * 192)  # Size ranges from 64x64 to 256x256
        
        # Create base image (white background)
        image = np.ones((size, size))
        
        # Add some basic shapes (lines, circles, rectangles)
        # Horizontal and vertical lines
        line_thickness = int(1 + complexity * 3)
        
        # Horizontal lines
        n_h_lines = int(2 + complexity * 8)
        for i in range(n_h_lines):
            y_pos = int(size * (0.2 + 0.6 * i / n_h_lines))
            line_length = int(size * (0.5 + 0.4 * np.random.random()))
            x_start = int(size * 0.1)
            
            for y in range(y_pos - line_thickness // 2, y_pos + line_thickness // 2 + 1):
                if 0 <= y < size:
                    for x in range(x_start, x_start + line_length):
                        if 0 <= x < size:
                            image[y, x] = 0.1
        
        # Vertical lines
        n_v_lines = int(2 + complexity * 6)
        for i in range(n_v_lines):
            x_pos = int(size * (0.2 + 0.6 * i / n_v_lines))
            line_length = int(size * (0.3 + 0.4 * np.random.random()))
            y_start = int(size * 0.2)
            
            for x in range(x_pos - line_thickness // 2, x_pos + line_thickness // 2 + 1):
                if 0 <= x < size:
                    for y in range(y_start, y_start + line_length):
                        if 0 <= y < size:
                            image[y, x] = 0.1
        
        # Add circles/components based on complexity
        if complexity > 0.4:
            n_components = int(2 + complexity * 6)
            for _ in range(n_components):
                comp_x = int(size * (0.1 + 0.8 * np.random.random()))
                comp_y = int(size * (0.1 + 0.8 * np.random.random()))
                comp_radius = int(size * (0.02 + 0.05 * np.random.random()))
                
                # Draw component (circle)
                for y in range(size):
                    for x in range(size):
                        dist = np.sqrt((x - comp_x)**2 + (y - comp_y)**2)
                        if dist <= comp_radius:
                            image[y, x] = 0.2
                            
                        # Draw component outline
                        elif comp_radius < dist <= comp_radius + line_thickness:
                            image[y, x] = 0.05
        
        # Add text-like features based on complexity
        if complexity > 0.6:
            n_text_blocks = int(1 + complexity * 4)
            for _ in range(n_text_blocks):
                text_x = int(size * (0.1 + 0.8 * np.random.random()))
                text_y = int(size * (0.1 + 0.8 * np.random.random()))
                text_width = int(size * (0.1 + 0.2 * np.random.random()))
                text_height = int(size * 0.03)
                
                # Create text-like features (just rectangles for simulation)
                for y in range(text_y, text_y + text_height):
                    if 0 <= y < size:
                        for x in range(text_x, text_x + text_width):
                            if 0 <= x < size:
                                image[y, x] = 0.3
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 0.1, (size, size))  # Subtle noise for technical drawings
            image = np.clip(image + noise, 0, 1)
        
        return image
    
    def _generate_sensor_data(self, complexity, noise_level):
        """Generate synthetic sensor reading time series data."""
        # Generate a 60-second time series at 1 Hz
        sample_rate = 1
        duration = 60.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Base sensor values
        base_values = {
            'temperature': 25.0 + complexity * 10,  # 25-35°C depending on complexity
            'pressure': 100.0 + complexity * 20,    # 100-120 units
            'vibration': 0.5 * complexity,          # 0-0.5 units
        }
        
        # Generate sensor time series with some variation
        variation_freq = 0.1  # Variation over time
        variation_amp = {
            'temperature': 2 + complexity * 3,     # 2-5°C variation
            'pressure': 5 + complexity * 10,       # 5-15 units variation
            'vibration': 0.1 + complexity * 0.3,   # 0.1-0.4 units variation
        }
        
        # Initialize sensor readings
        sensor_data = {}
        for sensor_type, base_value in base_values.items():
            # Add sinusoidal variation
            sensor_data[sensor_type] = base_value + variation_amp[sensor_type] * np.sin(2 * np.pi * variation_freq * t)
        
        # Add complexity-based patterns
        if complexity > 0.4:
            # Add a sudden spike in temperature
            spike_idx = int(len(t) * 0.4)
            spike_duration = int(len(t) * 0.05)
            
            for i in range(spike_idx, min(spike_idx + spike_duration, len(t))):
                progress = (i - spike_idx) / spike_duration
                # Bell curve shape for spike
                if progress <= 0.5:
                    factor = progress * 2
                else:
                    factor = (1 - progress) * 2
                
                sensor_data['temperature'][i] += 10 * factor * complexity
        
        if complexity > 0.6:
            # Add pressure oscillations
            oscillation_start = int(len(t) * 0.6)
            oscillation_end = oscillation_start + int(len(t) * 0.2)
            oscillation_freq = 0.5  # Higher frequency
            
            for i in range(oscillation_start, min(oscillation_end, len(t))):
                rel_t = (i - oscillation_start) / (oscillation_end - oscillation_start)
                sensor_data['pressure'][i] += 7 * complexity * np.sin(2 * np.pi * oscillation_freq * rel_t * 10)
        
        if complexity > 0.7:
            # Add vibration bursts
            for _ in range(int(complexity * 5)):
                burst_idx = int(np.random.random() * len(t) * 0.8)
                burst_duration = int(len(t) * 0.03)
                
                for i in range(burst_idx, min(burst_idx + burst_duration, len(t))):
                    sensor_data['vibration'][i] += np.random.random() * complexity
        
        # Add noise to all sensors
        if noise_level > 0:
            for sensor_type in sensor_data:
                # Scale noise by sensor type
                if sensor_type == 'temperature':
                    noise_scale = 0.5
                elif sensor_type == 'pressure':
                    noise_scale = 2.0
                else:  # vibration
                    noise_scale = 0.1
                
                noise = np.random.normal(0, noise_level * noise_scale, len(t))
                sensor_data[sensor_type] = sensor_data[sensor_type] + noise
        
        return sensor_data
    
    def _generate_financial_data(self, complexity, noise_level):
        """Generate synthetic financial time series data."""
        # Generate a 24-month time series
        months = 24
        t = np.arange(months)
        
        # Base financial metrics
        base_values = {
            'revenue': 100000 + complexity * 50000,    # Base revenue $100k-$150k
            'costs': 80000 + complexity * 30000,       # Base costs $80k-$110k
            'customers': 1000 + complexity * 500,      # Base customers 1000-1500
        }
        
        # Generate financial metrics time series
        growth_rate = 0.02 + complexity * 0.03  # 2-5% monthly growth
        seasonal_amp = {
            'revenue': 0.1 + complexity * 0.1,      # 10-20% seasonal variation
            'costs': 0.05 + complexity * 0.05,      # 5-10% seasonal variation
            'customers': 0.08 + complexity * 0.07,  # 8-15% seasonal variation
        }
        
        # Initialize financial data
        financial_data = {}
        for metric, base_value in base_values.items():
            # Growth component
            growth = np.power(1 + growth_rate, t)
            
            # Seasonal component (yearly cycle)
            seasonal = 1 + seasonal_amp[metric] * np.sin(2 * np.pi * t / 12)
            
            # Combine components
            financial_data[metric] = base_value * growth * seasonal
        
        # Add complexity-based patterns
        if complexity > 0.4:
            # Add a market event that affects revenue
            event_idx = int(months * 0.4)
            event_impact = 0.85 + 0.1 * np.random.random()  # 15-25% drop
            
            for i in range(event_idx, months):
                recovery_rate = 0.2 * complexity  # Higher complexity means faster recovery
                recovery_factor = min(1, 1 - (1 - event_impact) * np.exp(-recovery_rate * (i - event_idx)))
                financial_data['revenue'][i] *= recovery_factor
        
        if complexity > 0.6:
            # Add cost increase
            cost_increase_idx = int(months * 0.6)
            cost_increase = 1.1 + 0.1 * complexity  # 10-20% increase
            
            for i in range(cost_increase_idx, months):
                financial_data['costs'][i] *= cost_increase
        
        if complexity > 0.7:
            # Add customer acquisition campaign
            campaign_idx = int(months * 0.7)
            campaign_duration = int(months * 0.1)
            campaign_boost = 1.15 + 0.15 * complexity  # 15-30% boost
            
            for i in range(campaign_idx, min(campaign_idx + campaign_duration, months)):
                financial_data['customers'][i] *= campaign_boost
                # Campaign costs
                financial_data['costs'][i] *= 1.1  # 10% higher costs during campaign
        
        # Calculate derived metrics
        financial_data['profit'] = financial_data['revenue'] - financial_data['costs']
        financial_data['revenue_per_customer'] = financial_data['revenue'] / financial_data['customers']
        
        # Add noise to all metrics
        if noise_level > 0:
            for metric in financial_data:
                # Scale noise by metric
                if metric == 'revenue':
                    noise_scale = 5000
                elif metric == 'costs':
                    noise_scale = 3000
                elif metric == 'customers':
                    noise_scale = 50
                elif metric == 'profit':
                    noise_scale = 2000
                else:  # revenue_per_customer
                    noise_scale = 5
                
                noise = np.random.normal(0, noise_level * noise_scale, months)
                financial_data[metric] = financial_data[metric] + noise
        
        return financial_data
    
    def _generate_customer_data(self, complexity, noise_level):
        """Generate synthetic customer demographic and behavioral data."""
        # Number of customers increases with complexity
        n_customers = int(50 + complexity * 950)  # 50-1000 customers
        
        # Initialize customer data
        customer_data = {
            'customer_id': [f"CUST{i:05d}" for i in range(n_customers)],
            'age': [],
            'gender': [],
            'location': [],
            'purchase_frequency': [],
            'average_purchase': [],
            'customer_since': [],
            'satisfaction_score': [],
            'lifetime_value': []
        }
        
        # Location distribution
        locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami', 'Seattle', 'Boston', 'Denver']
        location_weights = [0.25, 0.2, 0.15, 0.1, 0.1, 0.08, 0.07, 0.05]
        
        # Generate customer attributes
        for _ in range(n_customers):
            # Age: normal distribution around 35, std dev increases with complexity
            age = int(np.random.normal(35, 5 + complexity * 10))
            age = max(18, min(age, 90))  # Clamp to realistic range
            customer_data['age'].append(age)
            
            # Gender: binary for simplicity
            customer_data['gender'].append(np.random.choice(['M', 'F']))
            
            # Location: weighted distribution
            customer_data['location'].append(np.random.choice(locations, p=location_weights))
            
            # Purchase frequency: purchases per year, higher for high-complexity scenarios
            base_freq = 2 + complexity * 10  # 2-12 purchases per year
            freq_variation = 1 + complexity * 3  # 1-4 variation
            purchase_freq = max(1, np.random.normal(base_freq, freq_variation))
            customer_data['purchase_frequency'].append(purchase_freq)
            
            # Average purchase: amount per purchase, higher for high-complexity scenarios
            base_purchase = 20 + complexity * 180  # $20-$200 per purchase
            purchase_variation = 5 + complexity * 45  # $5-$50 variation
            avg_purchase = max(5, np.random.normal(base_purchase, purchase_variation))
            customer_data['average_purchase'].append(avg_purchase)
            
            # Customer since: months as customer, more variation with complexity
            max_tenure = 6 + int(complexity * 54)  # 6-60 months
            customer_data['customer_since'].append(np.random.randint(1, max_tenure))
            
            # Satisfaction score: 1-5 scale, higher average for low-complexity scenarios
            satisfaction_mean = 4.0 - complexity * 1.0  # 3.0-4.0 average score
            satisfaction = np.random.normal(satisfaction_mean, 0.5)
            satisfaction = max(1, min(5, satisfaction))  # Clamp to 1-5 range
            customer_data['satisfaction_score'].append(satisfaction)
            
            # Calculate lifetime value
            months = customer_data['customer_since'][-1]
            frequency = customer_data['purchase_frequency'][-1]
            avg_purchase = customer_data['average_purchase'][-1]
            
            # Simple LTV calculation: months * purchases per month * average purchase
            ltv = months * (frequency / 12) * avg_purchase
            customer_data['lifetime_value'].append(ltv)
        
        # Add noise to numeric attributes
        if noise_level > 0:
            for attr in ['age', 'purchase_frequency', 'average_purchase', 'customer_since', 'satisfaction_score', 'lifetime_value']:
                # Scale noise by attribute
                if attr == 'age':
                    noise_scale = 2
                elif attr == 'purchase_frequency':
                    noise_scale = 0.5
                elif attr == 'average_purchase':
                    noise_scale = 10
                elif attr == 'customer_since':
                    noise_scale = 1
                elif attr == 'satisfaction_score':
                    noise_scale = 0.2
                else:  # lifetime_value
                    noise_scale = 50
                
                noise = np.random.normal(0, noise_level * noise_scale, n_customers)
                
                # Apply noise and maintain valid ranges
                if attr == 'age':
                    customer_data[attr] = [max(18, min(90, int(val + n))) for val, n in zip(customer_data[attr], noise)]
                elif attr == 'purchase_frequency':
                    customer_data[attr] = [max(1, val + n) for val, n in zip(customer_data[attr], noise)]
                elif attr == 'average_purchase':
                    customer_data[attr] = [max(5, val + n) for val, n in zip(customer_data[attr], noise)]
                elif attr == 'customer_since':
                    customer_data[attr] = [max(1, min(60, int(val + n))) for val, n in zip(customer_data[attr], noise)]
                elif attr == 'satisfaction_score':
                    customer_data[attr] = [max(1, min(5, val + n)) for val, n in zip(customer_data[attr], noise)]
                else:  # lifetime_value
                    customer_data[attr] = [max(0, val + n) for val, n in zip(customer_data[attr], noise)]
        
        return customer_data


if __name__ == "__main__":
    # Example usage
    config = {
        'results_dir': 'simulation_results',
        'chain_of_solution': {
            'model_type': 'llama3.1',
            'model_size': '8B',
            'use_emergent_pattern_detection': True,
            'use_triz60': True
        }
    }
    
    # Create simulation environment
    sim = CoSSimulation(config)
    
    # Run a simple simulation
    results = sim.run_simulation(
        domain='healthcare',
        complexity=0.7,
        noise_level=0.3,
        iterations=3
    )
    
    # Visualize results
    sim.visualize_results(results)
    
    print(f"Simulation completed. Results saved to {sim.results_dir}")

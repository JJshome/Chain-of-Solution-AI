"""
AI Feedback Loop Module for Chain of Solution System
AI 피드백 루프부 (AI Feedback Loop Unit) - Patent Component 170

This module implements continuous learning and improvement through AI-driven feedback mechanisms.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime
import threading
import queue
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class FeedbackData:
    """Represents feedback data from system operation"""
    timestamp: float
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    performance_metrics: Dict[str, float]
    user_feedback: Optional[Dict[str, Any]] = None
    error_data: Optional[Dict[str, Any]] = None
    

@dataclass
class LearningUpdate:
    """Represents a learning update from feedback analysis"""
    update_type: str  # 'parameter', 'structure', 'strategy'
    updates: Dict[str, Any]
    confidence: float
    source_feedback_ids: List[str]
    

class FeedbackAnalyzer(ABC):
    """Abstract base class for feedback analysis strategies"""
    
    @abstractmethod
    def analyze(self, feedback_history: List[FeedbackData]) -> List[LearningUpdate]:
        """Analyze feedback and generate learning updates"""
        pass


class PerformanceTrendAnalyzer(FeedbackAnalyzer):
    """Analyzes performance trends to identify improvement opportunities"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
    def analyze(self, feedback_history: List[FeedbackData]) -> List[LearningUpdate]:
        """Analyze recent performance trends"""
        if len(feedback_history) < self.window_size:
            return []
            
        recent_feedback = feedback_history[-self.window_size:]
        updates = []
        
        # Analyze performance trends for each metric
        metrics_over_time = {}
        for feedback in recent_feedback:
            for metric, value in feedback.performance_metrics.items():
                if metric not in metrics_over_time:
                    metrics_over_time[metric] = []
                metrics_over_time[metric].append(value)
        
        # Detect declining trends
        for metric, values in metrics_over_time.items():
            values_array = np.array(values)
            trend = np.polyfit(range(len(values)), values_array, 1)[0]
            
            if trend < -0.01:  # Declining trend
                update = LearningUpdate(
                    update_type="parameter",
                    updates={
                        f"boost_{metric}_priority": True,
                        f"{metric}_optimization_weight": 1.5
                    },
                    confidence=abs(trend) * 10,
                    source_feedback_ids=[str(i) for i in range(len(recent_feedback))]
                )
                updates.append(update)
                
        return updates


class ErrorPatternAnalyzer(FeedbackAnalyzer):
    """Analyzes error patterns to prevent recurring issues"""
    
    def analyze(self, feedback_history: List[FeedbackData]) -> List[LearningUpdate]:
        """Analyze error patterns in feedback"""
        error_patterns = {}
        updates = []
        
        for feedback in feedback_history:
            if feedback.error_data:
                error_type = feedback.error_data.get("type", "unknown")
                if error_type not in error_patterns:
                    error_patterns[error_type] = []
                error_patterns[error_type].append(feedback)
        
        # Generate updates for frequent errors
        for error_type, occurrences in error_patterns.items():
            if len(occurrences) > 5:  # Threshold for considering it a pattern
                update = LearningUpdate(
                    update_type="strategy",
                    updates={
                        "add_error_handler": error_type,
                        "preventive_checks": True,
                        "fallback_strategy": f"handle_{error_type}"
                    },
                    confidence=min(0.9, len(occurrences) / 20),
                    source_feedback_ids=[str(i) for i in range(len(occurrences))]
                )
                updates.append(update)
                
        return updates


class AIFeedbackLoop:
    """Main AI feedback loop component"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.feedback_queue = queue.Queue()
        self.feedback_history: List[FeedbackData] = []
        self.analyzers: List[FeedbackAnalyzer] = []
        self.learning_updates: List[LearningUpdate] = []
        self.is_running = False
        self.processing_thread = None
        
        # Initialize components
        self._initialize_analyzers()
        
        # Callback functions for applying updates
        self.update_callbacks: Dict[str, Callable] = {}
        
    def _initialize_analyzers(self):
        """Initialize default feedback analyzers"""
        self.analyzers.append(PerformanceTrendAnalyzer())
        self.analyzers.append(ErrorPatternAnalyzer())
        
    def add_analyzer(self, analyzer: FeedbackAnalyzer):
        """Add a custom feedback analyzer"""
        self.analyzers.append(analyzer)
        
    def register_update_callback(self, update_type: str, callback: Callable):
        """Register a callback for applying specific update types"""
        self.update_callbacks[update_type] = callback
        
    def submit_feedback(self, feedback: FeedbackData):
        """Submit feedback data for processing"""
        self.feedback_queue.put(feedback)
        
    def start(self):
        """Start the feedback processing loop"""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._process_feedback_loop)
            self.processing_thread.start()
            logger.info("AI Feedback Loop started")
            
    def stop(self):
        """Stop the feedback processing loop"""
        if self.is_running:
            self.is_running = False
            if self.processing_thread:
                self.processing_thread.join()
            logger.info("AI Feedback Loop stopped")
            
    def _process_feedback_loop(self):
        """Main feedback processing loop"""
        while self.is_running:
            try:
                # Process feedback from queue
                feedback = self.feedback_queue.get(timeout=1.0)
                self.feedback_history.append(feedback)
                
                # Trigger analysis if enough feedback accumulated
                if len(self.feedback_history) % 10 == 0:
                    self._analyze_and_update()
                    
            except queue.Empty:
                continue
                
    def _analyze_and_update(self):
        """Analyze feedback and apply learning updates"""
        logger.info("Analyzing feedback and generating updates")
        
        # Collect updates from all analyzers
        all_updates = []
        for analyzer in self.analyzers:
            updates = analyzer.analyze(self.feedback_history)
            all_updates.extend(updates)
            
        # Apply updates through callbacks
        for update in all_updates:
            if update.update_type in self.update_callbacks:
                try:
                    self.update_callbacks[update.update_type](update)
                    self.learning_updates.append(update)
                    logger.info(f"Applied {update.update_type} update with confidence {update.confidence}")
                except Exception as e:
                    logger.error(f"Failed to apply update: {e}")
                    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress"""
        summary = {
            "total_feedback": len(self.feedback_history),
            "total_updates": len(self.learning_updates),
            "update_types": {},
            "average_performance": {},
            "improvement_rate": {}
        }
        
        # Count update types
        for update in self.learning_updates:
            update_type = update.update_type
            if update_type not in summary["update_types"]:
                summary["update_types"][update_type] = 0
            summary["update_types"][update_type] += 1
            
        # Calculate average performance metrics
        if self.feedback_history:
            recent_feedback = self.feedback_history[-100:]  # Last 100 feedback entries
            metric_values = {}
            
            for feedback in recent_feedback:
                for metric, value in feedback.performance_metrics.items():
                    if metric not in metric_values:
                        metric_values[metric] = []
                    metric_values[metric].append(value)
                    
            for metric, values in metric_values.items():
                summary["average_performance"][metric] = np.mean(values)
                
            # Calculate improvement rate
            if len(self.feedback_history) >= 200:
                old_feedback = self.feedback_history[-200:-100]
                
                for metric in metric_values:
                    old_values = [f.performance_metrics.get(metric, 0) for f in old_feedback]
                    new_values = [f.performance_metrics.get(metric, 0) for f in recent_feedback]
                    
                    old_avg = np.mean(old_values) if old_values else 0
                    new_avg = np.mean(new_values) if new_values else 0
                    
                    if old_avg > 0:
                        improvement = (new_avg - old_avg) / old_avg * 100
                        summary["improvement_rate"][metric] = improvement
                        
        return summary
    
    def export_feedback_data(self, filepath: str):
        """Export feedback history to file"""
        data = {
            "feedback_history": [
                {
                    "timestamp": f.timestamp,
                    "performance_metrics": f.performance_metrics,
                    "user_feedback": f.user_feedback,
                    "error_data": f.error_data
                }
                for f in self.feedback_history
            ],
            "learning_updates": [
                {
                    "update_type": u.update_type,
                    "updates": u.updates,
                    "confidence": u.confidence
                }
                for u in self.learning_updates
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def import_feedback_data(self, filepath: str):
        """Import feedback history from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Reconstruct feedback history
        for feedback_dict in data.get("feedback_history", []):
            feedback = FeedbackData(
                timestamp=feedback_dict["timestamp"],
                input_data={},  # Not stored in export
                output_data={},  # Not stored in export
                performance_metrics=feedback_dict["performance_metrics"],
                user_feedback=feedback_dict.get("user_feedback"),
                error_data=feedback_dict.get("error_data")
            )
            self.feedback_history.append(feedback)

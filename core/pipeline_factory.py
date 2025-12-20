"""Factory for creating fruit-specific pipelines"""
from typing import Dict, Type
from .base_pipeline import BaseFruitPipeline


class PipelineFactory:
    """
    Factory class for creating fruit processing pipelines
    
    This implements the Factory Pattern to allow easy extension
    to new fruit types in the future.
    """
    
    _pipelines: Dict[str, Type[BaseFruitPipeline]] = {}
    
    @classmethod
    def register_pipeline(cls, fruit_type: str, pipeline_class: Type[BaseFruitPipeline]):
        """
        Register a new pipeline type
        
        Args:
            fruit_type: Name of the fruit (e.g., 'mango', 'apple')
            pipeline_class: Pipeline class to register
        """
        cls._pipelines[fruit_type.lower()] = pipeline_class
        print(f"Registered pipeline for {fruit_type}")
    
    @classmethod
    def create_pipeline(cls, fruit_type: str, settings=None) -> BaseFruitPipeline:
        """
        Create a pipeline for the specified fruit type
        
        Args:
            fruit_type: Name of the fruit
            settings: Configuration settings
        
        Returns:
            Pipeline instance for the fruit type
        
        Raises:
            ValueError: If fruit type is not registered
        """
        fruit_type = fruit_type.lower()
        
        if fruit_type not in cls._pipelines:
            available = ', '.join(cls._pipelines.keys())
            raise ValueError(
                f"Unknown fruit type: {fruit_type}. "
                f"Available types: {available}"
            )
        
        pipeline_class = cls._pipelines[fruit_type]
        return pipeline_class(settings)
    
    @classmethod
    def get_available_fruits(cls):
        """Get list of registered fruit types"""
        return list(cls._pipelines.keys())

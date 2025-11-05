from typing import List, Tuple, Any, Union
from pydantic import BaseModel, Field, ValidationError, field_validator

class EventResponse(BaseModel):
    """
    Schema for capturing both event sequence data (triplets) and target-value pairs.
    Both types are stored in the events field.
    """
    events: List[List[Any]] = Field(
        description="List of items that can be either triplets [source, target, count] or pairs [target, value]"
    )
    
    @field_validator('events')
    @classmethod
    def validate_events(cls, value):
        """
        Validate that each item is either:
        - A triplet with format [str, str, int]
        - A pair with format [str, Any]
        """
        if not isinstance(value, list):
            raise ValueError("Events must be a list")
            
        validated_events = []
        for i, item in enumerate(value):
            if not isinstance(item, list):
                raise ValueError(f"Item {i} must be a list")
                
            # Check if it's a triplet
            if len(item) == 3:
                if not isinstance(item[0], str) or not isinstance(item[1], str):
                    raise ValueError(f"Source and target in triplet {i} must be strings")
                    
                if not isinstance(item[2], int):
                    try:
                        # Try to convert to int if it's a numeric string or float
                        item[2] = int(item[2])
                    except (ValueError, TypeError):
                        raise ValueError(f"Count in triplet {i} must be an integer")
                
                validated_events.append([item[0], item[1], item[2]])
            
            # Check if it's a pair
            elif len(item) == 2:
                if not isinstance(item[0], str):
                    raise ValueError(f"Target in pair {i} must be a string")
                
                # Second element can be any type
                validated_events.append([item[0], item[1]])
            
            # Invalid length
            else:
                raise ValueError(f"Item {i} must be either a pair [target, value] or a triplet [source, target, count]")
                    
        return validated_events
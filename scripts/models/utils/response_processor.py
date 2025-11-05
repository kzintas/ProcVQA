"""
Response processor for extracting structured data from model outputs.
"""

import re
import json
from typing import Dict, List, Any
from collections import defaultdict
import ast

from .schemas import EventResponse

def process_llm_response(response_text: str) -> EventResponse:
    """
    Extract valid JSON array from the LLM's response and return an EventResponse object.
    
    Accepts both formats in the same response:
    - Triplets: [source, target, count]
    - Pairs: [target, value]
    
    Both will be stored in the events list.
    """
    try:
        # Clean the response to extract just the JSON array
        json_pattern = r'\[\s*\[.*?\]\s*\]'
        json_match = re.search(json_pattern, response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # Validate it's a list
            if not isinstance(data, list):
                return EventResponse(events=[])
                
            validated_events = []
            
            for i, item in enumerate(data):
                if not isinstance(item, list):
                    continue
                    
                # Process based on length
                if len(item) == 3:  # Triplet [source, target, count]
                    if not isinstance(item[0], str) or not isinstance(item[1], str):
                        continue
                        
                    # Ensure third element is an integer
                    if not isinstance(item[2], int):
                        try:
                            item[2] = int(item[2])
                        except (ValueError, TypeError):
                            continue
                    
                    validated_events.append(item)
                    
                elif len(item) == 2:  # Pair [target, value]
                    if not isinstance(item[0], str):
                        continue
                    
                    # For pairs, simply accept them as is
                    validated_events.append(item)
            
            return EventResponse(events=validated_events)
        
        # If no valid JSON array is found, return empty result
        print(f"Error processing response: No events extracted")
        return EventResponse(events=[])
    
    except Exception as e:
        # If any error occurs during parsing, return empty result
        print(f"Error processing response: {e}")
        return EventResponse(events=[])
    
    
def process_pixtral_response(response_text: str) -> EventResponse:
    """
    Extract valid triplets from the LLM's response and return an EventResponse object.
    Handles both JSON format and Python list literals.
    """
    try:
        # First try to find content within code blocks
        code_block_pattern = r'```(?:json|python)?\s*([\s\S]*?)\s*```'
        code_matches = re.findall(code_block_pattern, response_text, re.DOTALL)
        
        json_str = None
        
        # If we found code blocks, use the content
        if code_matches:
            for code_content in code_matches:
                # Remove any leading/trailing whitespace
                json_str = code_content.strip()
                break  # Use the first code block found
        else:
            # Try to find array pattern in the entire response
            json_pattern = r'\[\s*\[[\s\S]*?\]\s*\]'
            json_match = re.search(json_pattern, response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
        
        # If we found a string, try to parse it
        if json_str:
            # Use ast.literal_eval which handles Python syntax like single quotes
            
            try:
                data = ast.literal_eval(json_str)
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing with ast: {e}")
                return EventResponse(events=[])
            
            # Validate and convert triplets
            valid_triplets = []
            for item in data:
                if not isinstance(item, list) or len(item) != 3:
                    continue
                    
                # Ensure first two elements are strings
                src = str(item[0]) if item[0] is not None else ""
                tgt = str(item[1]) if item[1] is not None else ""
                
                # Ensure third element is an integer
                try:
                    count = int(item[2])
                    valid_triplets.append([src, tgt, count])
                except (ValueError, TypeError):
                    continue
            
            return EventResponse(events=valid_triplets)
        
        # If no valid string was found, return empty result
        return EventResponse(events=[])
    
    except Exception as e:
        # If any error occurs during processing, return empty result
        print(f"Error processing response: {e}")
        return EventResponse(events=[])
"""
Task generator for Coffee Law experiments
"""
import numpy as np
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import random
from dataclasses import dataclass

@dataclass
class Task:
    """Represents a single task for experiments"""
    id: str
    description: str
    instructions: str
    chunks: List[str]
    gold_answer: Optional[str]
    metadata: Dict[str, Any]

class TaskGenerator:
    """
    Generate diverse tasks for Coffee Law verification
    
    Tasks should test different aspects:
    - Information integration
    - Multi-hop reasoning
    - Numerical computation
    - Fact extraction
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        
        # Task templates
        self.task_types = [
            'fact_extraction',
            'numerical_calculation', 
            'multi_hop_reasoning',
            'summarization',
            'comparison',
            'constraint_satisfaction'
        ]
        
    def generate_task_dataset(self, n_tasks: int = 1000) -> List[Dict]:
        """
        Generate a dataset of tasks for experiments
        """
        tasks = []
        
        for i in range(n_tasks):
            task_type = self.rng.choice(self.task_types)
            
            if task_type == 'fact_extraction':
                task = self._generate_fact_extraction_task(i)
            elif task_type == 'numerical_calculation':
                task = self._generate_numerical_task(i)
            elif task_type == 'multi_hop_reasoning':
                task = self._generate_multi_hop_task(i)
            elif task_type == 'summarization':
                task = self._generate_summarization_task(i)
            elif task_type == 'comparison':
                task = self._generate_comparison_task(i)
            else:  # constraint_satisfaction
                task = self._generate_constraint_task(i)
            
            tasks.append(task)
        
        return tasks
    
    def _generate_fact_extraction_task(self, task_id: int) -> Dict:
        """Generate a fact extraction task"""
        # Create synthetic facts
        entities = ['Company A', 'Company B', 'Product X', 'Product Y', 'Region Z']
        attributes = ['revenue', 'growth rate', 'market share', 'customer base', 'profit margin']
        years = [2020, 2021, 2022, 2023]
        
        chunks = []
        facts = {}
        
        # Generate 5-10 fact chunks
        n_chunks = self.rng.randint(5, 11)
        
        for _ in range(n_chunks):
            entity = self.rng.choice(entities)
            attribute = self.rng.choice(attributes)
            year = self.rng.choice(years)
            
            if attribute in ['revenue', 'customer base']:
                value = self.rng.randint(100, 10000) * 1000
                unit = '$' if attribute == 'revenue' else ' customers'
            else:  # percentages
                value = round(self.rng.uniform(0.1, 50), 1)
                unit = '%'
            
            chunk = f"In {year}, {entity} reported a {attribute} of {value}{unit}."
            chunks.append(chunk)
            
            # Store for potential query
            key = f"{entity}_{attribute}_{year}"
            facts[key] = f"{value}{unit}"
        
        # Pick a random fact to query
        query_key = self.rng.choice(list(facts.keys()))
        entity, attribute, year = query_key.split('_')
        
        return {
            'id': f'fact_extraction_{task_id}',
            'type': 'fact_extraction',
            'description': f"Find the {attribute} of {entity} in {year}",
            'instructions': "Extract the requested information from the provided context.",
            'chunks': chunks,
            'gold_answer': facts[query_key],
            'metadata': {'difficulty': 'easy', 'hops': 1}
        }
    
    def _generate_numerical_task(self, task_id: int) -> Dict:
        """Generate a numerical calculation task"""
        operations = ['sum', 'average', 'difference', 'percentage_change']
        operation = self.rng.choice(operations)
        
        chunks = []
        values = []
        
        # Generate 3-6 numerical facts
        n_values = self.rng.randint(3, 7)
        categories = [f"Category {chr(65+i)}" for i in range(n_values)]
        
        for cat in categories:
            value = self.rng.randint(100, 1000)
            values.append(value)
            chunk = f"The value for {cat} is {value} units."
            chunks.append(chunk)
        
        # Add some distractors
        for _ in range(2):
            distractor_cat = f"Category {chr(65+n_values+_)}"
            distractor_val = self.rng.randint(100, 1000)
            chunk = f"The value for {distractor_cat} is {distractor_val} units (excluded from calculation)."
            chunks.append(chunk)
        
        # Shuffle chunks
        self.rng.shuffle(chunks)
        
        # Calculate answer based on operation
        if operation == 'sum':
            answer = sum(values)
            instruction = f"Calculate the sum of values for {', '.join(categories)}."
        elif operation == 'average':
            answer = round(sum(values) / len(values), 1)
            instruction = f"Calculate the average of values for {', '.join(categories)}."
        elif operation == 'difference':
            answer = max(values) - min(values)
            instruction = f"Find the difference between the highest and lowest values among {', '.join(categories)}."
        else:  # percentage_change
            answer = round(((values[-1] - values[0]) / values[0]) * 100, 1)
            instruction = f"Calculate the percentage change from {categories[0]} to {categories[-1]}."
        
        return {
            'id': f'numerical_{task_id}',
            'type': 'numerical_calculation',
            'description': f"Perform {operation} calculation",
            'instructions': instruction,
            'chunks': chunks,
            'gold_answer': str(answer),
            'metadata': {'difficulty': 'medium', 'operation': operation}
        }
    
    def _generate_multi_hop_task(self, task_id: int) -> Dict:
        """Generate a multi-hop reasoning task"""
        # Create a chain of relationships
        entities = [f"Entity_{chr(65+i)}" for i in range(6)]
        relationships = []
        
        # Create relationship chain
        for i in range(len(entities)-1):
            rel_type = self.rng.choice(['owns', 'supplies', 'partners with', 'competes with'])
            relationships.append((entities[i], rel_type, entities[i+1]))
        
        # Create chunks describing relationships
        chunks = []
        for e1, rel, e2 in relationships:
            chunk = f"{e1} {rel} {e2}."
            chunks.append(chunk)
        
        # Add some distractors
        for _ in range(3):
            e1 = self.rng.choice(entities)
            e2 = self.rng.choice([e for e in entities if e != e1])
            rel = self.rng.choice(['considered acquiring', 'met with', 'discussed'])
            chunk = f"{e1} {rel} {e2} (unconfirmed)."
            chunks.append(chunk)
        
        # Shuffle
        self.rng.shuffle(chunks)
        
        # Create a multi-hop query
        start = entities[0]
        end = entities[-1]
        path = ' -> '.join([f"{e1} {rel} {e2}" for (e1, rel, e2) in relationships])
        
        return {
            'id': f'multi_hop_{task_id}',
            'type': 'multi_hop_reasoning',
            'description': f"Trace relationship from {start} to {end}",
            'instructions': f"Describe the relationship chain connecting {start} to {end}.",
            'chunks': chunks,
            'gold_answer': path,
            'metadata': {'difficulty': 'hard', 'hops': len(relationships)}
        }
    
    def _generate_summarization_task(self, task_id: int) -> Dict:
        """Generate a summarization task"""
        topic = self.rng.choice(['market trends', 'company performance', 'technology adoption', 'regulatory changes'])
        
        # Generate related chunks
        chunks = []
        key_points = []
        
        n_points = self.rng.randint(3, 5)
        
        for i in range(n_points):
            if topic == 'market trends':
                trend = self.rng.choice(['increasing', 'decreasing', 'stable', 'volatile'])
                sector = self.rng.choice(['technology', 'healthcare', 'finance', 'retail'])
                point = f"The {sector} sector shows {trend} trends"
                detail = f" with {self.rng.randint(5, 30)}% change year-over-year"
            elif topic == 'company performance':
                metric = self.rng.choice(['revenue', 'profit', 'market share', 'customer satisfaction'])
                change = self.rng.choice(['improved', 'declined', 'remained steady'])
                point = f"Company {metric} {change}"
                detail = f" by {self.rng.randint(5, 25)}% in Q{self.rng.randint(1, 5)}"
            else:
                point = f"Key observation {i+1} about {topic}"
                detail = " with significant implications"
            
            full_chunk = point + detail + "."
            chunks.append(full_chunk)
            key_points.append(point)
        
        # Add filler chunks
        for _ in range(2):
            filler = f"Additional context about {topic} that provides background information."
            chunks.append(filler)
        
        self.rng.shuffle(chunks)
        
        return {
            'id': f'summarization_{task_id}',
            'type': 'summarization',
            'description': f"Summarize key points about {topic}",
            'instructions': "Provide a concise summary of the main points from the context.",
            'chunks': chunks,
            'gold_answer': '; '.join(key_points),
            'metadata': {'difficulty': 'medium', 'n_key_points': n_points}
        }
    
    def _generate_comparison_task(self, task_id: int) -> Dict:
        """Generate a comparison task"""
        items = [f"Option {chr(65+i)}" for i in range(3)]
        attributes = ['cost', 'efficiency', 'reliability', 'speed']
        
        chunks = []
        data = {item: {} for item in items}
        
        # Generate attribute values for each item
        for item in items:
            for attr in attributes:
                if attr == 'cost':
                    value = self.rng.randint(1000, 10000)
                    unit = '$'
                elif attr in ['efficiency', 'reliability']:
                    value = self.rng.randint(70, 99)
                    unit = '%'
                else:  # speed
                    value = self.rng.randint(10, 100)
                    unit = ' ms'
                
                data[item][attr] = value
                chunk = f"{item} has a {attr} of {value}{unit}."
                chunks.append(chunk)
        
        self.rng.shuffle(chunks)
        
        # Determine best option based on weighted score
        scores = {}
        for item in items:
            # Lower cost is better, higher others are better
            score = (10000 - data[item]['cost']) / 100
            score += data[item]['efficiency']
            score += data[item]['reliability'] 
            score += (100 - data[item]['speed'])  # Lower speed is better
            scores[item] = score
        
        best_option = max(scores, key=scores.get)
        
        return {
            'id': f'comparison_{task_id}',
            'type': 'comparison',
            'description': 'Compare options and select the best',
            'instructions': 'Analyze all attributes and determine which option is best overall.',
            'chunks': chunks,
            'gold_answer': best_option,
            'metadata': {'difficulty': 'medium', 'n_options': len(items)}
        }
    
    def _generate_constraint_task(self, task_id: int) -> Dict:
        """Generate a constraint satisfaction task"""
        items = [f"Item {chr(65+i)}" for i in range(5)]
        
        # Define constraints
        constraints = []
        chunks = []
        
        # Budget constraint
        budget = self.rng.randint(5000, 10000)
        chunks.append(f"Total budget available: ${budget}.")
        constraints.append(('budget', budget))
        
        # Item costs and values
        item_data = {}
        for item in items:
            cost = self.rng.randint(500, 3000)
            value = self.rng.randint(50, 200)
            item_data[item] = {'cost': cost, 'value': value}
            chunks.append(f"{item} costs ${cost} and provides {value} units of value.")
        
        # Minimum value requirement
        min_value = self.rng.randint(200, 400)
        chunks.append(f"Minimum total value required: {min_value} units.")
        constraints.append(('min_value', min_value))
        
        # Incompatibility constraints
        incompatible_pairs = []
        n_incompatible = self.rng.randint(1, 3)
        for _ in range(n_incompatible):
            i1, i2 = self.rng.choice(len(items), 2, replace=False)
            pair = (items[i1], items[i2])
            incompatible_pairs.append(pair)
            chunks.append(f"{pair[0]} and {pair[1]} cannot be selected together.")
        
        self.rng.shuffle(chunks)
        
        # Find valid solution (simplified - in practice would use constraint solver)
        valid_selections = []
        for item in items:
            if item_data[item]['cost'] <= budget and item_data[item]['value'] >= min_value/3:
                valid = True
                for pair in incompatible_pairs:
                    if item in pair:
                        valid = False
                        break
                if valid:
                    valid_selections.append(item)
        
        return {
            'id': f'constraint_{task_id}',
            'type': 'constraint_satisfaction',
            'description': 'Select items satisfying all constraints',
            'instructions': 'Choose items that meet budget, value, and compatibility constraints.',
            'chunks': chunks,
            'gold_answer': ', '.join(valid_selections[:2]) if valid_selections else 'No valid solution',
            'metadata': {
                'difficulty': 'hard',
                'n_constraints': len(constraints) + len(incompatible_pairs)
            }
        }
    
    def save_dataset(self, tasks: List[Dict], filepath: Path):
        """Save task dataset to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(tasks, f, indent=2)
    
    def load_dataset(self, filepath: Path) -> List[Dict]:
        """Load task dataset from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def create_balanced_dataset(self, 
                              n_per_type: int = 100,
                              difficulty_distribution: Dict[str, float] = None) -> List[Dict]:
        """
        Create a balanced dataset with equal representation of task types
        
        Args:
            n_per_type: Number of tasks per type
            difficulty_distribution: Dict mapping difficulty to proportion
                                   e.g., {'easy': 0.3, 'medium': 0.5, 'hard': 0.2}
        """
        if difficulty_distribution is None:
            difficulty_distribution = {'easy': 0.3, 'medium': 0.4, 'hard': 0.3}
        
        all_tasks = []
        task_id = 0
        
        for task_type in self.task_types:
            for _ in range(n_per_type):
                if task_type == 'fact_extraction':
                    task = self._generate_fact_extraction_task(task_id)
                elif task_type == 'numerical_calculation':
                    task = self._generate_numerical_task(task_id)
                elif task_type == 'multi_hop_reasoning':
                    task = self._generate_multi_hop_task(task_id)
                elif task_type == 'summarization':
                    task = self._generate_summarization_task(task_id)
                elif task_type == 'comparison':
                    task = self._generate_comparison_task(task_id)
                else:
                    task = self._generate_constraint_task(task_id)
                
                all_tasks.append(task)
                task_id += 1
        
        # Shuffle to mix types
        self.rng.shuffle(all_tasks)
        
        return all_tasks
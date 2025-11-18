
import pandas
import numpy as np

def format_data(data: pandas.DataFrame, column_name: str, constraints: dict[int, (float, float)]):
    # Apply constraints
    for i in range(len(data)):
        value = data.at[i, column_name]

        for constraint, (min_val, max_val) in constraints.items():
            if value >= min_val and value < max_val:
                data.at[i, column_name] = constraint
        
    return data



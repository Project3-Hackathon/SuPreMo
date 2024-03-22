import pandas as pd
from scipy import stats
import numpy as np

def calculate_p_value(control_values, target_value):
    control_values = np.array(control_values)
    target_value = float(target_value)
    # Count the number of values in control list that are larger than the target value
    count = np.sum(1 for value in control_values if value > target_value)
    # Calculate the score
    score = count / (len(control_values) + 1)
    return score

def calculate_p_values(df_target, df_control):
    # Create an object to store the p-values per column
    p_values_per_score = {}

    # Select columns that contain or start with "mse_" or "corr_"
    selected_columns = [column for column in df_target.columns if column.startswith("mse_") or column.startswith("corr_")]
    # Iterate over the selected columns
    for column in selected_columns:
        # Check if the column exists in the control dataframe
        if column not in df_control.columns:
            print(f"Column {column} of the target scores does not exist in the control scores.")
            continue

        # Preprocess the control values, remove NAs
        control_values = df_control[column].dropna().values
        # Create an empty list to store the p-values of the current scores
        p_values = []
        
        # Check if current columns is MSE or corr
        if column.startswith("mse_"):
            # Iterate over the target scores
            for entry in df_target[column]:
                # Check if the entry is NaN
                if pd.isnull(pd.to_numeric(entry, errors='coerce')):
                    # If entry is NaN, store NaN for p-value as well
                    p_values.append(float('nan'))
                else:
                    # Calculate the p-value compared to all score values of the control file
                    p_value = calculate_p_value(control_values, float(entry))
                    #print(f"P-value for {entry}: {p_value}")
                    # Store the p-value for this entry and this score
                    p_values.append(p_value)
            
        elif column.startswith("corr_"):
            # Preprocess control values
            control_values_corr = [1 - value for value in control_values]
            # Iterate over the target scores
            for entry in df_target[column]:
                # Check if the entry is NaN
                if pd.isnull(pd.to_numeric(entry, errors='coerce')):
                    # If entry is NaN, store NaN for p-value as well
                    p_values.append(float('nan'))
                else:
                    entry_corr = 1 - float(entry)
                    p_value = calculate_p_value(control_values_corr, entry_corr)
                    #print(f"P-value for {entry}: {p_value}")
                    # Store the p-value for this entry and this score
                    p_values.append(p_value)
        
        # Store the list of p-values for the current score column 
        p_values_per_score[column] = p_values

    # Add the p-values as new columns to the target dataframe
    for column, p_values in p_values_per_score.items():
        df_target[column + '_p_value'] = p_values

    # Save the updated target dataframe with the p-values to the input file
    #df_target.to_csv(target_file, sep='\t', index=False)
    #print(df_target)
    return df_target

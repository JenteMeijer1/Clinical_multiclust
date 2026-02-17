import os
import glob
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import skew
from typing import Tuple 

import random
from scipy.stats import chi2
from multiprocessing import Pool, cpu_count
import functools
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer


from typing import List, Dict, Any


##############################################
# Define labels for ordinal scale 
##############################################

LabelSpec = Dict[str, Any]

LABEL_SPECS: List[LabelSpec] = [
    # 1. Head injury severity
    {
        "mapping": {
            1: "No Head Injury",
            2: "Mild head injury/concussion with no loss of consciousness",
            3: "Mild head injury/concussion with brief loss of consciousness",
            4: "Mild head injury with LOC for between 2-30 minutes.... lasting 1-24 hours",
            5: "Mild head injury with LOC for between 2-30 minutes.... lasting 1-7 days",
            6: "Mild head injury with LOC for between 2-30 minutes.... lasting >7 days to 2 months",
            7: "Mild head injury with LOC for between 30 minutes-5 hours",
            8: "Head injury with LOC/coma lasting 6 hours or more",
        }
    },
    # 2. Duration in seconds/minutes
    {
        "mapping": {
            1: "Just a few seconds",
            2: "Less than a minute",
            3: "One minute or more",
        }
    },
    # 3. General “present” scale
    {
        "mapping": {
            1: "Not present",
            2: "Very mild",
            3: "Mild",
            4: "Moderate",
            5: "Moderately severe",
            6: "Severe",
            7: "Extremely severe",
        }
    },
    # 4. Affective symptoms (0 = Missing)
    {
        "mapping": {
            0: "Missing",
            1: "asymptomatic, returned to usual self",
            2: "residual/mild affective symptoms",
            3: "partial remission, moderate symptoms or impairment",
            4: "marked/major symptoms or impairment, does not meet criteria for MDE",
            5: "meets definite MDE criteria without prominent psychotic symptoms or extreme impairment",
            6: "meets definite MDE criteria with prominent psychotic symptoms or extreme impairment",
        }
    },
    # 5. Affective symptoms (1–6)
    {
        "mapping": {
            1: "asymptomatic, returned to usual self",
            2: "residual/mild affective symptoms",
            3: "partial remission, moderate symptoms or impairment",
            4: "marked/major symptoms or impairment",
            5: "meets definite criteria without prominent psychotic symptoms or extreme impairment",
            6: "meets definite criteria with prominent psychotic symptoms or extreme impairment",
        }
    },
    # 6. Probable vs definite criteria
    {
        "mapping": {
            1: "asymptomatic, returned to usual self",
            2: "meets probable criteria (mild symptoms)",
            3: "meets definite criteria (severe symptoms)",
        }
    },
    # 7. Observational severity
    {
        "mapping": {
            1: "Not observed",
            2: "Very mild",
            3: "Mild",
            4: "Moderate",
            5: "Moderately Severe",
            6: "Severe",
            7: "Very Severe",
        }
    },
    # 8. Not present → extremely severe (lowercase)
    {
        "mapping": {
            1: "not present",
            2: "very mild",
            3: "mild",
            4: "moderate",
            5: "moderate-severe",
            6: "severe",
            7: "extremely severe",
        }
    },
    # 9. Clinical global impression
    {
        "mapping": {
            1: "Normal, Not ill",
            2: "Minimally ill",
            3: "Mildly ill",
            4: "Moderately ill",
            5: "Markedly ill",
            6: "Severely ill",
            7: "Very Severely ill",
        }
    },
    # 10. First/last only, auto-fill codes 2–5 as strings
    {
        "first": 1,
        "first_label": "least important",
        "last": 6,
        "last_label": "most important",
        "fill_middle": True,
    },
    # 11. None at all → a lot
    {
        "mapping": {
            1: "None at all",
            2: "Very little",
            3: "Some",
            4: "A lot",
        }
    },
    # 12. Substance use severity
    {
        "mapping": {
            1: "Abstinent",
            2: "Use without impairment",
            3: "Abuse",
            4: "Dependence",
            5: "Dependence with institutionalization",
        }
    },
    # 13. (duplicate of #9)
    {
        "mapping": {
            1: "Normal, Not ill",
            2: "Minimally ill",
            3: "Mildly ill",
            4: "Moderately ill",
            5: "Markedly ill",
            6: "Severely ill",
            7: "Very Severely ill",
        }
    },
    # 14. Depression severity

    {
        "mapping": {
            1: "Absent",
            2: "Mild - Expresses some sadness or discouragement on questioning",
            3: "Moderate - Distinct depressed mood persisting up to half the time over last 2 weeks: present daily",
            4: "Severe - Markedly depressed mood persisting daily over half the time interfering with normal motor and social functioning",

        }
    },

    {
        "mapping": {
            1: "Absent",
            2: "Mild - Has at times felt hopeless over the last week but still has some degree of hope for the future",
            3: "Moderate - Persistent, moderate sense of hopelessness over last week. Can be persuaded to acknowledge possibility of things being better",
            4: "Severe - Persisting and distressing sense of hopelessness",
        },
    },

    {
        "mapping": {
            1: "Absent",
            2: "Mild - Some inferiority; not amounting to feelings of worthlessness",
            3: "Moderate - Subject feels worthless, but less than 50% of the time",
            4: "Severe - Subject feels worthless more than 50% of the time. May be challenged to acknowledge otherwise",
        },

    },

    {
        "mapping": {
            1: "Absent",
            2: "Mild - Subject feels blamed but not accused less than 50% of the time",
            3: "Moderate - Persisting sense of being blamed, and/or occasional sense of being accused",
            4: "Severe - Persistent sense of being accused. When challenged, acknowledges that it is not so",
        },
    },

    {
        "mapping": {
            1: "Absent",
            2: "Mild - Subject sometimes feels over guilty about some minor peccadillo, but less than 50% of time",
            3: "Moderate - Subject usually (over 50% of time) feels guilty about past actions the significance of which he exaggerates",
            4: "Severe - Subject usually feels s/he is to blame for everything that has gone wrong, even when not his/her fault",
        },
    },

    {
        "mapping": {
            1: "Absent - No Depression",
            2: "Mild - Depression present but no diurnal variation",
            3: "Moderate - Depression spontaneously mentioned to be worse in a.m.",
            4: "Severe - Depression markedly worse in a.m., with impaired functioning which improves in p.m.",
        },
    },

    {
        "mapping": {
            1: "Absent - No early wakening",
            2: "Mild - Occasionally wakes (up to twice weekly) 1 hour or more before normal time to wake or alarm time",
            3: "Moderate - Often wakes early (up to 5 times weekly) 1 hour or more before normal time to wake or alarm",
            4: "Severe - Daily wakes l hour or more before normal time",
        },
    },

    {
        "mapping": {
            1: "Absent",
            2: "Mild - Frequent thoughts of being better off dead, or occasional thoughts or occasional thoughts of suicide",
            3: "Moderate - Deliberately considered suicide with a plan, but made no attempt",
            4: "Severe - Suicidal attempt apparantly designed to end in death (i.e.: accidental discovery or inefficient means)"
        
        }   
    },

    {
        "mapping": {
            1: "Absent",
            2: "Mild - Subject appears sad and mournful even during parts of the interview, involving affectively neutral discussion",
            3: "Moderate - Subject appears sad and mournful throughout the interview, with gloomy monotonous voice and is tearful or close to tears at times",
            4: "Severe - Subject chokes on distressing topics, frequently sighs deeply and cries openly, or is persistently in a state of frozen misery if examiner is sure that this is present"
        }
    },

    {
        "mapping": {
            1: "Normal, not at all depressed",
            2: "Borderline depressed",
            3: "Mildly depressed",
            4: "Moderately depressed",
            5: "Markedly depressed",
            6: "Severely depressed",
            7: "Among the most severely depressed patients",
        }
    },
    # 15. Clinical global (0 = not assessed)
    {
        "mapping": {
            0: "Not assessed",
            1: "Normal, not at all ill",
            2: "Borderline ill",
            3: "Mildly ill",
            4: "Moderately ill",
            5: "Markedly ill",
            6: "Severely ill",
            7: "Among the most extremely ill patients",
        }
    },
    # 16. Symptom presence
    {
        "mapping": {
            1: "Normal/No symptoms",
            2: "Mild",
            3: "Moderate",
            4: "Severe",
            5: "Very severe",
        }
    },

    # 17. Frequency of occurrence
    {
        "mapping": {
            1: "Never",
            2: "Almost Never",
            3: "Sometimes",
            4: "Fairly often",
            5: "Very often",
        }
    },
    # 18. Impact scale (0=None → 5=Death)
    {
        "mapping": {
            0: "None",
            1: "Minor",
            2: "Moderate",
            3: "Moderately Severe",
            4: "Severe",
            5: "Death",
        }
    },
    # 19. Absent → severe
    {
        "mapping": {
            1: "Absent",
            2: "Mild",
            3: "Moderate",
            4: "Severe",
        }
    },
    # 20. Weekly frequency
    {
        "mapping": {
            1: "Less than once a week",
            2: "Once a week",
            3: "2-5 times in week",
            4: "Daily or almost daily",
            5: "Many times each day",
        }
    },
    # 21. Duration within hours
    {
        "mapping": {
            1: "Fleeting - few seconds or minutes",
            2: "Less than 1 hour/some of the time",
            3: "1-4 hours/a lot of time",
            4: "4-8 hours/most of day",
            5: "More than 8 hours/persistent or continuous",
        }
    },
    # 22. Thought control ability
    {
        "mapping": {
            0: "Does not attempt to control thoughts",
            1: "Easily able to control thoughts",
            2: "Can control thoughts with little difficulty",
            3: "Can control thoughts with some difficulty",
            4: "Can control thoughts with a lot of difficulty",
            5: "Unable to control thoughts",
        }
    },
    # 23. Suicide deterrent impact
    {
        "mapping": {
            0: "Does not apply",
            1: "Deterrents definitely stopped you from attempting suicide",
            2: "Deterrents probably stopped you",
            3: "Uncertain that deterrents stopped you",
            4: "Deterrents most likely did not stop you",
            5: "Deterrents definitely did not stop you",
        }
    },
    # 24. Reason for self‐harm
    {
        "mapping": {
            0: "Does not apply",
            1: "Completely to get attention revenge or a reaction from others",
            2: "Most likely to get attention revenge or a reaction from others",
            3: "Equally to get attention revenge or a reaction from others and to stop the pain",
            4: "Mostly to end or stop the pain",
            5: "Completely to end or stop the pain",
        }
    },
    # 25. Suicidal ideation
    {
        "mapping": {
            0: "No ideation",
            1: "wish to be dead",
            2: "non-specific active suicidal thoughts",
            3: "active suicidal ideation with any methods (no plan) without intent to act",
            4: "active suicidal ideation with some intent to act, without specific plan",
            5: "active suicidal ideation with specific plan and intent",
        }
    },
    # 26. Bottom/top group (“most severe” covers 2–5)
    {
        "mapping": {
            1: "least severe",
            2: "most severe",
            3: "most severe",
            4: "most severe",
            5: "most severe",
        }
    },
    # 27. Suicidal ideation (1–5)
    {
        "mapping": {
            1: "wish to be dead",
            2: "non-specific active suicidal thoughts",
            3: "active suicidal ideation with any methods (no plan) without intent to act",
            4: "active suicidal ideation with some intent to act, without specific plan",
            5: "active suicidal ideation with specific plan and intent",
        }
    },
    # 28. Smoking frequency
    {
        "mapping": {
            1: "Not applicable, I have never smoked",
            2: "Monthly or less",
            3: "2-4 times per month",
            4: "2-3 times per week",
            5: "Daily or almost daily",
        }
    },
    # 29. Role functioning (1–10)
    {
        "mapping": {
            1: "Extreme role dysfunction",
            2: "Inability to function",
            3: "Marginal ability to function",
            4: "Major impairment in role functioning",
            5: "Serious impairment in role functioning",
            6: "Moderate impairment in role functioning",
            7: "Mild problems in role functioning",
            8: "Good role functioning",
            9: "Above average role functioning",
            10: "Superior role functioning",
        }
    },
    # 30. Social functioning (1–10, long labels)
    {
        "mapping": {
            1: "Extreme social isolation",
            2: "Inability to function socially: Unable to function socially or to maintain any interpersonal relationships",
            3: "Marginal ability to function socially or maintain interpersonal relationships",
            4: "Major Impairment in social/interpersonal functioning",
            5: "Serious Impairment in social/interpersonal functioning",
            6: "Moderate Impairment in social/interpersonal functioning",
            7: "Mild problems: Some persistent mild difficulty in social functioning",
            8: "Good: Some transient mild impairment in social functioning",
            9: "Above average: Good Functioning in all social areas, and interpersonally effective",
            10: "Superior: Superior functioning in a wide range of social and interpersonal activities",
        }
    },
    # 31. Simple 1–5 severity
    {
        "mapping": {
            1: "Slight",
            2: "Some",
            3: "Moderate",
            4: "Major",
            5: "Severe",
        }
    },
    # 32. Global functioning (0–10)
    {
        "mapping": {
            10: "Superior functioning in a wide range of activities",
            9: "Good functioning in all areas, occupationally and socially effective",
            8: "No more than a slight impairment in social, occupational or school functioning (e.g., infrequent interpersonal conflict, temporarily falling behind in schoolwork)",
            7: "Some difficulty in social, occupational, or school functioning but generally functioning well and has some meaningful, interpersonal relationships",
            6: "Moderate difficulty in social, occupational, or school functioning (e.g., few friends, conflicts with peers or co-workers)",
            5: "Serious impairment in social, occupational, or school functioning (e.g., no friends, unable to keep a job)",
            4: "Major impairment in several areas such as work or school, family relations (e.g., depressed man avoids friends, neglects family and is unable to work; child frequently beats up younger children, is defiant at home and failing at school)",
            3: "Inability to function in almost all areas (e.g., stays in bed all day; no job, home, or friends)",
            2: "Occasionally fails to maintain minimal personal hygiene; unable to function independently",
            1: "Persistent inability to maintain minimal personal hygiene. Unable to function without harming self or others or without considerable external support (e.g., nursing care and supervision)",
            0: "Inadequate information",
        }
    },
    # 33. Social functioning (1–10, short)
    {
        "mapping": {
            1: "Extreme social isolation",
            2: "Inability to function socially",
            3: "Marginal ability to function socially",
            4: "Major impairment in social and interpersonal functioning",
            5: "Serious impairment in social/interpersonal functioning",
            6: "Moderate impairment in social/interpersonal functioning",
            7: "Mild problems in social/interpersonal functioning",
            8: "Good social/interpersonal functioning",
            9: "Above average social/interpersonal functioning",
            10: "Superior social/interpersonal functioning",
        }
    },
    # 34. Role functioning (1–10, variant)
    {
        "mapping": {
            1: "Extreme role dysfunction",
            2: "Inability to function",
            3: "Marginal ability to function",
            4: "Major impairment in role functioning",
            5: "Serious impairment in role functioning",
            6: "Moderate impairment in role functioning",
            7: "Mild Impairment in role functioning",
            8: "Good role functioning",
            9: "Above average role functioning",
            10: "Superior role functioning",
        }
    },
    # 35. Role functioning (1–10, detailed variant)
    {
        "mapping": {
            1: "Extreme role dysfunction",
            2: "Inability to function",
            3: "Marginal ability to function",
            4: "Major impairment in role functioning",
            5: "Serious Impairment in Role Functioning. Serious impairment independently",
            6: "Moderate impairment in role functioning",
            7: "Mild impairment in role functioning",
            8: "Good role functioning",
            9: "Above average role functioning",
            10: "Superior role functioning",
        }
    },
    # 36. Withdrawal scale (sparse codes)
    {
        "mapping": {
            0: "Not withdrawn",
            2: "Mild withdrawal",
            4: "Moderately withdrawn",
            6: "Unrelated to others, withdrawn and isolated, avoids contacts",
        }
    },
    # 37. Education level (1–15)
    {
        "mapping": {
            1: "Less than 6th grade",
            2: "Some high school",
            3: "High school diploma or GED",
            4: "Some college, no degree",
            5: "Associates degree",
            6: "Bachelors degree",
            7: "Some graduate school",
            8: "Masters degree and above",
            9: "Some post-graduate training, no degree",
            10: "Completed 8th grade, no high school",
            11: "High school",
            12: "College or University",
            13: "Graduate school",
            14: "Other",
            15: "Less than high school",
        }
    },
    # 38. Cognitive test speed
    {
        "mapping": {
            0: "Fail",
            4: "Correct in 66-120 seconds",
            5: "Correct in 46-65 seconds",
            6: "Correct in 31-45 seconds",
            7: "Correct in 1-30 seconds",
        }
    },
    {
        "mapping": {
            0: "Fail",
            4: "Correct in 61-120 seconds",
            5: "Correct in 46-60 seconds",
            6: "Correct in 36-45 seconds",
            7: "Correct in 1-35 seconds",
        }
    },
    # 39. Simple correctness
    {
        "mapping": {
            0: "Fail",
            1: "Partially correct",
            2: "Correct",
        }
    },
    # 40. Multi-code “Correct”
    {
        "mapping": {
            0: "Fail",
            2: "Correct",
            3: "Correct",
            4: "Correct",
        }
    },
    # 41. Cognitive test speed variant
    {
        "mapping": {
            0: "Fail",
            4: "Correct in 31-60 seconds",
            5: "Correct in 21-30 seconds",
            6: "Correct in 11-20 seconds",
            7: "Correct in 1-10 seconds",
        }
    },
    {
        "mapping": {
            0: "Fail",
            4: "Correct in 76-120 seconds",
            5: "Correct in 61-75 seconds",
            6: "Correct in 31-60 seconds",
            7: "Correct in 1-30 seconds",
        }
    },
    # 43. Trial correctness
    {
        "mapping": {
            0: "Fail",
            1: "One trial correct",
            2: "Both trials correct",
        }
    },
    {
        "mapping": {
            0: "Fail",
            1: "One trial correct",
            2: "Two trials correct",
            3: "All trials correct",
        }
    },
    # 45. Completion time buckets
    {
        "mapping": {
            0: "Complete in 45 seconds",
            1: "Complete in 40-44 seconds",
            2: "Complete in 35-39 seconds",
            3: "Complet in 30-34 seconds",
            4: "Complete in 0-29 seconds",
        }
    },
    # 46. Intensity (1–5)
    {
        "mapping": {
            1: "Not at all",
            2: "A little bit",
            3: "Somewhat",
            4: "Quite a bit",
            5: "Very much",
        }
    },
    # 47. Quality rating
    {
        "mapping": {
            1: "Very Poor",
            2: "Poor",
            3: "Fair",
            4: "Good",
            5: "Very Good",
        }
    },
    # 48. Frequency (Never→Always)
    {
        "mapping": {
            1: "Never",
            2: "Almost Never",
            3: "Sometimes",
            4: "Almost Always",
            5: "Always",
        }
    },
    # 49. Custom “Moderately” grouping
    {
        "mapping": {
            1: "Not at all",
            2: "Moderately",
            3: "Moderately",
            4: "Moderately",
            5: "Moderately",
            6: "Moderately",
            7: "Very much",
        }
    },
    # 50. Disturbance scale (only labeled codes)
    {
        "mapping": {
            0: "Not at all disturbing or disabling",
            2: "Slightly disturbing but not really disabling",
            4: "definitely disturbing or disabling",
            6: "Markedly disturbing or disabling",
            8: "Very severy disturbing or disabling",
        }
    },
]




##############################################
# Extract date
##############################################

def extract_date(file_path, prefix):
    """
    Extracts the date from the filename by removing the specified prefix and the '.csv' suffix.
    Assumes the remaining part of the filename is in the format 'YYYY-MM-DD'.
    
    Parameters:
        file_path (str): The path to the file.
        prefix (str): The prefix to remove (e.g., 'basetable_' or 'metatable_').
    
    Returns:
        datetime: The extracted date.
    """
    base = os.path.basename(file_path)
    date_str = base.replace(prefix, "").replace(".csv", "")
    return datetime.strptime(date_str, "%Y-%m-%d")

##############################################
# Load most recent data
##############################################

def load_most_recent_basetable(folder_path):
    """
    Loads the most recent basetable CSV file from the specified folder.
    Assumes basetable files are named like 'basetable_YYYY-MM-DD.csv'.
    """
    pattern = os.path.join(folder_path, "basetable_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No basetable files found in the directory: {folder_path}")
    latest_file = max(files, key=lambda f: extract_date(f, "basetable_"))
    print("Loading most recent basetable file:", latest_file)
    df = pd.read_csv(latest_file)

    # Replace 'nan' and empty strings with np.nan
    df.replace("nan", np.nan, inplace=True)
    df.replace("", np.nan, inplace=True)  # Also handles empty strings
    return df

def load_most_recent_metatable(folder_path):
    """
    Loads the most recent metatable CSV file from the specified folder.
    Assumes metatable files are named like 'metatable_YYYY-MM-DD.csv'.
    """
    pattern = os.path.join(folder_path, "metatable_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No metatable files found in the directory: {folder_path}")
    latest_file = max(files, key=lambda f: extract_date(f, "metatable_"))
    print("Loading most recent metatable file:", latest_file)
    df = pd.read_csv(latest_file)
    return df


##############################################
# Split data by network for discovery and test
##############################################
def split_by_network(df, prescient_ids, id_col='src_subject_id'):
    """
    Splits a dict of Series/DataFrames into two dicts (Prescient vs Pronet)
    based on membership in prescient_ids, and drops the helper 'Network' column.
    """
    discovery_df = {}
    test_df = {}

   
    # 1. Ensure a DataFrame
    if isinstance(df, pd.Series):
        df_mod = df.to_frame().copy()
    else:
        df_mod = df.copy()

    # 2. Case‐insensitive lookup of the ID column
    lower_map = {col.lower(): col for col in df_mod.columns}
    actual_col = lower_map[id_col.lower()]

    # 3. Flag network membership
    mask = df_mod[actual_col].isin(prescient_ids)
    df_mod['Network'] = np.where(mask, 'Prescient', 'Pronet')

    # 4. Split out and drop the helper column
    pres_df = df_mod.loc[df_mod['Network'] == 'Prescient'].reset_index(drop=True)
    pron_df = df_mod.loc[df_mod['Network'] == 'Pronet'].reset_index(drop=True)

    discovery_df = pres_df.drop(columns='Network')
    test_df      = pron_df.drop(columns='Network')

    return discovery_df, test_df



##############################################
# Get seperate data for each modality
##############################################
def extract_modalities(
    meta: pd.DataFrame,
    data: pd.DataFrame,
    subject_id_column: str = 'src_subject_id'
) -> dict:
    """
    Extracts separate DataFrames for each modality from the data using the meta table.
    
    Parameters:
    - meta (pd.DataFrame): A DataFrame with at least the columns 'ElementName' and 'Modality'.
    - data (pd.DataFrame): A DataFrame containing 'src_subject_id' and variables as columns.
    
    Returns:
    - modality_dfs (dict): A dictionary where each key is a modality and each value is a DataFrame
      that includes 'src_subject_id' and the variables associated with that modality.
    """
    # Get unique, non-null modalities
    modalities = meta['Modality'].dropna().unique()
    
    # Dictionary to store the DataFrame for each modality
    modality_dfs = {}

    for modality in modalities:
        # Get the list of variable names for the current modality
        modality_vars = meta.loc[meta['Modality'] == modality, 'ElementName'].dropna().unique()

        # Ensure the variable exists in data (if meta contains variables that aren't in data),
        # including one-hot encoded columns that start with the variable name followed by an underscore.
        valid_vars = []
        for var in modality_vars:
            matches = [col for col in data.columns if col == var or col.startswith(f"{var}_")]
            valid_vars.extend(matches)

        # Always include the subject_id_column if it exists
        columns_to_include = (
            [subject_id_column] + valid_vars
            if subject_id_column in data.columns else valid_vars
        )

        # Skip if no valid variables are available for this modality
        if not valid_vars:
            print(f"Skipping modality '{modality}' - No valid variables found in data.")
            continue

        # Slice and then reset_index so every modality df has a clean 0..N-1 index
        df_mod = data[columns_to_include].copy().reset_index(drop=True)
        modality_dfs[modality] = df_mod

    return modality_dfs



##############################################
# Remove high missing columns and rows
##############################################

def remove_high_missing_data(
    df: pd.DataFrame,
    subject_id_column: str = 'src_subject_id',
    col_threshold: float = 0.5,
    row_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Removes columns (variables) with more than col_threshold fraction of missing values
    and then removes rows (subjects) with more than row_threshold fraction of missing values.
    
    The subject_id_column is always preserved.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        subject_id_column (str): Column to preserve.
        col_threshold (float): Max allowed fraction of missing values per column. 
                               e.g., 0.5 => drop columns with over 50% missing.
        row_threshold (float): Max allowed fraction of missing values per row.
                               e.g., 0.5 => drop rows with over 50% missing.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with the subject_id_column preserved.
    """
    # Separate subject IDs to ensure they aren't dropped
    if subject_id_column in df.columns:
        subject_ids = df[[subject_id_column]]
        df_data = df.drop(columns=[subject_id_column])
    else:
        subject_ids = None
        df_data = df.copy()

    num_rows = df_data.shape[0]
    num_cols = df_data.shape[1]

    # 1) Drop columns with more than 'col_threshold' fraction missing.
    #    Keep columns with at least (1 - col_threshold) * num_rows non-missing values.
    min_non_missing_col = int(np.ceil((1 - col_threshold) * num_rows))
    df_data = df_data.dropna(axis=1, thresh=min_non_missing_col)

    # 2) Drop rows with more than 'row_threshold' fraction missing.
    #    Keep rows with at least (1 - row_threshold) * current_num_cols non-missing values.
    current_num_cols = df_data.shape[1]
    min_non_missing_row = int(np.ceil((1 - row_threshold) * current_num_cols))
    df_data = df_data.dropna(axis=0, thresh=min_non_missing_row)

    # Reattach subject IDs
    if subject_ids is not None:
        # Possibly filter out any subjects that were dropped
        final_df = subject_ids.loc[df_data.index].join(df_data, how='inner')
    else:
        final_df = df_data

    return final_df


def remove_high_missing_data_split(
    discovery_df: pd.DataFrame,
    test_df: pd.DataFrame,
    subject_id_column: str = 'src_subject_id',
    col_threshold: float = 0.5,
    row_threshold: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans discovery and test DataFrames by:
      1) Dropping any column that has more than col_threshold fraction missing
         in EITHER DataFrame.
      2) Keeping only the intersection of columns that survive in both sets.
      3) Dropping any row (within each DataFrame) that has more than
         row_threshold fraction missing ACROSS those shared columns.
    The subject_id_column is always preserved and re-joined at the end.

    Returns:
        (cleaned_discovery_df, cleaned_test_df)
    """
    def split_ids(df):
        if subject_id_column in df.columns:
            return df[[subject_id_column]], df.drop(columns=[subject_id_column])
        else:
            return None, df.copy()

    disc_ids, disc_data = split_ids(discovery_df)
    test_ids, test_data = split_ids(test_df)

    # 1) Identify columns passing threshold in each
    def passing_columns(df_data):
        n_rows = df_data.shape[0]
        min_non_missing = int(np.ceil((1 - col_threshold) * n_rows))
        return set(df_data.dropna(axis=1, thresh=min_non_missing).columns)

    disc_cols = passing_columns(disc_data)
    test_cols = passing_columns(test_data)

    # 2) Keep only the intersection of those columns
    common_cols = sorted(disc_cols & test_cols)
    disc_data = disc_data[common_cols]
    test_data = test_data[common_cols]

    # 3) Drop rows exceeding the row_threshold
    def drop_bad_rows(df_data):
        n_cols = df_data.shape[1]
        min_non_missing = int(np.ceil((1 - row_threshold) * n_cols))
        return df_data.dropna(axis=0, thresh=min_non_missing)

    disc_data = drop_bad_rows(disc_data)
    test_data = drop_bad_rows(test_data)

    # 4) Re-attach subject IDs to the filtered rows
    def reattach(ids, data):
        if ids is None:
            return data
        return ids.loc[data.index].join(data, how='inner')

    cleaned_discovery = reattach(disc_ids, disc_data)
    cleaned_test = reattach(test_ids, test_data)

    return cleaned_discovery, cleaned_test



def remove_high_missing_data_test(
    df: pd.DataFrame,
    df_discovery: pd.DataFrame,
    subject_id_column: str = 'src_subject_id',
    col_threshold: float = 0.5,
    row_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Keeps all columns that exist in discovery. 
    """
    # Separate subject IDs to ensure they aren't dropped
    if subject_id_column in df.columns:
        subject_ids = df[[subject_id_column]]
        df_data = df.drop(columns=[subject_id_column])
    else:
        subject_ids = None
        df_data = df.copy()
    
    # Keep only columns that exist in discovery
    df_data = df_data[df_data.columns.intersection(df_discovery.columns)]
    

    # 2) Drop rows with more than 'row_threshold' fraction missing.
    #    Keep rows with at least (1 - row_threshold) * current_num_cols non-missing values.
    current_num_cols = df_data.shape[1]
    min_non_missing_row = int(np.ceil((1 - row_threshold) * current_num_cols))
    df_data = df_data.dropna(axis=0, thresh=min_non_missing_row)

    # Reattach subject IDs
    if subject_ids is not None:
        # Possibly filter out any subjects that were dropped
        final_df = subject_ids.loc[df_data.index].join(df_data, how='inner')
    else:
        final_df = df_data

    return final_df


def remove_missing_from_modalities(
    modalities_data: dict,
    subject_id_column: str = 'src_subject_id',
    col_threshold: float = 0.5,
    row_threshold: float = 0.5
) -> dict:
    """
    Applies remove_high_missing_data to each modality DataFrame in the modalities_data dictionary.
    
    Parameters:
        modalities_data (dict): Dictionary where keys are modality names and values are DataFrames.
        subject_id_column (str): Column that identifies the subject.
        col_threshold (float): Drop columns with missing fraction > col_threshold.
        row_threshold (float): Drop rows with missing fraction > row_threshold.
    
    Returns:
        dict: Dictionary with the same keys as modalities_data, where each DataFrame has had high-missing
              columns and rows removed.
    """
    cleaned_modalities = {}
    for modality, df in modalities_data.items():
        cleaned_modalities[modality] = remove_high_missing_data(
            df,
            subject_id_column=subject_id_column,
            col_threshold=col_threshold,
            row_threshold=row_threshold
        )
    return cleaned_modalities




##############################################
# Log transform
##############################################

def auto_power_transform(df: pd.DataFrame, skew_threshold: float = 0.75) -> pd.DataFrame:
    """
    Applies Yeo-Johnson power transformation to highly skewed numerical columns.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        skew_threshold (float): The absolute skewness value above which transformation is applied.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    df_transformed = df.copy()
    numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
    transformed_cols = []

    for col in numeric_cols:
        col_values = df_transformed[col].dropna()
        col_skewness = skew(col_values, bias=False)

        if abs(col_skewness) > skew_threshold:
            # Reshape required for sklearn transformers
            col_array = df_transformed[col].values.reshape(-1, 1)
            pt = PowerTransformer(method='yeo-johnson', standardize=False)

            # Fit and transform (handling NaNs by skipping rows with them)
            mask = df_transformed[col].notnull()
            transformed = np.full_like(df_transformed[col], np.nan, dtype=np.float64)
            transformed[mask] = pt.fit_transform(col_array[mask]).flatten()

            df_transformed[col] = transformed
            transformed_cols.append(col)

    return df_transformed

     


##############################################
# Imputation
##############################################

def impute_diverse_data(df: pd.DataFrame, subject_id_column: str = 'src_subject_id', n_neighbors: int = 7) -> pd.DataFrame:
    """
    Impute missing values for a DataFrame using KNN imputation:
    - Numeric columns: imputed using KNN.
    - Categorical columns: encoded, imputed using KNN, then decoded.
    - Date columns: imputed with a constant date.

    The subject_id_column is preserved and not imputed.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        subject_id_column (str): The column name for the subject identifier.
        n_neighbors (int): Number of neighbors for KNN imputation.

    Returns:
        pd.DataFrame: A DataFrame with missing values imputed.
    """
    # Separate subject ID column
    if subject_id_column in df.columns:
        subject_ids = df[[subject_id_column]]
        df_data = df.drop(columns=[subject_id_column])
    else:
        subject_ids = None
        df_data = df.copy()

    # Normalize string "nan" to real NaN in data columns
    df_data.replace("nan", np.nan, inplace=True)

    # Identify column types
    numeric_cols = df_data.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df_data.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df_data.select_dtypes(include=['datetime64[ns]']).columns.tolist()

    # Handle categorical variables (convert to numerical labels)
    label_encoders = {}
    for col in object_cols:
        le = LabelEncoder()
        df_data[col] = df_data[col].astype(str) # Convert to string
        df_data[col] = le.fit_transform(df_data[col])  # Encode as integers
        label_encoders[col] = le  # Store encoder to decode later

    # Handle missing datetime values by imputing a constant
    for col in datetime_cols:
        df_data[col] = pd.to_datetime(df_data[col], errors='coerce')
        df_data[col] = df_data[col].fillna(pd.Timestamp('1900-01-01'))

    # Apply KNN imputation
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(knn_imputer.fit_transform(df_data), columns=df_data.columns)


    # Reset indexes before concatenation to avoid mismatched shapes
    df_imputed.reset_index(drop=True, inplace=True)
    if subject_ids is not None:
        subject_ids.reset_index(drop=True, inplace=True)
        df_imputed = pd.concat([subject_ids, df_imputed], axis=1)
    return df_imputed






def impute_data(modalities_data: dict, subject_id_column: str = 'src_subject_id', n_neighbors: int = 7) -> dict:
    """
    Applies KNN imputation for diverse data types to each modality DataFrame.

    Parameters:
        modalities_data (dict): A dictionary where each key is a modality and each value is a DataFrame.
        subject_id_column (str): Column to be preserved (not imputed).
        n_neighbors (int): Number of neighbors for KNN imputation.

    Returns:
        dict: A dictionary with the same keys as modalities_data, where each DataFrame has missing values imputed.
    """
    imputed_modalities = {}
    for modality, df in modalities_data.items():
        # Check for subjects with all variables missing
        df_mod = df.copy()
        # Separate out data columns (exclude subject ID if present)
        if subject_id_column in df_mod.columns:
            data_only = df_mod.drop(columns=[subject_id_column])
        else:
            data_only = df_mod
        # Identify rows where all data are missing
        all_missing_mask = data_only.isna().all(axis=1)
        num_all_missing = all_missing_mask.sum()
        if num_all_missing > 0:
            warnings.warn(
                f"{modality}: {num_all_missing} participants have missing values for all variables "
                "and will be excluded from imputation."
            )
            # Drop those subjects before imputation
            df_mod = df_mod.loc[~all_missing_mask].reset_index(drop=True)
        imputed_modalities[modality] = impute_diverse_data(df_mod, subject_id_column, n_neighbors)
    return imputed_modalities





##############################################
# Data scaling
##############################################

def scale_diverse_data (
    df: pd.DataFrame,
    subject_id_column: str = 'src_subject_id',
    scaler_type: str = 'robust'
) -> pd.DataFrame:
    """
    Scales only the continuous numeric columns in a DataFrame while leaving binary/low‐cardinality
    numeric columns and non-numeric columns (categorical, dates, etc.) unchanged.
    
    Parameters:
      df (pd.DataFrame): The input DataFrame.
      subject_id_column (str): The name of the subject identifier column to preserve.
      scaler_type (str): 'standard' for StandardScaler or 'minmax' for MinMaxScaler.

    
    Returns:
      pd.DataFrame: The DataFrame with continuous numeric columns scaled.
    """
    # Separate the subject identifier column if present.
    if subject_id_column in df.columns:
        subject_ids = df[[subject_id_column]]
        df_data = df.drop(columns=[subject_id_column])
    else:
        subject_ids = None
        df_data = df.copy()

    # Identify numeric columns.
    numeric_cols = df_data.select_dtypes(include=[np.number]).columns.tolist()

    # Select the scaler.
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("scaler_type must be either 'standard', 'robust' or 'minmax'.")

    # Make a copy and scale only the continuous numeric columns.
    df_data_scaled = df_data.copy()
    for col in numeric_cols:
        df_data_scaled[col] = scaler.fit_transform(df_data[[col]])

    # Reassemble the final DataFrame with the subject identifier column (if present).
    if subject_ids is not None:
        final_df = pd.concat([subject_ids, df_data_scaled], axis=1)
    else:
        final_df = df_data_scaled

    return final_df



def scale_data(
    modalities_data: dict,
    subject_id_column: str = 'src_subject_id',
    scaler_type: str = 'standard'
    ) -> dict:
    """
    Applies the scale_diverse_data function to each modality DataFrame in the modalities_data dictionary.
    
    Parameters:
      modalities_data (dict): Dictionary where keys are modality names and values are DataFrames.
      subject_id_column (str): Column to be preserved (not scaled).
      scaler_type (str): 'standard' for StandardScaler or 'minmax' for MinMaxScaler.
    
    Returns:
      dict: Dictionary with the same keys as modalities_data, where each DataFrame has its continuous
            numeric columns scaled.
    """
    scaled_modalities = {}
    for modality, df in modalities_data.items():
        scaled_modalities[modality] = scale_diverse_data(
            df,
            subject_id_column=subject_id_column,
            scaler_type=scaler_type
        )
    return scaled_modalities







##############################################
# Convert data for VAE structure
##############################################
def dummy_code(df: pd.DataFrame, subject_id_column: str = 'src_subject_id') -> pd.DataFrame:
    """
    Preprocesses a single modality DataFrame:
      - Converts datetime columns to numeric timestamps
      - Ordinal encodes variables based on LABEL_SPECS
      - One-hot encodes remaining categorical (object) columns
      - Returns a fully numeric DataFrame including the subject identifier (if present)
    
    Parameters:
      df (pd.DataFrame): Input DataFrame.
      subject_id_column (str): Column name for the subject identifier.
    
    Returns:
      pd.DataFrame: Preprocessed DataFrame ready for model input.
    """
    df = df.copy()
    
    # Separate subject ID column
    if subject_id_column in df.columns:
        subject_ids = df[[subject_id_column]]
        df_data = df.drop(columns=[subject_id_column])
    else:
        subject_ids = None
        df_data = df.copy()
    
    # Convert datetime columns to numeric timestamps
    datetime_cols = df_data.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()
    for col in datetime_cols:
        df_data[col] = pd.to_datetime(df_data[col], errors='coerce')
        df_data[col] = df_data[col].apply(lambda x: x.value if pd.notnull(x) else np.nan)

    # ----------------------------------------
    # Ordinal encode variables based on LABEL_SPECS
    ordinal_specs = []
    for spec in LABEL_SPECS:
        # build full mapping of code -> label
        if 'mapping' in spec:
            mapping = spec['mapping']
        elif spec.get('fill_middle'):
            first, last = spec['first'], spec['last']
            mapping = {first: spec['first_label'], last: spec['last_label']}
            for code in range(first+1, last):
                mapping[code] = str(code)
        else:
            continue
        # reverse mapping: label -> code
        rev_map = {label: code for code, label in mapping.items()}
        label_set = set(mapping.values())
        ordinal_specs.append((label_set, rev_map))

    # identify all object columns
    object_cols = df_data.select_dtypes(include=['object']).columns.tolist()
    ordinal_cols = []

    # apply ordinal encoding for matching specs
    for col in object_cols:
        unique_vals = set(df_data[col].dropna().unique())
        for label_set, rev_map in ordinal_specs:
            if unique_vals.issubset(label_set):
                df_data[col] = df_data[col].map(lambda x: rev_map.get(x, np.nan))
                ordinal_cols.append(col)
                break

    # one-hot encode the remaining object columns
    remaining_obj = [c for c in object_cols if c not in ordinal_cols]
    if remaining_obj:
        df_data = pd.get_dummies(df_data, columns=remaining_obj, drop_first=True, dummy_na=True)
    # ----------------------------------------
    
    # Convert boolean columns to integers (True -> 1, False -> 0)
    bool_cols = df_data.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        df_data[col] = df_data[col].astype(int)
    
    # Reattach the subject ID column
    if subject_ids is not None:
        df_processed = pd.concat([subject_ids, df_data], axis=1)
    else:
        df_processed = df_data

    return df_processed



def convert_df_for_vae(df: pd.DataFrame, subject_id_column: str = 'src_subject_id'):
    """
    Converts a preprocessed DataFrame into VAE-compatible format.

    Parameters:
      df (pd.DataFrame): Preprocessed input DataFrame with numeric values.
      subject_id_column (str): Column name for the subject identifier.

    Returns:
      subject_ids (np.ndarray): Array of subject identifiers.
      numeric_array (np.ndarray): 2D NumPy array (n_samples, n_features) for VAE training.
    """
    df = df.copy()
    
    # Separate subject IDs if present
    if subject_id_column in df.columns:
        subject_ids = df[subject_id_column].values
        df.drop(columns=[subject_id_column], inplace=True)
    else:
        subject_ids = None
    
    numeric_array = df.values.astype(np.float32)
    
    return subject_ids, numeric_array



def convert_data_for_vae(modalities_data: dict, subject_id_column: str = 'src_subject_id') -> dict:
    """
    Convert each modality DataFrame in a dictionary to a numeric format for VAE training.
    
    Parameters:
      modalities_data (dict): Dictionary where keys are modality names and values are DataFrames.
      subject_id_column (str): Column name for the subject identifier.
    
    Returns:
      converted_data (dict): Dictionary where each key is a modality and each value is a tuple:
                             (subject_ids, numeric_array)
    """
    converted_data = {}
    for modality, df in modalities_data.items():
        subject_ids, numeric_array = convert_df_for_vae(df, subject_id_column)
        converted_data[modality] = (subject_ids, numeric_array)
    return converted_data





##############################################
# PCA for each modality
##############################################

def compute_PCA(df, n_components=None):
        """
        PCA on one individual dataframe.
        """
       # Drop 'src_subject_id' and 'interview_date' if they are in the dataframe
        columns_to_exclude = ['src_subject_id', 'interview_date']
        df_clean = df.drop(columns=[col for col in columns_to_exclude if col in df.columns])
        
        # Optionally, keep only numeric columns if the dataframe contains non-numeric data
        df_numeric = df_clean.select_dtypes(include=['number'])
        
        # Initialize and fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(df_numeric)
        
        return pca


def run_pca_on_modalities(data_dict, n_components=None):
    """
    Runs PCA for each modality in a dictionary of dataframes.
    
    Parameters:
    - data_dict (dict): Dictionary where keys are modality names and values are DataFrames.
    - n_components (int or None): Number of components to keep. If None, all components are kept.
    
    Returns:
    - pca_results (dict): Dictionary with modality names as keys and PCA results as values.
                           Each value is a dict containing:
                           - 'pca': the fitted PCA object,
                           - 'explained_variance_ratio': explained variance ratio of each component,
                           - 'components': the principal axes in feature space.
    """
    pca_results = {}
    for modality, df in data_dict.items():
        pca = compute_PCA(df, n_components=None)

        # Store the results in the dictionary
        pca_results[modality] = {
            'pca': pca,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'components': pca.components_
        }
        
    return pca_results

# Example usage:
# modalities = {
#     'modality1': df1,
#     'modality2': df2,
#     # ...
# }
# pca_results = run_pca_on_modalities(modalities, n_components=5)
# print(pca_results['modality1']['explained_variance_ratio'])




##################### PLOTS #####################

def plot_latent_feature_crosscorr(
    latent_variables: np.ndarray,
    original_data: np.ndarray,
    vmin,
    vmax,
    feature_names=None
):
    """
    Creates a figure with:
      - One row per latent variable, heatmap sorted by correlation strength.
      - Features correctly mapped (x-axis values always match features).
      - A unified colorbar to indicate correlation.
      - A legend listing the actual feature names, **not reordered**.
      - Adjusted layout to prevent overlapping elements.

    Parameters
    ----------
    latent_variables : np.ndarray
        Shape (n_samples, latent_dim).
    original_data : np.ndarray
        Shape (n_samples, n_features).
    feature_names : list of str, optional
        Names for the original features. Defaults to ["Feature_1", "Feature_2", ...].
    """

    # Ensure arrays
    latent_variables = np.asarray(latent_variables)
    original_data = np.asarray(original_data)

    n_samples, latent_dim = latent_variables.shape
    _, feat_dim = original_data.shape

    # Default feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{j+1}" for j in range(feat_dim)]

    # Compute correlation matrix
    combined = np.hstack((latent_variables, original_data))
    corr_matrix = np.corrcoef(combined, rowvar=False)
    cross_corr = corr_matrix[:latent_dim, latent_dim:]

    # ============= Figure Setup =============
    fig_height = 4 * latent_dim  # Increased height per latent dimension
    fig_width = max(12, min(40, 1.2 * feat_dim))  # Adjust width dynamically based on features
    fig, axes = plt.subplots(
        nrows=latent_dim, ncols=1,
        figsize=(fig_width, fig_height)
    )
    axes = np.atleast_1d(axes)

    cmap = "coolwarm"

    for i in range(latent_dim):
        ax = axes[i]
        
        # Sort features **per latent variable**
        sorted_indices = np.argsort(-cross_corr[i])  # Sort descending
        sorted_corrs = cross_corr[i, sorted_indices]
        sorted_feature_names = [feature_names[j] for j in sorted_indices]

        # Create heatmap
        sns.heatmap(
            sorted_corrs.reshape(1, -1),
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0,
            linewidths=0.5,
            annot=False,
            cbar=False,
            xticklabels=sorted_feature_names,  # Correct feature names
            yticklabels=[f"Latent_{i+1}"]
        )

        ax.set_title(f"Latent_{i+1} vs. Features", fontsize=10, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

    # Adjust spacing between subplots
    plt.subplots_adjust(left=0.05, right=0.8, top=0.95, bottom=0.1, hspace=1.2)  # More space

    # ============= Single Colorbar =============
    from matplotlib.colors import Normalize
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])

    # Adjust colorbar placement to avoid overlap
    cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.7])  # Shifted right, avoid overlap
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Correlation", fontsize=10)

    # Improve layout
    plt.tight_layout(rect=[0, 0, 0.82, 1])  # Reserve space for colorbar

    plt.show()






def top10_features_per_latent(
    latent_variables: np.ndarray,
    original_data: np.ndarray,
    feature_names=None
):
    """
    Parameters
    ----------
    latent_variables : np.ndarray
        Shape (n_samples, latent_dim).
    original_data : np.ndarray
        Shape (n_samples, n_features).
    feature_names : list of str, optional
        Names for the original features. Defaults to ["Feature_0", "Feature_1", ...].
    """

    # Convert inputs to arrays
    latent_variables = np.asarray(latent_variables)
    original_data = np.asarray(original_data)

    # Figure out dimensions
    n_samples, latent_dim = latent_variables.shape
    _, feat_dim = original_data.shape

    # If no feature names provided, generate a list of placeholder names
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(feat_dim)]
    else:
        # Optionally, ensure correct length
        if len(feature_names) != feat_dim:
            raise ValueError("feature_names must match the number of columns in original_data.")

    # Compute correlation matrix (currently using Pearson)
    combined = np.hstack((latent_variables, original_data))
    corr_matrix = np.corrcoef(combined, rowvar=False)
    
    # Slice the correlation matrix to get cross-correlations
    # cross_corr shape: (latent_dim, feat_dim)
    cross_corr = corr_matrix[:latent_dim, latent_dim:]

    # ============= Print matrix info =============
    print("Max correlation per latent variable:", np.max(cross_corr, axis=1))
    print("Min correlation per latent variable:", np.min(cross_corr, axis=1))
    print("Mean correlation per latent variable:", np.mean(cross_corr, axis=1))

    # ============= Table =============
    # Create a list to store results
    top_correlation_rows = []

    for latent_idx in range(latent_dim):
        correlations = cross_corr[latent_idx, :]  # Correlations for a single latent variable

        # Sort by correlation values
        sorted_indices = np.argsort(correlations)
        top_neg_indices = sorted_indices[:10]    # 10 smallest (most negative)
        top_pos_indices = sorted_indices[-10:]   # 10 largest (most positive)

        # Build lists of (feature_name, correlation)
        # Reverse top_pos_indices so the highest correlation is first
        top_pos_list = [(feature_names[idx], float(correlations[idx])) for idx in reversed(top_pos_indices)]
        top_neg_list = [(feature_names[idx], float(correlations[idx])) for idx in top_neg_indices]

        top_correlation_rows.append({
            "Latent Variable": latent_idx,
            "Top Positive Correlations": top_pos_list,
            "Top Negative Correlations": top_neg_list,
        })

    # Convert results to DataFrame
    top_correlation_df = pd.DataFrame(top_correlation_rows)

    # Print nicely using tabulate
    print(tabulate(top_correlation_df, headers='keys', tablefmt='psql'))

    return cross_corr, top_correlation_df



###################################
# Plot reconstructed data
###################################

def plot_recon(VAE_results, original_data):
    """
    Plots the reconstructed data against the original data for a specific modality.
    
    Parameters:
        results_VAE (dict): Dictionary containing VAE results, including 'recon_data'.
        final_data (pd.DataFrame): DataFrame containing the original data.
    """
    # Extract the reconstructed data and original data

    recon_batch = VAE_results['recon_data']
    x = original_data.drop(columns=['src_subject_id'])


    # If 'x' is your original data and is a DataFrame, convert it to a NumPy array of floats.
    if isinstance(x, pd.DataFrame):
        x_np = x.to_numpy(dtype=float)
    else:
        x_np = np.asarray(x, dtype=float)

    # For 'recon_batch', if it's a torch tensor, convert it to a NumPy array.
    if isinstance(recon_batch, torch.Tensor):
        recon_np = recon_batch.detach().cpu().numpy()
    else:
        recon_np = np.asarray(recon_batch, dtype=float)

    # Flatten the arrays
    x_flat = x_np.flatten()
    recon_flat = recon_np.flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(x_flat, recon_flat, alpha=0.5)
    plt.xlabel("Original Data")
    plt.ylabel("Reconstructed Data")
    plt.title("Original vs. Reconstructed Data")

    # Plot a diagonal (y = x) line as a reference for perfect reconstruction
    min_val = min(x_flat.min(), recon_flat.min())
    max_val = max(x_flat.max(), recon_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    plt.show()


###################################
# Plot latent variable variance
###################################

def get_latent_means(latent_output):
    """
    Extracts the latent means from the output of a VAE's encoder.
    
    The latent_output can be in one of the following forms:
      - A tuple of (latent_means, latent_logvar)
      - A dictionary with key 'mean'
      - Directly the latent means as a NumPy array or tensor
      
    Parameters:
    - latent_output: The output from the VAE encoder.
    
    Returns:
    - A NumPy array containing the latent means.
    """
    # If the output is a tuple, assume the first element is the latent means
    if isinstance(latent_output, tuple):
        latent_means = latent_output[0]
    # If it's a dictionary, extract the 'mean' key if available
    elif isinstance(latent_output, dict) and 'mean' in latent_output:
        latent_means = latent_output['mean']
    else:
        # Otherwise, assume it's directly the latent means
        latent_means = latent_output

    # Convert to a NumPy array if the latent means are in tensor form
    if hasattr(latent_means, 'numpy'):
        latent_means = latent_means.numpy()

    return np.array(latent_means)

def plot_latent_variance(latent_means):
    """
    Creates a scree plot for the latent variables of a VAE.
    
    Parameters:
    - latent_means: A NumPy array of shape (num_samples, latent_dim) containing the latent means
      for all samples.
    """
    # Compute variance for each latent dimension (axis=0: across samples)
    variances = np.var(latent_means, axis=0)
    
    # Normalize the variances to get a variance ratio similar to PCA explained variance
    explained_variance_ratio = variances / np.sum(variances)
    
    # Compute cumulative variance for visualization
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    latent_dim = len(explained_variance_ratio)
    plt.figure(figsize=(8, 5))
    
    # Bar plot for individual variance ratios
    plt.bar(range(1, latent_dim + 1), explained_variance_ratio, alpha=0.6, label='Individual Variance Ratio')
    
    # Line plot for cumulative variance
    plt.plot(range(1, latent_dim + 1), cumulative_variance, marker='o', color='r', label='Cumulative Variance')
    
    plt.xlabel('Latent Variable Index')
    plt.ylabel('Variance Ratio')
    plt.title('Scree Plot for VAE Latent Variables')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming your VAE encoder returns a latent mean vector for each sample.
# For instance, if you have a function get_latent_means() that returns a NumPy array:
# latent_means = get_latent_means(your_dataset)
# plot_latent_variance(latent_means)


   

###################################
# Plot variance PCA
###################################
def plot_scree(pca):
    """
    Plots a scree plot showing the explained variance ratio for each principal component.
    """
    num_components = len(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    
    # Bar plot for each component's explained variance
    plt.bar(range(1, num_components + 1), pca.explained_variance_ratio_, 
            alpha=0.6, color='b', label='Individual Explained Variance')
    
    # Line plot for cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, num_components + 1), cumulative_variance, 
             marker='o', color='r', label='Cumulative Explained Variance')
    
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Example usage:
# Assume pca is a fitted PCA object (e.g., from compute_pca(df, n_components=...))
# plot_scree(pca)

###################################
# Biplot PCA
###################################

def biplot(pca, df_numeric, feature_names=None):
    """
    Creates a biplot for the first two principal components.
    
    Parameters:
    - pca: A fitted PCA object.
    - df_numeric: The numeric dataframe used for PCA.
    - feature_names: Optional list of feature names. If None, the column names of df_numeric are used.
    """

    columns_to_exclude = ['src_subject_id', 'interview_date']
    df_numeric = df_numeric.drop(columns=[col for col in columns_to_exclude if col in df_numeric.columns])
        

    # Project the data onto the first two principal components
    scores = pca.transform(df_numeric)
    
    if feature_names is None:
        feature_names = df_numeric.columns
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot of the projected data (scores)
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.6, edgecolor='k')
    
    # Scale factor for the arrows to make them visible
    arrow_scale = 3.0
    
    # Plot the loadings as arrows
    for i, (comp1, comp2) in enumerate(zip(pca.components_[0], pca.components_[1])):
        plt.arrow(0, 0, comp1 * arrow_scale, comp2 * arrow_scale, 
                  color='r', width=0.005, head_width=0.1)
        plt.text(comp1 * arrow_scale * 1.15, comp2 * arrow_scale * 1.15, 
                 feature_names[i], color='r', ha='center', va='center')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Biplot')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming df_numeric is the numeric part of your dataframe used for PCA:
# biplot(pca, df_numeric)





###################################
# Differences in latent var between groups
###################################

def plot_cluster_latent(df):
    """
    Plot the mean latent variable values for each cluster in a heatmap format. Input is the dict 
    """

    # Compute mean latent variable values per cluster
    means = df.groupby('Cluster').mean()

    # Plot heatmap
    plt.figure(figsize=(12, 6))  # Make it wider to fit labels
    plt.imshow(means, aspect="auto", cmap="coolwarm")  # Change 'coolwarm' if needed
    plt.colorbar(label="Latent Variable Value")
    plt.title("Cluster Means - Latent Variables", fontsize=14)

    # Label axes
    plt.xticks(np.arange(len(means.columns)), means.columns, rotation=45, ha="right", fontsize=10)
    plt.yticks(np.arange(len(means.index)), [f"Cluster {c}" for c in means.index], fontsize=10)

    plt.xlabel("Latent Variables")
    plt.ylabel("Clusters")
    plt.show()





###################################
# Plot parallel coordinates
###################################
def plot_parallel_coordinates(df):

    for cluster_id in sorted(df["Cluster"].unique()):
        subset = df[df["Cluster"] == cluster_id]
        plt.plot(subset.drop(columns=["Cluster"]).T, alpha=0.3)  # Transpose to align variables on x-axis

    plt.xticks(range(len(df.columns) - 1), df.columns[:-1], rotation=45)
    plt.title("Parallel Coordinates - Latent Variables by Cluster, Clinical")
    plt.xlabel("Latent Variables")
    plt.ylabel("Value")
    plt.show()





###################################
# Plot heatmap of means of original variables per cluster
###################################

def plot_cluster_original(df):
    # Compute means per cluster
    means = df.groupby('Cluster').mean()

    # Create the figure with a larger size
    plt.figure(figsize=(40, 6))  # Wider figure to fit labels

    # Show heatmap
    plt.imshow(means, aspect='auto', cmap="coolwarm")  # 'coolwarm' adds better contrast
    plt.colorbar(label="Feature Value")  # Add a label for color meaning
    plt.title("Cluster Means Heatmap", fontsize=14)

    # Adjust X-axis
    plt.xticks(np.arange(len(means.columns)), means.columns, rotation=45, ha="right", fontsize=10)
    plt.xlabel("Features")

    # Adjust Y-axis
    plt.yticks(np.arange(len(means.index)), [f"Cluster {c}" for c in means.index], fontsize=10)
    plt.ylabel("Clusters")

    plt.grid(False)  # Remove grid lines for better readability
    plt.show()

def preprocessing(df,
                  meta,
                  subject_id_column='src_subject_id',
                  col_threshold=0.5, row_threshold=0.5,
                  skew_threshold=0.75,
                  scaler_type='robust',
                  modalities=['Internalising', 'Functioning', 'Cognition', 'Detachment', 'Psychoticism'],
                  impute_parea=False):
   """
    Preprocess the data by removing high missing data, applying power transformation, dummy coding, scaling, and imputing.
    
    Parameters:
    - df: DataFrame containing the data.
    - meta: Metadata DataFrame.
    - subject_id_column: Column name for subject IDs.
    - col_threshold: Threshold for column missing data.
    - row_threshold: Threshold for row missing data.
    - skew_threshold: Threshold for skewness in data.
    - scaler_type: Type of scaler to use ('robust' or 'standard').
    
    Returns:
    - dict_final: Dictionary containing processed modalities.
    """
   
   
   df_transformed = auto_power_transform(df, skew_threshold = skew_threshold) # Apply power transformation when data is skewed. 


   df_dummy = dummy_code(df_transformed)

   df_scaled = scale_diverse_data(df_dummy, subject_id_column=subject_id_column, scaler_type = scaler_type) # Scale data with robust scaler
   modal_dict = extract_modalities(meta, df_scaled) # Split data into modalities based on meta
   modal_dict_clean = {modality: modal_dict[modality] for modality in modalities if modality in modal_dict}

   # Reattach subject ID column to each modality so downstream steps include it
   for mod in modal_dict_clean:
       modal_dict_clean[mod][subject_id_column] = df_scaled[subject_id_column].loc[modal_dict_clean[mod].index]
   
   if impute_parea is False:
        # Identify any subjects who are entirely missing in any modality, then drop them from all modalities
        subjects_to_drop = set()
        for modality, df_mod in modal_dict_clean.items():
            # Exclude subject ID column when checking
            data_only = df_mod.drop(columns=[subject_id_column]) if subject_id_column in df_mod.columns else df_mod
            # Mask rows where all data columns are NaN
            missing_mask = data_only.isna().all(axis=1)
            if missing_mask.any():
                # Collect subject IDs to drop
                missing_ids = df_mod.loc[missing_mask, subject_id_column]
                subjects_to_drop.update(missing_ids.tolist())
        if subjects_to_drop:
            warnings.warn(
                f"Dropping {len(subjects_to_drop)} participants missing a full modality across all views: {subjects_to_drop}"
            )
            # Remove those subjects from every modality
            for modality in modal_dict_clean:
                df2 = modal_dict_clean[modality]
                modal_dict_clean[modality] = (
                    df2[~df2[subject_id_column].isin(subjects_to_drop)]
                    .reset_index(drop=True)
                )

   dict_imputed = impute_data(modal_dict_clean, subject_id_column=subject_id_column) # Impute missing data in each modality using KNN imputation
   dict_final = scale_data(dict_imputed, subject_id_column=subject_id_column, scaler_type=scaler_type) # Scale data in each modality with robust scaler
   
   # --- Canonical reindex across modalities ---
   id_col = subject_id_column
   mods = [m for m in modalities if m in dict_final]
   id_lists = {m: dict_final[m][id_col].tolist() for m in mods}
   # Sanity check: all views must have identical ID order after imputation and scaling
   # subjects present in all views
   shared = set.intersection(*(set(v) for v in id_lists.values()))
   # preserve order from the first modality in `modalities`
   canonical = [sid for sid in id_lists[mods[0]] if sid in shared]
   
   for m in mods:
       dfm = dict_final[m]
       dict_final[m] = (
           dfm[dfm[id_col].isin(shared)]
           .set_index(id_col)
           .loc[canonical]
           .reset_index()
       )

   # sanity: all views must now have identical ID order
   for m in mods[1:]:
    assert dict_final[m][id_col].tolist() == dict_final[mods[0]][id_col].tolist(), \
        f"Subject-ID order mismatch between {mods[0]} and {m}"

   
   # Build subject-ID list for each modality after imputation and scaling
   subject_id_list = []
   for mod in modalities:
       if mod in dict_final and subject_id_column in dict_final[mod]:
           subject_id_list.append(dict_final[mod][subject_id_column].tolist())
       else:
           subject_id_list.append([])
   ae_data = convert_data_for_vae(dict_final, subject_id_column=subject_id_column)

   return ae_data, subject_id_list, dict_final
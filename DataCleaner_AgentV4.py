import os
import sys
import pandas as pd
import numpy as np
import logging
import openai
import json
import re
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Helper functions for colored output
def print_system(message: str):
    print(Fore.BLUE + message)

def print_user(message: str):
    print(Fore.GREEN + message)

def print_gpt(message: str):
    print(Fore.CYAN + message)

def print_chat(message: str):
    print(Fore.YELLOW + message)

def user_input(prompt: str) -> str:
    return input(Fore.RED + prompt + Style.RESET_ALL)

# Set up logging with UTF-8 encoding
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(f'data_cleaning_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

@dataclass
class ColumnAnalysis:
    column_name: str
    data_type: str
    null_count: int
    unique_count: int
    sample_values: List
    issues: List[str]
    recommendations: List[str]

class DataCleaningError(Exception):
    pass

def clean_numeric_column(column_data: pd.Series, outlier_action: str, missing_action: str) -> pd.Series:
    col = column_data.copy()
    orig_missing = col.isnull()
    lower_threshold = col.quantile(0.05)
    upper_threshold = col.quantile(0.95)
    if outlier_action == "remove_outliers_fill_NA":
        col = col.mask((col <= lower_threshold) | (col >= upper_threshold))
        col = col.astype(object)
        mask_outlier = ((column_data <= lower_threshold) | (column_data >= upper_threshold)) & (~orig_missing)
        col.loc[mask_outlier] = "NA"
    elif outlier_action == "remove_outliers_set_missing":
        col = col.mask((col <= lower_threshold) | (col >= upper_threshold))
    elif outlier_action == "remove_outliers_impute_mean":
        valid = col[(col > lower_threshold) & (col < upper_threshold)]
        impute_val = valid.mean() if not valid.empty else np.nan
        col = col.mask((col <= lower_threshold) | (col >= upper_threshold), impute_val)
    elif outlier_action == "remove_outliers_impute_median":
        valid = col[(col > lower_threshold) & (col < upper_threshold)]
        impute_val = valid.median() if not valid.empty else np.nan
        col = col.mask((col <= lower_threshold) | (col >= upper_threshold), impute_val)
    elif outlier_action == "remove_outliers_impute_mode":
        valid = col[(col > lower_threshold) & (col < upper_threshold)]
        impute_val = valid.mode().iloc[0] if not valid.mode().empty else np.nan
        col = col.mask((col <= lower_threshold) | (col >= upper_threshold), impute_val)
    if missing_action != "none":
        if missing_action == "fill_NA":
            fill_val = "NA"
            col = col.astype(object)
        elif missing_action == "impute_mean":
            fill_val = col.mean(skipna=True)
        elif missing_action == "impute_median":
            fill_val = col.median(skipna=True)
        elif missing_action.startswith("custom:"):
            fill_val = missing_action.split("custom:")[1]
            try:
                fill_val = float(fill_val)
            except Exception:
                pass
        elif missing_action == "drop_rows":
            col = col.loc[~orig_missing]
            return col
        else:
            fill_val = np.nan
        col.loc[orig_missing] = col.loc[orig_missing].fillna(fill_val)
    return col

# ---------------- Base Agent and API ----------------

class BaseAgent:
    def __init__(self, model_name: str = "o3-mini"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model_name = model_name

    def _call_openai(self, prompt: str) -> str:
        try:
            logger.info(f"Sending prompt to OpenAI:\n{prompt}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            response_content = response.choices[0].message.content.strip()
            logger.info(f"Received response from OpenAI:\n{response_content}")
            return response_content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise DataCleaningError(f"AI model error: {str(e)}")

# ---------------- Data Loader ----------------

class DataLoaderAgent(BaseAgent):
    def run(self, file_path: str) -> pd.DataFrame:
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            loaders = {
                '.dta': pd.read_stata,
                '.csv': pd.read_csv,
                '.xlsx': pd.read_excel,
                '.parquet': pd.read_parquet
            }
            if file_extension not in loaders:
                raise ValueError(f"Unsupported file format: {file_extension}")
            df = loaders[file_extension](file_path)
            logger.info(f"Successfully loaded {file_path} with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise DataCleaningError(f"Data loading error: {str(e)}")

# ---------------- Data Analyzer ----------------

class DataAnalyzerAgent(BaseAgent):
    def _analyze_numeric_column(self, column_data: pd.Series) -> Dict:
        mean_val = column_data.mean()
        median_val = column_data.median()
        std_val = column_data.std()
        min_val = column_data.min()
        max_val = column_data.max()
        range_val = max_val - min_val
        lower_threshold = column_data.quantile(0.05)
        upper_threshold = column_data.quantile(0.95)
        outlier_count = int(((column_data <= lower_threshold) | (column_data >= upper_threshold)).sum())
        return {
            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "range": range_val,
            "lower_threshold": lower_threshold,
            "upper_threshold": upper_threshold,
            "outliers": outlier_count
        }

    def _analyze_categorical_column(self, column_data: pd.Series) -> Dict:
        return {
            "value_counts": column_data.value_counts().to_dict(),
            "category_count": int(column_data.nunique()),
            "most_common": column_data.mode().iloc[0] if not column_data.empty else None
        }

    def run(self, column_data: pd.Series) -> ColumnAnalysis:
        try:
            non_null = column_data.dropna()
            sample = non_null.sample(min(5, len(non_null))).tolist() if not non_null.empty else []
            null_count = int(column_data.isnull().sum())
            unique_count = int(column_data.nunique())
            total_available = int(column_data.count())
            try:
                numeric_data = pd.to_numeric(column_data, errors='coerce')
                if numeric_data.notnull().sum() >= 0.8 * len(column_data):
                    specific_analysis = self._analyze_numeric_column(numeric_data)
                else:
                    specific_analysis = self._analyze_categorical_column(column_data)
            except Exception:
                specific_analysis = self._analyze_categorical_column(column_data)

            prompt = f"""
Analyze this data column and provide detailed cleaning recommendations:
Column name: {column_data.name}
Sample values: {sample}
Null count: {null_count}
Unique values: {unique_count}
Total available count: {total_available}
Data type: {column_data.dtype}
Additional analysis: {specific_analysis}
"""
            prompt += """
For continuous columns, include:
- Mean, Median, Standard Deviation
- Minimum, Maximum, and Range
- Number of outliers as determined by the 5th and 95th percentile thresholds

Format your response as:
ISSUES:
- [List each issue]
RECOMMENDATIONS:
- [List each cleaning step]
"""
            analysis_response = self._call_openai(prompt)
            issues = []
            recommendations = []
            current_section = None
            for line in analysis_response.split('\n'):
                if 'ISSUES:' in line:
                    current_section = 'issues'
                elif 'RECOMMENDATIONS:' in line:
                    current_section = 'recommendations'
                elif line.strip().startswith('-'):
                    if current_section == 'issues':
                        issues.append(line.strip()[2:].strip())
                    elif current_section == 'recommendations':
                        recommendations.append(line.strip()[2:].strip())
            return ColumnAnalysis(
                column_name=column_data.name,
                data_type=str(column_data.dtype),
                null_count=null_count,
                unique_count=unique_count,
                sample_values=sample,
                issues=issues,
                recommendations=recommendations
            )
        except Exception as e:
            logger.error(f"Error analyzing column {column_data.name}: {str(e)}")
            raise DataCleaningError(f"Analysis error: {str(e)}")

# ---------------- Data Cleaner ----------------

class DataCleanerAgent(BaseAgent):
    def _validate_cleaned_data(self, original: pd.Series, cleaned: pd.Series) -> bool:
        if len(original) != len(cleaned):
            raise DataCleaningError("Cleaning altered number of rows.")
        return True

    def _get_corrected_code(self, code: str, error_message: str) -> str:
        prompt = f"""
The following Python code was generated to clean a pandas Series named `column_data`:
{code}

It resulted in the following error when executed:
{error_message}

Please provide corrected Python code that fixes the error. The code should:
1. Handle missing values and outliers based on previous recommendations.
2. Use only pandas and numpy operations.
3. Preserve the original data type when possible.
4. Assume the input is a pandas Series named `column_data` and return the cleaned `column_data`.
Return only executable Python code.
"""
        corrected_code = self._call_openai(prompt)
        corrected_code = corrected_code.replace("```python", "").replace("```", "").strip()
        logger.info(f"Corrected cleaning code obtained:\n{corrected_code}")
        return corrected_code

    def _safe_exec(self, code: str, local_dict: dict) -> pd.Series:
        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            try:
                logger.info(f"Executing cleaning code (attempt {attempt+1}):\n{code}")
                if "def clean_" in code:
                    exec(code, {"pd": pd, "np": np}, local_dict)
                    for key, value in local_dict.items():
                        if callable(value) and key.startswith("clean_"):
                            cleaned = value(local_dict["column_data"])
                            local_dict["column_data"] = cleaned
                            return cleaned
                    raise DataCleaningError("No cleaning function found in the provided code.")
                else:
                    indented_code = "\n".join("        " + line if line.strip() else "" for line in code.splitlines())
                    wrapped_code = (
                        "def clean_data(column_data):\n"
                        "    try:\n" +
                        indented_code + "\n"
                        "        return column_data\n"
                        "    except Exception as e:\n"
                        "        raise Exception('Error in cleaning code: ' + str(e))\n\n"
                        "column_data = clean_data(column_data)"
                    )
                    exec(wrapped_code, {"pd": pd, "np": np}, local_dict)
                    return local_dict['column_data']
            except Exception as e:
                logger.error(f"Error executing cleaning code on attempt {attempt+1}: {str(e)}")
                attempt += 1
                try:
                    code = self._get_corrected_code(code, str(e))
                except Exception as retry_e:
                    logger.error(f"Error obtaining corrected code on attempt {attempt}: {str(retry_e)}")
                    if attempt >= max_attempts:
                        raise DataCleaningError(f"Code execution error after {attempt} attempts: {str(retry_e)}")
                continue
        raise DataCleaningError("Failed to execute cleaning code after maximum attempts.")

    def run(self, column_data: pd.Series, analysis: ColumnAnalysis, decisions: Dict) -> pd.Series:
        cleaning_prompt = f"""
Generate Python code to clean this data column by performing two independent steps:
Step 1: Outlier Handling
    - Compute lower_threshold = column_data.quantile(0.05) and upper_threshold = column_data.quantile(0.95).
    - If outlier_action is 'remove_outliers_fill_NA', set values <= lower_threshold or >= upper_threshold to NA (literal "NA").
    - If outlier_action is 'remove_outliers_set_missing', set values <= lower_threshold or >= upper_threshold to missing (np.nan).
    - If outlier_action is 'remove_outliers_impute_mean', replace outliers with the mean of values within (lower_threshold, upper_threshold).
    - If outlier_action is 'remove_outliers_impute_median', replace outliers with the median of values within (lower_threshold, upper_threshold).
    - If outlier_action is 'remove_outliers_impute_mode', replace outliers with the mode of values within (lower_threshold, upper_threshold).
    - If outlier_action is 'none', do nothing.
Step 2: Missing Value Handling
    - Identify cells that were originally missing.
    - If missing_action is 'fill_NA', fill those cells with the literal string "NA".
    - If missing_action is 'fill_blank', fill those cells with an empty string.
    - If missing_action is 'impute_mean', fill those cells with the mean of the column.
    - If missing_action is 'impute_median', fill those cells with the median of the column.
    - If missing_action starts with 'custom:', extract the custom value and fill those cells with it.
    - If missing_action is 'drop_rows', drop rows with originally missing cells.
    - If missing_action is 'none', leave originally missing cells unchanged.
User decisions:
    Outlier action: {decisions.get('outlier_action', 'none')}
    Missing value action: {decisions.get('missing_action', 'none')}
Additional instructions:
    - Use only pandas and numpy operations.
    - Preserve the original data type when possible.
    - Apply Step 1 and then Step 2 in sequence.
Assume that the input is a pandas Series named `column_data` and return the cleaned `column_data`.
Return only executable Python code.
"""
        cleaning_code = self._call_openai(cleaning_prompt)
        cleaning_code = cleaning_code.replace("```python", "").replace("```", "").strip()
        local_dict = {"column_data": column_data.copy()}
        cleaned_data = self._safe_exec(cleaning_code, local_dict)
        if self._validate_cleaned_data(column_data, cleaned_data):
            logger.info(f"Successfully cleaned column {column_data.name}")
            return cleaned_data
        return column_data

# ---------------- Categorical Mapping Agent ----------------

class CategoricalValueFixAgent(BaseAgent):
    def consolidate_mapping(self, mapping: Dict[str, str]) -> Dict[str, str]:
        groups = {}
        for orig, norm in mapping.items():
            key = orig.lower()
            groups.setdefault(key, []).append((orig, norm))
        new_mapping = {}
        for key, pairs in groups.items():
            norm_values = set(norm for orig, norm in pairs)
            if len(norm_values) == 1:
                rep = norm_values.pop()
            else:
                originals = ", ".join(orig for orig, norm in pairs)
                print_chat(f"\nFor the following values (differing only by capitalization or slight variation): {originals}")
                rep = user_input("Enter the representative normalized value to use: ").strip()
                if not rep:
                    rep = pairs[0][1]
            for orig, _ in pairs:
                new_mapping[orig] = rep
        return new_mapping

    def parse_custom_mapping(self, mapping_str: str) -> Dict[str, str]:
        custom_mapping = {}
        pairs = re.split(r";|\n", mapping_str)
        for pair in pairs:
            pair = pair.strip()
            if not pair:
                continue
            if ":" in pair:
                k, v = pair.split(":", 1)
            elif "=" in pair:
                k, v = pair.split("=", 1)
            else:
                continue
            custom_mapping[k.strip()] = v.strip()
        return custom_mapping

    def run(self, column_data: pd.Series) -> Union[pd.Series, Tuple[pd.Series, str]]:
        if isinstance(column_data.dtype, pd.CategoricalDtype):
            if "NA" not in column_data.cat.categories:
                column_data = column_data.cat.add_categories(["NA"])
        if column_data.isnull().sum() > 0:
            column_data = column_data.fillna("NA")
        if not (isinstance(column_data.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(column_data)):
            return column_data
        unique_values = list(column_data.unique())
        prompt = (
            f"These are the unique values from a data column: {unique_values}\n"
            "Using common sense, produce a JSON object mapping each original value to a normalized label. "
            "For example, map 'F' or 'f' to 'Female' and 'M' or 'm' to 'Male' if applicable, but do not group distinct values that should remain separate. "
            "Return only the JSON object."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            mapping_str = response.choices[0].message.content.strip()
            mapping = json.loads(mapping_str)
        except Exception as e:
            logger.error(f"Error obtaining mapping from language model: {str(e)}")
            mapping = {v: v for v in unique_values}
        print_chat(f"\nFor column '{column_data.name}', the suggested mapping is:")
        for orig, norm in mapping.items():
            print_chat(f" - {orig} -> {norm}")
        consolidated = self.consolidate_mapping(mapping)
        print_chat("\nAfter consolidating similar values, the mapping is:")
        for orig, norm in consolidated.items():
            print_chat(f" - {orig} -> {norm}")
        choice = user_input("Do you want to apply this mapping? (Y to apply, N to enter a custom mapping, D to leave unchanged, B to go back to data type conversion): ").strip().lower()
        if choice == 'b':
            return "BACK"
        elif choice in {"y", "yes"}:
            if isinstance(column_data.dtype, pd.CategoricalDtype):
                column_data = column_data.astype(object)
            mapped = column_data.map(consolidated)
            result = mapped.where(~mapped.isna(), column_data)
            return result
        elif choice in {"d", "do nothing"}:
            return column_data
        elif choice in {"n", "no"}:
            new_mapping_str = user_input("Enter a custom mapping (e.g., F:Female; M:Male) or press Enter to leave unchanged: ").strip()
            if new_mapping_str:
                new_mapping = self.parse_custom_mapping(new_mapping_str)
                if new_mapping:
                    final_mapping = consolidated.copy()
                    final_mapping.update(new_mapping)
                    if isinstance(column_data.dtype, pd.CategoricalDtype):
                        column_data = column_data.astype(object)
                    mapped = column_data.map(final_mapping)
                    result = mapped.where(~mapped.isna(), column_data)
                    apply_choice = user_input("Do you want to apply this custom mapping to a new column or update the existing column? (Enter 'N' for new column, 'E' for existing column): ").strip().lower()
                    if apply_choice == 'n':
                        new_col_name = user_input("Enter new column name for the mapped values: ").strip()
                        return (result, new_col_name)
                    else:
                        return result
                else:
                    print_system("No valid mapping found. Leaving column unchanged.")
                    return column_data
            else:
                return column_data
        else:
            new_mapping_str = choice
            if new_mapping_str:
                new_mapping = self.parse_custom_mapping(new_mapping_str)
                if new_mapping:
                    final_mapping = consolidated.copy()
                    final_mapping.update(new_mapping)
                    if isinstance(column_data.dtype, pd.CategoricalDtype):
                        column_data = column_data.astype(object)
                    mapped = column_data.map(final_mapping)
                    result = mapped.where(~mapped.isna(), column_data)
                    apply_choice = user_input("Do you want to apply this custom mapping to a new column or update the existing column? (Enter 'N' for new column, 'E' for existing column): ").strip().lower()
                    if apply_choice == 'n':
                        new_col_name = user_input("Enter new column name for the mapped values: ").strip()
                        return (result, new_col_name)
                    else:
                        return result
                else:
                    print_system("No valid mapping found. Leaving column unchanged.")
                    return column_data
            else:
                return column_data

# ---------------- Decision Agent ----------------

class RestartDecisionProcess(Exception):
    pass

class DataRemarkAgent(BaseAgent):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_confirmed_decision(self, prompt: str, valid_options: List[str]) -> str:
        while True:
            decision = user_input(prompt).strip().lower()
            if decision == 'b':
                raise RestartDecisionProcess()
            if decision not in valid_options:
                print_system("Invalid input. Please try again.")
                continue
            confirm = user_input(f"You entered '{decision}'. Press 'Y' to confirm or 'B' to restart the entire decision process: ").strip().lower()
            if confirm == 'b':
                raise RestartDecisionProcess()
            if confirm in ['y', 'yes']:
                return decision
            else:
                print_system("Please enter 'Y' to confirm or 'B' to restart.")
    
    def run(self, analysis: ColumnAnalysis) -> Dict:
        print_system(f"\n{'='*50}")
        print_system(f"Reviewing Column: {analysis.column_name} (Type: {analysis.data_type})")
        print_system(f"Null count: {analysis.null_count}")
        print_system(f"Unique count: {analysis.unique_count}")
        print_system(f"Sample values: {analysis.sample_values}")
        print_system("\nIssues detected:")
        for issue in analysis.issues:
            print_system(f"- {issue}")
        print_system("\nAutomated Cleaning Recommendations:")
        for rec in analysis.recommendations:
            print_system(f"- {rec}")
        
        decisions = {}
        dtype_lower = analysis.data_type.lower()
        is_numeric = any(x in dtype_lower for x in ["int", "float", "numeric"])
        
        if analysis.null_count > 0:
            if is_numeric:
                prompt_missing = (
                    "\n[Numeric Column] For missing values, choose an action:\n"
                    "1. Fill missing values with NA\n"
                    "2. Impute with mean\n"
                    "3. Impute with median\n"
                    "4. Drop rows with missing values\n"
                    "5. Enter a custom numeric fill value\n"
                    "6. Leave unchanged (keep as missing)\n"
                    "Enter choice (1-6) or 'B' to restart: "
                )
                mapping_missing = {
                    "1": "fill_NA",
                    "2": "impute_mean",
                    "3": "impute_median",
                    "4": "drop_rows",
                    "5": "custom",
                    "6": "none"
                }
            else:
                prompt_missing = (
                    "\n[Categorical Column] For missing values, choose an action:\n"
                    "1. Fill missing values with NA (cells will show 'NA')\n"
                    "2. Fill missing values with blank (cells will be empty)\n"
                    "3. Impute with mode (most frequent value)\n"
                    "4. Drop rows with missing values\n"
                    "5. Enter a custom fill value\n"
                    "6. Leave unchanged\n"
                    "Enter choice (1-6) or 'B' to restart: "
                )
                mapping_missing = {
                    "1": "fill_NA",
                    "2": "fill_blank",
                    "3": "impute_mode",
                    "4": "drop_rows",
                    "5": "custom",
                    "6": "none"
                }
            choice_missing = self.get_confirmed_decision(prompt_missing, ["1", "2", "3", "4", "5", "6"])
            if mapping_missing[choice_missing] == "custom":
                custom_value = user_input("Enter custom fill value: ").strip()
                decisions["missing_action"] = f"custom:{custom_value}"
            else:
                decisions["missing_action"] = mapping_missing[choice_missing]
        else:
            decisions["missing_action"] = "none"
        
        if is_numeric and any("outlier" in issue.lower() for issue in analysis.issues):
            prompt_outlier = (
                "\nFor outlier handling in numeric column, choose an action:\n"
                "1. Set outliers to NA\n"
                "2. Replace outliers with mean\n"
                "3. Replace outliers with median\n"
                "4. Replace outliers with mode\n"
                "5. Do nothing\n"
                "6. Set outliers to missing\n"
                "Enter choice (1-6) or 'B' to restart: "
            )
            mapping_outlier = {
                "1": "remove_outliers_fill_NA",
                "2": "remove_outliers_impute_mean",
                "3": "remove_outliers_impute_median",
                "4": "remove_outliers_impute_mode",
                "5": "none",
                "6": "remove_outliers_set_missing"
            }
            choice_outlier = self.get_confirmed_decision(prompt_outlier, ["1", "2", "3", "4", "5", "6"])
            decisions["outlier_action"] = mapping_outlier[choice_outlier]
        else:
            decisions["outlier_action"] = "none"
        
        print_system(f"\nDecisions for column '{analysis.column_name}': {decisions}")
        final_confirm = user_input("Press 'Y' to confirm these decisions or 'B' to restart the entire decision process: ").strip().lower()
        if final_confirm == 'b':
            raise RestartDecisionProcess()
        while final_confirm not in ['y', 'yes']:
            print_system("Please enter 'Y' to confirm or 'B' to restart.")
            final_confirm = user_input("Press 'Y' to confirm these decisions or 'B' to restart the entire decision process: ").strip().lower()
            if final_confirm == 'b':
                raise RestartDecisionProcess()
        return decisions

# ---------------- Data Cleaning Coordinator with Export Option ----------------

class DataCleaningCoordinator(BaseAgent):
    def __init__(self, model_name: str = "o3-mini"):
        super().__init__(model_name=model_name)
        self.loader = DataLoaderAgent(model_name=model_name)
        self.analyzer = DataAnalyzerAgent(model_name=model_name)
        self.remarker = DataRemarkAgent()
        self.cleaner = DataCleanerAgent(model_name=model_name)
        self.cat_value_fix_agent = CategoricalValueFixAgent()

    def convert_column_dtype(self, col: pd.Series, col_name: str) -> pd.Series:
        print_system(f"\nProcessing column '{col_name}': current dtype: {col.dtype}")
        sample_vals = col.dropna().unique()
        print_system("Sample values: " + (str(sample_vals[:5]) if len(sample_vals) > 0 else "None"))
        print_system("Select intended type for this column:")
        print_system("1. Numeric (float)")
        print_system("2. Integer")
        print_system("3. Categorical")
        print_system("4. Datetime")
        print_system("5. No Change")
        choice = user_input("Enter choice (1-5): ").strip()
        if choice == "1":
            converted = pd.to_numeric(col, errors='coerce')
            problematic = [str(x).strip().lower() for x in col[(col.notna()) & (pd.to_numeric(col, errors='coerce').isna())].unique()]
            if len(problematic) > 0:
                print_system(f"Column '{col_name}' has values that could not be converted to numeric: {problematic}")
                replacement_map = {}
                for val in problematic:
                    rep = user_input(f"Provide a numeric replacement for '{val}' (or leave blank to set as missing): ").strip()
                    if rep == "":
                        replacement_map[val] = np.nan
                    else:
                        try:
                            replacement_map[val] = float(rep)
                        except:
                            print_system(f"Invalid input, setting '{val}' as missing.")
                            replacement_map[val] = np.nan
                replacement_map_lower = {k.lower(): v for k, v in replacement_map.items()}
                col = col.apply(lambda x: replacement_map_lower.get(str(x).strip().lower(), x) if isinstance(x, str) else x)
                converted = pd.to_numeric(col, errors='coerce')
            return converted.astype(float)
        elif choice == "2":
            converted = pd.to_numeric(col, errors='coerce')
            problematic = [str(x).strip().lower() for x in col[(col.notna()) & (converted.isna())].unique()]
            if len(problematic) > 0:
                print_system(f"Column '{col_name}' has values that could not be converted to integer: {problematic}")
                replacement_map = {}
                for val in problematic:
                    rep = user_input(f"Provide an integer replacement for '{val}' (or leave blank to set as missing): ").strip()
                    if rep == "":
                        replacement_map[val] = np.nan
                    else:
                        try:
                            replacement_map[val] = int(rep)
                        except:
                            print_system(f"Invalid input, setting '{val}' as missing.")
                            replacement_map[val] = np.nan
                replacement_map_lower = {k.lower(): v for k, v in replacement_map.items()}
                col = col.apply(lambda x: replacement_map_lower.get(str(x).strip().lower(), x) if isinstance(x, str) else x)
                converted = pd.to_numeric(col, errors='coerce')
            return converted.astype("Int64")
        elif choice == "3":
            return col.astype("category")
        elif choice == "4":
            return pd.to_datetime(col, errors='coerce')
        elif choice == "5":
            print_system(f"Leaving column '{col_name}' no change.")
            return col
        else:
            print_system("Invalid choice, no change.")
            return col

    def export_data(self, df: pd.DataFrame, cleaned_columns: Dict[str, pd.Series], output_file: str):
        final_df = df.copy()
        for col, series in cleaned_columns.items():
            final_df[col] = series
        cleaned_names = list(cleaned_columns.keys())
        non_cleaned_names = [col for col in df.columns if col not in cleaned_columns]
        message = (f"Exporting data...\nCleaned columns: {cleaned_names}\nNon-cleaned columns: {non_cleaned_names}")
        print_system(message)
        logger.info(message)
        file_extension = os.path.splitext(output_file)[1].lower()
        if file_extension == '.dta':
            final_df.to_stata(output_file)
        elif file_extension == '.csv':
            final_df.to_csv(output_file, index=False)
        elif file_extension == '.xlsx':
            final_df.to_excel(output_file, index=False)
        elif file_extension == '.parquet':
            final_df.to_parquet(output_file)
        print_system("Export completed.")
        return final_df

    # Updated apply_binning with new labelling format
    def apply_binning(self, col_name: str, series: pd.Series) -> Optional[pd.Series]:
        if not pd.api.types.is_numeric_dtype(series):
            return None
        bin_choice = user_input(f"Do you want to create a binned categorical column for '{col_name}'? (Y to categorize, N to skip): ").strip().lower()
        if bin_choice != 'y':
            return None
        bp_input = user_input("Enter breakpoints separated by commas (e.g., 18,40,60): ")
        parts = [p.strip() for p in bp_input.split(",") if p.strip()]
        if not parts:
            print_system("No valid breakpoints provided. Skipping binning.")
            return None
        try:
            sorted_breaks = sorted([float(p) for p in parts])
        except Exception as e:
            print_system(f"Error converting breakpoints: {str(e)}. Skipping binning.")
            return None
        # Create bins: [-inf, bp1, bp2, ..., bpN, inf]
        bins = [-np.inf] + sorted_breaks + [np.inf]
        labels = []
        # First bin: >-inf<=first_break
        labels.append(f">-inf<={sorted_breaks[0]}")
        # Intermediate bins: >prev<=current
        for i in range(1, len(sorted_breaks)):
            labels.append(f">{sorted_breaks[i-1]}<={sorted_breaks[i]}")
        # Last bin: >last_break
        labels.append(f">{sorted_breaks[-1]}")
        try:
            binned_series = pd.cut(series, bins=bins, right=True, include_lowest=True, labels=labels)
            print_system(f"Binned column '{col_name}_binned' created successfully.")
            return binned_series
        except Exception as e:
            print_system(f"Error in binning column '{col_name}': {str(e)}")
            return None

    def run(self, input_file: str, output_file: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        cleaned_columns = {}
        try:
            df = self.loader.run(input_file)
            all_missing = df.isnull().all(axis=1)
            if all_missing.sum() > 0:
                print_system(f"\nThere are {all_missing.sum()} rows with all missing values in columns.")
                drop_choice = user_input("Do you want to drop these rows? (y/n): ").strip().lower()
                if drop_choice == "y":
                    df = df.dropna(how='all')
                    print_system("Rows with all missing values have been dropped.")
            identifier_cols = [col for col in df.columns if col.lower().endswith("id")]
            if identifier_cols:
                print_system("The following identifier columns will be skipped from cleaning but will be exported:")
                print_system(str(identifier_cols))
                logger.info(f"Identifier columns to be skipped from cleaning: {identifier_cols}")
            else:
                print_system("No identifier columns (ending with 'id') were found.")
                logger.info("No identifier columns were skipped.")

            # Determine mode based on column selection with re-prompting for invalid columns
            if columns is None:
                while True:
                    col_choice = user_input("Enter column names to clean (comma-separated) or press Enter to clean all columns: ").strip()
                    if not col_choice:
                        columns = list(df.columns)
                        break
                    chosen = [c.strip() for c in col_choice.split(",") if c.strip()]
                    valid = [c for c in chosen if c in df.columns]
                    invalid = [c for c in chosen if c not in df.columns]
                    print_system(f"Valid columns entered: {valid}")
                    if invalid:
                        print_system(f"Columns not found: {invalid}")
                        option = user_input("Enter R to re-enter column names, V to proceed with valid columns, or A to proceed with all columns: ").strip().upper()
                        if option == "R":
                            continue
                        elif option == "V":
                            if valid:
                                columns = valid
                                break
                            else:
                                print_system("No valid columns entered. Please try again.")
                                continue
                        elif option == "A":
                            columns = list(df.columns)
                            break
                        else:
                            print_system("Invalid option. Please try again.")
                            continue
                    else:
                        columns = chosen
                        break
            else:
                manual_selection = True

            print_system(f"Columns selected for cleaning: {columns}")

            # Use a loop to process cleaning rounds
            current_columns = columns.copy()
            while current_columns:
                for col_name in current_columns:
                    if col_name.lower().endswith("id"):
                        logger.info(f"Skipping cleaning for column '{col_name}' (identifier column).")
                        print_system(f"Skipping cleaning for column '{col_name}' (identifier column).")
                        cleaned_columns[col_name] = df[col_name]
                    else:
                        while True:
                            try:
                                print_system(f"\n{'='*50}\nProcessing column: {col_name}\n{'='*50}")
                                while True:
                                    col_data = self.convert_column_dtype(df[col_name], col_name)
                                    confirm_dtype = user_input(f"Are you satisfied with the datatype conversion for column '{col_name}'? (Y to confirm, B to re-enter): ").strip().lower()
                                    if confirm_dtype in ['y', 'yes']:
                                        break
                                    elif confirm_dtype == 'b':
                                        continue
                                    else:
                                        print_system("Invalid input. Please enter Y or B.")
                                analysis = self.analyzer.run(col_data)
                                decisions = self.remarker.run(analysis)
                                print_system(f"Decisions for column '{col_name}': {decisions}")
                                final_confirm = user_input(f"Press 'Y' to final confirm these decisions for column '{col_name}', or 'B' to restart this column's processing: ").strip().lower()
                                if final_confirm not in ['y', 'yes']:
                                    raise RestartDecisionProcess()
                                if not isinstance(decisions, dict):
                                    raise RestartDecisionProcess()
                                if pd.api.types.is_numeric_dtype(col_data):
                                    outlier_decision = decisions.get("outlier_action", "none")
                                    missing_decision = decisions.get("missing_action", "none")
                                    orig_missing = col_data.isnull()
                                    outlier_handled = clean_numeric_column(col_data, outlier_decision, "none")
                                    if outlier_decision == "remove_outliers_fill_NA":
                                        outlier_handled = outlier_handled.astype(object)
                                        lower = col_data.quantile(0.05)
                                        upper = col_data.quantile(0.95)
                                        mask_outlier = ((col_data <= lower) | (col_data >= upper)) & (~orig_missing)
                                        outlier_handled.loc[mask_outlier] = "NA"
                                    if missing_decision != "none":
                                        filled = outlier_handled.copy()
                                        if missing_decision == "fill_NA":
                                            missing_fill = "NA"
                                            filled = filled.astype(object)
                                        elif missing_decision == "impute_mean":
                                            missing_fill = outlier_handled.mean(skipna=True)
                                        elif missing_decision == "impute_median":
                                            missing_fill = outlier_handled.median(skipna=True)
                                        elif missing_decision.startswith("custom:"):
                                            missing_fill = missing_decision.split("custom:")[1]
                                            try:
                                                missing_fill = float(missing_fill)
                                            except Exception:
                                                pass
                                        elif missing_decision == "drop_rows":
                                            filled = outlier_handled.loc[~orig_missing]
                                            outlier_handled = filled
                                        else:
                                            missing_fill = np.nan
                                        if missing_decision != "drop_rows":
                                            filled.loc[orig_missing] = filled.loc[orig_missing].fillna(missing_fill)
                                        cleaned_numeric = filled
                                    else:
                                        cleaned_numeric = outlier_handled
                                    cleaned = cleaned_numeric
                                else:
                                    if decisions["missing_action"] == "fill_NA":
                                        if isinstance(col_data.dtype, pd.CategoricalDtype):
                                            col_data = col_data.cat.add_categories(["NA"])
                                            col_data = col_data.fillna("NA")
                                        else:
                                            col_data = col_data.fillna("NA")
                                    elif decisions["missing_action"] == "fill_blank":
                                        col_data = col_data.fillna("")
                                    elif decisions["missing_action"] == "impute_mode":
                                        mode_val = col_data.mode().iloc[0] if not col_data.mode().empty else None
                                        col_data = col_data.fillna(mode_val)
                                    elif decisions["missing_action"] == "drop_rows":
                                        col_data = col_data.dropna()
                                    elif decisions["missing_action"].startswith("custom:"):
                                        custom_val = decisions["missing_action"].split("custom:")[1]
                                        if isinstance(col_data.dtype, pd.CategoricalDtype):
                                            col_data = col_data.cat.add_categories([custom_val])
                                        col_data = col_data.fillna(custom_val)
                                    mapping_result = self.cat_value_fix_agent.run(col_data)
                                    if isinstance(mapping_result, str) and mapping_result == "BACK":
                                        raise RestartDecisionProcess()
                                    else:
                                        if isinstance(mapping_result, tuple):
                                            mapped, new_col_name = mapping_result
                                            cleaned_columns[new_col_name] = mapped
                                            cleaned = col_data
                                        else:
                                            cleaned = mapping_result
                                break
                            except RestartDecisionProcess:
                                print_system(f"Restarting processing for column: {col_name}")
                                continue
                        logger.info(f"User decisions for column '{col_name}': {decisions}")
                        cleaned_columns[col_name] = cleaned
                        if pd.api.types.is_numeric_dtype(col_data):
                            binned = self.apply_binning(col_name, cleaned)
                            if binned is not None:
                                cleaned_columns[col_name + "_binned"] = binned
                # End of round: check for remaining columns
                remaining_in_df = [col for col in df.columns if col not in cleaned_columns]
                if not remaining_in_df:
                    print_system("No columns remaining to clean. Proceeding to export.")
                    break
                prompt = user_input("Do you want to clean additional columns? (Type 'C' to specify columns, 'A' to clean all remaining, or 'E' to export cleaned data): ").strip().upper()
                if prompt == "E":
                    break
                elif prompt == "C":
                    while True:
                        new_cols = user_input(f"Enter column names to clean from remaining {remaining_in_df} (comma-separated): ").strip()
                        new_cols_list = [c.strip() for c in new_cols.split(",") if c.strip() in remaining_in_df]
                        if new_cols_list:
                            current_columns = new_cols_list
                            break
                        else:
                            print_system("No valid columns entered. Please try again.")
                elif prompt == "A":
                    current_columns = remaining_in_df
                else:
                    break
            # Export summary and return the final DataFrame.
            for col, series in cleaned_columns.items():
                logger.info(f"\nColumn: {col}")
                if col in df.columns:
                    logger.info(f"Original null count: {df[col].isnull().sum()}")
                    logger.info(f"Original unique count: {df[col].nunique()}")
                else:
                    logger.info("Original data: N/A (new column)")
                logger.info(f"Cleaned null count: {series.isnull().sum()}")
                logger.info(f"Cleaned unique count: {series.nunique()}")
            final_cleaned_df = self.export_data(df, cleaned_columns, output_file)
            logger.info(f"\n=== Cleaning Process Summary ===")
            logger.info(f"Final number of rows: {len(final_cleaned_df)}")
            return final_cleaned_df
        except Exception as e:
            logger.error(f"Error in cleaning coordination: {str(e)}")
            raise DataCleaningError(f"Coordination error: {str(e)}")

# ---------------- Extended Agents ----------------

class DataRemarkAgentExtended(DataRemarkAgent):
    pass

class DataCleaningCoordinatorExtended(DataCleaningCoordinator):
    def __init__(self, model_name: str = "o3-mini"):
        super().__init__(model_name=model_name)
        self.remarker = DataRemarkAgentExtended()

# ---------------- Main ----------------

def main():
    coordinator = DataCleaningCoordinatorExtended()
    try:
        cleaned_df = coordinator.run(
            input_file="data.csv",
            output_file="cleaned_data.csv",
            columns=None
        )
        print_system("Data cleaning completed successfully!")
        print_system(str(cleaned_df))
    except DataCleaningError as e:
        print_system(f"Data cleaning failed: {str(e)}")

if __name__ == "__main__":
    main()

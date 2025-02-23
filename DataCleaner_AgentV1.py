import os
import pandas as pd
import numpy as np
import logging
import difflib
import openai
import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data_cleaning_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
    """Custom exception for data cleaning errors."""
    pass


class BaseAgent:
    def __init__(self, model_name: str = "o3-mini"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        # Create a client using the new OpenAI interface
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model_name = model_name

    def _call_openai(self, prompt: str) -> str:
        try:
            logger.info(f"Sending prompt to OpenAI:\n{prompt}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                # temperature=0
            )
            response_content = response.choices[0].message.content.strip()
            logger.info(f"Received response from OpenAI:\n{response_content}")
            return response_content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise DataCleaningError(f"AI model error: {str(e)}")


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


class DataAnalyzerAgent(BaseAgent):
    def _analyze_numeric_column(self, column_data: pd.Series) -> Dict:
        return {
            "mean": column_data.mean(),
            "std": column_data.std(),
            "outliers": int(len(column_data[np.abs(column_data - column_data.mean()) > 3 * column_data.std()]))
        }

    def _analyze_categorical_column(self, column_data: pd.Series) -> Dict:
        return {
            "value_counts": column_data.value_counts().head().to_dict(),
            "category_count": int(column_data.nunique()),
            "most_common": column_data.mode().iloc[0] if not column_data.empty else None
        }

    def run(self, column_data: pd.Series) -> ColumnAnalysis:
        try:
            non_null = column_data.dropna()
            sample = non_null.sample(min(5, len(non_null))).tolist() if not non_null.empty else []
            null_count = int(column_data.isnull().sum())
            unique_count = int(column_data.nunique())

            if pd.api.types.is_numeric_dtype(column_data):
                specific_analysis = self._analyze_numeric_column(column_data)
            else:
                specific_analysis = self._analyze_categorical_column(column_data)

            prompt = f"""
Analyze this data column and provide detailed cleaning recommendations:
Column name: {column_data.name}
Sample values: {sample}
Null count: {null_count}
Unique values: {unique_count}
Data type: {column_data.dtype}
Additional analysis: {specific_analysis}

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


class DataRemarkAgent:
    """
    Presents the analysis to the user and asks for cleaning decisions regarding missing values and outliers.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def run(self, analysis: ColumnAnalysis) -> Dict:
        print(f"\n{'='*50}")
        print(f"Reviewing Column: {analysis.column_name} (Type: {analysis.data_type})")
        print(f"Null count: {analysis.null_count}")
        print(f"Unique count: {analysis.unique_count}")
        print(f"Sample values: {analysis.sample_values}")
        print("\nIssues detected:")
        for issue in analysis.issues:
            print(f"- {issue}")
        print("\nAutomated Cleaning Recommendations:")
        for rec in analysis.recommendations:
            print(f"- {rec}")

        decisions = {}
        if analysis.null_count > 0:
            print("\nFor missing values, please choose an action:")
            if any(x in analysis.data_type.lower() for x in ["int", "float"]):
                print("1. Impute with mean")
                print("2. Impute with median")
                print("3. Drop rows with missing values")
                print("4. Fill missing values with 'NA'")
                choice = input("Enter choice (1-4): ").strip()
                if choice == "1":
                    decisions["missing_action"] = "impute_mean"
                elif choice == "2":
                    decisions["missing_action"] = "impute_median"
                elif choice == "3":
                    decisions["missing_action"] = "drop_rows"
                elif choice == "4":
                    decisions["missing_action"] = "fill_NA"
                else:
                    decisions["missing_action"] = "none"
            else:
                print("1. Fill missing values with 'NA'")
                print("2. Drop rows with missing values")
                print("3. Impute with mode (most common value)")
                choice = input("Enter choice (1-3): ").strip()
                if choice == "1":
                    decisions["missing_action"] = "fill_NA"
                elif choice == "2":
                    decisions["missing_action"] = "drop_rows"
                elif choice == "3":
                    decisions["missing_action"] = "impute_mode"
                else:
                    decisions["missing_action"] = "none"

        if any("outlier" in issue.lower() for issue in analysis.issues):
            print("\nOutliers have been detected. Choose an action:")
            print("1. Remove outliers")
            print("2. Cap outliers")
            print("3. Do nothing")
            choice = input("Enter choice (1-3): ").strip()
            if choice == "1":
                decisions["outlier_action"] = "remove_outliers"
            elif choice == "2":
                decisions["outlier_action"] = "cap_outliers"
            elif choice == "3":
                decisions["outlier_action"] = "none"
            else:
                decisions["outlier_action"] = "none"

        print(f"Decisions for column '{analysis.column_name}': {decisions}\n{'='*50}\n")
        return decisions


class DataCleanerAgent(BaseAgent):
    def _validate_cleaned_data(self, original: pd.Series, cleaned: pd.Series) -> bool:
        if len(original) != len(cleaned):
            raise DataCleaningError("Cleaning altered number of rows.")
        if cleaned.isnull().sum() > original.isnull().sum() * 1.1:
            raise DataCleaningError("Cleaning introduced too many null values")
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
4. Assume the input is a pandas Series called `column_data` and return the cleaned `column_data`.

Return only executable Python code (no markdown formatting).
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

    def run(self, column_data: pd.Series, analysis: ColumnAnalysis, remarks: Dict) -> pd.Series:
        try:
            cleaning_prompt = f"""
Generate Python code to clean this data column based on these recommendations and user decisions:
Column name: {column_data.name}
Data type: {analysis.data_type}
Automated recommendations: {analysis.recommendations}
User decisions: {remarks}

The code should:
1. Handle missing values according to the user's decision.
2. Handle outliers as specified by the user's decision.
3. Use only pandas and numpy operations.
4. Preserve the original data type when possible.
5. Assume that the input is a pandas Series called `column_data` and return the cleaned `column_data`.

Return only executable Python code (no markdown formatting).
"""
            cleaning_code = self._call_openai(cleaning_prompt)
            cleaning_code = cleaning_code.replace("```python", "").replace("```", "").strip()
            local_dict = {"column_data": column_data.copy()}
            cleaned_data = self._safe_exec(cleaning_code, local_dict)

            if self._validate_cleaned_data(column_data, cleaned_data):
                logger.info(f"Successfully cleaned column {column_data.name}")
                logger.info(f"Cleaning results for {column_data.name}:")
                logger.info(f"Original null count: {column_data.isnull().sum()}")
                logger.info(f"Cleaned null count: {cleaned_data.isnull().sum()}")
                logger.info(f"Original unique count: {column_data.nunique()}")
                logger.info(f"Cleaned unique count: {cleaned_data.nunique()}")
                return cleaned_data
        except Exception as e:
            logger.error(f"Error in cleaning process for {column_data.name}: {str(e)}")
            return column_data


class CategoricalValueFixAgent:
    """
    Uses commonsense reasoning via a language model to propose a mapping for 
    normalizing categorical values.
    """
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
                print(f"\nFor the following values (differing only by capitalization or slight variation): {originals}")
                rep = input("Enter the representative normalized value to use: ").strip()
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

    def run(self, column_data: pd.Series) -> pd.Series:
        # Automatically fill missing values.
        if column_data.isnull().sum() > 0:
            column_data = column_data.fillna("NA")

        if not (pd.api.types.is_object_dtype(column_data) or isinstance(column_data.dtype, pd.CategoricalDtype)):
            return column_data

        unique_values = list(column_data.unique())
        prompt = (
            f"These are the unique values from a data column: {unique_values}\n"
            "Using common sense, produce a JSON object mapping each original value to a normalized label. "
            "For example, map 'F' or 'f' to 'Female' and 'M' or 'm' to 'Male' if applicable, but do not group distinct values that should remain separate. "
            "Return only the JSON object."
        )
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            mapping_str = response.choices[0].message.content.strip()
            mapping = json.loads(mapping_str)
        except Exception as e:
            logger.error(f"Error obtaining mapping from language model: {str(e)}")
            mapping = {v: v for v in unique_values}

        print(f"\nFor column '{column_data.name}', the suggested mapping is:")
        for orig, norm in mapping.items():
            print(f" - {orig} -> {norm}")

        consolidated = self.consolidate_mapping(mapping)
        print("\nAfter consolidating similar values, the mapping is:")
        for orig, norm in consolidated.items():
            print(f" - {orig} -> {norm}")
        choice = input("Do you want to apply this mapping? (Y/n): ").strip().lower()
        if choice in ["", "y", "yes"]:
            return column_data.map(consolidated).fillna(column_data)
        else:
            new_mapping_str = input("Enter a custom mapping (e.g., F:Female; fem:Female; M:Male) or press Enter to leave unchanged: ").strip()
            if new_mapping_str:
                new_mapping = self.parse_custom_mapping(new_mapping_str)
                if new_mapping:
                    return column_data.map(new_mapping).fillna(column_data)
                else:
                    print("No valid mapping found. Leaving column unchanged.")
                    return column_data
            else:
                return column_data


class DataCleaningCoordinator(BaseAgent):
    def __init__(self, model_name: str = "o3-mini"):
        super().__init__(model_name=model_name)
        self.loader = DataLoaderAgent(model_name=model_name)
        self.analyzer = DataAnalyzerAgent(model_name=model_name)
        self.remarker = DataRemarkAgent()
        self.cleaner = DataCleanerAgent(model_name=model_name)
        self.cat_value_fix_agent = CategoricalValueFixAgent()

    def run(self, input_file: str, output_file: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        try:
            df = self.loader.run(input_file)
            # Check for rows where all column values are missing.
            all_missing = df.isnull().all(axis=1)
            if all_missing.sum() > 0:
                print(f"\nThere are {all_missing.sum()} rows with all values missing.")
                drop_choice = input("Do you want to drop these rows? (y/n): ").strip().lower()
                if drop_choice == "y":
                    df = df.dropna(how='all')
                    print("Rows with all missing values have been dropped.")

            cleaned_columns = {}
            global_index = df.index
            columns_to_clean = columns or list(df.columns)

            for column in columns_to_clean:
                logger.info(f"\n{'='*50}\nProcessing column: {column}\n{'='*50}")
                try:
                    analysis = self.analyzer.run(df[column])
                    logger.info(f"Analysis results for {column}:")
                    logger.info(f"Issues found: {analysis.issues}")
                    logger.info(f"Recommendations: {analysis.recommendations}")

                    remarks = self.remarker.run(analysis)
                    cleaned_series = self.cleaner.run(df[column], analysis, remarks)

                    if set(cleaned_series.index) != set(df[column].index):
                        global_index = global_index.intersection(cleaned_series.index)

                    cleaned_columns[column] = cleaned_series
                except Exception as e:
                    logger.error(f"Error processing column {column}: {str(e)}")
                    cleaned_columns[column] = df[column]

            for column in df.columns:
                if column not in cleaned_columns:
                    cleaned_columns[column] = df[column]

            final_cleaned_df = pd.DataFrame({
                col: series.reindex(global_index) for col, series in cleaned_columns.items()
            })

            # Apply generic categorical value fixing on all object or categorical columns.
            for col in final_cleaned_df.columns:
                if pd.api.types.is_object_dtype(final_cleaned_df[col]) or isinstance(final_cleaned_df[col].dtype, pd.CategoricalDtype):
                    print(f"\nAuto-fixing categorical values for column: {col}")
                    final_cleaned_df[col] = self.cat_value_fix_agent.run(final_cleaned_df[col])

            file_extension = os.path.splitext(output_file)[1].lower()
            if file_extension == '.dta':
                final_cleaned_df.to_stata(output_file)
            elif file_extension == '.csv':
                final_cleaned_df.to_csv(output_file, index=False)
            elif file_extension == '.xlsx':
                final_cleaned_df.to_excel(output_file, index=False)
            elif file_extension == '.parquet':
                final_cleaned_df.to_parquet(output_file)

            logger.info("\n=== Cleaning Process Summary ===")
            logger.info(f"Final number of rows: {len(final_cleaned_df)}")
            for col, series in cleaned_columns.items():
                logger.info(f"\nColumn: {col}")
                logger.info(f"Original null count: {df[col].isnull().sum()}")
                logger.info(f"Cleaned null count: {series.isnull().sum()}")
                logger.info(f"Original unique count: {df[col].nunique()}")
                logger.info(f"Cleaned unique count: {series.nunique()}")

            return final_cleaned_df
        except Exception as e:
            logger.error(f"Error in cleaning coordination: {str(e)}")
            raise DataCleaningError(f"Coordination error: {str(e)}")


def main():
    coordinator = DataCleaningCoordinator()
    try:
        cleaned_df = coordinator.run(
            input_file="data.csv",
            output_file="cleaned_data.csv",
            columns=None
        )
        print("Data cleaning completed successfully!")
        print(cleaned_df)
    except DataCleaningError as e:
        print(f"Data cleaning failed: {str(e)}")


if __name__ == "__main__":
    main()

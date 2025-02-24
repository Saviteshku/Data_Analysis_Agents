Version 1
Date: 12/05/2024

## Retrieves the API key and initializes the OpenAI client.
## Sends a prompt to OpenAI and returns its response.
## Loads a file into a DataFrame based on its extension.
## Computes numeric statistics and identifies outliers.
## Summarizes categorical data with counts and mode.
## Analyzes a column to fetch cleaning issues and suggestions.
## Prepares the agent for user interaction.
## Displays analysis results and collects user cleaning choices.
## Validates that cleaned data retains its structure and quality.
## Requests revised cleaning code from OpenAI after errors.
## Safely executes cleaning code with retries if needed.
## Generates and applies cleaning code based on recommendations and user decisions.
## Merges similar categorical mappings with user input.
## Converts a custom mapping string into a dictionary.
## Normalizes categorical values using automatic or custom mappings.
## Sets up all sub-agents for the cleaning process.
## Orchestrates data loading, analysis, cleaning, and saving.
## Initiates the overall data cleaning workflow.
## USAGE GPT-4-Turbo
## Fixed the missing action and outlier action.
## Added extra options to missing action and outlier action.
## Updated per column wise cleaning instead of steps wise.
## Added custom input options.
## Used quantiles for outlier detection and removal and imputation.
## USAGE GPT-4o-mini
## Added the loop back mechanism for verification at each action.
## The loop back will help to fix errors if typed or selected by mistake.
## USAGE GPT-4o-mini
## Added option to "do nothing" to categorical mapping.
## Added option to auto skip "ID" columns
## USAGE GPT-4o-mini

Version 2
Date 12/02/2025


## Added custom halt fucntion after each column to export cleaned data.
## Added non missing cell count for column.
## Refined the mapping to proceed with given input and use rest as suggested in console by GPT.
## USAGE GPT-o3-mini (reasoning model)


Version 3
Date 24/02/2025

## Colour codes added to the outputs in console





flowchart
    subgraph "Data Loading & Initialization"
        A[User] -->|Provides input file| B(DataLoaderAgent)
        B --> C[Raw DataFrame]
    end

    subgraph "Column Processing (Iterative)"
        C --> D(DataCleaningCoordinator)
        D --> E[For Each Column]
        E --> F(DataAnalyzerAgent)
        F --> G[Column Analysis]
        G --> H(DataRemarkAgent)
        H --> I[User Decisions]
        I --> J{Column Type?}
        J -- "Numeric" --> K(DataCleanerAgent)
        J -- "Categorical" --> L(CategoricalValueFixAgent)
        K --> M[Clean Numeric Column]
        L --> N[Clean Categorical Column]
        M --> O[Cleaned Column]
        N --> O
        O --> P[Update DataFrame with Cleaned Column]
    end

    subgraph "Post-Processing & Export"
        P --> Q{More Columns?}
        Q -- "Yes" --> E
        Q -- "No" --> R[Prompt: Export & Exit?]
        R -- "Yes" --> S[Export Final DataFrame]
        S --> T[Output File Generated]
        R -- "No" --> U[Continue Processing Next Column]
        U --> E
    end

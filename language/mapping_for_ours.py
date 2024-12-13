import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import openpyxl
from src.api_manager import llm_chain
from src.template_loader import get_template


def get_reports(file: str | Path, template_file: str | Path) -> pd.DataFrame:
    # Read the accession numbers from the text file into a set
    with open(template_file, 'r') as f:
        # print(f.read())
        # for line in f:
        #     print(line.strip() )
        accession_numbers = {int(line.strip()) for line in f}
    print(accession_numbers)
    # Load the CSV into a DataFrame
    df = pd.read_csv(file)

    print(f"Number of rows before filtering: {len(df)}")

    # Filter the DataFrame to only include rows with accession in the set
    print(df['accession'].isin(accession_numbers))
    filtered_df = df[df['accession'].isin(accession_numbers)]
       
    print(f"Number of rows after filtering: {len(filtered_df)}")

    return filtered_df


def run_gpt_vague_incorrect_analysis(template_file: str | Path, reports_file: str | Path,
                                     output_file: str | Path, accession_file) -> pd.DataFrame:
    template = get_template(template_file)
    chain = llm_chain(template, temperature=0.2, deployment_name="GPT4-32k")
    df_reports = get_reports(reports_file, accession_file)

    if output_file.exists():
        df = pd.read_csv(output_file)

        # Skip already processed reports
        df_reports = df_reports[~df_reports["raw_report"].isin(df["accession"])]
    else:
        df = pd.DataFrame()

    for i, row in tqdm(df_reports.iterrows(), total=len(df_reports), desc="Processing reports"):
        if "raw_report" not in row:
            continue
        template_params = {"report": row["raw_report"]}

        response = chain.invoke(template_params)
        response["raw_report"] = row["raw_report"]

        # Add template to every row (not efficient)
        # response["template"] = template

        # Save progress after each iteration in case server fails without overwriting the dataframe
        df_new = pd.DataFrame([response])
        df_new = df_new.rename(columns={"text": "gpt_output"})

        df = pd.concat([df, df_new], ignore_index=True)
        # print(df)
        df.to_csv(output_file, index=False)

    
    return df



if __name__ == "__main__":
    # Run each template 3 times
    accession_file = "./unique_accession_numbers.txt"
    template_file = "./datamapping/prompt.txt"
    reports_file = "./label_report/reports.csv"
    output_dir = "./datamapping/experiment_l_1"
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

 
    output_file = output_dir / f"results.csv"
    # filtered_reports = get_reports(reports_file, accession_file)
    df = run_gpt_vague_incorrect_analysis(template_file, reports_file, output_file, accession_file)



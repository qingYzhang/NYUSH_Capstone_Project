import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.api_manager import llm_chain
from src.template_loader import get_template


def get_reports(file: str | Path) -> pd.DataFrame:
    df = pd.read_csv(file)

    # # Add accession if missing
    # if "accession" not in df.columns:
    #     # Start from 1 for easier understanding by non-technical users
    #     df["accession"] = df.index + 1

    return df


def run_gpt_vague_incorrect_analysis(template_file: str | Path, reports_file: str | Path,
                                     output_file: str | Path) -> pd.DataFrame:
    template = get_template(template_file)
    chain = llm_chain(template, temperature=0.2, deployment_name="GPT4-32k")
    df_reports = get_reports(reports_file)

    if output_file.exists():
        df = pd.read_csv(output_file)

        # Skip already processed reports
        df_reports = df_reports[~df_reports["accession"].isin(df["accession"])]
    else:
        df = pd.DataFrame()

    for i, row in tqdm(df_reports.iterrows(), total=len(df_reports), desc="Processing reports"):
        # if "FINDINGS" not in row:
        #     continue
        template_params = {"report": row["raw_report"],"impression": row["impression"]}

        response = chain.invoke(template_params)
        response["accession"] = row["accession"]

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
    template_file = "./label_report/prompt.txt"
    reports_file = "./label_report/reports.csv"
    output_dir = "./label_report/experiment_2"
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

 
    output_file = output_dir / f"results.csv"
    df = run_gpt_vague_incorrect_analysis(template_file, reports_file, output_file)
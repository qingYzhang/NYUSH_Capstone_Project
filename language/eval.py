import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
import numpy as np
nltk.download('punkt_tab')
from scipy.stats import t


# import nltk
nltk.download('wordnet')  # Download the WordNet resource
nltk.download('punkt')    # Ensure the punkt tokenizer is available

def calculate_bleu(reference, candidate):
    """Calculate BLEU score."""
    reference = [reference.split()]
    candidate = candidate.split()
    return sentence_bleu(reference, candidate)

def calculate_rouge(reference, candidate):
    """Calculate ROUGE score."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, candidate)

# def calculate_meteor(ground_truth, generated):
#     return meteor_score([ground_truth], generated)

def calculate_meteor(ground_truth, generated):
    """Calculate METEOR score after tokenizing the sentences."""
    # Tokenize both the ground truth and generated sentences
    ground_truth_tokens = nltk.word_tokenize(ground_truth)
    generated_tokens = nltk.word_tokenize(generated)
    return meteor_score([ground_truth_tokens], generated_tokens)
def calculate_confidence_interval(data, confidence=0.95):
    """Calculate 95% confidence interval for a given data series."""
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))  # Standard error
    t_crit = t.ppf((1 + confidence) / 2, df=len(data) - 1)  # t critical value
    margin = t_crit * std_err
    return mean, mean - margin, mean + margin
def calculate_scores(file1, file2, output_file):
    """Calculate BLEU, ROUGE, METEOR scores for each pair of reports."""
    # Load the CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Merge the two dataframes on the 'accession' column
    merged_df = pd.merge(df1, df2, on="accession", how='inner', suffixes=('_gt', '_gen'))
    # print(merged_df.columns)
    # print(merged_df.head())
    # Prepare a list to store the results
    results = []

    # Iterate through each row in the merged dataframe
    for _, row in merged_df.iterrows():
        ground_truth = row['gpt_output_gt']  # Ground truth report
        generated = row['gpt_output_gen']    # Generated report

        # Calculate BLEU, ROUGE, METEOR scores
        bleu_score = calculate_bleu(ground_truth, generated)
        rouge_scores = calculate_rouge(ground_truth, generated)
        meteor_score = calculate_meteor(ground_truth, generated)

        # Store the results
        results.append({
            'accession': row['accession'],
            'bleu_score': bleu_score,
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure,
            'meteor_score': meteor_score
        })

    # Convert results into a DataFrame
    results_df = pd.DataFrame(results)
    
    average_scores = {}
    confidence_intervals = {}

    for metric in ['bleu_score', 'rouge1', 'rouge2', 'rougeL', 'meteor_score']:
        mean, lower, upper = calculate_confidence_interval(results_df[metric])
        average_scores[f'average_{metric}'] = mean
        confidence_intervals[metric] = (lower, upper)

    print("Average Scores and 95% Confidence Intervals:")
    for metric, mean in average_scores.items():
        ci_lower, ci_upper = confidence_intervals[metric.replace('average_', '')]
        print(f"{metric}: {mean:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")

    # Calculate average scores
    average_scores = {
        'average_bleu': results_df['bleu_score'].mean(),
        'average_rouge1': results_df['rouge1'].mean(),
        'average_rouge2': results_df['rouge2'].mean(),
        'average_rougeL': results_df['rougeL'].mean(),
        'average_meteor': results_df['meteor_score'].mean()
    }

    # Save the results to a CSV file
    results_df.to_csv(output_file, index=False)
    print(f"Scores saved to {output_file}")

    # # Print the average scores
    # print("Average Scores:")
    # for metric, score in average_scores.items():
    #     print(f"{metric}: {score:.4f}")

    return average_scores

if __name__ == "__main__":
    # Define input and output file paths
    file1 = './rep/gt_new.csv'  # Replace with your first CSV file
    file2 = './rep/results.csv'     # Replace with your second CSV file
    output_file = 'evaluation_scores.csv'

    # Calculate and save the scores
    calculate_scores(file1, file2, output_file)

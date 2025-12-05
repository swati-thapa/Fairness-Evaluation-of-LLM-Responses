# Fairness-Evaluation-of-LLM-Responses

# LLM Fairness Evaluation with HolisticBias

This project evaluates group fairness of an LLM using the HolisticBias dataset and external sentiment/toxicity classifiers. We focus on gender & sex–related buckets and compute Demographic Parity, Statistical Parity Difference, Disparate Impact Ratio, and Toxicity Disparity.

## 1. Setup

- Install dependencies:
  - `datasets`
  - `transformers`
  - `pandas`, `numpy`
  - `plotly` (for plots)
- Import libraries and set a random seed for reproducibility.

## 2. Load Dataset

- Load the `fairnlp/holistic-bias` dataset (configuration: `sentences`) from Hugging Face.
- Select only the `gender_and_sex` axis.
- Keep the `text` and the demographic `bucket` (e.g., *binary*, *cisgender*, *non_binary_or_gnc*, etc.).

## 3. Pre-processing

- Drop rows where `bucket` is `"none"` or `"(none)"`.
- Group by `bucket` and sample up to 20 examples per group (`groupby("bucket").head(20)`) to control the number of LLM calls.
- Store the result in `df_sampled`.

## 4. Generate LLM Responses

- Use `google/flan-t5-large` with a `text2text-generation` pipeline.
- Wrap each input sentence with a simple “helpful assistant” instruction prompt.
- Save the model output in a new column `llm_response`.

## 5. Sentiment & Favorable Outcome

- Use `cardiffnlp/twitter-roberta-base-sentiment-latest` with a `sentiment-analysis` pipeline.
- Run sentiment on each `llm_response` and store the label in `sentiment` (`positive`, `neutral`, `negative`).
- Define a binary outcome:
  - `favorable = 1` if sentiment is `positive` or `neutral`
  - `favorable = 0` if sentiment is `negative`

## 6. Toxicity Scoring

- Use `unitary/toxic-bert` with a `text-classification` pipeline.
- For each `llm_response`, extract the score associated with the `toxic` label.
- Store it as `toxicity_score` (float in [0, 1]).

## 7. Group Fairness Metrics

Using `df_sampled` grouped by `bucket`:

1. **Demographic Parity (DP)**
   - `DP(group) = mean(favorable | bucket = group)`
   - Also compute overall mean DP and flag groups within a ±5% tolerance band.

2. **Statistical Parity Difference (SPD)**
   - Choose a reference group (e.g., the largest group or `binary`).
   - `SPD(group) = DP(group) - DP(reference_group)`

3. **Disparate Impact Ratio (DIR)**
   - `DIR(group) = DP(group) / DP(reference_group)`

4. **Toxicity Disparity**
   - `mean_toxicity(group) = mean(toxicity_score | bucket = group)`
   - `ToxDisparity(group) = mean_toxicity(group) - mean_toxicity(reference_group)`

- Collect these into a summary table `group_stats_gender` with:
  - `n_examples`, `group_benefit` (DP), `SPD`, `DIR`, `mean_toxicity`, `tox_disparity`.

## 8. Visualization

- Use Plotly bar charts to visualize:
  - Demographic Parity per bucket.
  - SPD and DIR per bucket.
  - Mean toxicity and toxicity disparity per bucket.

## 9. Interpretation

- Inspect DP, SPD, DIR, and toxicity disparity:
  - Check if some gender/sex buckets have systematically lower favorable rates or higher toxicity.
  - Use the tolerance band and ratios to flag potential fairness issues that may need mitigation.

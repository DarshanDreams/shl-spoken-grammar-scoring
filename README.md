# SHL Hiring Assessment 2026 — Spoken English Grammar Scoring

## Project Overview

This project predicts grammar scores from spoken English audio recordings. It uses audio representations and linguistic features to estimate the grammar score for each recording.

## Methodology

- **Audio Feature Extraction:** Used Wav2Vec2 to extract speech embeddings from audio recordings.
- **Statistical Pooling:** Combined mean and standard deviation pooling to represent audio characteristics and their variation over time.
- **Linguistic Features:** Incorporated features derived from speech transcripts.
- **Modeling:** Experimented with Ridge Regression and XGBoost.
- **Ensemble:** Combined model predictions to balance their strengths.
- **Evaluation:** Used five-fold cross-validation to compare model performance and experiment with regularization and blending weights.

## Technologies Used

- Python
- Wav2Vec2
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Audio and speech processing

## Results

**Best public leaderboard score: 0.5263**

The best-performing submission used a blend of model predictions based on Wav2Vec2 mean and standard deviation embeddings and linguistic features.

## Repository Contents

- `SHL_Grammar_Scoring.ipynb` — Initial experimentation and modeling notebook.
- `shl_mean_std_blend_candidate.csv` — Best-performing Kaggle submission file.

**Note:** The initial notebook does not, by itself, reproduce the best leaderboard score. The final feature extraction and model-blending pipeline should also be included for full reproducibility.

## Objective

The goal was to build a regression pipeline that uses speech-related features to predict spoken English grammar scores and improve performance through validation and model ensembling.

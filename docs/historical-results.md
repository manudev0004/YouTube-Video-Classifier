# Historical Results and Evaluation Limits

[Project overview](../README.md) · [Current workflow](workflow.md) · [Original training notebook](../ML_AI_Train_Test_Code.ipynb)

The original notebook preserves an experiment with 4,218 rows: 2,384 negative and 1,834 positive category-proxy labels. Its saved outputs record:

| Model | Historical accuracy |
|---|---:|
| Naive Bayes | 89.57% |
| Random Forest | 91.71% |
| Support Vector Classifier | 88.51% |
| K-Nearest Neighbours | 86.14% |
| Decision Tree | 91.11% |
| XGBoost | 91.23% |

The original dataset is absent, so these results have not been reproduced in the current repository.

## What Limits the Comparison

TF-IDF was fitted on all titles before the 80/20 split. This exposes preprocessing to test vocabulary and IDF statistics. The original procedure also lacks a fixed split/model seed and exact-title deduplication, so the recorded scores do not provide a clean, reproducible held-out estimate.

Labels come from YouTube categories 27 and 28. They measure a category proxy rather than human assessments of educational value. Some saved examples appear to include extra keyword text in the title; input cleanliness cannot be established without the CSV.

The historical work also explores multiclass category prediction, which is a separate task. The earlier script's binary `premodel.model` artifact referred to XGBoost, while the 91.71% result belongs to Random Forest. Neither that artifact nor the removed script is needed for the current CLI.

## The Current Evaluation Path

`predict.py` validates and deduplicates raw titles, splits before preprocessing and saves the fitted vectoriser and estimator together. Fixed seeds, a majority baseline and metrics metadata make new runs easier to inspect. The synthetic sample checks that this workflow runs; a real dataset is needed to evaluate model quality.

A meaningful new result needs a real dataset with provenance and permission to use it. Separate related videos or channels where possible, review label quality, inspect precision/recall and error cases, and reserve a final test set after model selection. Record the data revision and cleaning decisions with the results.

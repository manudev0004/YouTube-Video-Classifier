# Historical Results and Evaluation Limits

[Overview](../README.md) · [Original training notebook](../ML_AI_Train_Test_Code.ipynb) · [Archived script](../archive/predict-original.py.txt)

The original training notebook preserves a 4,218-row experiment with 2,384 negative and 1,834 positive proxy labels. Its saved outputs report:

| Model | Recorded historical accuracy |
|---|---:|
| Naive Bayes | 89.57% |
| Random Forest | 91.71% |
| Support Vector Classifier | 88.51% |
| K-Nearest Neighbours | 86.14% |
| Decision Tree | 91.11% |
| XGBoost | 91.23% |

These are historical notebook outputs. The original dataset is absent, so the results have not been reproduced in this revision.

## Why these are not the new pipeline's benchmark

The original experiment fits TF-IDF on all titles before creating the 80/20 split. This exposes preprocessing to test vocabulary and IDF statistics. It also has no fixed split/model seed or exact-title deduplication. The recorded result is therefore not a clean, reproducible held-out estimate.

The binary target is **Category_ID 27 or 28**, rather than a human annotation of educational quality. Some saved input examples appear to include extra keyword text in the Title field; original input cleanliness cannot be established without the CSV. There is no transcript, audio or video-content analysis in the executable classifier.

The archived script also trains separate multiclass category models. Those are a different task from the binary benchmark. Its saved `premodel.model` is the binary XGBoost estimator, not the Random Forest associated with 91.7%.

## What changed

The executable `predict.py` now handles the binary proxy task explicitly. It validates and deduplicates raw titles, splits them before preprocessing, fits a pipeline on training data, and saves the vectoriser and estimator together. Fixed seeds, a majority baseline and metrics metadata make new runs inspectable.

The included `examples/demo.csv` is synthetic and tests the workflow only. Its scores must not be presented as project accuracy or used on a resume. New meaningful results require a real, appropriately licensed dataset and evaluation under the corrected procedure.

## Further evaluation worth doing

- Separate related videos/channels when measuring generalisation beyond familiar channels.
- Review label quality with human annotations if the desired task is educational relevance.
- Check class balance, precision/recall and errors, rather than accuracy alone.
- Use a validation set or cross-validation for model selection and reserve a final test set.
- Record the data revision, cleaning decisions, channel/video grouping and dependency versions.

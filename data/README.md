# Training Data

The original dataset is not included. Supply a CSV using this schema:

| Column | Meaning |
|---|---|
| `Title` | Non-empty video title text |
| `Category_ID` | Integer YouTube category ID |
| `Data_Source` | Optional provenance marker; the supplied demo uses `synthetic` |

Categories 27 (Education) and 28 (Science & Technology) map to label 1; other categories map to label 0. This is a category proxy, not a judgment about factual correctness or teaching quality.

The CLI rejects blank titles and non-integer category IDs, removes exact normalized title duplicates and rejects duplicate titles with contradictory proxy labels. It needs both classes with enough examples for the chosen stratified split. Related videos may still cross the split.

Put a real dataset at `data/videos.csv`, or pass another path. CSVs in this directory are ignored by Git so a private dataset is not added accidentally. The original `category_id.JSON` and `categories.csv` are not required by the new binary workflow.

The API-analysis notebook is a separate historical exploration. Its collector does not currently emit Category_ID, so its CSV is not directly compatible with this schema. Add the actual category IDs from the API before using collected rows as training data.

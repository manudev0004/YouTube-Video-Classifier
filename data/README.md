# Training Data

[Project overview](../README.md) · [Training workflow](../docs/workflow.md)

Supply a CSV containing video titles and their YouTube category IDs. The original 4,218-row dataset is not included; [the 40-row sample](../examples/demo.csv) uses synthetic titles to demonstrate the workflow.

## CSV Format

| Column | Required | Meaning |
|---|---|---|
| `Title` | Yes | Non-empty video title text |
| `Category_ID` | Yes | Integer YouTube category ID |
| `Data_Source` | No | Provenance marker; every supplied demo row uses `synthetic` |

```csv
Title,Category_ID
Understanding electric circuits,27
An introduction to computer processors,28
Live music performance,10
```

## Labels

| YouTube category | Label | Prediction name |
|---|---:|---|
| 27: Education | 1 | `education-or-science` |
| 28: Science & Technology | 1 | `education-or-science` |
| Other categories | 0 | `other` |

These labels describe category membership. Educational material can appear in other categories, and category membership does not establish teaching quality.

## Validation

The CLI rejects blank titles and non-integer category IDs. It normalises case and whitespace for duplicate detection, removes exact normalised duplicates and rejects duplicates with conflicting labels. Both classes need enough unique titles for the selected stratified split; at least two per class are required during loading.

Related titles or videos from the same channel can still appear on both sides of the split. For stronger evaluation, record the data's provenance and separate related videos or channels where appropriate.

Pass the CSV path with `--data`. Files under `data/` are ignored by Git except for this guide, so an unpublished dataset can stay local.

The API-analysis notebook does not currently export `Category_ID`. Add category IDs from the API before using its collected rows with this classifier. The binary CLI does not require the original `category_id.JSON` or `categories.csv` files.

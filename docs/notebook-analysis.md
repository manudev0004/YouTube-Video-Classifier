# YouTube Notebook Analysis

[Project overview](../README.md) · [Analysis notebook](../YT-Analysis.ipynb) · [Historical classification results](historical-results.md)

The analysis notebook explores YouTube channel and video metadata. It uses API pagination, tabular preparation, sorting and aggregation to look at channel totals, popular uploads and publishing patterns. These plots preserve the notebook's saved outputs at the time of the experiment.

## Channel Comparisons

<table>
<tr><th width="50%">Subscribers</th><th width="50%">Total channel views</th></tr>
<tr><td><img src="../assets/analysis-1.png" width="420" alt="Original subscriber-count comparison between channels"></td><td><img src="../assets/analysis-2.png" width="420" alt="Original total-view comparison between channels"></td></tr>
</table>

![Original notebook chart comparing published video counts](../assets/analysis-3.png)

## Video and Publishing Patterns

![Original notebook chart showing ten most-viewed video titles](../assets/analysis-4.png)

![Original notebook chart showing uploads by month](../assets/analysis-5.png)

These are exploratory metadata views. Subscriber and view counts are historical snapshots, and none of these charts measures classifier accuracy.

## Working with the Notebook

```bash
python -m pip install -r requirements-optional.txt
export YOUTUBE_API_KEY="your-own-key"
jupyter lab
```

On PowerShell, use `$env:YOUTUBE_API_KEY="your-own-key"`. The notebook reads this value from the environment; `.env.example` is a reference file and is not loaded automatically.

The notebook keeps original machine-specific paths, and some optional API statistics need defensive handling. Its exported metadata lacks the `Category_ID` required by the classification CSV, so collection and training need an explicit data-preparation step. Live API collection has not been revalidated in this documentation pass.

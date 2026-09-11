import pandas as pd
from common.data_loader import dataset_summary


def test_dataset_summary_counts_labels():
    df = pd.DataFrame({"message": ["a", "b", "c"], "label": [0, 1, 1]})
    summary = dataset_summary(df)
    assert summary["rows"] == 3
    assert summary["ham"] == 1
    assert summary["spam"] == 2

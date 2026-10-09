"""Dataset loading and splitting for the SMS Spam Collection."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FILE = "SMSSpamCollection"


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the raw TSV into a DataFrame with columns `text` and `label` (1 = spam)."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], encoding="utf-8")
    df["label"] = (df["label"] == "spam").astype(int)
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated messages so copies cannot leak across the train/test split."""
    return df.drop_duplicates(subset="text").reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split that preserves the spam ratio."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    return train.reset_index(drop=True), test.reset_index(drop=True)

import pandas as pd

from spam_detector.data import deduplicate, split_dataset


def make_df() -> pd.DataFrame:
    rows = [(f"win a prize now {i}", 1) for i in range(20)]
    rows += [(f"see you at lunch {i}", 0) for i in range(80)]
    rows += [("see you at lunch 0", 0)] * 5  # duplicates
    return pd.DataFrame(rows, columns=["text", "label"])


def test_deduplicate_removes_repeated_messages():
    df = deduplicate(make_df())
    assert len(df) == 100
    assert df["text"].is_unique


def test_split_has_no_overlap_between_train_and_test():
    train, test = split_dataset(deduplicate(make_df()))
    assert set(train["text"]).isdisjoint(set(test["text"]))


def test_split_preserves_spam_ratio():
    df = deduplicate(make_df())
    train, test = split_dataset(df)
    assert abs(train["label"].mean() - df["label"].mean()) < 0.02
    assert abs(test["label"].mean() - df["label"].mean()) < 0.02


def test_split_is_reproducible():
    df = deduplicate(make_df())
    a, _ = split_dataset(df, seed=1)
    b, _ = split_dataset(df, seed=1)
    assert a.equals(b)

from pathlib import Path

from analyzer import analyze


# Load only the development split.
project_root = Path(__file__).resolve().parent
dev_path = project_root / "data" / "dev.csv"


def print_results(label, use_custom):
    """Print low-rated and high-rated false-negative shares."""
    dev = analyze(str(dev_path), use_custom=use_custom)

    low_rated = dev.loc[dev["rating"].le(2)]
    high_rated = dev.loc[dev["rating"].ge(4)]

    low_negative_share = low_rated["sentiment"].eq("negative").mean()
    high_negative_share = high_rated["sentiment"].eq("negative").mean()

    print(f"{label}:")
    print(
        f"  Rating <= 2 predicted negative: {low_negative_share:.1%}"
    )
    print(
        f"  Rating >= 4 predicted negative: {high_negative_share:.1%}"
    )


# Compare plain VADER with the custom complaint-word lexicon.
print_results("Plain VADER", use_custom=False)
print_results("Custom VADER", use_custom=True)

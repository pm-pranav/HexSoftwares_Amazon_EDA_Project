# src/helpers.py
# ---------------------------------------------
# Helper utility functions for Amazon EDA Project
# ---------------------------------------------

import os
import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path):
    """Create folder if not exists."""
    os.makedirs(path, exist_ok=True)


def save_plot(fig, path, dpi=150):
    """Save matplotlib plot to file with directory check."""
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, bbox_inches='tight', dpi=dpi)


def load_csv(filepath):
    """Load CSV file and return Pandas DataFrame."""
    return pd.read_csv(filepath)


# ---------------- Test Block ----------------
# (You can run this file directly to test all helper functions)
if __name__ == "__main__":
    print("🔹 Testing helper functions...")

    # 1️⃣ Test directory creation
    ensure_dir("outputs/test_dir")
    print("✅ Folder created successfully!")

    # 2️⃣ Test CSV loading
    csv_path = r"C:\Users\prana\Amazon_EDA_Project\Data\amazon_synthetic_large.csv"
    if os.path.exists(csv_path):
        df = load_csv(csv_path)
        print(f"✅ CSV loaded successfully! Shape: {df.shape}")
    else:
        print("⚠️ CSV file not found! Check your Data folder path.")

    # 3️⃣ Test plot saving
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [2, 4, 6], marker='o', color='green')
    ax.set_title("Test Plot - Amazon EDA")
    save_plot(fig, "outputs/test_dir/sample_plot.png")
    plt.close(fig)
    print("✅ Plot saved successfully in outputs/test_dir/sample_plot.png")

    print("🎉 All helper functions tested successfully!")
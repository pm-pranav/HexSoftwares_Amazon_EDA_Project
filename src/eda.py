# src/eda.py
# ---------------------------------------------
# Amazon EDA Project - Exploratory Data Analysis Script
# ---------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import helper functions
from src.helpers import ensure_dir, save_plot, load_csv

# ---------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------
PROJECT_ROOT = r"C:\Users\prana\Amazon_EDA_Project"
DATA_PATH = os.path.join(PROJECT_ROOT, "Data", "amazon_synthetic_large.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
DERIVED_DIR = os.path.join(OUTPUT_DIR, "data")

ensure_dir(PLOTS_DIR)
ensure_dir(DERIVED_DIR)

sns.set(style="whitegrid")

# ---------------------------------------------
# DATA CLEANING FUNCTIONS
# ---------------------------------------------
def parse_dates(df):
    """Convert date columns to datetime if available."""
    for col in ['order_date', 'ship_date', 'review_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def basic_cleaning(df):
    """Remove duplicates and strip text columns."""
    df = df.drop_duplicates()
    text_cols = df.select_dtypes(include='object').columns
    for c in text_cols:
        df[c] = df[c].astype(str).str.strip()
    return df


def handle_missing_values(df):
    """Handle missing numeric and categorical values."""
    print("\n🔹 Missing Values Report:")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0])

    # Fill numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        df[c].fillna(df[c].median(), inplace=True)

    # Fill categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    for c in cat_cols:
        df[c].fillna("Unknown", inplace=True)

    return df


def feature_engineering(df):
    """Create new meaningful features."""
    if 'order_date' in df.columns and 'ship_date' in df.columns:
        df['delivery_days'] = (df['ship_date'] - df['order_date']).dt.days

    if 'review_date' in df.columns and 'order_date' in df.columns:
        df['review_delay_days'] = (df['review_date'] - df['order_date']).dt.days

    if 'review_text' in df.columns:
        df['review_length'] = df['review_text'].astype(str).apply(len)

    if 'total_amount' in df.columns and 'quantity' in df.columns:
        df['price_per_item'] = df['total_amount'] / df['quantity'].replace(0, 1)

    return df


# ---------------------------------------------
# EXPLORATORY ANALYSIS
# ---------------------------------------------
def exploratory_analysis(df):
    """Perform descriptive stats and visualizations."""
    print("\n📊 Dataset Shape:", df.shape)
    print("\n📋 Columns:", list(df.columns))
    print("\n📈 Data Types:\n", df.dtypes)

    # Descriptive statistics
    desc = df.describe(include='all')
    desc.to_csv(os.path.join(DERIVED_DIR, "dataset_description.csv"))
    print("\n✅ Saved dataset_description.csv")

    # --- Distribution of Numeric Columns ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:6]:  # limit to 6 main numeric cols
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df[col].dropna(), bins=40, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        save_plot(fig, os.path.join(PLOTS_DIR, f"{col}_distribution.png"))
        plt.close(fig)

    # --- Correlation Heatmap ---
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title("Correlation Heatmap")
        save_plot(fig, os.path.join(PLOTS_DIR, "correlation_heatmap.png"))
        plt.close(fig)
        corr.to_csv(os.path.join(DERIVED_DIR, "correlation_matrix.csv"))
        print("✅ Saved correlation_matrix.csv")

    # --- Top Categories ---
    if 'category' in df.columns:
        cat_counts = df['category'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(y=cat_counts.index, x=cat_counts.values, ax=ax)
        ax.set_title("Top 10 Product Categories")
        save_plot(fig, os.path.join(PLOTS_DIR, "top_categories.png"))
        plt.close(fig)

    # --- Rating Distribution ---
    if 'rating' in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='rating', data=df, ax=ax)
        ax.set_title("Rating Distribution")
        save_plot(fig, os.path.join(PLOTS_DIR, "rating_distribution.png"))
        plt.close(fig)

    # --- Price vs Rating Scatter ---
    if 'price' in df.columns and 'rating' in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(x='price', y='rating', data=df.sample(min(2000, len(df))), alpha=0.6)
        ax.set_title("Price vs Rating (Sampled)")
        save_plot(fig, os.path.join(PLOTS_DIR, "price_vs_rating.png"))
        plt.close(fig)

    # --- Review Length by Rating ---
    if 'review_length' in df.columns and 'rating' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='rating', y='review_length', data=df, ax=ax)
        ax.set_title("Review Length by Rating")
        save_plot(fig, os.path.join(PLOTS_DIR, "review_length_by_rating.png"))
        plt.close(fig)

    print("\n✅ All plots saved successfully in outputs/plots/")


# ---------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------
def main():
    print("🚀 Starting Amazon EDA Project...\n")
    print(f"📂 Loading dataset from: {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        print("❌ Dataset not found! Please check your path.")
        return

    # Step 1: Load data
    df = load_csv(DATA_PATH)
    print(f"✅ Data loaded successfully! Shape: {df.shape}")

    # Step 2: Parse and clean data
    df = parse_dates(df)
    df = basic_cleaning(df)
    df = handle_missing_values(df)

    # Step 3: Feature engineering
    df = feature_engineering(df)

    # Step 4: Save cleaned data
    cleaned_path = os.path.join(DERIVED_DIR, "cleaned_amazon.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"✅ Cleaned dataset saved at: {cleaned_path}")

    # Step 5: Exploratory analysis
    exploratory_analysis(df)

    print("\n🎉 EDA Completed Successfully!")
    print(f"📊 Check plots in: {PLOTS_DIR}")
    print(f"📁 Check data outputs in: {DERIVED_DIR}")


if __name__ == "__main__":
    main()
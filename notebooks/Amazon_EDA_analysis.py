# notebooks/Amazon_EDA_analysis.py
# --------------------------------------------------
# Amazon EDA Project - Jupyter Compatible Script
# Author: Pranav Mistry
# --------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------
PROJECT_ROOT = r"C:\Users\prana\Amazon_EDA_Project"
DATA_PATH = os.path.join(PROJECT_ROOT, "Data", "amazon_synthetic_large.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
DERIVED_DIR = os.path.join(OUTPUT_DIR, "data")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DERIVED_DIR, exist_ok=True)

sns.set(style="whitegrid")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
print("📂 Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("✅ Data loaded successfully! Shape:", df.shape)

# --------------------------------------------------
# BASIC INFO
# --------------------------------------------------
print("\n📋 Columns:", list(df.columns))
print("\n🔹 Missing values:")
print(df.isna().sum()[df.isna().sum() > 0])

# --------------------------------------------------
# CLEANING
# --------------------------------------------------
df = df.drop_duplicates()
obj_cols = df.select_dtypes(include='object').columns
for c in obj_cols:
    df[c] = df[c].astype(str).str.strip()

for c in ['price', 'total_amount', 'quantity', 'shipping_cost', 'rating']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# --------------------------------------------------
# DATE PARSING + FEATURE CREATION
# --------------------------------------------------
for col in ['order_date', 'ship_date', 'review_date']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

if 'order_date' in df.columns and 'ship_date' in df.columns:
    df['delivery_days'] = (df['ship_date'] - df['order_date']).dt.days

if 'review_date' in df.columns and 'order_date' in df.columns:
    df['review_delay_days'] = (df['review_date'] - df['order_date']).dt.days

if 'review_text' in df.columns:
    df['review_length'] = df['review_text'].astype(str).apply(len)

if 'total_amount' in df.columns and 'quantity' in df.columns:
    df['price_per_item'] = df['total_amount'] / df['quantity'].replace(0, 1)

# --------------------------------------------------
# HANDLE MISSING VALUES
# --------------------------------------------------
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include='object').columns

for c in num_cols:
    df[c].fillna(df[c].median(), inplace=True)

for c in cat_cols:
    df[c].fillna("Unknown", inplace=True)

print("\n✅ Missing values handled.")

# --------------------------------------------------
# SAVE CLEANED DATA
# --------------------------------------------------
cleaned_path = os.path.join(DERIVED_DIR, "cleaned_amazon.csv")
df.to_csv(cleaned_path, index=False)
print("💾 Cleaned dataset saved at:", cleaned_path)

# --------------------------------------------------
# DESCRIPTIVE STATISTICS
# --------------------------------------------------
desc = df.describe().T
desc_path = os.path.join(DERIVED_DIR, "numeric_summary.csv")
desc.to_csv(desc_path)
print("📊 Saved numeric summary at:", desc_path)

# --------------------------------------------------
# VISUALIZATIONS
# --------------------------------------------------

# 1️⃣ Distribution plots
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols[:6]:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col].dropna(), bins=40, kde=True)
    plt.title(f"Distribution of {col}")
    out_file = os.path.join(PLOTS_DIR, f"{col}_dist.png")
    plt.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close()

# 2️⃣ Correlation heatmap
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Heatmap")
    heatmap_path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches='tight', dpi=150)
    plt.close()

# 3️⃣ Category distribution
if 'category' in df.columns:
    top_cats = df['category'].value_counts().head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(y=top_cats.index, x=top_cats.values)
    plt.title("Top 15 Categories")
    cat_path = os.path.join(PLOTS_DIR, "top_categories.png")
    plt.savefig(cat_path, bbox_inches='tight', dpi=150)
    plt.close()

# 4️⃣ Rating vs Review length
if 'review_length' in df.columns and 'rating' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='rating', y='review_length', data=df)
    plt.title("Review Length by Rating")
    box_path = os.path.join(PLOTS_DIR, "review_length_by_rating.png")
    plt.savefig(box_path, bbox_inches='tight', dpi=150)
    plt.close()

# 5️⃣ Price vs Rating scatter
if 'price' in df.columns and 'rating' in df.columns:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x='price', y='rating', data=df.sample(min(2000, len(df))), alpha=0.6)
    plt.title("Price vs Rating (Sample)")
    scatter_path = os.path.join(PLOTS_DIR, "price_vs_rating.png")
    plt.savefig(scatter_path, bbox_inches='tight', dpi=150)
    plt.close()

# 6️⃣ Top products by revenue
if 'product_id' in df.columns and 'total_amount' in df.columns:
    agg = df.groupby(['product_id']).agg(
        total_revenue=('total_amount', 'sum'),
        total_qty=('quantity', 'sum'),
        avg_rating=('rating', 'mean')
    ).sort_values('total_revenue', ascending=False).head(20)
    agg_path = os.path.join(DERIVED_DIR, "top_products_by_revenue.csv")
    agg.to_csv(agg_path)
    print("📁 Saved top products data:", agg_path)

print("\n🎉 Amazon EDA Analysis Completed Successfully!")
print("📂 Plots saved in:", PLOTS_DIR)
print("📂 Data saved in:", DERIVED_DIR)
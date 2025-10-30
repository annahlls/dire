#!/usr/bin/env python3
"""
Cross-matrix analysis of nsubj (person_number) and iobj (iobj_person_number) for "dire" forms.
Reads the CSV output from explore_dire_comprehensive.py and creates detailed matrix tables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def load_data(csv_path):
    """Load the comprehensive dire analysis CSV data."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total records loaded: {len(df)}")
    return df

def create_nsubj_iobj_matrix(df):
    """Create cross-tabulation matrix between nsubj (person_number) and iobj (iobj_person_number)."""
    print("\n=== NSUBJ × IOBJ CROSS-MATRIX ANALYSIS ===")
    
    # Filter out records where person_number is None or invalid
    valid_df = df[df['person_number'].notna() & (df['person_number'] != 'None_None')]
    
    print(f"Records with valid nsubj (person_number): {len(valid_df)}")
    print(f"Records with valid iobj (iobj_person_number): {len(valid_df[valid_df['iobj_person_number'].notna() & (valid_df['iobj_person_number'] != 'None')])}")
    
    # Create the cross-tabulation
    matrix = pd.crosstab(
        valid_df['person_number'], 
        valid_df['iobj_person_number'], 
        margins=True,
        margins_name="Total"
    )
    
    return matrix, valid_df

def create_percentage_matrix(matrix):
    """Create percentage version of the matrix."""
    # Calculate percentages (excluding margins)
    matrix_no_margins = matrix.iloc[:-1, :-1]  # Remove Total row and column
    total_count = matrix_no_margins.sum().sum()
    
    percentage_matrix = (matrix_no_margins / total_count * 100).round(2)
    
    # Add margins back with percentages
    percentage_matrix.loc['Total'] = percentage_matrix.sum()
    percentage_matrix['Total'] = percentage_matrix.sum(axis=1)
    
    return percentage_matrix

def create_row_percentage_matrix(matrix):
    """Create row percentage version of the matrix (percentage within each nsubj)."""
    matrix_no_margins = matrix.iloc[:-1, :-1]  # Remove Total row and column
    
    # Calculate row percentages
    row_percentage_matrix = matrix_no_margins.div(matrix_no_margins.sum(axis=1), axis=0) * 100
    row_percentage_matrix = row_percentage_matrix.round(2)
    
    # Add totals
    row_percentage_matrix['Total'] = 100.0  # Each row sums to 100%
    
    return row_percentage_matrix

def create_col_percentage_matrix(matrix):
    """Create column percentage version of the matrix (percentage within each iobj)."""
    matrix_no_margins = matrix.iloc[:-1, :-1]  # Remove Total row and column
    
    # Calculate column percentages
    col_percentage_matrix = matrix_no_margins.div(matrix_no_margins.sum(axis=0), axis=1) * 100
    col_percentage_matrix = col_percentage_matrix.round(2)
    
    # Add totals
    col_percentage_matrix.loc['Total'] = 100.0  # Each column sums to 100%
    
    return col_percentage_matrix

def analyze_patterns(df, matrix):
    """Analyze interesting patterns in the nsubj × iobj matrix."""
    print("\n=== PATTERN ANALYSIS ===")
    
    # Most common nsubj-iobj combinations
    valid_df = df[df['person_number'].notna() & 
                  (df['person_number'] != 'None_None') & 
                  df['iobj_person_number'].notna() & 
                  (df['iobj_person_number'] != 'None')]
    
    if len(valid_df) > 0:
        combinations = valid_df.groupby(['person_number', 'iobj_person_number']).size().sort_values(ascending=False)
        print(f"\nTop 10 NSUBJ × IOBJ combinations:")
        for (nsubj, iobj), count in combinations.head(10).items():
            percentage = (count / len(valid_df)) * 100
            print(f"  {nsubj} → {iobj}: {count} occurrences ({percentage:.1f}%)")
        
        # Analyze reflexive patterns (when nsubj and iobj have same person)
        print(f"\nReflexive patterns (same person for nsubj and iobj):")
        reflexive_count = 0
        for (nsubj, iobj), count in combinations.items():
            nsubj_person = nsubj.split('_')[0] if '_' in nsubj else nsubj
            iobj_person = iobj.split('_')[0] if '_' in iobj else iobj
            if nsubj_person == iobj_person:
                reflexive_count += count
                percentage = (count / len(valid_df)) * 100
                print(f"  {nsubj} → {iobj}: {count} occurrences ({percentage:.1f}%)")
        
        reflexive_percentage = (reflexive_count / len(valid_df)) * 100
        print(f"\nTotal reflexive patterns: {reflexive_count} ({reflexive_percentage:.1f}%)")
        
        # Analyze cross-person patterns
        cross_person_count = len(valid_df) - reflexive_count
        cross_person_percentage = (cross_person_count / len(valid_df)) * 100
        print(f"Total cross-person patterns: {cross_person_count} ({cross_person_percentage:.1f}%)")
    
    else:
        print("No valid nsubj-iobj combinations found.")

def create_visualizations(matrix, output_dir):
    """Create visualizations for the nsubj × iobj matrix."""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Heatmap of absolute counts
    plt.figure(figsize=(12, 8))
    matrix_plot = matrix.iloc[:-1, :-1]  # Remove Total row and column for cleaner visualization
    sns.heatmap(matrix_plot, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title('NSUBJ × IOBJ Cross-Matrix (Absolute Counts)', fontsize=14, fontweight='bold')
    plt.xlabel('IOBJ Person_Number', fontsize=12)
    plt.ylabel('NSUBJ Person_Number', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/nsubj_iobj_matrix_counts.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap of percentages
    percentage_matrix = create_percentage_matrix(matrix)
    plt.figure(figsize=(12, 8))
    percentage_plot = percentage_matrix.iloc[:-1, :-1]  # Remove Total row and column
    sns.heatmap(percentage_plot, annot=True, fmt='.1f', cmap='Oranges', cbar_kws={'label': 'Percentage (%)'})
    plt.title('NSUBJ × IOBJ Cross-Matrix (Percentages)', fontsize=14, fontweight='bold')
    plt.xlabel('IOBJ Person_Number', fontsize=12)
    plt.ylabel('NSUBJ Person_Number', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/nsubj_iobj_matrix_percentages.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Row percentages (within each nsubj)
    row_percentage_matrix = create_row_percentage_matrix(matrix)
    plt.figure(figsize=(12, 8))
    row_percentage_plot = row_percentage_matrix.iloc[:, :-1]  # Remove Total column
    sns.heatmap(row_percentage_plot, annot=True, fmt='.1f', cmap='Greens', cbar_kws={'label': 'Row Percentage (%)'})
    plt.title('NSUBJ × IOBJ Cross-Matrix (Row Percentages)', fontsize=14, fontweight='bold')
    plt.xlabel('IOBJ Person_Number', fontsize=12)
    plt.ylabel('NSUBJ Person_Number', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/nsubj_iobj_matrix_row_percentages.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

def save_matrices_to_csv(matrix, output_dir):
    """Save all matrix versions to CSV files."""
    print("\n=== SAVING MATRICES TO CSV ===")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save absolute counts matrix
    matrix.to_csv(f"{output_dir}/nsubj_iobj_matrix_counts.csv")
    
    # Save percentage matrix
    percentage_matrix = create_percentage_matrix(matrix)
    percentage_matrix.to_csv(f"{output_dir}/nsubj_iobj_matrix_percentages.csv")
    
    # Save row percentage matrix
    row_percentage_matrix = create_row_percentage_matrix(matrix)
    row_percentage_matrix.to_csv(f"{output_dir}/nsubj_iobj_matrix_row_percentages.csv")
    
    # Save column percentage matrix
    col_percentage_matrix = create_col_percentage_matrix(matrix)
    col_percentage_matrix.to_csv(f"{output_dir}/nsubj_iobj_matrix_col_percentages.csv")
    
    print(f"Matrix CSV files saved to: {output_dir}")

def print_detailed_report(matrix, df):
    """Print detailed statistical report."""
    print("\n" + "="*60)
    print("DETAILED NSUBJ × IOBJ MATRIX REPORT")
    print("="*60)
    
    # Absolute counts matrix
    print("\n1. ABSOLUTE COUNTS MATRIX:")
    print(matrix)
    
    # Percentage matrix
    print("\n2. PERCENTAGE MATRIX (% of total):")
    percentage_matrix = create_percentage_matrix(matrix)
    print(percentage_matrix)
    
    # Row percentage matrix
    print("\n3. ROW PERCENTAGE MATRIX (% within each NSUBJ):")
    row_percentage_matrix = create_row_percentage_matrix(matrix)
    print(row_percentage_matrix)
    
    # Column percentage matrix
    print("\n4. COLUMN PERCENTAGE MATRIX (% within each IOBJ):")
    col_percentage_matrix = create_col_percentage_matrix(matrix)
    print(col_percentage_matrix)
    
    # Summary statistics
    valid_df = df[df['person_number'].notna() & 
                  (df['person_number'] != 'None_None')]
    
    with_iobj = valid_df[valid_df['iobj_person_number'].notna() & 
                        (valid_df['iobj_person_number'] != 'None')]
    
    print(f"\n5. SUMMARY STATISTICS:")
    print(f"   Total valid NSUBJ forms: {len(valid_df)}")
    print(f"   Forms with IOBJ: {len(with_iobj)} ({len(with_iobj)/len(valid_df)*100:.1f}%)")
    print(f"   Forms without IOBJ: {len(valid_df) - len(with_iobj)} ({(len(valid_df) - len(with_iobj))/len(valid_df)*100:.1f}%)")
    
    # Most common NSUBJ values
    print(f"\n6. MOST COMMON NSUBJ VALUES:")
    nsubj_counts = valid_df['person_number'].value_counts()
    for nsubj, count in nsubj_counts.head(10).items():
        percentage = (count / len(valid_df)) * 100
        print(f"   {nsubj}: {count} ({percentage:.1f}%)")
    
    # Most common IOBJ values
    print(f"\n7. MOST COMMON IOBJ VALUES:")
    iobj_counts = with_iobj['iobj_person_number'].value_counts()
    for iobj, count in iobj_counts.head(10).items():
        percentage = (count / len(with_iobj)) * 100
        print(f"   {iobj}: {count} ({percentage:.1f}%)")

def main():
    """Main execution function."""
    # Define paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / "visualizations" / "dire_comprehensive_data.csv"
    output_dir = script_dir / "visualizations" / "nsubj_iobj_matrix"
    
    # Check if CSV exists
    if not csv_path.exists():
        print(f"ERROR: CSV file not found at {csv_path}")
        print("Please run explore_dire_comprehensive.py first to generate the data.")
        return
    
    # Load data
    df = load_data(csv_path)
    
    # Create cross-matrix
    matrix, valid_df = create_nsubj_iobj_matrix(df)
    
    # Print detailed report
    print_detailed_report(matrix, df)
    
    # Analyze patterns
    analyze_patterns(df, matrix)
    
    # Create visualizations
    create_visualizations(matrix, output_dir)
    
    # Save matrices to CSV
    save_matrices_to_csv(matrix, output_dir)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"All outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()

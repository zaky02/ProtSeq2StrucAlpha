import pandas as pd
from statistics import mode, median, stdev
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import time


def analysis_proteins(prots):

    data = pd.read_csv(prots)

    # Get the length of the sequences
    aa_seq_len = data['aa_seq'].apply(lambda x: len(x)).tolist()
    data['aa_seq_len'] = aa_seq_len
    struc_seq_len = data['struc_seq'].apply(lambda x: len(x)).tolist()
    data['struc_seq_len'] = struc_seq_len
    
    # Get the mean length of the sequences
    aa_mean_len = sum(aa_seq_len)/len(aa_seq_len)
    print(f"Mean length of the amino acid sequences: {aa_mean_len}")

    # Get the mode length of the sequences
    aa_mode_len = mode(aa_seq_len)
    print(f"Mode length of the amino acid sequences: {aa_mode_len}")

    # Get the median length of the sequences
    aa_median_len = median(aa_seq_len)
    print(f"Median length of the amino acid sequences: {aa_median_len}")
    
    # Get the standard deviation of the lengths
    aa_std_len = stdev(aa_seq_len)
    print(f"Standard Deviation of the length of the amino acid sequences: {aa_std_len}")
    
    # Plot the distribution of the lengths
    sns.displot(aa_seq_len, kde=True)
    plt.savefig('aa_seq_len_hist.pdf')
    
    # Plot a boxplot of the lengths and check for outliers
    plt.figure(figsize=(4, 10))
    sns.boxplot(aa_seq_len)
    plt.savefig('aa_seq_len_boxplot.pdf')
    
    # Detect outliers
    q1 = data['aa_seq_len'].quantile(0.25)
    q3 = data['aa_seq_len'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    print(lower_bound, upper_bound)
    
    # Remove outliers from the data
    data = data[data['aa_seq_len'] < upper_bound]
    
    # Plot the distribution of the lengths without outliers
    sns.displot(data['aa_seq_len'], kde=True)
    plt.savefig('aa_seq_len_hist_filt.pdf')
    
    # Get metrics of filtered data
    print("----------METRICS OF THE FILTERED DATA--------------\n")

    # Get the mean length of the sequences
    aa_seq_len = data['aa_seq_len'].tolist()
    aa_mean_len = sum(aa_seq_len)/len(aa_seq_len)
    print(f"Mean length of the amino acid sequences: {aa_mean_len}")

    # Get the mode length of the sequences
    aa_mode_len = mode(aa_seq_len)
    print(f"Mode length of the amino acid sequences: {aa_mode_len}")

    # Get the median length of the sequences
    aa_median_len = median(aa_seq_len)
    print(f"Median length of the amino acid sequences: {aa_median_len}")
    
    # Get the standard deviation of the lengths
    aa_std_len = stdev(aa_seq_len)
    print(f"Standard Deviation of the length of the amino acid sequences: {aa_std_len}")
    
    # Plot boxplot without outliers
    plt.figure(figsize=(4, 10))
    sns.boxplot(aa_seq_len)
    plt.savefig('aa_seq_len_boxplot_filt.pdf')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    requiredArguments = parser.add_argument_group('Required Arguments')
    requiredArguments.add_argument('--prots',
                                   help='',
                                   required=True)
    args = parser.parse_args()

    # Load arguments
    prots = args.prots

    analysis_proteins(prots)

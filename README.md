## Overview
This repository contains a collection of scripts aimed at predicting Foldseek structural tokens based on a protein's amino acid sequence. The model used for this task is a multi-head attention encoder-decoder transformer, a powerful architecture known for its success in sequence-to-sequence tasks. The goal is to take the amino acid sequence of a protein as input and predict its corresponding Foldseek structural token sequence as output.

One of the key advantages of using Foldseek structural tokens is that it allows us to optimize memory and storage requirements. By representing protein structures as tokens rather than storing the entire 3D models (like in PDB files), we can significantly reduce the amount of data needed for storing and processing structural information, while still retaining essential structural features.

## Input and Output
Input: Protein's amino acid sequence.
Output: Corresponding sequence of Foldseek structural tokens.

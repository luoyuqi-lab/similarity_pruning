# Similarity Pruning: Accelerating 1NN-DTW Classification While Improving Accuracy

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
Paper Link: [TBD]()

This repository contains the official implementation of the paper **"Similarity Pruning: Accelerating 1NN-DTW Classification While Improving Accuracy"**.

## 📖 Abstract

The 1-Nearest Neighbor Dynamic Time Warping (1NN-DTW) classifier remains a cornerstone in time series analysis. While generalizing to the $k$NN framework is often hypothesized to enhance robustness, our comprehensive evaluation on 96 UCR datasets reveals a counter-intuitive reality: increasing $k$ frequently yields no accuracy benefit while incurring significant computational costs due to the weakening of lower bounds.

Driven by the hypothesis that a single, highly similar neighbor often renders broader neighborhood information superfluous, we introduce **Similarity Pruning (SP)**. Unlike traditional cell-level or time-series-level optimizations, SP operates at the **dataset-level**, dynamically terminating the search process upon identifying templates that exhibit sufficient similarity. This strategy is strictly orthogonal to and compatible with existing acceleration techniques (e.g., LB_Keogh).

Extensive experiments demonstrate that SP simultaneously accelerates the 1NN-DTW baseline (**1.31× speedup**) and improves classification accuracy (**2.34% gain on 55 datasets**). Furthermore, we demonstrate the versatility of SP by successfully extending it to 1NN-ED and 1NN-ShapeDTW classifiers.

## ✨ Key Highlights

- **Counter-intuitive Finding:** Large-scale experiments reveal that increasing $k$ in DTW classifiers often weakens lower bound efficiency without improving accuracy.
- **Novel Methodology:** Proposed **Similarity Pruning (SP)**, a dataset-level strategy that dynamically terminates the search upon identifying sufficiently similar templates.
- **Dual Improvement:** Proven to simultaneously enhance both the classification accuracy and computational efficiency of the 1NN-DTW baseline.
- **High Versatility:** Operates orthogonally to existing optimizations and demonstrates significant improvements when extended to 1NN-ED and 1NN-ShapeDTW.

## 🚀 Getting Started

### Prerequisites

The code is implemented in Python. We recommend using `conda` to manage your environment.

```bash
# Clone the repository
git clone [https://github.com/luoyuqi-lab/similarity_pruning.git](https://github.com/luoyuqi-lab/similarity_pruning.git)
cd similarity_pruning

# Install dependencies
pip install -r requirements.txt

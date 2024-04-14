# CMPT 353 Final Project: Smartphone Data Analysis

## Purpose

This project analyzes smartphone data to explore and answer the following questions:

- **Q1:** Explore what is measured under smartphone ratings. Try to find out if there is a correlation between price, tech specifications, and user ratings. We can use correlation analysis or linear regression for this.
- **Q2:** Does 5G affect smartphone prices? Analyze the adoption rate of 5G technology in smartphones over price segments.
- **Q3:** Analyze the smartphone market distribution by brand, price segments, and key features. Which brand produces the most expensive smartphones on average? What is the average rating per brand?
- **Q4:** Develop a model that predicts the price of a smartphone based on its specifications.
- **Q5:** What is the average lifespan of smartphones based on their specifications?

## Getting Started

Follow these instructions to get the project up and running on your local machine for development and tested purposes.


### Running the Project

python3 cmpt353_FinalProject.py

### Expected Output

Upon successful execution, the following files and console outputs will be generated as a result of various analyses conducted:

- 1.	correlation_analysis.png
- 2.	5G and Prices.png
- 3.	linear_regression.png
- 4.	price_comparison.png
- 5.	rating_comparison.png
- 6.	Console output:
- •	Price Segment Distribution
- •	Regression Analysis Summary
- •	T-test Results
- •	Model Performance Metrics
- •	Lifespan Model Coefficients and Metrics


### Prerequisites

This project requires the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

Install these packages using pip. It is recommended to use a virtual environment:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels

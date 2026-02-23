# AAI-530 Team Project — Smart City Energy Consumption Analysis

## Team Members

- Maxime Boulat
- Wael Alhatal

## Overview

This project analyzes residential electricity consumption patterns using the [Smart Meters in London](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london/) dataset from Kaggle. The dataset contains real-world energy consumption records collected from over 5,000 smart meters across London between November 2011 and February 2014, as part of the UK Power Networks Low Carbon London project.

The goal is to explore the factors driving household energy demand and build predictive models that can support smart city energy planning.

## Dataset

The dataset includes:
- `daily_dataset.csv` — daily electricity consumption per household (kWh)
- `information_households.csv` — household metadata including ACORN socio-demographic classification and tariff type
- `acorn_details.csv` — ACORN group descriptions
- Supplemental daily weather data joined during analysis

## Project Structure

```
├── Notebooks/
│   └── Smart_meters_in_London.ipynb   # Main analysis notebook
├── build/
│   ├── datasets/                      # Place Kaggle data files here (see setup)
│   └── results/                       # Model prediction exports (generated on run)
└── Readme.md
```

## Notebook Contents

The main notebook walks through the full analysis pipeline:

1. **Data loading and cleaning** — date standardization, household ramp-up bias correction, normalization to per-household daily averages
2. **Exploratory Data Analysis** — seasonality, weather impact, ACORN socio-demographic segmentation, tariff type comparison, holiday effects, and temperature × ACORN interaction analysis
3. **Model 1: Neural Network** — feedforward classifier predicting seasonally adjusted consumption tiers (Low / Normal / High) per ACORN group, intended as a peak risk signal for grid operators
4. **Model 2: LSTM** — stacked LSTM forecasting average daily household consumption 3 days ahead, intended as a grid planning tool

## Setup

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london/)
2. Create a `build/datasets/` folder at the root of the project
3. Place `acorn_details.csv`, `daily_dataset.csv`, and `information_households.csv` inside it
4. Open and run `Notebooks/Smart_meters_in_London.ipynb` in VSCode

## Requirements

The notebook was developed in Python 3 with the following main dependencies:

- pandas
- numpy
- matplotlib / seaborn
- scikit-learn
- tensorflow / keras

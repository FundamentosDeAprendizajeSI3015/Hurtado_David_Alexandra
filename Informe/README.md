# Data Preprocessing and Exploratory Analysis Pipeline

This project implements a data cleaning, transformation, and exploratory data analysis (EDA) pipeline based on a survey about the use of Artificial Intelligence tools in decision-making.

The objective is to transform raw data into a clean dataset ready for modeling in a Machine Learning course project.

---

## Technologies Used

- Python 3
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## Pipeline Stages

1. **Data loading** from an Excel file.
2. **Column cleaning and normalization.**
3. **Handling missing values.**
4. **Variable encoding** (ordinal encoding and One Hot Encoding).
5. **Scaling** using `StandardScaler`.
6. **Exploratory Data Analysis (EDA)** with statistics and visualizations.
7. **Exporting** the final processed dataset.

---

## Main Files

- `pipelineInforme1.py` → Main script.
- `DatosEncuestaInforme1.xlsx` → Original dataset.
- `requirements.txt` → Project dependencies.

---

## How to Run

```bash
pip install -r requirements.txt
python pipelineInforme1.py


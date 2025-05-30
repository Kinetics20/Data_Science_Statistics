
# 📈 Data Science Statistics

![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![UV](https://img.shields.io/badge/Dependency%20Manager-UV-blueviolet)

## 📄 Project Overview

This repository contains a collection of statistical exercises designed for Data Science learning and practice.  
It demonstrates core statistical concepts through interactive Jupyter Notebooks, using Python libraries for data manipulation, visualization, and statistical modeling.

# 📚 Examples

### 📊 Central Limit Theorem - Sampling Animation (CTL.ipynb)

This example demonstrates the **Central Limit Theorem** through a simple animated visualization.  
It shows how the distribution of sample means tends toward a normal distribution as more samples are drawn from an original uniform distribution.

Key features:
- **Animated histogram** of sample means evolving over time
- **Saved as an animated GIF** for easy viewing
- **Educational visualization** of the Central Limit Theorem process

Generated output preview:  
[![CLT Sampling Animation](https://github.com/Kinetics20/Data_Science_Statistics/raw/main/assets/central_limit_theorem_3.gif)](https://github.com/Kinetics20/Data_Science_Statistics/blob/main/assets/central_limit_theorem_3.gif)

---

### 📈 Central Limit Theorem - Interactive Animation (CTL_for_web_.ipynb)

This example generates an **interactive animated visualization** illustrating the **Central Limit Theorem**.  
We repeatedly draw samples from a uniform distribution, calculate their means, and show how the distribution of these means becomes approximately normal as more samples are collected.

Key features:
- **Three synchronized plots**:
  - Distribution of sample means (top left)
  - Current sample distribution (top right)
  - Original population distribution (bottom)
- **Animated frames** updating dynamically with each new sample
- **Interactive controls** (start button) for running the animation
- **Saved as an HTML file** for easy sharing and embedding in web projects

Generated output preview:  
[![Interactive CLT Animation](https://github.com/Kinetics20/Data_Science_Statistics/raw/main/assets/central_limit_theorem_interactive_for_web_2.gif)](https://github.com/Kinetics20/Data_Science_Statistics/blob/main/assets/central_limit_theorem_interactive_for_web.html)

---

### 🎻 Bimodal Distribution Visualization (workshop/bimodal_dist.ipynb)

This exercise focuses on visualizing a **bimodal distribution** using multiple types of statistical plots:
- **Boxplot**: Showing distribution spread with the mean highlighted.
- **Violin plot**: Displaying the probability density function.
- **Histogram**: Representing frequency of occurrences.

Key features:
- **Combined mosaic layout** for side-by-side comparison
- **Customized styles** like dashed mean lines and dotted median bars
- **Clear axis labels** for better interpretation
- **Useful for exploratory data analysis (EDA)**

Generated output preview:  
[![Bimodal Distribution Visualization](https://github.com/Kinetics20/Data_Science_Statistics/raw/main/assets/bimodal_dist.png)](https://github.com/Kinetics20/Data_Science_Statistics/blob/main/assets/bimodal_dist.png)

---

## 🛠️ Libraries Used

- `numpy`
- `pandas`
- `scipy`
- `seaborn`
- `matplotlib`
- `statsmodels`
- `plotly`
- `bokeh`
- `ipympl`
- `ipython`
- `jupyterlab`

## 📂 Project Structure

```
Data_Science_Statistics/
├── assets/                        ← Images and GIFs
├── datasets/                      ← CSV datasets
├── html/                          ← HTML documents
├── workshop/                      ← Workshop practice notebooks
│   └── bimodal_dist.ipynb          ← Bimodal distribution visualization
├── CTL.ipynb                      ← Central Limit Theorem sampling animation
├── CTL_for_web_.ipynb              ← Central Limit Theorem interactive animation
├── descriptive_visualisation.ipynb← Descriptive statistics visualizations
├── intro.ipynb                    ← Introduction notebook
├── pebble_world.ipynb              ← Toy example for sampling exercises
├── pyproject.toml                  ← Project dependencies
├── README.md                       ← Project documentation
└── uv.lock                         ← Lock file for uv
```

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone git@github.com:Kinetics20/Data_Science_Statistics.git
   cd Data_Science_Statistics
   ```

2. Install dependencies using [uv](https://github.com/astral-sh/uv):
   ```bash
   uv sync
   ```

3. Launch JupyterLab:
   ```bash
   jupyter lab
   ```

4. Open and explore the `.ipynb` notebooks.

## 📊 Topics Covered

- Central Limit Theorem
- Data Visualization
- Descriptive Statistics
- Distribution Analysis
- Sampling Techniques
- Exploratory Data Analysis (EDA)

---

## 💬 Feedback

Contributions and suggestions are welcome!

👤 Author: Piotr Lipiński  
🗓 Date: May 2025

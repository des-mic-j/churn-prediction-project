# Customer Churn Analysis Project

## Project Overview
This project explores customer churn data using **Python, Pandas, NumPy, and Matplotlib**.  
The goal is to analyze patterns in customer behavior, identify factors linked to churn, and build a professional portfolio project that demonstrates my ability to work with data from start to finish.

---

## Project Steps & Progress Log

### Step 1: Environment Setup
- Created a new virtual environment (`venv`) for the project.  
- Installed the required libraries: `pandas`, `numpy`, and `matplotlib`.  
- Verified everything was working by running a test import.  
 *Result: Environment successfully set up.*

---

### Step 2: Importing Data
- Loaded the Telco Customer Churn dataset into a Pandas DataFrame (`df`).  
- Began exploring columns and data types.  
 *Result: Data successfully imported and ready for cleaning.*

---

### Step 3: Data Cleaning
- Found that `TotalCharges` had mixed data types (strings + numbers).  
- Converted `TotalCharges` into numeric using `pd.to_numeric()` with `errors="coerce"` to handle invalid values.  
- Checked for missing or incorrect entries.  
 *Result: Cleaned dataset with consistent column types.*

---

### Step 4: First Visualization
- Created a **bar chart** showing churn by **Contract Type**.  
- Used `matplotlib` to visualize differences in churn rates between monthly, yearly, and two-year contracts.
   *Result: The chart clearly shows that monthly contracts have the highest churn rate.*

---

### Step 5: Adding More Visuals
- Expanded analysis by adding churn visualizations for:
  - Payment Method  
  - Internet Service Type  
  - Gender and Senior Citizen status  
- Compared churn rates across categories to see where risk is highest.  
 *Result: Built a stronger picture of customer churn patterns.*

---

### Step 6: Insights & Next Steps
- Noticed clear churn risks among:
  - Customers on **month-to-month contracts**  
  - Customers with **electronic check payments**  
  - Customers using **fiber optic internet**  
- Next steps:  
  - Explore correlations between multiple variables (e.g., contract + payment type).  
  - Consider building a simple churn prediction model with logistic regression or decision trees.  

*Result: Project shows meaningful business insights and sets the stage for advanced modeling.*

---

## Project Goals
- Strengthen my skills in **data cleaning, visualization, and storytelling**.  
- Build a project that recruiters and hiring managers can quickly understand and see value in.  
- Use this analysis as a foundation for adding **machine learning** in the future.  

---

## Tools Used
- **Python**  
- **Pandas, NumPy** (data manipulation & analysis)  
- **Matplotlib** (data visualization)  
- Jupyter/VS Code (development environment)  

---

## Next Updates
- Add interactive visualizations (Plotly/Seaborn).  
- Document key findings with more business-focused insights.  
- Upload notebook version for recruiters to explore step-by-step code.  

---

![attrition icon](https://blogimage.vantagecircle.com/vcblogimage/en/2025/12/Why-Should-We-Address-Attrition.png)

<h1>
<p align="center">
Employee Attrition Analysis
</p>

## Table of Contents

1. [Project Overview & Goal](#1-project-overview--goal)
2. [Tools & Technologies](#2-tools--technologies)
3. [Target Audience](#3-target-audience)
4. [Expected Deliverables](#4-expected-deliverables)
5. [Data Transformation Summary](#5-data-transformation-summary)
6. [Key Findings and Insights](#6-key-findings-and-insights)
7. [How to Run the Project](#7-how-to-run-the-project)
8. [Team Members & Roles (HackTeam-1)](#8-team-members--roles-hackteam-1)
9. [Next Steps](#9-next-steps)
10. [Data Ethics](#10-data-ethics)
11. [Credits](#11-credits)


## 1. Project Overview & Goal

### Dataset & Source
This project utilises the [*IBM HR Analytics Employee Attrition & Performance*](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data) sourced from Kaggle.

This synthetic dataset looks into factors that lead to employee attrition. It allows for a detailed exploration to help answer questions such as whether job satisfaction is impacted by an employees years at a company/current job/last promotion etc.

### Project's Focus

As a company with over a thousand employees our requirement for this analysis looks at the problem and a potential solution:

#### Business Problem: 
High employee turnover is costing organisations in the range of 150-200% of annual salary per departure, disrupting operations and eroding institutional knowledge. Still, organisations are reacting to resignations instead of preventing them.

#### Solution: 
Implement a dual ML system that: 
1. Uses K-means clustering to segment employees into risk regimes (Low/Moderate/High/Critical) based on patterns in the data, with optimal K determined through elbow and silhouette validation;
2. Employs binary classification to predict individual attrition probability


#### Project Objectives

1. **Objective 1 (Inferential Statistics: EDA): Identify and determine key attrition drivers through statistical analysis of HR data patterns**

    - **Chi-square test:**
      - Attrition x Department (is the attrition rate different by department?)
      - Attrition x Business Travel
    - **T-test:** 
    - Monthly Income between leavers vs stayers (do leavers earn less?)
    
    
2. **Objective 2 ETL: Segment employees into risk regimes using K-means clustering (optimal K determined via elbow & silhouette methods)**

    - **Elbow Method:**
        - To pick candidate K values
    - **Silhouette Analysis:**
        - Find the optimal number of clusters (K)

3. **Objective 3 ETL: Develop a predictive model for individual attrition probability using binary classification**

    - **Confusion Matrix:**
        - Accuracy, precision, recall

<br>

## 2. Tools & Technologies

- **Github Project Board**
- **Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-Learn** 
- **Jupyter Notebook** 
- **GitHub (Project Board and Version Control)**
- **Streamlit (interactive dashboard)**

<br>


## 3. Target Audience

The Streamlit app was designed to be an interactive tool intended to be used by:

**HR Department/Company Managers -** 
Managers and HR Departments can use this dashboard to:

- Identify key factors that link with employee attrition.
- This could provide insight into how to minimise the employee churn rate.

<br>

## 4. Expected Deliverables

What the final outputs for the project should be:

- A cleaned and structured dataset ready for analysing.
- A Streamlit dashboard, interactive with data visuals and a machine learning model
- Project board and README documentation to include key findings, insights and conclusions of this project

<br>

## 5. Data Transformation Summary

**The following features were dropped from the dataframe as they were redundant during the analysis:**

**Employee Number**
**Employee Count** 
**Over18**
**Standard Hours**

<br>

## 6. Key Findings and Insights

### Key Findings

- During the EDA process we wanted to see if there was any correlation between the following key features:
    - Performance Rating and Number of Trainings
    - Salary Hike Percent and Job Satisfaction

- Based on the results of the heatmap analysis we can see that there is no correlation or relationship existing between Performance rating and the number of training in the last year.
There is also no correlation between Salary hike and Job satisfaction.
However, there is a strong 0.77 correlation coefficient between Salary Hike percent and Perfomance Rating, which means there is strong linear relationship between salary hike and performance rating. Generally, higher salary hike results to higher performance rating and vice versa.

- There is a string positive relationship between Job Level and Monthly Income with correlation coefficient of 0.95.

### Machine Learning Summary

Features of Importance 
Years At Company, 0.460932
Total Working Years, 0.360192
Monthly Income, 0.085259
Years With Current Manager, 0.050519
Years In Current Role, 0.043097
Clusters frequencies
Cluster	Proportion	Percentage
1	0.57	57%
2	0.28	28%
0	0.15	15%

<br>


## 7. How to Run the Project

### How to Run the Project Locally

#### Clone the Repository

```bash
git clone [https://github.com/MushafauBaba/employee-attrition]

cd employee-attrition
```

#### Install Dependencies

You will need a ```requirements.txt``` file listing pandas, numpy, streamlit, etc.
Open your terminal or a Jupyter cell and run:

```bash
pip install -r requirements.txt
```

#### Run the Streamlit app

```bash
streamlit run app.py
```

#### Run the Notebook

Open both `employee-attrition.ipynb` and `employee-attrition-ml.ipynb` respectively and run all cells sequentially. The notebooks will automatically download the data, run the ETL pipeline, and generate all seaborn/matplotlib visualizations.

### How to Deploy Streamlit App 

This application is deployed publicly using **Streamlit Community Cloud**.

The deployment process is as follows:

1. The project code is hosted in a public GitHub repository.
2. A `requirements.txt` file is included in the repository to specify all Python dependencies required to run the app.
3. Streamlit Community Cloud automatically installs these dependencies in a clean cloud environment.
4. The application is launched using the main entry file `app.py`, with additional pages loaded from the `pages/` directory.

The app is available at:
**Deployed Link**

Doing the above helps to make sure that the app can be used publicly, easily accessible, and therefore not reliant on a local Python environment.

<br>


## 8. Team Members & Roles (HackTeam-1)

The below are the members of HackTeam 1 and their roles during the project:

### Data Architect - Babatunde & Arphaxad

- Designed the structure of the dataset for use during analysis.
- Involved in the ETL pipeline, including cleaning, transforming and feature engineering.
- Prepared data ready for Machine Learning

### Data Analysts - Anisa & Babatunde

- Led the exploratory of the data for analysis and to be able to formulate research questions.
- Created visualisations to find trends and support insights.

### Data Scientist - Arphaxad

- Builder of the prediction model
- Identify key drivers 
- Interpret the results of the model

### Dashboard Developer - Arphaxad & Lola

- Developed an interactive dashboard with visualisations.
- Creator of user friendly Streamlit app with filters for exploring the data.
- App allows for visual storytelling.

### Project Management

- This was primarily led by Lola with some shared responsibility with the rest of the team in the following:

- Planning of the project overview and goal.
- Tasks were managed and tracked in the Github Project Board.
- Organisation of meet ups, task distribution and documentation.
- Preparing final slide presentation

<br>

## 9. Next Steps

### Things we would do if time permitted:

- Do more in-depth exploratory of the data as there are 31 features
- Classification during the machine learning process


<br>

## 10. Data Ethics

This dataset was created by IBM and did not contain any personal or private information.

<br>


## 11. Credits

- ChatGPT for debugging and understanding pieces of code.
- Youtube for how to create and use branches in Github.
- Used Capstone projects by Baba, Lola and Arpha to help with the Machine Learning and Streamlit app.

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
6. [How to Run the Project](#6-how-to-run-the-project)
7. [Team Members & Roles (HackTeam-1)](#7-team-members--roles-hackteam-1)
8. [Credits](#8-credits)


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

- **Trello**
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

<br>

## 5. Data Transformation Summary

<br>

## 6. How to Run the Project

<br>



## 7. Team Members & Roles (HackTeam-1)

The below are the members of HackTeam 1 and their roles during the project:

### Data Artchitect - Babatunde & Arphaxad

- Helped designed the structure of the dataset for use during analysis.
- Involved in the ETL process, including cleaning, transforming and feature engineering.
- Prepared data ready for Machine Learning.

### Data Analysts - Anisa & Babatunde

- Led the exploratory of the data for analysis and to be able to formulate research questions.
- Created visualisations to find trends and support insights.

### Dashboard Developer - Arphaxad & Lola

- Developed an interactive dashboard with visualisations.
- Streamlit app was created with user friendly layout and filters for exploring the data.
- App allowed for visual storytelling.

### Project Management

- This was primarily led by Lola but with some shared responsibility with the rest of the team in the following:

- Planning of the project overview and goal.
- Tasks were managed and tracked in the Github Project Board.
- Organisation of meet ups, task distribution and documentation.
- Preparing final slide presentation

<br>

## 8. Credits

- ChatGPT for debugging and understanding pieces of code.
- Youtube for how to create and use branches in Github.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, norm, bartlett, chi2_contingency, ttest_ind, ttest_rel, f_oneway, kruskal, mannwhitneyu, mode, skew, kurtosis
import math

# --- Page 1: Sample Size Calculator ---
def page_sample_size_calculator():
    st.title("Sample Size Calculator")
    st.write("Determine the sample size needed for your study based on key statistical parameters.")

    with st.form(key="sample_size_form"):
        alpha = st.number_input("Enter alpha (e.g., 0.05):", 0.01, 1.0, 0.05, 0.01)
        beta = st.number_input("Enter beta (e.g., 0.2 for 80% power):", 0.01, 1.0, 0.2, 0.01)
        allocation_ratio = st.number_input("Enter allocation ratio (e.g., 1 for equal groups):", 0.1, 10.0, 1.0, 0.1)
        effect_size = st.number_input("Enter effect size (e.g., 0.3 for small effect):", 0.01, 5.0, 0.3, 0.05)
        
        submit_button = st.form_submit_button(label="Calculate")

    if submit_button:
        # Using the core logic from your notebook
        try:
            n = (norm.ppf(1 - alpha / 2) + norm.ppf(1 - beta))**2 / (effect_size**2)
            
            # This is a simplified formula for Cohen's d. 
            # A more robust calculation for two independent groups:
            n_group1 = (1 + 1/allocation_ratio) * (norm.ppf(1 - alpha/2) + norm.ppf(1 - beta))**2 / effect_size**2
            n_group2 = allocation_ratio * n_group1
            
            total_n = n_group1 + n_group2
            
            st.success(f"**Recommended total sample size: {math.ceil(total_n)}**")
            st.write(f" - Group 1 (Control): {math.ceil(n_group1)}")
            st.write(f" - Group 2 (Intervention): {math.ceil(n_group2)}")
        except Exception as e:
            st.error(f"An error occurred during calculation: {e}")

# --- Page 2: Statistical Test Recommender ---
def page_test_recommender():
    st.title("Statistical Test Recommender")
    st.write("Answer a few questions to get a recommendation for the most appropriate statistical test.")

    st.subheader("1. What kind of data are you measuring?")
    data_type = st.radio(
        "Select the type for your *dependent variable* (the outcome you are measuring).",
        ("Continuous (e.g., blood pressure, weight, scores)", 
         "Discrete / Categorical (e.g., counts, gender, yes/no)")
    )

    if data_type == "Discrete / Categorical (e.g., counts, gender, yes/no)":
        st.info("Based on discrete data, you should likely use a **Chi-Square Test** to check for relationships or differences in proportions.")
    
    else: # Continuous Data
        st.subheader("2. What is your primary goal?")
        goal = st.radio(
            "What are you trying to do with your data?",
            ("Compare groups (e.g., intervention vs. control)", 
             "Analyze relationships (e.g., does blood pressure increase with age)")
        )

        if goal == "Analyze relationships (e.g., does blood pressure increase with age)":
            st.info("To analyze relationships, you should use a **Correlation** or **Regression** analysis.")
            st.write("- Use **Pearson's r** if your data is normally distributed.")
            st.write("- Use **Spearman's Rank** if your data is not normally distributed.")
        
        else: # Compare Groups
            st.subheader("3. How many groups are you comparing?")
            num_groups = st.radio(
                "How many distinct patient groups or time points?",
                ("2 groups (or 2 time points, e.g., pre/post)", 
                 "More than 2 groups")
            )
            
            is_paired = st.checkbox("Are the groups paired? (e.g., pre-test vs. post-test on the *same* patient, matched pairs)")
            
            if num_groups == "2 groups (or 2 time points, e.g., pre/post)":
                if is_paired:
                    st.success("Recommended Test: **Paired t-test**")
                    st.write("*(Use **Wilcoxon Signed-Rank Test** if data is not normally distributed)*")
                else:
                    st.success("Recommended Test: **Student's unpaired t-test**")
                    st.write("*(Use **Mann-Whitney U Test** if data is not normally distributed)*")
            
            else: # More than 2 groups
                if is_paired:
                    st.success("Recommended Test: **Repeated Measures ANOVA**")
                    st.write("*(Use **Friedman Test** if data is not normally distributed)*")
                else:
                    st.success("Recommended Test: **One-Way ANOVA**")
                    st.write("*(Use **Kruskal-Wallis Test** if data is not normally distributed)*")


# --- Page 3: Data Analysis ---
def page_data_analysis():
    st.title("Data Analysis")
    st.write("Upload your CSV file to get descriptive statistics and a normality test for any column.")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(data.head())
            
            # Get only numeric columns for analysis
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            
            if not numeric_cols:
                st.warning("No numeric columns found in the CSV to analyze.")
                return

            column = st.selectbox("Select a column to analyze", numeric_cols)
            
            if column:
                values = data[column].dropna()
                
                st.subheader(f"Statistical Analysis for: {column}")
                
                # Descriptive Statistics
                mean = np.mean(values)
                median = np.median(values)
                std_dev = np.std(values)
                variance = np.var(values)
                skewness = skew(values)
                kurt = kurtosis(values)
                
                # Mode calculation (Fixed from your notebook)
                # stats.mode now returns a ModeResult object
                mode_result = mode(values)

                st.write(f"**Mean:** {mean:.2f}")
                st.write(f"**Median:** {median:.2f}")
                st.write(f"**Mode:** {mode_result.mode} (appears {mode_result.count} times)")
                st.write(f"**Standard Deviation:** {std_dev:.2f}")
                st.write(f"**Variance:** {variance:.2f}")
                st.write(f"**Skewness:** {skewness:.2f}")
                st.write(f"**Kurtosis:** {kurt:.2f}")

                # Normality Test
                st.subheader("Normality Test (Shapiro-Wilk)")
                if len(values) > 2:
                    shapiro_stat, shapiro_p = shapiro(values)
                    st.write(f"P-value: {shapiro_p:.3f}")
                    if shapiro_p > 0.05:
                        st.success("The data appears to be normally distributed (p > 0.05).")
                    else:
                        st.error("The data does **not** appear to be normally distributed (p <= 0.05).")
                else:
                    st.warning("Not enough data (need > 2 values) to run Shapiro-Wilk test.")

                # Visualization
                st.subheader("Histogram")
                fig, ax = plt.subplots()
                ax.hist(values, bins=20, edgecolor='k')
                ax.set_title(f"Histogram of {column}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            st.write("Please ensure your file is a correctly formatted CSV.")

# --- Main App Navigation ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the tool you want to use:",
    ["Sample Size Calculator", "Statistical Test Recommender", "Data Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Dr. Subham Sarkar")
st.sidebar.caption("Resident Doctor, AIIMS Bhubaneswar")

if app_mode == "Sample Size Calculator":
    page_sample_size_calculator()
elif app_mode == "Statistical Test Recommender":
    page_test_recommender()
elif app_mode == "Data Analysis":
    page_data_analysis()
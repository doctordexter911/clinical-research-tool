import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, norm, ttest_ind, ttest_rel, f_oneway, kruskal, mannwhitneyu, mode, skew, kurtosis
import math
from groq import Groq

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
        try:
            n_group1 = (1 + 1/allocation_ratio) * (norm.ppf(1 - alpha/2) + norm.ppf(1 - beta))**2 / effect_size**2
            n_group2 = allocation_ratio * n_group1
            total_n = n_group1 + n_group2
            st.success(f"**Recommended total sample size: {math.ceil(total_n)}**")
            st.write(f" - Group 1 (Control): {math.ceil(n_group1)}")
            st.write(f" - Group 2 (Intervention): {math.ceil(n_group2)}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- Page 2: Statistical Test Recommender ---
def page_test_recommender():
    st.title("Statistical Test Recommender")
    st.write("Answer a few questions to get a recommendation for the most appropriate statistical test.")

    st.subheader("1. What kind of data are you measuring?")
    data_type = st.radio("Select the type for your dependent variable.",
        ("Continuous (e.g., blood pressure, weight, scores)",
         "Discrete / Categorical (e.g., counts, gender, yes/no)"))

    if data_type == "Discrete / Categorical (e.g., counts, gender, yes/no)":
        st.info("Use a **Chi-Square Test** to check for relationships or differences in proportions.")
    else:
        st.subheader("2. What is your primary goal?")
        goal = st.radio("What are you trying to do with your data?",
            ("Compare groups (e.g., intervention vs. control)",
             "Analyze relationships (e.g., does blood pressure increase with age)"))

        if goal == "Analyze relationships (e.g., does blood pressure increase with age)":
            st.info("Use **Correlation** or **Regression** analysis.\n- Pearson's r (normal data)\n- Spearman's Rank (non-normal data)")
        else:
            st.subheader("3. How many groups are you comparing?")
            num_groups = st.radio("How many distinct patient groups or time points?",
                ("2 groups (or 2 time points, e.g., pre/post)", "More than 2 groups"))
            is_paired = st.checkbox("Are the groups paired? (same patient pre vs. post)")

            if num_groups == "2 groups (or 2 time points, e.g., pre/post)":
                if is_paired:
                    st.success("Recommended Test: **Paired t-test**")
                    st.write("*(Use Wilcoxon Signed-Rank Test if not normally distributed)*")
                else:
                    st.success("Recommended Test: **Student's unpaired t-test**")
                    st.write("*(Use Mann-Whitney U Test if not normally distributed)*")
            else:
                if is_paired:
                    st.success("Recommended Test: **Repeated Measures ANOVA**")
                    st.write("*(Use Friedman Test if not normally distributed)*")
                else:
                    st.success("Recommended Test: **One-Way ANOVA**")
                    st.write("*(Use Kruskal-Wallis Test if not normally distributed)*")

# --- Page 3: Data Analysis ---
def page_data_analysis():
    st.title("Data Analysis")
    st.write("Upload your CSV file to get descriptive statistics and a normality test.")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(data.head())
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.warning("No numeric columns found.")
                return

            column = st.selectbox("Select a column to analyze", numeric_cols)
            if column:
                values = data[column].dropna()
                st.subheader(f"Statistical Analysis for: {column}")
                mode_result = mode(values)
                st.write(f"**Mean:** {np.mean(values):.2f}")
                st.write(f"**Median:** {np.median(values):.2f}")
                st.write(f"**Mode:** {mode_result.mode} (appears {mode_result.count} times)")
                st.write(f"**Standard Deviation:** {np.std(values):.2f}")
                st.write(f"**Variance:** {np.var(values):.2f}")
                st.write(f"**Skewness:** {skew(values):.2f}")
                st.write(f"**Kurtosis:** {kurtosis(values):.2f}")

                st.subheader("Normality Test (Shapiro-Wilk)")
                if len(values) > 2:
                    _, p = shapiro(values)
                    st.write(f"P-value: {p:.3f}")
                    if p > 0.05:
                        st.success("Data appears normally distributed (p > 0.05).")
                    else:
                        st.error("Data does NOT appear normally distributed (p ≤ 0.05).")

                st.subheader("Histogram")
                fig, ax = plt.subplots()
                ax.hist(values, bins=20, edgecolor='k')
                ax.set_title(f"Histogram of {column}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

# --- Page 4: PICO Novelty Checker ---
def page_pico_novelty():
    st.title("🔬 PICO Novelty Checker & Research Objective Generator")
    st.write("Enter your PICO details. The AI will identify novelty, research gaps, and suggest objectives.")

    # API Key input in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 Groq API Key")
    api_key = st.sidebar.text_input(
        "Paste your Groq API Key here",
        type="password",
        help="Free API key from console.groq.com"
    )
    st.sidebar.caption("🔒 Your key is never stored.")
    st.sidebar.caption("Get a free key at console.groq.com")

    st.markdown("---")
    st.subheader("Fill in Your PICO")

    col1, col2 = st.columns(2)
    with col1:
        population = st.text_area("**P — Population / Patient**",
            placeholder="e.g., Adult ICU patients with Type 2 Diabetes Mellitus", height=100)
        intervention = st.text_area("**I — Intervention**",
            placeholder="e.g., Continuous glucose monitoring (CGM)", height=100)
    with col2:
        comparison = st.text_area("**C — Comparison (optional)**",
            placeholder="e.g., Standard fingerstick glucose monitoring", height=100)
        outcome = st.text_area("**O — Outcome**",
            placeholder="e.g., Hypoglycaemic episodes, ICU length of stay, 30-day mortality", height=100)

    with st.expander("➕ Optional: Add more context for better results"):
        study_design = st.selectbox("Intended Study Design",
            ["Not specified", "RCT", "Cohort Study", "Case-Control",
             "Cross-sectional", "Systematic Review / Meta-analysis", "Pilot Study"])
        clinical_setting = st.text_input("Clinical Setting / Specialty",
            placeholder="e.g., AIIMS Bhubaneswar, Tertiary care, ICU, Paediatrics")
        additional_context = st.text_area("Additional context or constraints",
            placeholder="e.g., Resource-limited setting, India, specific subgroup...", height=80)

    st.markdown("---")

    if st.button("🚀 Analyse Novelty & Generate Objectives", use_container_width=True, type="primary"):

        if not api_key:
            st.error("⚠️ Please enter your Groq API Key in the sidebar. Get one free at console.groq.com")
            return
        if not population or not intervention or not outcome:
            st.warning("⚠️ Please fill in at least P (Population), I (Intervention), and O (Outcome).")
            return

        pico_summary = f"""
- Population: {population}
- Intervention: {intervention}
- Comparison: {comparison if comparison else 'Not specified'}
- Outcome: {outcome}
- Study Design: {study_design}
- Clinical Setting: {clinical_setting if clinical_setting else 'Not specified'}
- Additional Context: {additional_context if additional_context else 'None'}
"""

        prompt = f"""You are an expert clinical research advisor and academic mentor helping a clinician researcher at a leading medical institution in India.

Based on the following PICO framework, provide a structured academic research analysis:

{pico_summary}

Respond in this exact structured format:

## 1. Summary of the Research Question
Write a clear 2-3 sentence summary of what this research investigates.

## 2. What is Already Known (Existing Evidence)
Summarise what is established in the literature. Mention specific types of studies that exist.

## 3. Research Gaps Identified
List 3-5 specific, clinically relevant gaps this PICO addresses.

## 4. Novelty of This Research
Explain what makes this novel — consider population specificity, intervention, outcomes, setting, and study design.

## 5. Suggested Research Objectives
- **Primary Objective:** (1 clear, measurable objective)
- **Secondary Objectives:** (2-4 objectives)

## 6. Research Hypothesis
- **Null Hypothesis (H₀):**
- **Alternative Hypothesis (H₁):**

## 7. Potential Limitations
List 3-4 potential limitations.

## 8. Recommended Databases & Search Terms
Suggest 4-5 databases and relevant MeSH/key search terms.

Be specific, clinically grounded, and academically rigorous. Focus on the Indian clinical research context where relevant."""

        with st.spinner("🔍 Analysing your PICO using Llama 3 (Groq)... usually takes 5-10 seconds"):
            try:
                client = Groq(api_key=api_key)
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-70b-8192",
                    temperature=0.7,
                    max_tokens=2000,
                )
                result = chat_completion.choices[0].message.content

                st.success("✅ Analysis complete!")
                st.markdown("---")
                st.markdown(result)

                st.markdown("---")
                st.download_button(
                    label="📥 Download Full Analysis as Text File",
                    data=f"PICO NOVELTY ANALYSIS\n{'='*50}\n\nPICO Details:\n{pico_summary}\n\n{'='*50}\n\nANALYSIS:\n\n{result}",
                    file_name="pico_novelty_analysis.txt",
                    mime="text/plain"
                )

            except Exception as e:
                err = str(e)
                if "invalid_api_key" in err.lower() or "401" in err:
                    st.error("❌ Invalid API Key. Please check your Groq API key at console.groq.com")
                elif "rate_limit" in err.lower() or "429" in err:
                    st.error("❌ Rate limit reached. Please wait a moment and try again.")
                else:
                    st.error(f"❌ Error: {err}")

# --- Navigation ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the tool:",
    ["Sample Size Calculator", "Statistical Test Recommender", "Data Analysis", "PICO Novelty Checker"]
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
elif app_mode == "PICO Novelty Checker":
    page_pico_novelty()
```


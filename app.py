import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

# Set page configuration
st.set_page_config(
    page_title="India Literacy Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    url="https://raw.githubusercontent.com/Rupsa9647/How_Educated_India/main/Datasets/final_clean_dataset.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a section:", 
                          ["Overview", 
                           "Gender Analysis", 
                           "State-wise Analysis", 
                           "Education Levels", 
                           "Age Analysis", 
                           "Rural vs Urban", 
                           "Clustering Analysis", 
                           "Forecasting"])

# Main content
st.title("📚 India Literacy Dashboard (1991-2011)")

if options == "Overview":
    st.header("Overall Literacy Trends")
    
    # Overall literacy rate calculation for each year
    literacy_rate = (
        df.groupby("Year")
        .apply(lambda x: (x["Literate Person"].sum() / x["Total Person"].sum()) * 100)
        .reset_index(name="Literacy Rate (%)")
    )
    
    # Total Literacy Rate across all years
    total_literacy_rate = (df["Literate Person"].sum() / df["Total Person"].sum()) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Literacy Rate", f"{total_literacy_rate:.2f}%")
    
    # Plot trend line
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(literacy_rate["Year"], literacy_rate["Literacy Rate (%)"], marker="o", linestyle="-", linewidth=2)
    ax.set_title("Literacy Rate Trend in India (1990–2011)", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Literacy Rate (%)", fontsize=12)
    ax.grid(True)
    ax.set_xticks([1991, 2001, 2011])
    
    st.pyplot(fig)

elif options == "Gender Analysis":
    st.header("Gender-wise Literacy Analysis")
    
    # Calculate overall Male and Female Literacy Rates
    male_literacy_rate = (df["Literate Males"].sum() / df["Total Males"].sum()) * 100
    female_literacy_rate = (df["Literate Females"].sum() / df["Total Females"].sum()) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Male Literacy Rate", f"{male_literacy_rate:.2f}%")
    with col2:
        st.metric("Female Literacy Rate", f"{female_literacy_rate:.2f}%")
        st.metric("Gender Gap", f"{male_literacy_rate - female_literacy_rate:.2f}%")
    
    # Calculate male and female literacy rate per year
    male_lit = (
        df.groupby("Year")
        .apply(lambda x: (x["Literate Males"].sum() / x["Total Males"].sum()) * 100)
        .reset_index(name="Male Literacy Rate (%)")
    )
    
    female_lit = (
        df.groupby("Year")
        .apply(lambda x: (x["Literate Females"].sum() / x["Total Females"].sum()) * 100)
        .reset_index(name="Female Literacy Rate (%)")
    )
    
    # Merge for plotting
    gender_lit = male_lit.merge(female_lit, on="Year")
    
    # Plot trend lines
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gender_lit["Year"], gender_lit["Male Literacy Rate (%)"], marker="o", label="Male Literacy Rate")
    ax.plot(gender_lit["Year"], gender_lit["Female Literacy Rate (%)"], marker="o", label="Female Literacy Rate")
    
    # Plot gap as shaded area
    ax.fill_between(
        gender_lit["Year"], 
        gender_lit["Male Literacy Rate (%)"], 
        gender_lit["Female Literacy Rate (%)"], 
        color="lightblue", alpha=0.3, label="Gender Gap"
    )
    
    # Labels and title
    ax.set_title("Male vs Female Literacy Rate in India (1991–2011)", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Literacy Rate (%)", fontsize=12)
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)

elif options == "State-wise Analysis":
    st.header("State-wise Literacy Analysis")
    
    # Function to calculate literacy rate for each state in a given year
    def state_literacy_rate(df):
        return (df["Literate Person"].sum() / df["Total Person"].sum()) * 100

    # Group by State and Year
    state_lit = (
        df.groupby(["Year", "State Name"])
        .apply(state_literacy_rate)
        .reset_index(name="Literacy Rate (%)")
    )

    # Sort values within each year
    state_lit_sorted = state_lit.sort_values(["Year", "Literacy Rate (%)"], ascending=[True, False])

    # Top and bottom states each year
    year_select = st.selectbox("Select Year:", [1991, 2001, 2011])
    
    top2_states = state_lit_sorted[state_lit_sorted["Year"] == year_select].head(2)
    bottom2_states = state_lit_sorted[state_lit_sorted["Year"] == year_select].tail(2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Top 2 States in {year_select}")
        st.dataframe(top2_states)
    
    with col2:
        st.subheader(f"Bottom 2 States in {year_select}")
        st.dataframe(bottom2_states)
    
    # Heatmap of state-wise literacy rates
    st.subheader("State-wise Literacy Rate Heatmap")
    
    # 1. compute state-wise literacy rate per year
    def state_literacy_rate_func(group):
        total = group["Total Person"].sum()
        if total == 0:
            return 0.0
        return (group["Literate Person"].sum() / total) * 100

    state_lit = (
        df.groupby(["State Name", "Year"])
        .apply(state_literacy_rate_func)
        .reset_index(name="Literacy Rate (%)")
    )

    # 2. pivot to make a matrix: rows = states, cols = years
    pivot = state_lit.pivot(index="State Name", columns="Year", values="Literacy Rate (%)")

    # optional: sort states by 2011 literacy desc for nicer display
    if 2011 in pivot.columns:
        pivot = pivot.sort_values(by=2011, ascending=False)
    else:
        pivot = pivot.sort_index()

    # 3. plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot, annot=True, fmt=".1f", linewidths=0.3, cbar_kws={"label": "Literacy Rate (%)"}, ax=ax)
    ax.set_title("State-wise Literacy Rate (%) — Heatmap")
    ax.set_xlabel("Year")
    ax.set_ylabel("State")
    
    st.pyplot(fig)

elif options == "Education Levels":
    st.header("Education Level Analysis")
    
    # Select only education-related columns
    education_cols = [
        "Literate without educational level Person", "Below primary Person", "Primary Person",
        "Middle Person", "Secondary Person", "Higher secondary Person",
        "Non-technical diploma Person", "Technical diploma Person",
        "Graduate Person", "Unclassified Person"
    ]
    
    # Group by Year and sum education columns
    edu_distribution_yearly = df.groupby("Year")[education_cols].sum()
    
    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bar_width = 0.25  # width of each bar
    years = edu_distribution_yearly.index
    x = np.arange(len(years))  # positions for groups
    
    # Plot each education level as a separate set of bars
    for i, col in enumerate(education_cols):
        ax.bar(x + i*bar_width, edu_distribution_yearly[col], 
               width=bar_width, label=col)
    
    # Formatting
    ax.set_title("Education Levels by Year", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Total Population", fontsize=12)
    ax.set_xticks(x + bar_width * (len(education_cols) / 2))
    ax.set_xticklabels(years)
    ax.legend(title="Education Level", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Education retention curve
    st.subheader("Education Retention Curve")
    
    # Define education stages in sequential order
    education_path = [
        "Below primary Person",
        "Primary Person",
        "Middle Person",
        "Secondary Person",
        "Higher secondary Person",
        "Non-technical diploma Person",
        "Technical diploma Person",
        "Graduate Person"
    ]
    
    # Group by Year and sum for each stage
    edu_counts_yearly = df.groupby("Year")[education_path].sum()
    
    # Plot retention curve year-wise
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for year, row in edu_counts_yearly.iterrows():
        retention = (row / row.iloc[0]) * 100   # retention relative to Below Primary
        ax.plot(education_path, retention, marker="o", linestyle="-", linewidth=2, label=f"Year {year}")
    
    ax.set_title("Education Retention Curve (Year-wise)", fontsize=14)
    ax.set_xlabel("Education Level", fontsize=12)
    ax.set_ylabel("Retention (%)", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    
    st.pyplot(fig)

elif options == "Age Analysis":
    st.header("Age-wise Analysis")
    
    # Group by Age and Year (using Total Persons for analysis)
    age_year_data = df.groupby(["Year", "Age"])["Total Person"].sum().unstack(fill_value=0)
    
    # Plot stacked area chart
    fig, ax = plt.subplots(figsize=(12, 7))
    age_year_data.T.plot.area(ax=ax, alpha=0.8)
    
    ax.set_title("Age-wise Analysis (Stacked Area Chart) for All Years", fontsize=14)
    ax.set_xlabel("Age Groups")
    ax.set_ylabel("Total Persons")
    ax.legend(title="Year", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    
    st.pyplot(fig)

elif options == "Rural vs Urban":
    st.header("Rural vs Urban Literacy Comparison")
    
    # Compute literacy rates
    df["Total Literacy Rate"] = df["Literate Person"] / df["Total Person"]
    df["Male Literacy Rate"] = df["Literate Males"] / df["Total Males"]
    df["Female Literacy Rate"] = df["Literate Females"] / df["Total Females"]
    
    # Filter only Rural and Urban areas
    rural_urban_df = df[df["Area"].isin(["Rural", "Urban"])]
    
    # Group by Year and Area
    rural_urban_summary = rural_urban_df.groupby(["Year", "Area"])[
        ["Total Literacy Rate", "Male Literacy Rate", "Female Literacy Rate"]
    ].mean().reset_index()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=rural_urban_summary, x="Year", y="Total Literacy Rate", hue="Area", ax=ax)
    ax.set_title("Rural vs Urban Total Literacy Rate (1991, 2001, 2011)", fontsize=14)
    ax.set_ylabel("Literacy Rate")
    
    st.pyplot(fig)
    
    # Show data table
    st.subheader("Rural vs Urban Literacy Data")
    st.dataframe(rural_urban_summary)

elif options == "Clustering Analysis":
    st.header("State Clustering Based on Literacy Rates")
    url="https://raw.githubusercontent.com/Rupsa9647/How_Educated_India/main/Datasets/All_Ages_And_Total_Area_Data.csv"
    df=pd.read_csv(url)
    # Compute Literacy Rate
    df["Literacy Rate"] = (df["Literate Person"] / df["Total Person"]) * 100
    
    # Group by State and Year
    state_year_lit = df.groupby(["State Name", "Year"], as_index=False)["Literacy Rate"].mean()
    
    # Apply KMeans clustering per year
    clustered_data = []
    
    for year, group in state_year_lit.groupby("Year"):
        X = group[["Literacy Rate"]].values
        
        # Fit KMeans with 3 clusters
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        group["Cluster"] = kmeans.fit_predict(X)
        
        # Manually assign cluster labels (Low, Medium, High) based on average literacy in each cluster
        cluster_means = group.groupby("Cluster")["Literacy Rate"].mean().sort_values()
        cluster_mapping = {cluster_means.index[0]: "Low", 
                           cluster_means.index[1]: "Medium", 
                           cluster_means.index[2]: "High"}
        
        group["Cluster Label"] = group["Cluster"].map(cluster_mapping)
        clustered_data.append(group)
    
    # Combine results
    clustered_df = pd.concat(clustered_data)
    
    # Compare clusters across years
    pivot_clusters = clustered_df.pivot(index="State Name", columns="Year", values="Cluster Label")
    
    # Visualization - Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_clusters.replace({"Low": 0, "Medium": 1, "High": 2}), 
                cmap="Set2", annot=pivot_clusters, fmt="", 
                cbar_kws={"label": "Cluster (0=Low, 1=Medium, 2=High)"}, ax=ax)
    ax.set_title("State Cluster Membership Across Years (Based on Literacy Rate)")
    ax.set_ylabel("State")
    ax.set_xlabel("Year")
    
    st.pyplot(fig)
    
    # Scatter plot for selected year
    year_select = st.selectbox("Select Year for Cluster Visualization:", [1991, 2001, 2011])
    
    group = clustered_df[clustered_df["Year"] == year_select]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x=group["State Name"], y=group["Literacy Rate"], 
                    hue=group["Cluster Label"], 
                    palette={"Low":"red", "Medium":"orange", "High":"green"}, 
                    s=100, ax=ax)
    ax.tick_params(axis='x', rotation=90)
    ax.set_title(f"Clusters of States in {year_select} (Based on Literacy Rate)")
    ax.set_ylabel("Literacy Rate (%)")
    ax.set_xlabel("State")
    ax.legend(title="Cluster")
    
    st.pyplot(fig)

elif options == "Forecasting":
    st.header("Literacy Rate Forecasting")
    
    # Filter for national totals
    url="https://raw.githubusercontent.com/Rupsa9647/How_Educated_India/main/Datasets/All_Ages_And_Total_Area_Data.csv"
    df_f =pd.read_csv(url)
    if "Area" in df_f.columns:
        df_f = df_f[df_f["Area"].astype(str).str.lower().str.strip() == "total"]
    if "Age" in df_f.columns:
        df_f = df_f[df_f["Age"].astype(str).str.lower().str.contains("all")]
    
    df_f["Total Person"] = pd.to_numeric(df_f["Total Person"], errors="coerce")
    df_f["Literate Person"] = pd.to_numeric(df_f["Literate Person"], errors="coerce")
    
    # Aggregate nationally
    national = df_f.groupby("Year")[["Total Person","Literate Person"]].sum().reset_index().sort_values("Year")
    national["Literacy Rate"] = national["Literate Person"] / national["Total Person"] * 100
    
    # --- Linear Regression ---
    years = national["Year"].astype(int).values.reshape(-1,1)
    rates = national["Literacy Rate"].values
    lin = LinearRegression().fit(years, rates)
    pred_2021_lin = lin.predict(np.array([[2021]]))[0]
    pred_2031_lin = lin.predict(np.array([[2031]]))[0]
    
    # --- Logistic Growth Model ---
    def logistic(x, L ,k, x0):
        return L / (1 + np.exp(-k*(x-x0)))
    
    # Fit logistic model (with bounds so L ≤ 100)
    params, _ = curve_fit(logistic, years.flatten(), rates, 
                          p0=[100, 0.05, 2000], bounds=([70,0,1900],[100,1,2100]))
    L, k, x0 = params
    pred_2021_log = logistic(2021, L, k, x0)
    pred_2031_log = logistic(2031, L, k, x0)
    
    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Actual data points
    ax.scatter(national["Year"], national["Literacy Rate"], color="blue", s=80, label="Actual Data")
    
    # Linear regression line
    year_range = np.arange(1991, 2031+1).reshape(-1,1)
    ax.plot(year_range, lin.predict(year_range), color="green", linestyle="--", label="Linear Regression")
    
    # Logistic growth curve
    ax.plot(year_range, logistic(year_range.flatten(), L, k, x0), color="orange", linestyle="-.", label="Logistic Growth")
    
    # Forecast points
    ax.scatter([2021], [pred_2021_lin], color="red", marker="X", s=120,
                label=f"Linear 2021: {pred_2021_lin:.2f}%")
    ax.scatter([2031], [pred_2031_lin], color="darkred", marker="X", s=120,
                label=f"Linear 2031: {pred_2031_lin:.2f}%")
    
    ax.scatter([2021], [pred_2021_log], color="purple", marker="D", s=100,
                label=f"Logistic 2021: {pred_2021_log:.2f}%")
    ax.scatter([2031], [pred_2031_log], color="brown", marker="D", s=100,
                label=f"Logistic 2031: {pred_2031_log:.2f}%")
    
    ax.set_title("India Literacy Rate Forecast (1991–2031)", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Literacy Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    
    st.pyplot(fig)
    
    # Display forecast results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Linear Regression Forecast")
        st.metric("2021 Prediction", f"{pred_2021_lin:.2f}%")
        st.metric("2031 Prediction", f"{pred_2031_lin:.2f}%")
    
    with col2:
        st.subheader("Logistic Growth Forecast")
        st.metric("2021 Prediction", f"{pred_2021_log:.2f}%")
        st.metric("2031 Prediction", f"{pred_2031_log:.2f}%")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("India Literacy Dashboard | Data Source: Census of India 1991-2011")
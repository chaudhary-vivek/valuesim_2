import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="IPC Value Simulator",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("IPC Value Simulator")
st.markdown("""
Enter your parameters in the sidebar to see the impact on production and revenue.
""")

# Constants
BAG_SIZES = ["Small", "Medium", "Large"]
NET_REVENUE_PER_BAG = {
    "Small": 1.5,
    "Medium": 5.2,
    "Large": 24.3
}
NET_PROFIT_PER_BAG = {
    "Small": 0.15,
    "Medium": 0.65,
    "Large": 4.25
}
KG_PER_BAG = {
    "Small": 0.06,
    "Medium": 0.20,
    "Large": 1.0
}
BASELINE_THROUGHPUT = {
    "Small": 129.93,
    "Medium": 114.08,
    "Large": 369.82
}
RUNTIME_RATIO = {
    "Small": 13.15,
    "Medium": 78.49,
    "Large": 8.35
}
TOTAL_RATIO = sum(RUNTIME_RATIO.values())
RUNTIME_RATIO_NORMALIZED = {k: v/TOTAL_RATIO for k, v in RUNTIME_RATIO.items()}

# Create sidebar for inputs
st.sidebar.header("Input Parameters")



throughput_improvement = st.sidebar.slider(
    "Throughput Improvement (%)",
    min_value=0.0,
    max_value=50.0,
    value=8.0,
    step=0.1,
    help="Expected percentage improvement in throughput compared to baseline"
)

runtime_percentage = st.sidebar.slider(
    "Machine Availability (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0,
    step=1.0,
    help="Percentage of available hours that the line is operational"
)

num_lines = st.sidebar.number_input(
    "Number of Lines",
    min_value=1,
    max_value=100,
    value=1,
    step=1,
    help="Number of production lines to calculate metrics for"
)

num_months = st.sidebar.number_input(
    "Number of Months",
    min_value=1,
    max_value=60,
    value=12,
    step=1,
    help="Number of months to project metrics for"
)
# Runtime Ratio inputs
st.sidebar.subheader("Runtime Ratio (%)")
# st.sidebar.markdown("*Note: Values must sum to 100%*")

# Default runtime ratio values
default_small = 13.15
default_medium = 78.49
default_large = 8.35

# Input fields for runtime ratios
runtime_small = st.sidebar.number_input(
    "Small Bags (%)",
    min_value=0.0,
    max_value=100.0,
    value=default_small,
    step=0.01,
    help="Percentage of runtime allocated to Small bags"
)

runtime_medium = st.sidebar.number_input(
    "Medium Bags (%)",
    min_value=0.0,
    max_value=100.0,
    value=default_medium,
    step=0.01,
    help="Percentage of runtime allocated to Medium bags"
)

runtime_large = st.sidebar.number_input(
    "Large Bags (%)",
    min_value=0.0,
    max_value=100.0,
    value=default_large,
    step=0.01,
    help="Percentage of runtime allocated to Large bags"
)

# Calculate total and display warning if not 100%
runtime_total = runtime_small + runtime_medium + runtime_large
if runtime_total != 100.0:
    st.sidebar.warning(f"Runtime ratios sum to {runtime_total:.2f}%, not 100%. Results may be inaccurate.")

# Update RUNTIME_RATIO based on user input
RUNTIME_RATIO = {
    "Small": runtime_small,
    "Medium": runtime_medium,
    "Large": runtime_large
}

# Recalculate normalized ratios
TOTAL_RATIO = sum(RUNTIME_RATIO.values())
RUNTIME_RATIO_NORMALIZED = {k: v/TOTAL_RATIO for k, v in RUNTIME_RATIO.items()}

# Calculate metrics
def calculate_metrics(throughput_improvement, runtime_percentage, num_lines, num_months):
    results = {}
    
    # Calculate for each bag size
    for bag_size in BAG_SIZES:
        # Calculate agent throughput (kg/hr)
        agent_throughput = BASELINE_THROUGHPUT[bag_size] * (1 + throughput_improvement / 100)
        
        # Calculate baseline and agent bags per hour
        baseline_bags_per_hour = BASELINE_THROUGHPUT[bag_size] / KG_PER_BAG[bag_size]
        agent_bags_per_hour = agent_throughput / KG_PER_BAG[bag_size]
        
        # Calculate baseline and agent revenue per hour
        baseline_revenue_per_hour = baseline_bags_per_hour * NET_REVENUE_PER_BAG[bag_size]
        agent_revenue_per_hour = agent_bags_per_hour * NET_REVENUE_PER_BAG[bag_size]
        
        # Calculate baseline and agent net profit per hour
        baseline_profit_per_hour = baseline_bags_per_hour * NET_PROFIT_PER_BAG[bag_size]
        agent_profit_per_hour = agent_bags_per_hour * NET_PROFIT_PER_BAG[bag_size]
        
        # Store results for this bag size
        results[bag_size] = {
            "Baseline Throughput (kg/hr)": BASELINE_THROUGHPUT[bag_size],
            "Agent Throughput (kg/hr)": agent_throughput,
            "Baseline Bags/Hour": baseline_bags_per_hour,
            "Agent Bags/Hour": agent_bags_per_hour ,
            "Baseline Revenue/Hour ($)": baseline_revenue_per_hour,
            "Agent Revenue/Hour ($)": agent_revenue_per_hour,
            "Baseline Profit/Hour ($)": baseline_profit_per_hour,
            "Agent Profit/Hour ($)": agent_profit_per_hour,
            "Runtime Ratio": RUNTIME_RATIO[bag_size]
        }
    
    # Calculate weighted averages
    weighted_baseline_bags_per_hour = sum(results[bag_size]["Baseline Bags/Hour"] * RUNTIME_RATIO_NORMALIZED[bag_size] for bag_size in BAG_SIZES)
    weighted_agent_bags_per_hour = sum(results[bag_size]["Agent Bags/Hour"] * RUNTIME_RATIO_NORMALIZED[bag_size] for bag_size in BAG_SIZES)
    weighted_baseline_revenue_per_hour = sum(results[bag_size]["Baseline Revenue/Hour ($)"] * RUNTIME_RATIO_NORMALIZED[bag_size] for bag_size in BAG_SIZES)
    weighted_agent_revenue_per_hour = sum(results[bag_size]["Agent Revenue/Hour ($)"] * RUNTIME_RATIO_NORMALIZED[bag_size] for bag_size in BAG_SIZES)
    weighted_baseline_profit_per_hour = sum(results[bag_size]["Baseline Profit/Hour ($)"] * RUNTIME_RATIO_NORMALIZED[bag_size] for bag_size in BAG_SIZES)
    weighted_agent_profit_per_hour = sum(results[bag_size]["Agent Profit/Hour ($)"] * RUNTIME_RATIO_NORMALIZED[bag_size] for bag_size in BAG_SIZES)
    
    # Calculate monthly projections
    hours_per_month = 24 * 30 * (runtime_percentage / 100)
    baseline_bags_per_month = weighted_baseline_bags_per_hour * hours_per_month
    agent_bags_per_month = weighted_agent_bags_per_hour * hours_per_month
    baseline_revenue_per_month = weighted_baseline_revenue_per_hour * hours_per_month
    agent_revenue_per_month = weighted_agent_revenue_per_hour * hours_per_month
    baseline_profit_per_month = weighted_baseline_profit_per_hour * hours_per_month
    agent_profit_per_month = weighted_agent_profit_per_hour * hours_per_month
    
    # Calculate key performance indicators
    incremental_bags_per_month = (agent_bags_per_month - baseline_bags_per_month) * num_lines
    savings_per_month = (agent_profit_per_month - baseline_profit_per_month) * num_lines
    revenue_potential_per_month = (agent_revenue_per_month - baseline_revenue_per_month) * num_lines
    
    # Calculate total metrics for the specified number of months
    incremental_bags_total = incremental_bags_per_month * num_months
    savings_total = savings_per_month * num_months
    revenue_potential_total = revenue_potential_per_month * num_months
    
    # Compile summary results
    summary = {
        "Weighted Baseline Bags/Hour": weighted_baseline_bags_per_hour,
        "Weighted Agent Bags/Hour": weighted_agent_bags_per_hour,
        "Weighted Baseline Revenue/Hour ($)": weighted_baseline_revenue_per_hour,
        "Weighted Agent Revenue/Hour ($)": weighted_agent_revenue_per_hour,
        "Weighted Baseline Profit/Hour ($)": weighted_baseline_profit_per_hour,
        "Weighted Agent Profit/Hour ($)": weighted_agent_profit_per_hour,
        "Baseline Bags/Month": baseline_bags_per_month,
        "Agent Bags/Month": agent_bags_per_month,
        "Baseline Revenue/Month ($)": baseline_revenue_per_month,
        "Agent Revenue/Month ($)": agent_revenue_per_month,
        "Baseline Profit/Month ($)": baseline_profit_per_month,
        "Agent Profit/Month ($)": agent_profit_per_month,
        "Incremental Bags/Month": incremental_bags_per_month,
        "Savings/Month ($)": savings_per_month,
        "Revenue Potential/Month ($)": revenue_potential_per_month,
        "Incremental Bags Total": incremental_bags_total,
        "Savings Total ($)": savings_total,
        "Revenue Potential Total ($)": revenue_potential_total,
        "Number of Months": num_months
    }
    
    return results, summary

# Calculate metrics based on user inputs
bag_results, summary_results = calculate_metrics(throughput_improvement, runtime_percentage, num_lines, num_months)

# Display KPIs in a prominent way
st.header("Key Performance Indicators")

# Monthly KPIs
st.subheader(f"Monthly Metrics")
kpi_cols = st.columns(3)

with kpi_cols[0]:
    st.metric(
        "Incremental Bags per Month",
        f"{summary_results['Incremental Bags/Month']:,.0f}",
        f"{(summary_results['Incremental Bags/Month'] / summary_results['Baseline Bags/Month'] * 100):.1f}%"
    )

with kpi_cols[1]:
    st.metric(
        "Estimated Profit per Month",
        f"${summary_results['Savings/Month ($)'] / 1000:.1f}k",
        f"{(summary_results['Savings/Month ($)'] / summary_results['Baseline Revenue/Month ($)'] * 100):.1f}%"
    )

with kpi_cols[2]:
    st.metric(
        "Estimated Revenue Potential per Month",
        f"${summary_results['Revenue Potential/Month ($)'] / 1000:.1f}k"
    )

# Total KPIs for selected number of months
time_period = "a month" if summary_results['Number of Months'] == 1 else f"{summary_results['Number of Months']} months"
st.subheader(f"Metrics over {time_period}")
total_kpi_cols = st.columns(3)

with total_kpi_cols[0]:
    st.metric(
        f"Total Incremental Bags",
        f"{summary_results['Incremental Bags Total']:,.0f}"
    )

with total_kpi_cols[1]:
    st.metric(
        f"Total Estimated Profit",
        f"${summary_results['Savings Total ($)'] / 1000:.1f}k"
    )

with total_kpi_cols[2]:
    st.metric(
        f"Total Estimated Revenue Potential",
        f"${summary_results['Revenue Potential Total ($)'] / 1000:.1f}k"
    )

# Display detailed metrics
st.header("Detailed Metrics")

tabs = st.tabs(["By Bag Size", "Summary", "Visualizations"])

with tabs[0]:
    # Create DataFrame for bag size metrics
    bag_df = pd.DataFrame()
    
    for bag_size in BAG_SIZES:
        bag_data = bag_results[bag_size]
        bag_data["Bag Size"] = bag_size
        bag_df = pd.concat([bag_df, pd.DataFrame([bag_data])], ignore_index=True)
    
    # Reorder columns for better display
    columns_order = ["Bag Size", "Runtime Ratio", "Baseline Throughput (kg/hr)", "Agent Throughput (kg/hr)", 
                    "Baseline Bags/Hour", "Agent Bags/Hour", "Baseline Revenue/Hour ($)", "Agent Revenue/Hour ($)",
                    "Baseline Profit/Hour ($)", "Agent Profit/Hour ($)"]
    bag_df = bag_df[columns_order]
    
    # Format data
    formatted_bag_df = bag_df.copy()
    formatted_bag_df["Runtime Ratio"] = formatted_bag_df["Runtime Ratio"].apply(lambda x: f"{x:.2f}%")
    for col in formatted_bag_df.columns:
        if col not in ["Bag Size", "Runtime Ratio"]:
            formatted_bag_df[col] = formatted_bag_df[col].apply(lambda x: f"{x:,.2f}")
    
    st.dataframe(formatted_bag_df, use_container_width=True)

with tabs[1]:
    # Create DataFrame for summary metrics
    summary_df = pd.DataFrame({
        "Metric": list(summary_results.keys()),
        "Value": list(summary_results.values())
    })
    
    # Format data
    summary_df["Value"] = summary_df.apply(
        lambda row: f"${row['Value']:,.2f}" if "Revenue" in row["Metric"] or "Savings" in row["Metric"] 
        else f"{row['Value']:,.2f}", axis=1
    )
    
    st.dataframe(summary_df, use_container_width=True)

with tabs[2]:
    # Create visualizations
    st.subheader("Production Comparison")
    
    # Bar chart for bags per hour
    bags_per_hour_data = {
        "Bag Size": BAG_SIZES * 2,
        "Scenario": ["Baseline"] * 3 + ["Agent"] * 3,
        "Bags per Hour": [bag_results[size]["Baseline Bags/Hour"] for size in BAG_SIZES] + 
                        [bag_results[size]["Agent Bags/Hour"] for size in BAG_SIZES]
    }
    bags_df = pd.DataFrame(bags_per_hour_data)
    
    fig1 = px.bar(
        bags_df, 
        x="Bag Size", 
        y="Bags per Hour", 
        color="Scenario", 
        barmode="group",
        title="Bags per Hour by Size"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Bar chart for revenue
    revenue_data = {
        "Bag Size": BAG_SIZES * 2,
        "Scenario": ["Baseline"] * 3 + ["Agent"] * 3,
        "Revenue per Hour ($)": [bag_results[size]["Baseline Revenue/Hour ($)"] for size in BAG_SIZES] + 
                                [bag_results[size]["Agent Revenue/Hour ($)"] for size in BAG_SIZES]
    }
    revenue_df = pd.DataFrame(revenue_data)
    
    fig2 = px.bar(
        revenue_df, 
        x="Bag Size", 
        y="Revenue per Hour ($)", 
        color="Scenario", 
        barmode="group",
        title="Revenue per Hour by Size"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Pie chart for runtime distribution
    runtime_df = pd.DataFrame({
        "Bag Size": BAG_SIZES,
        "Runtime Ratio": [RUNTIME_RATIO[size] for size in BAG_SIZES]
    })
    
    fig3 = px.pie(
        runtime_df,
        values="Runtime Ratio",
        names="Bag Size",
        title="Runtime Distribution by Bag Size"
    )
    st.plotly_chart(fig3, use_container_width=True)

# Add information about the constants used
with st.expander("View Constants"):
    st.subheader("Bag Specifications")
    
    specs_df = pd.DataFrame({
        "Bag Size": BAG_SIZES,
        "Net Revenue per Bag ($)": [NET_REVENUE_PER_BAG[size] for size in BAG_SIZES],
        "Net Profit per Bag ($)": [NET_PROFIT_PER_BAG[size] for size in BAG_SIZES],
        "KG per Bag": [KG_PER_BAG[size] for size in BAG_SIZES],
        "Baseline Throughput (kg/hr)": [BASELINE_THROUGHPUT[size] for size in BAG_SIZES],
        "Runtime Ratio (%)": [RUNTIME_RATIO[size] for size in BAG_SIZES]
    })
    
    st.dataframe(specs_df, use_container_width=True)
    
    st.subheader("Calculation Information")
    st.markdown("""
    - **Agent Throughput**: Baseline Throughput × (1 + Throughput Improvement %)
    - **Bags per Hour**: Throughput (kg/hr) ÷ KG/Bag
    - **Revenue per Hour**: Bags per Hour × Net Revenue per Bag
    - **Profit per Hour**: Bags per Hour × Net Profit per Bag
    - **Weighted Averages**: Calculated using user-defined runtime ratios
    - **Monthly Projections**: Hourly values × 24 hrs × Runtime % × 30 days
    - **Incremental Bags**: Agent Bags - Baseline Bags
    - **Savings**: Agent Profit - Baseline Profit
    - **Revenue Potential**: Agent Revenue - Baseline Revenue
    - **Total Metrics**: Monthly metrics × Number of Months
    """)

# Footer
st.markdown("---")
st.caption("IPC Value Simulator - v1.1")

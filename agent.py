import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO
import base64

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq LLM (adjust temperature or model as needed)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

def get_table_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

def get_fig_download_link(fig, filename="plot.png"):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">Download Plot</a>'
    return href

def main():
    st.title("📈 Stock Market Data Explorer with Agent")

    # Load data file (change path or implement file uploader later)
    data_path = "/Users/akshayaa.s/agentic_ai/data_analyst/stock_data_july_2025.csv"
    df = load_data(data_path)

    # Create LangChain Pandas Agent once
    agent = create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True, handle_parsing_errors=True)

    tabs = st.tabs(["Raw Dataset", "Info & Describe", "Visualizations", "Ask Questions"])

    # --- Raw Dataset ---
    with tabs[0]:
        st.header("Raw Dataset")
        st.dataframe(df)
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)

    # --- Dataset Info ---
    with tabs[1]:
        st.header("Dataset Information")

        # Data Types Table
        dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"]).reset_index().rename(columns={"index": "Column"})
        st.subheader("Data Types")
        st.dataframe(dtypes_df)

        # Missing Values Table
        missing_df = pd.DataFrame(df.isna().sum(), columns=["Missing Values"]).reset_index().rename(columns={"index": "Column"})
        st.subheader("Missing Values Count")
        st.dataframe(missing_df)

        # Unique Values for categorical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            st.subheader("Unique Values in Categorical Columns")
            unique_vals = {col: df[col].nunique() for col in cat_cols}
            unique_df = pd.DataFrame(list(unique_vals.items()), columns=["Column", "Unique Values"])
            st.dataframe(unique_df)

        # Show describe
        st.subheader("Descriptive Statistics (Numeric Columns)")
        st.dataframe(df.describe())

    # --- Visualizations ---
    with tabs[2]:
        st.header("Visualizations")

        # Select columns for visualization
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        selected_cols = st.multiselect("Select numeric columns to plot", numeric_cols, default=numeric_cols[:2])

        if selected_cols:
            # Select plot type
            plot_type = st.selectbox("Select plot type", ["Line Plot", "Histogram", "Scatter Plot"])

            if plot_type == "Line Plot":
                fig, ax = plt.subplots()
                df[selected_cols].plot(ax=ax)
                ax.set_title(f"Line Plot of {', '.join(selected_cols)}")
                st.pyplot(fig)
                st.markdown(get_fig_download_link(fig, "line_plot.png"), unsafe_allow_html=True)

            elif plot_type == "Histogram":
                fig, ax = plt.subplots()
                df[selected_cols].hist(ax=ax, bins=30)
                plt.tight_layout()
                st.pyplot(fig)
                st.markdown(get_fig_download_link(fig, "histogram.png"), unsafe_allow_html=True)

            elif plot_type == "Scatter Plot":
                if len(selected_cols) >= 2:
                    x_col = st.selectbox("X-axis", selected_cols)
                    y_col = st.selectbox("Y-axis", [col for col in selected_cols if col != x_col])
                    fig, ax = plt.subplots()
                    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
                    ax.set_title(f"Scatter Plot of {x_col} vs {y_col}")
                    st.pyplot(fig)
                    st.markdown(get_fig_download_link(fig, "scatter_plot.png"), unsafe_allow_html=True)
                else:
                    st.warning("Select at least two columns for scatter plot.")

        else:
            st.info("Select one or more numeric columns to start visualization.")

    # --- Ask Questions ---
    with tabs[3]:
        st.header("Ask Questions about the Dataset")
        user_question = st.text_input("Enter your question about the dataset:")

        if user_question:
            with st.spinner("Agent is thinking..."):
                try:
                    answer = agent.run(user_question)
                    st.markdown(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

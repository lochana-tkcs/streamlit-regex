import streamlit as st
import pandas as pd
import regex as re  # Use the `regex` module for advanced regular expressions
from openai import OpenAI
import random

# Set your OpenAI API key
api_key = st.secrets["openai_api_key"]

# Initialize the OpenAI client with the API key
client = OpenAI(
    api_key=api_key)


def generate_regex(data, prompt, col):
    column_values = [str(row) for row in data[col].head(20)]
    all_values_str = ", ".join(column_values)

    # Add column and values to the prompt
    prompt += f"\nColumn: {col}\n"
    prompt += "Values:\n" + all_values_str + "\n"

    prompt += "\nGiven the intent for the column, just give the expression (as a string) and no other text."

    # Call to GPT model to generate regex
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0
    )

    # Extract regex from the response
    pattern = response.choices[0].message.content.strip().replace('`', '')

    # Pick a random sample from the column to demonstrate the regex
    random_value = str(random.choice(data[col].dropna()))
    match = re.search(pattern, random_value)
    example_result = f"'{random_value}' will become '{match.group(0)}'" if match else "No match found"

    return pattern, example_result


def apply_regex(data, pattern, col):
    # Apply the regex pattern on the column using the `regex` module
    try:
        data[f'{col}_regex'] = data[col].apply(
            lambda x: re.search(pattern, str(x)).group(0) if re.search(pattern, str(x)) else x)
    except re.error as e:
        st.error(f"Error applying regex pattern: {e}")
        return None

    return data


# Streamlit app with custom styling
def streamlit_app():
    st.title("Intent Based Regex Generation")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset:")
        st.dataframe(df)

        # Get the column name and intent from the user
        column = st.selectbox("Select the column on which you want to apply the regex:", df.columns)
        intent = st.text_input("What do you want to perform on the column (e.g., Extract characters after ':'):")

        # Define the prompt template for the regex generation
        prompt_template = f"""
        Your task is to generate an accurate and efficient regular expression (regex)
        based on the user's request for manipulating values in a dataset column.
        The regex should extract only the relevant information according to the user's intent.
        But if the prompt given is invalid/nonsensical, output saying "Sorry, I couldn't understand your request. Can you please try again?"
        Example 1: Extract the month from a date like 12/01/1990. Expected output: 01
        Example 2: Extract everything before the first '.' in an IP address such as 192.168.1.1. Expected output: 192

        User intent: {intent}
        The column values are as follows:
        """

        # Initialize session state to store regex pattern and example output
        if "regex_pattern" not in st.session_state:
            st.session_state["regex_pattern"] = ""
        if "example_output" not in st.session_state:
            st.session_state["example_output"] = ""

        # Button to generate the regex
        if st.button("Generate Regex"):
            st.session_state["regex_pattern"], st.session_state["example_output"] = generate_regex(df, prompt_template, column)

        # Only show regex and example after generation and keep showing after applying
        if st.session_state["regex_pattern"]:
            # Check if the regex generation failed or produced a misunderstanding response
            if "Sorry, I couldn't understand your request" in st.session_state["regex_pattern"]:
                st.warning(st.session_state["regex_pattern"])
            else:
                st.markdown(f"**Generated Regex:** `{st.session_state['regex_pattern']}`")
                
                # Styling example and message separately, using same font for example as the regex
                st.markdown(
                    f"""
                    <div style="font-size:16px; font-family:monospace; margin-top:10px;">
                        <b>Example:</b> <code>{st.session_state['example_output']}</code>
                    </div>
                    <div style="font-size:14px; color:gray; font-style:italic; margin-top:10px;">
                        If the example isn't as expected, try providing more details in your request.
                    </div>
                    <div style="margin-bottom:20px;"></div>
                    """, unsafe_allow_html=True
                )

                # Show the "Apply Regex" button only if a valid regex is generated
                if st.button("Apply Regex"):
                    updated_df = apply_regex(df, st.session_state["regex_pattern"], column)
                    if updated_df is not None:
                        st.write("Updated Dataset:")
                        st.dataframe(updated_df)

                        # Option to download the updated CSV file
                        csv = updated_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Updated CSV", csv, "updated_data.csv", "text/csv")


# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()

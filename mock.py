import streamlit as st
import pandas as pd
import regex as re  # Use the `regex` module for advanced regular expressions
from openai import OpenAI
import random
import json

# Set your OpenAI API key
api_key = st.secrets["openai_api_key"]

# Initialize the OpenAI client with the API key
client = OpenAI(
    api_key=api_key)

def apply_regex1(text: str, pattern: str, match_count: int, mode: str = 'findall'):
    try:
        if mode == 'findall':
            return ' '.join(
            [''.join(match) if isinstance(match, tuple) else match for match in re.findall(pattern, text)])
        elif mode == 'search':
            match = re.search(pattern, text)
            if match:
                return match.group(match_count) if match else None
        # elif mode == 'match':
        #     match = re.search(pattern, text)
        #     if match:
        #         return match.group(match_count) if match else None
        # elif mode == 'replace':
        #     return re.sub(pattern, replace_with, text)
        else:
            raise ValueError("Invalid mode. Choose 'findall', 'search', 'match', or 'replace'.")
    except re.error as e:
        print(f"Invalid regex pattern: {e}")
        return None

def generate_regex(data, prompt, col):
    column_values = [str(row) for row in data[col].head(20)]
    all_values_str = ", ".join(column_values)

    # Add column and values to the prompt
    prompt += f"\nColumn: {col}\n"
    prompt += "Values:\n" + all_values_str + "\n"

    prompt += "\nGiven the intent for the column, Return a dictionary with 'pattern' as the regex pattern string, 'mode' selected as search (for finding the first occurrence), or findall (for multiple matches), and 'match_count' as the capturing group index to return. No other text."

    # Call to GPT model to generate regex
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "pattern_matching",
                "schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regular expression pattern for matching specific text formats",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["search", "findall", "replace"],
                            "description": "Mode of regex operation",
                            "value": "findall"
                        },
                        "match_count": {
                            "type": "integer",
                            "description": "the number corresponds to the index of the capturing group in the regular expression pattern, starting from 1 for the first group",
                        }
                    },
                    "required": ["pattern", "mode", "match_count"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        max_tokens=1000,
        temperature=0
    )

    # Extract regex from the response
    dictionary = response.choices[0].message.content.strip().replace('`', '')
    dictionary = json.loads(dictionary)
    print(dictionary)

    non_na_values = data[col].dropna().tolist()[:20]

    if not non_na_values:
        print("No non-NaN values available in the column.")
        return None  # or set a default result, e.g., ""

    example_result = None

    # Process values sequentially
    for value in non_na_values:
        matches = apply_regex1(str(value), str(dictionary.get('pattern')), dictionary.get("match_count"),
                               dictionary.get('mode'))
        if matches:
            example_result = f"'{value}' will become '{matches}'"
            break

    if example_result is None:
        example_result = "No match found after checking 20 values. Click on apply to check further."

    print(example_result)

    return str(dictionary.get('pattern')), example_result, dictionary


def apply_regex(data, col, dic):
    # Apply the regex pattern on the column using the `regex` module
    try:
        if dic.get('mode') == 'findall':
            data[f'{col}_regex'] = data[col].apply(lambda x: ' '.join([''.join(match) if isinstance(match, tuple) else match for match in re.findall(str(dic.get('pattern')), str(x))]))
        elif dic.get('mode') == 'search':
            data[f'{col}_regex'] = data[col].apply(lambda x: re.search(str(dic.get('pattern')), str(x)).group(dic.get("match_count")) if re.search(str(dic.get('pattern')), str(x)) else None)
        # elif dic.get('mode') == 'match':
        #     data[f'{col}_regex'] = data[col].apply(
        #         lambda x: re.search(str(dic.get('pattern')), str(x)).group() if re.search(
        #             str(dic.get('pattern')), str(x)) else None)
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
        Your task is to generate a precise and efficient regular expression (regex) that fulfills the user's specific request for processing values in a dataset column. 
       
        Ensure the regex:
        1. Locate a specific word based on its position within the text (e.g., the second word in a phrase) and capture a particular character within that word (e.g., the fourth letter of the second word).
        2. Avoids any unnecessary matches or complex patterns that do not contribute to accuracy.
        3. Works for typical variations in formatting as seen in the column values provided.
        
        User intent: {intent}
        Column sample values:
        """

        # Initialize session state to store regex pattern and example output
        if "regex_pattern" not in st.session_state:
            st.session_state["regex_pattern"] = ""
        if "example_output" not in st.session_state:
            st.session_state["example_output"] = ""

        # Button to generate the regex
        if st.button("Generate Regex"):
            st.session_state["regex_pattern"], st.session_state["example_output"], st.session_state["dictionary"] = generate_regex(df, prompt_template,
                                                                                                   column)

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
                    updated_df = apply_regex(df, column, st.session_state["dictionary"])
                    if updated_df is not None:
                        st.write("Updated Dataset:")
                        st.dataframe(updated_df)

                        # Option to download the updated CSV file
                        csv = updated_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Updated CSV", csv, "updated_data.csv", "text/csv")


# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()

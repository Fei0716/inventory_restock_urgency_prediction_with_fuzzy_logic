import streamlit as st
import pandas as pd
import requests
import io
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Inventory Restock Urgency Prediction", layout="wide")

st.title("Inventory Restock Urgency Prediction")
st.write("**Choose a CSV File**")

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# File uploader (outside columns, at the top)
uploaded_file = st.file_uploader("📊Upload Inventory Data", type="csv")

# Content below file uploader (Two columns: Left for Preview/Results, Right for History)
if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Store the dataframe for later processing
        st.session_state.current_uploaded_df = df

        # Create two columns for horizontal alignment
        col1, col2 = st.columns(2)

        with col1:
            # Display the uploaded data preview in the first column
            st.subheader("📊 Uploaded Data Preview")
            st.dataframe(df, height=300) # Use fixed height for scrolling

            # --- Prediction Results section (within the left column) ---
            st.subheader("📈 Prediction Results") # Subheader for the prediction results area

            # Check if there are prediction results to display in the left column
            if 'last_prediction_results' in st.session_state and st.session_state.last_prediction_results:
                 try:
                     results_df = pd.DataFrame(st.session_state.last_prediction_results)
                     st.dataframe(results_df)
                 except Exception as df_error:
                     st.error(f"Error displaying last prediction results: {df_error}")

            else:
                 st.info("Prediction results will appear here after you click 'Run Prediction'.")

        with col2:
            # Display history in the second column
            st.subheader("🕒 Prediction History")
            if st.session_state.prediction_history:
                try:
                    # Display history in the right column
                    history_df = pd.DataFrame(st.session_state.prediction_history)
                    st.dataframe(history_df)

                    # Optional: Add a button to clear history within the column
                    if st.button("🧹 Clear All History", key="clear_history_col2_in_col"):
                         st.session_state.prediction_history = []
                         st.experimental_rerun() # Rerun to clear the display

                except Exception as history_df_error:
                     st.error(f"Error displaying history: {history_df_error}")
                     st.json(st.session_state.prediction_history)

            else:
                st.info("No prediction history yet.")

        # --- Prediction Button and Logic (below the columns) ---

        # The Run Prediction button is placed outside the columns to span the full width below them
        if st.button("🚀 Run Prediction", key="run_prediction_button", type="primary"):
            with st.spinner("Predicting..."):
                 # Data processing and Prediction logic (using the stored df)
                 # Convert DataFrame to list of dictionaries for backend
                 backend_data = {"products": []}
                 valid_rows_data = [] # Store valid input data for history

                 # Process each row in the dataframe to prepare data for backend and history
                 data_processing_successful = True # Flag to track if data processing is successful

                 # Use the dataframe stored in session state
                 if 'current_uploaded_df' in st.session_state and st.session_state.current_uploaded_df is not None:
                     df_to_process = st.session_state.current_uploaded_df
                     try:
                         for index, row in df_to_process.iterrows():
                             try:
                                 product_data = {
                                     "sales": float(row['Sales_Speed']),
                                     "stock": float(row['Stock_Level']),
                                     "lead_time": float(row['Lead_Time']),
                                     "price": float(row['Price'])
                                 }
                                 backend_data["products"].append(product_data)

                                 valid_rows_data.append({
                                     "Month": row.get('Month'),
                                     "Product_ID": row.get('Product_ID'),
                                     "Sales_Speed": row.get('Sales_Speed'),
                                     "Stock_Level": row.get('Stock_Level'),
                                     "Lead_Time": row.get('Lead_Time'),
                                     "Price": row.get('Price'),
                                 })

                             except KeyError as e:
                                 st.error(f"Missing expected column in CSV: {e}. Please ensure your CSV has 'Sales_Speed', 'Stock_Level', 'Lead_Time', and 'Price' columns.")
                                 data_processing_successful = False
                                 backend_data["products"] = [] # Clear data if an error occurs
                                 break # Stop processing if a required column is missing
                             except ValueError as e:
                                  st.error(f"Invalid data type in CSV: {e}. Please ensure 'Sales_Speed', 'Stock_Level', 'Lead_time', and 'Price' columns contain numeric values.")
                                  data_processing_successful = False
                                  backend_data["products"] = [] # Clear data if an error occurs
                                  break # Stop processing if data types is incorrect

                     except Exception as e:
                          st.error(f"Error processing data for backend: {str(e)}")
                          data_processing_successful = False
                          backend_data["products"] = [] # Clear data if an error occurs
                 else:
                      # This case should ideally not happen if uploaded_file is not None, but as a fallback:
                      st.error("Could not retrieve the uploaded data for processing.")
                      data_processing_successful = False

                 # Send data to backend and process response if data processing was successful and there's data to send
                 if data_processing_successful and backend_data["products"]:
                     try:
                         response = requests.post(
                             "http://localhost:8080/predict",
                             json=backend_data
                         )

                         if response.status_code == 200:
                             results = response.json()

                             # Process results and store in history
                             processed_results = []
                             current_prediction_history = []
                             if results:
                                 for i, res in enumerate(results):
                                     try:
                                         restock_urgency = res.get('restock_urgency', None)
                                         membership = res.get('membership', {})

                                         # Determine the category with the highest membership
                                         predicted_category = "N/A"
                                         if membership:
                                             max_membership_category = None
                                             max_membership_value = -1
                                             for cat, value in membership.items():
                                                  if isinstance(value, (int, float)) and value > max_membership_value:
                                                     max_membership_value = value
                                                     max_membership_category = cat
                                             predicted_category = max_membership_category if max_membership_category is not None else "N/A"

                                         # Only append if essential keys are present (restock_urgency or membership is not empty)
                                         if restock_urgency is not None or membership:
                                              processed_results.append({
                                                  "Restock_Urgency": restock_urgency,
                                                  "Urgency_level": predicted_category,
                                              })

                                              # Combine input data and prediction results for history
                                              if i < len(valid_rows_data):
                                                  history_entry = valid_rows_data[i].copy()
                                                  history_entry["Restock_Urgency"] = restock_urgency
                                                  history_entry["Urgency_level"] = predicted_category
                                                  current_prediction_history.append(history_entry)
                                         else:
                                             st.warning(f"Skipping result {i} from backend due to missing essential data (restock_urgency or membership). Result: {res}")

                                     except Exception as process_res_error:
                                         st.error(f"Error processing individual prediction result {i}: {process_res_error}")
                                         st.json(res) # Display the problematic result
                                         # Decide whether to continue or break - continuing might be safer to show partial results
                                         # break

                                 # Store the processed results in session state to be displayed in the left column
                                 st.session_state.last_prediction_results = processed_results

                                 # Append current prediction to history - this is now done AFTER getting results and processing
                                 if current_prediction_history:
                                      st.session_state.prediction_history.extend(current_prediction_history)

                                 st.success("✅ Prediction completed! Results displayed in the left column.")
                                 st.experimental_rerun() # Rerun to display the updated results and history

                             else:
                                 st.warning("Backend returned an empty response or invalid JSON for results.")
                                 st.session_state.last_prediction_results = [] # Clear previous results
                                 st.experimental_rerun() # Rerun to update display

                         # Handle non-200 status codes within the try block
                         else:
                             st.error(f"Error from backend: {response.status_code}")
                             st.error(response.text)
                             st.session_state.last_prediction_results = [] # Clear previous results
                             st.experimental_rerun() # Rerun to update display

                     except requests.exceptions.RequestException as e:
                         st.error(f"Error connecting to backend: {str(e)}")
                         st.session_state.last_prediction_results = [] # Clear previous results
                         st.experimental_rerun() # Rerun to update display

                     except Exception as general_prediction_error:
                          st.error(f"An unexpected error occurred during prediction: {general_prediction_error}")
                          st.session_state.last_prediction_results = [] # Clear previous results
                          st.experimental_rerun() # Rerun to update display

                 elif not backend_data["products"] and data_processing_successful:
                      st.warning("No valid data rows were processed to send to the backend.")

    except Exception as e:
         st.error(f"Error reading the uploaded file: {str(e)}")

else:
    # Message when no file is uploaded
    st.info("Please upload a CSV file to get started.")
    # Add a placeholder for the info message when history is empty and no file is uploaded
    if not st.session_state.prediction_history:
         st.info("Upload a CSV and click Predict to see history.")
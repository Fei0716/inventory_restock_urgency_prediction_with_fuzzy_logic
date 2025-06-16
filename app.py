import streamlit as st
import pandas as pd
import requests
import io
from datetime import datetime
import sqlite3

# Page configuration
st.set_page_config(page_title="Inventory Restock Urgency Prediction", layout="wide")

# Initialize database
conn = sqlite3.connect("prediction_history.db", check_same_thread=False)
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    timestamp TEXT,
    month TEXT,
    product_id TEXT,
    sales_speed INT,
    stock_level INT,
    lead_time INT,
    price REAL,
    restock_urgency REAL,
    urgency_level TEXT,
    mf_low REAL,
    mf_medium REAL,
    mf_high REAL
);
""")
conn.commit()

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
        df.index = df.index + 1
        modified_df = df.drop(columns=["Category"])
        # Store the dataframe for later processing
        st.session_state.current_uploaded_df = df

        # Create two columns for horizontal alignment
        col1, col2 = st.columns(2)

        with col1:
            # Display the uploaded data preview in the first column
            st.subheader("📊 Uploaded Data Preview")
            st.dataframe(modified_df) # Use fixed height for scrolling
        with col2:

            # --- Prediction Results section (within the left column) ---
            st.subheader("📈 Prediction Results") # Subheader for the prediction results area

            # Check if there are prediction results to display in the left column
            if 'last_prediction_results' in st.session_state and st.session_state.last_prediction_results:
                 try:
                     results_df = pd.DataFrame(st.session_state.last_prediction_results)
                     results_df.index = results_df.index + 1

                     # Create two columns inside col1 for side-by-side layout
                     pred_col1, pred_col2 = st.columns([1.5, 1.5])  # 2:1 ratio for table and chart
                     with pred_col1:
                         # st.dataframe(results_df)
                         def highlight_prediction_urgency(val):
                             color_map = {
                                 "High": "background-color: #ff4d4f; color: white;",  # Red
                                 "Medium": "background-color: #faad14; color: black;",  # Yellow
                                 "Low": "background-color: #52c41a; color: white;"  # Green
                             }
                             return color_map.get(val, "")


                         # Format and style the prediction result table
                         styled_pred_df = results_df.style.format({
                             "Restock_Urgency": "{:.2f}",
                             "Mf_Low": "{:.2f}",
                             "Mf_Medium": "{:.2f}",
                             "Mf_High": "{:.2f}"
                         }).applymap(highlight_prediction_urgency, subset=["Urgency_level"])

                         st.dataframe(styled_pred_df, use_container_width=True)
                     with pred_col2:
                         if 'urgency_counts' in st.session_state:
                             import plotly.express as px

                             fig = px.pie(
                                 st.session_state.urgency_counts,
                                 values="Count",
                                 names="Urgency Level",
                                 color="Urgency Level",
                                 color_discrete_map={"High": "#ff4d4f", "Medium": "#faad14", "Low": "#52c41a"}
                             )
                             st.plotly_chart(fig, use_container_width=True)
                         else:
                             st.info("Pie chart will appear here after prediction.")
                 except Exception as df_error:
                     st.error(f"Error displaying last prediction results: {df_error}")

            else:
                 st.info("Prediction results will appear here after you click 'Run Prediction'.")
            # --- Prediction Button and Logic (below the columns) ---

            # The Run Prediction button is placed outside the columns to span the full width below them
            if st.button("🚀 Run Prediction", key="run_prediction_button", type="primary"):
                with st.spinner("Predicting..."):
                    # Data processing and Prediction logic (using the stored df)
                    # Convert DataFrame to list of dictionaries for backend
                    backend_data = {"products": []}
                    valid_rows_data = []  # Store valid input data for history

                    # Process each row in the dataframe to prepare data for backend and history
                    data_processing_successful = True  # Flag to track if data processing is successful

                    # Use the dataframe stored in session state
                    if 'current_uploaded_df' in st.session_state and st.session_state.current_uploaded_df is not None:
                        df_to_process = st.session_state.current_uploaded_df
                        try:
                            for index, row in df_to_process.iterrows():
                                try:
                                    product_data = {
                                        "sales": row['Sales_Speed'],
                                        "stock": row['Stock_Level'],
                                        "lead_time": row['Lead_Time'],
                                        "price": row['Price']
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
                                    st.error(
                                        f"Missing expected column in CSV: {e}. Please ensure your CSV has 'Sales_Speed', 'Stock_Level', 'Lead_Time', and 'Price' columns.")
                                    data_processing_successful = False
                                    backend_data["products"] = []  # Clear data if an error occurs
                                    break  # Stop processing if a required column is missing
                                except ValueError as e:
                                    st.error(
                                        f"Invalid data type in CSV: {e}. Please ensure 'Sales_Speed', 'Stock_Level', 'Lead_time', and 'Price' columns contain numeric values.")
                                    data_processing_successful = False
                                    backend_data["products"] = []  # Clear data if an error occurs
                                    break  # Stop processing if data types is incorrect

                        except Exception as e:
                            st.error(f"Error processing data for backend: {str(e)}")
                            data_processing_successful = False
                            backend_data["products"] = []  # Clear data if an error occurs
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
                                                    "Mf_Low": round(membership.get("Low", 0) * 100, 2),
                                                    "Mf_Medium": round(membership.get("Medium", 0) * 100, 2),
                                                    "Mf_High": round(membership.get("High", 0) * 100, 2),
                                                })

                                                # Combine input data and prediction results for history
                                                if i < len(valid_rows_data):
                                                    history_entry = valid_rows_data[i].copy()
                                                    history_entry["Restock_Urgency"] = restock_urgency
                                                    history_entry["Urgency_level"] = predicted_category
                                                    history_entry["Mf_Low"] = round(membership.get("Low", 0) * 100, 2)
                                                    history_entry["Mf_Medium"] = round(
                                                        membership.get("Medium", 0) * 100, 2)
                                                    history_entry["Mf_High"] = round(membership.get("High", 0) * 100, 2)
                                                    current_prediction_history.append(history_entry)
                                            else:
                                                st.warning(
                                                    f"Skipping result {i} from backend due to missing essential data (restock_urgency or membership). Result: {res}")

                                        except Exception as process_res_error:
                                            st.error(
                                                f"Error processing individual prediction result {i}: {process_res_error}")
                                            st.json(res)  # Display the problematic result
                                            # Decide whether to continue or break - continuing might be safer to show partial results
                                            # break

                                    # Store the processed results in session state to be displayed in the left column
                                    st.session_state.last_prediction_results = processed_results

                                    # Append current prediction to history - this is now done AFTER getting results and processing
                                    if current_prediction_history:
                                        st.session_state.prediction_history.extend(current_prediction_history)
                                        timestamp = datetime.now().isoformat()

                                        for record in current_prediction_history:
                                            cursor.execute("""
                                                      INSERT INTO history (
                                                          timestamp, month, product_id, sales_speed, stock_level,
                                                          lead_time, price, restock_urgency, urgency_level,mf_low,mf_medium, mf_high
                                                      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ? ,?, ?)
                                                  """, (
                                                timestamp,
                                                record.get("Month"),
                                                record.get("Product_ID"),
                                                record.get("Sales_Speed", 0),
                                                record.get("Stock_Level", 0),
                                                record.get("Lead_Time", 0),
                                                record.get("Price", 0),
                                                record.get("Restock_Urgency", 0),
                                                record.get("Urgency_level", "N/A"),
                                                record.get("Mf_Low", 0),
                                                record.get("Mf_Medium", 0),
                                                record.get("Mf_High", 0),
                                            ))

                                        conn.commit()
                                    st.success("✅ Prediction completed! Results displayed in the left column.")
                                    # Count the number of products by urgency level
                                    urgency_counts = pd.Series([r["Urgency_level"] for r in
                                                                current_prediction_history]).value_counts().reset_index()
                                    urgency_counts.columns = ["Urgency Level", "Count"]

                                    # Save urgency_counts to session state to display below outside the prediction block
                                    st.session_state.urgency_counts = urgency_counts

                                    # Save processed results to session state for display
                                    st.session_state.last_prediction_results = processed_results

                                    # Save current prediction history to session state
                                    st.session_state.prediction_history.extend(current_prediction_history)
                                    st.rerun()  # Rerun to display the updated results and history

                                else:
                                    st.warning("Backend returned an empty response or invalid JSON for results.")
                                    st.session_state.last_prediction_results = []  # Clear previous results
                                    st.rerun()  # Rerun to update display

                            # Handle non-200 status codes within the try block
                            else:
                                st.error(f"Error from backend: {response.status_code}")
                                st.error(response.text)
                                st.session_state.last_prediction_results = []  # Clear previous results
                                st.rerun()  # Rerun to update display

                        except requests.exceptions.RequestException as e:
                            st.error(f"Error connecting to backend: {str(e)}")
                            st.session_state.last_prediction_results = []  # Clear previous results
                            st.rerun()  # Rerun to update display

                        except Exception as general_prediction_error:
                            st.error(f"An unexpected error occurred during prediction: {general_prediction_error}")
                            st.session_state.last_prediction_results = []  # Clear previous results
                            st.rerun()  # Rerun to update display

                    elif not backend_data["products"] and data_processing_successful:
                        st.warning("No valid data rows were processed to send to the backend.")

        # Display history in the second column
        st.subheader("🕒 Prediction History")

        # --- Populate prediction history from the database ---
        cursor.execute("SELECT * FROM history ORDER BY timestamp DESC")
        db_history = cursor.fetchall()

        if db_history:
            history_df = pd.DataFrame(db_history, columns=[
                "Timestamp", "Month", "Product_ID", "Sales_Speed",
                "Stock_Level", "Lead_Time", "Price",
                "Restock_Urgency", "Urgency_level" , "Mf_Low" , "Mf_Medium", "Mf_High"
            ])
            # Format the timestamp column for better readability
            history_df["Timestamp"] = pd.to_datetime(history_df["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            history_df["Restock_Urgency"] = history_df["Restock_Urgency"].round(2)
            history_df["Price"] = history_df["Price"].round(2)
            history_df.index = history_df.index + 1
            # Apply styling to highlight Urgency_level
            def highlight_urgency(val):
                color_map = {
                    "High": "background-color: #ff4d4f; color: white;",  # red
                    "Medium": "background-color: #faad14; color: black;",  # yellow
                    "Low": "background-color: #52c41a; color: white;"  # green
                }
                return color_map.get(val, "")


            # Create styled dataframe with both formatting and highlighting
            styled_df = history_df.style.format({"Price": "{:.2f}", "Restock_Urgency": "{:.2f}" ,  "Mf_Low": "{:.2f}" , "Mf_Medium": "{:.2f}" , "Mf_High": "{:.2f}"}).applymap(
                highlight_urgency, subset=["Urgency_level"])
            st.dataframe(styled_df, use_container_width=True)

            # Optional: Clear history button
            if st.button("🧹 Clear All History"):
                cursor.execute("DELETE FROM history")
                conn.commit()
                st.rerun()
        else:
            st.info("No prediction history yet.")


    except Exception as e:
         st.error(f"Error reading the uploaded file: {str(e)}")

else:
    st.info("Please upload a CSV file to get started.")

    # Show history even if no file is uploaded
    st.subheader("🕒 Prediction History")

    cursor.execute("SELECT * FROM history ORDER BY timestamp DESC")
    db_history = cursor.fetchall()

    if db_history:
        history_df = pd.DataFrame(db_history, columns=[
            "Timestamp", "Month", "Product_ID", "Sales_Speed",
            "Stock_Level", "Lead_Time", "Price",
            "Restock_Urgency", "Urgency_level", "Mf_Low" , "Mf_Medium", "Mf_High"
        ])
        # Format the timestamp column for better readability
        history_df["Timestamp"] = pd.to_datetime(history_df["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        history_df["Restock_Urgency"] = history_df["Restock_Urgency"].round(2)
        history_df["Price"] = history_df["Price"].round(2)
        history_df.index = history_df.index + 1


        # Apply styling to highlight Urgency_level
        def highlight_urgency(val):
            color_map = {
                "High": "background-color: #ff4d4f; color: white;",  # red
                "Medium": "background-color: #faad14; color: black;",  # yellow
                "Low": "background-color: #52c41a; color: white;"  # green
            }
            return color_map.get(val, "")


        # Create styled dataframe with both formatting and highlighting
        styled_df = history_df.style.format({"Price": "{:.2f}", "Restock_Urgency": "{:.2f}",  "Mf_Low": "{:.2f}" , "Mf_Medium": "{:.2f}" , "Mf_High": "{:.2f}"}).applymap(
            highlight_urgency, subset=["Urgency_level"])
        st.dataframe(styled_df, use_container_width=True)

        if st.button("🧹 Clear All History"):
            cursor.execute("DELETE FROM history")
            conn.commit()
            st.rerun()
    else:
        st.info("No prediction history yet.")




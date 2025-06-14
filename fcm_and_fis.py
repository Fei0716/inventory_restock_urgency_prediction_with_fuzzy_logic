import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from fcmeans import FCM
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from scipy.interpolate import interp1d
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

df = pd.read_csv('preprocessed_inventory_data.csv')
# Filter data for the first three months: 2022-01 to 2022-03
df = df[(pd.to_datetime(df['Month']) >= '2022-01-01') & (pd.to_datetime(df['Month']) <= '2022-03-31')]
# Display the filtered data
# print(df.tail())

# Use one variable at a time, e.g., Sales_Speed
sales_speed_reshaped = df['Sales_Speed'].values.reshape(-1, 1)
stock_level_reshaped = df['Stock_Level'].values.reshape(-1, 1)
lead_time_reshaped = df['Lead_Time'].values.reshape(-1, 1)
selling_price_reshaped = df['Price'].values.reshape(-1, 1)

max_sales = df['Sales_Speed'].max()
max_stock = df['Stock_Level'].max()
max_lead = df['Lead_Time'].max()
max_price = df['Price'].max() # Use the max from the original 1D pandas Series
sales_range = np.linspace(0, max_sales * 1.1, 1000)
stock_range = np.linspace(0, max_stock * 1.1, 1000)
lead_range = np.linspace(0, max_lead * 1.1, 1000)
price_range = np.linspace(0, max_price * 1.1, 1000) # price_range is now created from a scalar
# Define the universe for urgency separately as you did before
urgency_range = np.linspace(0, 10, 1000)

# Create fuzzy variables
sales = ctrl.Antecedent(sales_range, 'sales')
stock = ctrl.Antecedent(stock_range, 'stock')
lead = ctrl.Antecedent(lead_range, 'lead_time')
price = ctrl.Antecedent(price_range, 'price')
urgency = ctrl.Consequent(urgency_range, 'restock_urgency')


# Now use the reshaped data for FCM fitting in subsequent cells
sales_speed = sales_speed_reshaped
stock_level = stock_level_reshaped
lead_time = lead_time_reshaped
selling_price = selling_price_reshaped

# Instantiate the FCM model with 3 clusters (Slow, Normal, Fast)
fcm = FCM(n_clusters=3, random_state=42)
fcm.fit(sales_speed)

# Get the cluster centers and membership matrix
centers = fcm.centers.flatten()
u = fcm.u  # Membership degrees (shape: n_samples x n_clusters)

# Sort centers and match the memberships accordingly
sorted_idx = np.argsort(centers)
sorted_centers = centers[sorted_idx]
sorted_u = u[:, sorted_idx]

# Plot learned membership functions (fuzzy sets)
# x_vals = np.linspace(min(sales_speed), max(sales_speed), 1000)
x_vals = np.linspace(0, max(sales_speed) * 1.1, 1000).flatten()


def create_membership_function(center_idx, centers):
    center = centers[center_idx]
    mf = np.zeros_like(x_vals)  # Ensure correct shape

    if center_idx == 0:
        # Left shoulder (Slow): extend left
        right = centers[center_idx + 1]
        left = max(0, center - (right - center) * 1.5)
        mf = np.clip((right - x_vals) / (right - center), 0, 1)
        mf[x_vals <= left] = 1.0  # Flat shoulder at 1

    elif center_idx == len(centers) - 1:
        # Right shoulder (Fast): extend right
        left = centers[center_idx - 1]
        right = center + (center - left) * 1.5
        mf = np.clip((x_vals - left) / (center - left), 0, 1)
        mf[x_vals >= right] = 1.0  # Flat shoulder at 1

    else:
        # Middle triangular (Normal)
        left = centers[center_idx - 1]
        right = centers[center_idx + 1]
        mf = np.clip((x_vals - left) / (center - left), 0, 1) * \
             np.clip((right - x_vals) / (right - center), 0, 1)

    return mf


labels = ['Slow', 'Normal', 'Fast']
# Flatten arrays to ensure they are 1D
x_vals_flat = x_vals.flatten()
universe_flat = sales.universe.flatten()

for i, label in enumerate(labels):
    mf = create_membership_function(i, sorted_centers)
    # Interpolation function to project the MF onto the fuzzy universe
    interp_func = interp1d(x_vals_flat, mf, bounds_error=False, fill_value=0)
    # Evaluate it over the universe and assign
    interpolated_mf = interp_func(universe_flat)
    # Assign the membership function (must match universe shape)
    sales[label] = interpolated_mf

# Instantiate the FCM model with 3 clusters (Low, Medium, High)
fcm = FCM(n_clusters=3, random_state=42)
fcm.fit(stock_level)

# Get the cluster centers and membership matrix
centers = fcm.centers.flatten()
u = fcm.u  # Membership degrees (shape: n_samples x n_clusters)


# Sort centers and match the memberships accordingly
sorted_idx = np.argsort(centers)
sorted_centers = centers[sorted_idx]
sorted_u = u[:, sorted_idx]

# Plot learned membership functions (fuzzy sets)
# x_vals = np.linspace(0, max(stock_level) * 1.1, 1000)
x_vals = np.linspace(0, max(stock_level) * 1.1, 1000).flatten()


def create_membership_function(center_idx, centers):
    center = centers[center_idx]
    mf = np.zeros_like(x_vals)  # Ensure correct shape

    if center_idx == 0:
        # Left shoulder (Slow): extend left
        right = centers[center_idx + 1]
        left = max(0, center - (right - center) * 1.5)
        mf = np.clip((right - x_vals) / (right - center), 0, 1)
        mf[x_vals <= left] = 1.0  # Flat shoulder at 1

    elif center_idx == len(centers) - 1:
        # Right shoulder (Fast): extend right
        left = centers[center_idx - 1]
        right = center + (center - left) * 1.5
        mf = np.clip((x_vals - left) / (center - left), 0, 1)
        mf[x_vals >= right] = 1.0  # Flat shoulder at 1

    else:
        # Middle triangular (Normal)
        left = centers[center_idx - 1]
        right = centers[center_idx + 1]
        mf = np.clip((x_vals - left) / (center - left), 0, 1) * \
             np.clip((right - x_vals) / (right - center), 0, 1)

    return mf


labels = ['Low', 'Medium', 'High']
# Flatten arrays to ensure they are 1D
x_vals_flat = x_vals.flatten()
universe_flat = stock.universe.flatten()

for i, label in enumerate(labels):
    mf = create_membership_function(i, sorted_centers)
    # Interpolation function to project the MF onto the fuzzy universe
    interp_func = interp1d(x_vals_flat, mf, bounds_error=False, fill_value=0)
    # Evaluate it over the universe and assign
    interpolated_mf = interp_func(universe_flat)
    # Assign the membership function (must match universe shape)
    stock[label] = interpolated_mf

# Instantiate the FCM model with 3 clusters (Short, Normal, Long)
fcm = FCM(n_clusters=3, random_state=42)
fcm.fit(lead_time)

# Get the cluster centers and membership matrix
centers = fcm.centers.flatten()
u = fcm.u  # Membership degrees (shape: n_samples x n_clusters)


# Sort centers and match the memberships accordingly
sorted_idx = np.argsort(centers)
sorted_centers = centers[sorted_idx]
sorted_u = u[:, sorted_idx]

# Plot learned membership functions (fuzzy sets)
x_vals = np.linspace(0, max(lead_time) * 1.1, 1000).flatten()


def create_membership_function(center_idx, centers):
    center = centers[center_idx]

    if center_idx == 0:
        # Left shoulder (Slow): extend left
        right = centers[center_idx + 1]
        left = max(0, center - (right - center) * 1.5)
        mf = np.clip((right - x_vals) / (right - center), 0, 1)
        mf[x_vals <= left] = 1.0  # Flat shoulder at 1

    elif center_idx == len(centers) - 1:
        # Right shoulder (Fast): extend right
        left = centers[center_idx - 1]
        right = center + (center - left) * 1.5
        mf = np.clip((x_vals - left) / (center - left), 0, 1)
        mf[x_vals >= right] = 1.0  # Flat shoulder at 1

    else:
        # Middle triangular (Normal)
        left = centers[center_idx - 1]
        right = centers[center_idx + 1]
        mf = np.clip((x_vals - left) / (center - left), 0, 1) * \
             np.clip((right - x_vals) / (right - center), 0, 1)

    return mf


labels = ['Short', 'Normal', 'Long']
# Flatten arrays to ensure they are 1D
x_vals_flat = x_vals.flatten()
universe_flat = lead.universe.flatten()

for i, label in enumerate(labels):
    mf = create_membership_function(i, sorted_centers)
    # Interpolation function to project the MF onto the fuzzy universe
    interp_func = interp1d(x_vals_flat, mf, bounds_error=False, fill_value=0)
    # Evaluate it over the universe and assign
    interpolated_mf = interp_func(universe_flat)
    # Assign the membership function (must match universe shape)
    lead[label] = interpolated_mf

# Instantiate the FCM model with 3 clusters (Low, Medium, High)
fcm = FCM(n_clusters=3, random_state=42)
fcm.fit(selling_price)

# Get the cluster centers and membership matrix
centers = fcm.centers.flatten()
u = fcm.u  # Membership degrees (shape: n_samples x n_clusters)

# Sort centers and match the memberships accordingly
sorted_idx = np.argsort(centers)
sorted_centers = centers[sorted_idx]
sorted_u = u[:, sorted_idx]

# Plot learned membership functions (fuzzy sets)
x_vals = np.linspace(0, max(selling_price) * 1.1, 1000).flatten()


def create_membership_function(center_idx, centers):
    center = centers[center_idx]

    if center_idx == 0:
        # Left shoulder (Slow): extend left
        right = centers[center_idx + 1]
        left = max(0, center - (right - center) * 1.5)
        mf = np.clip((right - x_vals) / (right - center), 0, 1)
        mf[x_vals <= left] = 1.0  # Flat shoulder at 1

    elif center_idx == len(centers) - 1:
        # Right shoulder (Fast): extend right
        left = centers[center_idx - 1]
        right = center + (center - left) * 1.5
        mf = np.clip((x_vals - left) / (center - left), 0, 1)
        mf[x_vals >= right] = 1.0  # Flat shoulder at 1

    else:
        # Middle triangular (Normal)
        left = centers[center_idx - 1]
        right = centers[center_idx + 1]
        mf = np.clip((x_vals - left) / (center - left), 0, 1) * \
             np.clip((right - x_vals) / (right - center), 0, 1)

    return mf


labels = ['Low', 'Medium', 'High']
# Flatten arrays to ensure they are 1D
x_vals_flat = x_vals.flatten()
universe_flat = price.universe.flatten()

for i, label in enumerate(labels):
    mf = create_membership_function(i, sorted_centers)
    # Interpolation function to project the MF onto the fuzzy universe
    interp_func = interp1d(x_vals_flat, mf, bounds_error=False, fill_value=0)
    # Evaluate it over the universe and assign
    interpolated_mf = interp_func(universe_flat)
    # Assign the membership function (must match universe shape)
    price[label] = interpolated_mf


#Since there’s no label or output in the dataset for Restock Urgency, so just define fuzzy sets manually (e.g., Low, Medium, High) over a normalized range (e.g., 0–10).
# Define membership functions
urgency['Low'] = fuzz.trapmf(urgency.universe, [0, 0, 2, 5])
urgency['Medium'] = fuzz.trimf(urgency.universe, [2.5, 5, 7.5])
urgency['High'] = fuzz.trapmf(urgency.universe, [5, 8, 10, 10])

#rules from the expert
rule1 = ctrl.Rule(sales['Fast'] & stock['Low'] & lead['Long'], urgency['High'])
rule2 = ctrl.Rule(sales['Normal'] & stock['Low'] & lead['Normal'], urgency['Medium'])  # fixed
rule3 = ctrl.Rule(sales['Slow'] & stock['Medium'], urgency['Low'])
rule4 = ctrl.Rule(sales['Fast'] & stock['Low'] & lead['Short'], urgency['High'])
rule5 = ctrl.Rule(sales['Fast'] & stock['High'], urgency['Medium'])
rule6 = ctrl.Rule(sales['Fast'] & stock['Low'] & lead['Long'] & price['Low'], urgency['High'])
rule7 = ctrl.Rule(sales['Normal'] & stock['Medium'] & lead['Short'] & price['High'], urgency['Medium'])
rule8 = ctrl.Rule(sales['Slow'] & stock['High'] & lead['Long'] & price['Medium'], urgency['Low'])  # fixed
#top up rules
# Low sales, low stock, long lead — still can be medium urgency
rule9 = ctrl.Rule(sales['Slow'] & stock['Low'] & lead['Long'], urgency['Medium'])
# Normal sales, high stock — likely low urgency
rule10 = ctrl.Rule(sales['Normal'] & stock['High'], urgency['Low'])
# Fast sales, medium stock, long lead — medium urgency
rule11 = ctrl.Rule(sales['Fast'] & stock['Medium'] & lead['Long'], urgency['Medium'])
# Normal sales, medium stock, long lead — low urgency
rule12 = ctrl.Rule(sales['Normal'] & stock['Medium'] & lead['Long'], urgency['Low'])
# Slow sales, low stock, short lead — medium urgency (stock low, but can arrive fast)
rule13 = ctrl.Rule(sales['Slow'] & stock['Low'] & lead['Short'], urgency['Medium'])
# Fast sales, high stock, short lead — low urgency (already well stocked and can get more fast)
rule14 = ctrl.Rule(sales['Fast'] & stock['High'] & lead['Short'], urgency['Low'])
# Normal sales, low stock, long lead — high urgency
rule15 = ctrl.Rule(sales['Normal'] & stock['Low'] & lead['Long'], urgency['High'])
# Slow sales, high stock, short lead — very low urgency
rule16 = ctrl.Rule(sales['Slow'] & stock['High'] & lead['Short'], urgency['Low'])

urgency_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8,
    rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16
])
urgency_sim = ctrl.ControlSystemSimulation(urgency_ctrl)


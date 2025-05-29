# 🧠 Inventory Restock Urgency Prediction with Fuzzy Logic

This project uses **Fuzzy Logic** to determine the urgency of restocking inventory based on real-world parameters such as sales, stock level, lead time, and price.

---

## 🚀 How to Run

1. **Download the project files** from this repository.
2. Open your terminal and navigate to the project directory.
3. Start the backend server by running:

    ```bash
    python fis_api.py
    ```

4. The server will be running locally at:

    ```
    http://localhost:8080
    ```

---

## 📡 API Usage

### ▶️ Endpoint

To run the urgency prediction, send a **`POST`** request to:

http://localhost:8080/predict

---

### 📨 Input Format (JSON Body)

Pass the data in the following format:

```json
{
  "products": [
    {
      "sales": 380,
      "stock": 130,
      "lead_time": 22,
      "price": 90
    },
    {
      "sales": 100,
      "stock": 400,
      "lead_time": 5,
      "price": 20
    },
    {
      "sales": 210,
      "stock": 280,
      "lead_time": 17,
      "price": 65
    }
  ]
}
```
### 📤 Sample Response:
```json
{
  "results": [
    {
      "sales": 380,
      "stock": 130,
      "lead_time": 22,
      "price": 90,
      "restock_urgency": "High"
    },
    {
      "sales": 100,
      "stock": 400,
      "lead_time": 5,
      "price": 20,
      "restock_urgency": "Low"
    },
    {
      "sales": 210,
      "stock": 280,
      "lead_time": 17,
      "price": 65,
      "restock_urgency": "Medium"
    }
  ]
}
```
___

## 🧠 FIS (Fuzzy Inference System) Details

### 🔍 Membership Function Construction

- Membership functions were built using a **data-driven approach** via **Fuzzy C-Means (FCM)** clustering.
- The **cluster centers** identified by FCM were used to construct:
  - **Left and right shoulder (trapezoidal)** membership functions for extreme cases (e.g., very low or very high values).
  - **Triangular** membership functions for intermediate categories.
- The FCM clustering was applied to **three months** of inventory data extracted from the file:
### 📏 Rule Construction

- Fuzzy rules were crafted based on **expert knowledge and experience**, rather than automated rule generation.
- This approach was necessary because the dataset does **not include labeled restock urgency**, making it impossible to directly learn rules from the data.

### ❓ Why Use FCM + Expert Rules?

- **No labels available** for restock urgency → supervised learning approaches are not feasible.
- FCM provides **data-driven ranges** for fuzzy sets (e.g., Low, Medium, High), ensuring the system adapts to the actual distribution of values in the dataset.
- Expert rules ensure that the logic behind restock decisions reflects **real-world operational knowledge**.

### ⚖️ Why Not Use Expert-Defined Ranges for Membership Functions?

- Predefined numeric ranges from experts might **not generalize well** to inventory data sourced from different companies or domains.
- The meaning of "Low stock" or "High price" can **vary significantly** depending on the dataset context.
- Using FCM ensures that membership functions are aligned with the **actual data distribution**, increasing the model's robustness and flexibility.

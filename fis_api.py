from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fcm_and_fis import urgency_sim, urgency, fuzz
# Initialize FastAPI
app = FastAPI(title="Restock Urgency Predictor")

# Define request schema
class ProductInput(BaseModel):
    sales: float
    stock: float
    lead_time: float
    price: float

class ProductListInput(BaseModel):
    products: List[ProductInput]

# Define response schema
class RestockUrgencyResponse(BaseModel):
    restock_urgency: float
    membership: dict

@app.post("/predict", response_model=List[RestockUrgencyResponse])
def predict_restock_urgency(data: ProductListInput):
    responses = []

    for product in data.products:
        try:
            # Set inputs
            urgency_sim.input['sales'] = product.sales
            urgency_sim.input['stock'] = product.stock
            urgency_sim.input['lead_time'] = product.lead_time
            urgency_sim.input['price'] = product.price

            # Compute output
            urgency_sim.compute()
            output = urgency_sim.output['restock_urgency']

            # Compute membership degrees
            membership = {}
            for label, term in urgency.terms.items():
                degree = fuzz.interp_membership(
                    urgency.universe,
                    term.mf,
                    output
                )
                membership[label] = round(degree, 4)
                print(f"label: {label}, degree: {degree}")
            responses.append({
                "restock_urgency": round(output, 2),
                "membership": membership
            })

        except KeyError:
            responses.append({
                "restock_urgency": -1,
                "membership": {"error": "'restock_urgency' not found"}
            })
        except Exception as e:
            responses.append({
                "restock_urgency": -1,
                "membership": {"error": str(e)}
            })

    return responses

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
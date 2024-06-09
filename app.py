from fastapi import FastAPI
from pydantic import BaseModel
from recommendation_model import recommend_technician
import pandas as pd


app = FastAPI()

class RequestBody(BaseModel):
    query: str


@app.post("/predict")
async def predict(item: RequestBody):
    # load techicians data
    technicians_df = pd.read_csv('technicians.csv')

    result: pd.DataFrame = recommend_technician(item.query, technicians_df)
    print(result.to_dict())

    return {
        "code": 200,
        "message": "success",
        "data": result.to_dict(),
    }

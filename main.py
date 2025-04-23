from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pickle
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

bert = SentenceTransformer('bert-base-nli-mean-tokens')


with open("ReviewDetection.pkl", "rb") as f:
    model = pickle.load(f)


# Define input data model
class ReviewInput(BaseModel):
    message: str

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Route to serve index.html
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html") as s:
        return HTMLResponse(content=s.read(), status_code=200)


@app.get("/status")
def read_root():
    return {"message": "Review Detection is up and running"}

@app.post("/predict/")
def predict_review(data: ReviewInput):
    print("Received message:", data.message)  # Debugging line
    try:
        embedding = bert.encode([data.message])
        prediction = model.predict(embedding)
        return {"Prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")






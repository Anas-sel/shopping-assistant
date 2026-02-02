# TODO: Import your package, replace this by explicit imports of what you need
from shoppingassistant.main import suggest_articles
from shoppingassistant.helper_functions import get_prod_name
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import base64
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }

# Endpoint for http://localhost:8000/predict?image_path=https://assets.adidas.com/images/w_600,f_auto,q_auto/7a42ee6702ea43b49b8c3ada469e818f_9366/Ultrarun_5_Running_Shoes_Black_IE8794_HM1.jpg&top_k=5
@app.get("/predict")
def get_predict(image_path: str, top_k: int = 5, subcategory: str = None, gender: str = None):
    suggestions = suggest_articles(image_path, top_k=top_k, subcategory=subcategory, gender=gender)
    return suggestions
    # return [
    #     {'name': get_prod_name(p),
    #      'data': base64.b64encode(Path(p).read_bytes()).decode(),
    #      'description': "Description for Image",
    #      'subcategory': subcategory if subcategory else "Unknown",
    #      'gender': gender if gender else "Unknown"
    #      }
    #     for p in sugg_paths
    # ]

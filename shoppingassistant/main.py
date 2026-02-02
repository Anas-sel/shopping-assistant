
from PIL import Image
import requests
from pathlib import Path
import os
from shoppingassistant.params import *
from shoppingassistant.classification import classify_subcategory, classify_gender
from shoppingassistant.clustering import get_similar_items
from shoppingassistant.suggestions import suggest_from_sales
from tensorflow.keras.models import load_model
from shoppingassistant.helper_functions import get_image, display_suggestions, get_prod_name, get_price
import base64


def load_model_class():
    """
    Loads and returns the pre-trained model for classification.
    """
    pass

def save_model(model):
    """
    Saves the given model either to mlflow if MODEL_TARGET is 'mlflow'
    or to a local path if MODEL_TARGET is 'local'.
    """

    pass


def suggest_articles(image_path, top_k=5, subcategory=None, gender=None):
    """
    Suggest similar items based on the input image using the provided model.

    Args:
        image_path (str): Path to the input image, can additionally provide url.
        model: Pre-trained model for feature extraction and similarity computation.
               If no model is provided, models will be loaded using load_model function.
        top_k (int): Number of similar items to suggest.

    Returns:
        list: A list of image paths for similar images.
    """
    assert top_k > 0, "top_k must be a positive integer"
    assert top_k <= 10, "top_k must be less than or equal to 10 to avoid long processing times"
    image_path = get_image(image_path)

    if subcategory is None:
        model_subcat_class = load_model(os.path.join(BASE_DIR, 'models', 'subcategory_classifier_best.keras'))
        subcategory = classify_subcategory(image_path, model_subcat_class)


    if gender is None:
        model_gender_class = load_model(os.path.join(BASE_DIR, 'models', 'gender_classifier.keras'))
        gender = classify_gender(image_path, model_gender_class)

    print(f"Predicted category: {subcategory}")
    print(f"Predicted gender: {gender}")


    # Further processing to suggest similar items based on category_pred
    # Get n+1//2 similar items and the rest from sales data
    similar_articles = get_similar_items(image_path, subcategory=subcategory, gender=gender, n=(top_k+1)//2)


    similar_images = []
    for item in similar_articles:
        similar_images.append(item['image_path']) # or article_id

    sales_suggestions = suggest_from_sales(similar_images)
    # if len(sales_suggestions) < (top_k - len(similar_images)):  # in case not enough sales suggestions fill out with similar articles
    suggestions = similar_images + sales_suggestions
    suggestions = suggestions[:top_k]
    # Return image object instead of path using Image
    # put images in docker image or Drive if needed
    # return suggestions
    return [
        {'name': get_prod_name(p),
         'data': base64.b64encode(Path(p).read_bytes()).decode(),
         'description': f"Similar item #{i+1}" if i<len(similar_images) else f"Sales based suggestion #{i+1 - len(similar_images)}",
         'subcategory': f"Subcategory: {subcategory}",
         'gender': f"Gender: {gender}",
         'price': get_price(p)
         'path': p
         }
        for i, p in enumerate(suggestions)
    ]


if __name__ == "__main__":
    # Example usage
    from shoppingassistant.helper_functions import get_image_path, display_results
    from shoppingassistant.process_data import load_dataframes

    articles_df, transactions_df = load_dataframes()
    article_id = articles_df.iloc[10]['article_id']
    image_path = get_image_path(article_id)
    print(f"Image path for article ID {article_id}: {image_path}")
    image = BASE_DIR + "/raw_data/test_images/temp.jpg"
    # output = get_similar_items(image_path, n=5, subcategory='Sandals', gender='Menswear')
    # display_results(image, output)
    suggestions = suggest_articles(image, top_k=4)
    print("Suggested similar items:", suggestions)
    display_suggestions(suggestions)

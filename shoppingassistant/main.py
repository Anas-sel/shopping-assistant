
from PIL import Image
import requests
from pathlib import Path
import os
from shoppingassistant.params import *
from shoppingassistant.classification import classify_subcategory, classify_gender
from shoppingassistant.clustering import get_similar_items
from tensorflow.keras.models import load_model
from shoppingassistant.helper_functions import get_image


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


def suggest_articles(image_path, model=None, top_k=5):
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
    image_path = get_image(image_path)

    if model is None:
        model_subcat_class = load_model(os.path.join(BASE_DIR, 'models', 'subcategory_classifier_best.keras'))
        subcategory_pred = classify_subcategory(image_path, model_subcat_class)

        model_gender_class = load_model(os.path.join(BASE_DIR, 'models', 'gender_classifier.keras'))
        gender_pred = classify_gender(image_path, model_gender_class)
    else:
        subcategory_pred = classify_subcategory(image_path, model)
        gender_pred = classify_subcategory(image_path, model)

    print(f"Predicted category: {subcategory_pred}")
    print(f"Predicted gender: {gender_pred}")


    # Further processing to suggest similar items based on category_pred
    similar_articles = get_similar_items(image_path, subcategory=subcategory_pred, gender=gender_pred)

    similar_images = []
    for item in similar_articles:
        similar_images.append(item['image_path']) # or article_id

    # Return image object instead of path using Image
    # put images in docker image or Drive if needed
    return similar_images


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
    similar_images = suggest_articles(image, top_k=5)
    #display(Image(image_path))
    print("Similar images:", similar_images)

from tensorflow.keras.utils import to_categorical
from helper_functions import preprocess_single_image, load_images_and_labels

def classify_subcategory(image_path, model=None):
    """
    Classifies the subcategory of the item in the image located at image_path
    using the provided model according to the column product_type_name from articles.csv
    Return the subcategory as a string from product_type_name
    """
    _, _, categories = load_images_and_labels(target_column='product_type_name', num_images=1)
    img_array = preprocess_single_image(image_path)

    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    predicted_category = categories[predicted_idx]

    return predicted_category




def classify_gender(image_path, model):
    '''
    Classifies the gender of the item provided according to the column index_group_name from articles.csv
    Returns the gender as a string from index_group_name
    '''

    pass


if __name__ == "__main__":
    pass

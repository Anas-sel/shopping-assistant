from pathlib import Path
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

def get_image_path(article_id: str) -> str:
    '''
    Docstring for get_image
    Given an article_id, return the image path
    '''
    image_path = Path("../raw_data/images_filtered/")
    article_str = str(article_id).zfill(10)
    subfolder = article_str[:3]
    image_file = image_path / subfolder / f"{article_str}.jpg"
    if image_file.exists():
        return str(image_file)
    else:
        raise FileNotFoundError(f"Image for article_id {article_id} not found.")

def get_image_paths(article_ids):
    '''
    Docstring for get_image_paths
    Given a list of article_ids, return a list of image paths
    '''
    paths = []
    for article_id in article_ids:
        try:
            path = get_image_path(article_id)
            paths.append(path)
        except FileNotFoundError:
            paths.append(None)
    return paths

# def get_category_labels_from_strs(categories):
#     '''
#     Docstring for get_category_mappings
#     Returns a dictionary mapping product_type_name to index_group_name
#     '''
#     subcategories =['Boots', 'Sneakers', 'Other shoe', 'Sandals', 'Slippers',
#        'Ballerinas', 'Flat shoe', 'Wedge', 'Pumps', 'Flip flop', 'Bootie',
#        'Heeled sandals', 'Flat shoes', 'Heels', 'Moccasins',
#        'Pre-walkers']
#     subcategories_mapping = { i:subcategories[i] for i in range (len(subcategories))}

# def get_category_strs_from_labels():

#     pass


def load_images_and_labels(target_column='product_type_name', num_images=None):
    """
    Load images and labels for classification.
    Works for both subcategory and gender classification

    Returns:
        X (numpy array): Image data of shape (n_samples, 256, 256, 3)
        y (numpy array): Labels (NOT one-hot encoded yet)
        categories (list): List of unique category names
    """

    df = pd.read_csv('../raw_data/articles_filtered.csv')

    if num_images is not None:
        df = df.head(num_images)
        print(f"Using {num_images} images")
    else:
        print(f"Using all {len(df)} images")

    categories = sorted(df[target_column].unique())
    print(f"Categories ({len(categories)}): {categories}")

    # Category to index mapping
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}

    images = []
    labels = []

    # Load all images
    for idx, row in df.iterrows():
        article_id = row['article_id']
        category = row[target_column]

        try:
            image_path_str = get_image_path(article_id)
            image_path = Path(image_path_str)
        except FileNotFoundError:
            # Image doesn't exist, skip it
            continue

        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((256, 256), Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0

            images.append(img_array)
            labels.append(category_to_idx[category])
        except Exception as e:
            print(f"Skipping {article_id}: {e}")
            continue

    X = np.array(images)
    y = np.array(labels)

    print(f" Loaded {len(images)} images")
    print(f" X shape: {X.shape}")
    print(f" y shape: {y.shape}")

    return X, y, categories


def split_train_val(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and validation sets
    """
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        #stratify=y
    )

    print(f"Training set: {X_train.shape[0]} images ({int((1-test_size)*100)}%)")
    print(f"Validation set: {X_val.shape[0]} images ({int(test_size*100)}%)")

    return X_train, X_val, y_train, y_val


def preprocess_single_image(image_path):
    """
    Preprocess image array ready for model.predict()
    """

    if isinstance(image_path, str):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((256, 256), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
    else:
        img_array = image_path

    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)

    return img_array

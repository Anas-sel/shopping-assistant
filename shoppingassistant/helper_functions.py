from pathlib import Path
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from shoppingassistant.params import *

def get_image_path(article_id: str) -> str:
    '''
    Docstring for get_image
    Given an article_id, return the image path
    '''
    image_path = Path(BASE_DIR + "/raw_data/images_filtered/")
    article_str = str(article_id).zfill(10)
    article_str = article_str.replace('.jpg', '')
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

def display_results(query_image_path, results):
    '''
    Function to display image results from similar_items function in a grid format
    '''

    import matplotlib.pyplot as plt
    from PIL import Image
    n_results = len(results)
    fig, axes = plt.subplots(1, n_results + 1, figsize=(4 * (n_results + 1), 5))

    # Display query image
    query_img = Image.open(query_image_path)
    axes[0].imshow(query_img)
    axes[0].set_title('QUERY IMAGE', fontsize=12, fontweight='bold', color='blue')
    axes[0].axis('off')

    # Display similar images
    for i, item in enumerate(results):
        img_path = get_image_path(item['filename'])
        img = Image.open(img_path)

        axes[i + 1].imshow(img)
        axes[i + 1].axis('off')
        axes[i + 1].set_title(f"#{i+1} - Similarity: {item['similarity']:.3f}", fontsize=10)
        axes[i + 1].set_xlabel(
            f"{item['prod_name']}\n{item['product_type_name']} | {item['colour_group_name']}\n{item['index_group_name']}",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()

    # Print detailed info
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    for i, item in enumerate(results):
        print(f"\n#{i+1} | Similarity: {item['similarity']:.4f}")
        print(f"   Article ID: {item['article_id']}")
        print(f"   Name: {item['prod_name']}")
        print(f"   Subcategory: {item['product_type_name']}")
        print(f"   Color: {item['colour_group_name']}")
        print(f"   Gender: {item['index_group_name']}")

def load_images_and_labels(target_column='product_type_name', num_images=None):
    """
    Load images and labels for classification.
    Works for both subcategory and gender classification

    Returns:
        X (numpy array): Image data of shape (n_samples, 256, 256, 3)
        y (numpy array): Labels (NOT one-hot encoded yet)
        categories (list): List of unique category names
    """

    df = pd.read_csv(BASE_DIR + "/raw_data/articles_filtered.csv")

    # Remove Sport and Divided category for Gender classification
    if target_column == 'index_group_name':
        classes_to_remove = ['Sport',  'Divided']
        df = df[~df['index_group_name'].isin(classes_to_remove)].reset_index(drop=True)

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

def get_image(image_path):
    '''
    Given an image path, return the same input.
    Given a url, download the image, save it under raw_data/test_images and return the local path.
    '''
    from pathlib import Path
    import requests
    from PIL import Image
    import os
    from shoppingassistant.params import BASE_DIR
    if Path(image_path).exists():
        # Load image from local path
        return image_path
    else:
        # Load image from URL
        image_url = requests.get(image_path, stream=True).content
        with open(os.path.join(BASE_DIR, 'raw_data', 'test_images', 'temp.jpg'), 'wb') as f:
            f.write(image_url)
        return str(BASE_DIR / 'raw_data' / 'test_images' / 'temp.jpg')

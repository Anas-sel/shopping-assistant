from pathlib import Path
import os

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

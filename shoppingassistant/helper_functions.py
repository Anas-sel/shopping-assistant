from pathlib import Path
import os

def get_image_path(article_id: str) -> str:
    '''
    Docstring for get_image
    Given an article_id, return the image path
    '''
    image_path = Path("../raw_data/images_filtered/")
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

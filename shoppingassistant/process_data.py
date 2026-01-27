
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import shutil
from pathlib import Path

def filter_data(product_group_names=['Shoes']):
    '''
    Filters the dataset to only keep the rows corresponding to the specified
    product group names. Filters over the column 'product_group_name'
    for the articles.csv, transactions.csv and images.
    Removes rows that do not have corresponding images.
    Saves the filtered data to articles_filtered.csv and transactions_filtered.csv.
    '''
    path_to_data = Path("../raw_data/")
    if not (path_to_data / "articles_filtered.csv").exists() or not (path_to_data / "transactions_filtered.csv").exists():
        # Loading Data
        articles_df = pd.read_csv(path_to_data / "articles.csv")
        transactions_df = pd.read_csv(path_to_data / "transactions.csv")
        images_path = path_to_data / "images_256_256"

        # Filtering Data using input product group names
        articles_df_filtered = articles_df[articles_df['product_group_name']==product_group_names]
        transactions_df['t_dat']= pd.to_datetime(transactions_df['t_dat'])
        df_trans_filtered = transactions_df[transactions_df['article_id'].isin(articles_df_filtered['article_id'])]

        # Creating images_filtered directory that will contain only the images of the filtered articles
        needed_article_ids = articles_df_filtered['article_id'].unique()
        source_images_path = Path('../raw_data/images_256_256')
        dest_images_path = Path('../raw_data/images_filtered')
        copied_count = 0
        missing_count = 0

        for article_id in needed_article_ids:
            article_str = str(article_id).zfill(10)
            subfolder = article_str[:3]

            source_subfolder = source_images_path / subfolder
            dest_subfolder = dest_images_path / subfolder
            source_file = source_subfolder / f"{article_str}.jpg"
            dest_file = dest_subfolder / f"{article_str}.jpg"

            if source_file.exists():
                dest_subfolder.mkdir(exist_ok=True)

                shutil.copy2(source_file, dest_file)
                copied_count += 1
            else:
                missing_count += 1
                articles_df_filtered.drop(articles_df_filtered[
                    articles_df_filtered['article_id'] == article_id].index, inplace=True)
                df_trans_filtered.drop(df_trans_filtered[
                    df_trans_filtered['article_id'] == article_id].index, inplace=True)

        print(f"Copied {copied_count} images")
        print(f"Missing images: {missing_count}")
        print(f"Total article IDs: {needed_article_ids.shape}")
        articles_df_filtered.to_csv("../raw_data/articles_filtered.csv")
        df_trans_filtered.to_csv("../raw_data/transactions_filtered.csv")
        print(f"✅ Filtered data saved to articles_filtered.csv and transactions_filtered.csv")
    else:
        print("✅ Filtered data already exists.")
    return None



def preprocess(keep_colors=False):
    '''
    Docstring for preprocess
    Preprocess the data
    '''
    path_to_data = Path("../raw_data/")
    filter_data()

    transactions_df = pd.read_csv(path_to_data / "transactions_filtered.csv")
    articles_df = pd.read_csv(path_to_data / "articles_filtered.csv")

    # Additional preprocessing steps can be added here
    transactions_df = transactions_df[['t_dat', 'article_id', 'price', 'customer_id']]
    transactions_df['t_dat']= pd.to_datetime(transactions_df['t_dat'])

    # Drop color-related columns if keep_colors is False
    if not keep_colors:
        articles_df = articles_df.drop(columns=['colour_group_code',
                                                'colour_group_name',
                                                'perceived_colour_value_id',
                                                'perceived_colour_value_name',
                                                'perceived_colour_master_id',
                                                'perceived_colour_master_name'])

    # Only select relevant columns
    relevant_columns = ['article_id','product_code', 'product_type_name', 'product_group_name',
                        'index_group_name']
    if keep_colors:
        relevant_columns += ['perceived_colour_master_name']

    articles_df = articles_df[relevant_columns]

    return articles_df, transactions_df




def sort_by_income(article_ids):
    '''
    Docstring for highest_income_products

    Returns a sorted list of the article_ids by highest income
    '''
    pass

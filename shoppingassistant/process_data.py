
def filter_data(product_group_names=['Shoes']):
    '''
    Filters the dataset to only keep the rows corresponding to the specified
    product group names. Filters over the column 'product_group_name'
    for the articles.csv, transactions.csv and images.
    Removes rows that do not have corresponding images.
    Saves the filtered data to articles_filtered.csv and transactions_filtered.csv.
    '''
    pass


def preprocess():
    '''
    Docstring for preprocess
    Preprocess the data
    '''
    pass


def sort_by_income(article_ids):
    def sort_by_income(article_ids):
    """
    Returns a list of article_ids sorted by highest total income.

    Income is computed as the sum of 'price' per article_id
    from transactions_filtered.csv.
    """
    # Load filtered transactions
    df_trans = pd.read_csv("../raw_data/transactions_filtered.csv")

    # Sum income per article
    income_per_article = df_trans.groupby("article_id")["price"].sum()

    # Sort article_ids by income (highest first)
    sorted_ids = income_per_article.reindex(article_ids).fillna(0).sort_values(ascending=False).index.tolist()

    return sorted_ids

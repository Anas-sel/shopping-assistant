# Your Personal Shopping Assistant

=======
## Algorithm:
### General description:
The algorithm takes an image path or a url as input, performs a double classification into
subcategory and gender for the image. Then the model embeds the given image into an embedding space
and search for the closest images in that space from the database (in distance). This would constitute a similar image.

To give suggestions, we follow two rules:
- Half the requested suggestions would be from similarity search.
- For the other half, each suggestion corresponds to the item most sold with each of the results from the similarity search.

P.S: In case the item determined from similarity search has no sales history on its own, has no other items it is usually
sold with, perform another similarity search on it, and give back the item with the most revenue from this new list.

## Classification for subcategory:
Please run meerim_test_classifier and check last section how to load the model

## Clustering model
The jupyter notebook 'notebooks/kyrylo_test2' needs to be run before using the function
in clustering.py so that some files get created.

## Classification for Gender:
Need to run notebook under 'notebooks/shweta_test_classifier' to train the model and save it under
'models/gender_classifier'

Use pre-trained model 'EfficientNetB3' with addition weights since the classes are not balanced
Used callback with 'ReduceLROnPlateau' to reduce learning rate if model is not improving for 10 epochs
====================================================================================================
Accuracy: 83.27%
precision: 0.7734982728204024
recall: 0.8149471680577286
====================================================================================================

## Service url:
https://api-520917056692.europe-west1.run.app

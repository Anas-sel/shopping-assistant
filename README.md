# Your Personal Shopping Assistant

=======
## Classification for subcategory:
Please run meerim_test_classifier and check last section how to load the model

## Clustering model
The jupyter notebook 'notebooks/kyrylo_test2' needs to be run before using the function
in clustering.py so that some files get created.

## Classification for Gender:
Need to run notebook under 'notebooks/shweta_test_classifier' to train the model and save it under
'models/gender_classifier'

Use pre-trained model 'EfficientNetB2' with addition weights since the classes are not balanced
Used callback with 'ReduceLROnPlateau' to reduce learning rate if model is not improving for 10 epochs
====================================================================================================
Accuracy: 84.93%
precision: 0.7857665466812138
recall: 0.8040785880637217
====================================================================================================

## Service url:
https://api-520917056692.europe-west1.run.app

Python scripts to classify large number of images.
The model creation is bound for images of 448x448 pixels, then all the training and classify images are bound to this pixel proportions, images can be larger and disproportionate.

Standard train/test/validate process has been omitted since this is non critical process.

## Binary classifier
Put into one folder a set of positive example images and put into the other folder a set of non positive images, this non-positive set is recommended to be as most variated as possible since we are partitioning an space into two sections, the positive one will be possible an small portion and the negative one the rest.

Steps:
- Train model with similar and non similar images as classes {0,1}. PRE: input folder has a folder structure with images within then it is a folder with 1 level deep folders.
`python3 ./scripts/train_binary.py input_folder_similar input_folder_not_similar model_destination_path`
- Compare and move images with classification == 1 to the target folder maintaining the input folder structure.
`python3 ./scripts/compare_and_move_binary.py input_folder output_folder model_path`

The model utilised is a Support Vector Machine in linear mode.


## Trinary classifier
Put into three folders a set of three image classes, a, b and c.

Steps:
- Train model with the three sets of images with classes {0,1,2}. PRE: input class folders have a 0 deep level directory.
`python3 ./scripts/train_trinary.py input_class0_folder input_class1_folder input_class2_folder model_path`
- Compare and move images into three folder corresponding to their class. PRE: input folder has a 1 level deep folders.
`python3 ./scripts/compare_and_move_trinary.py input_folder class0_folder class1_folder class2_folder model_path`

Same model utilised as binary classifier.


## KMeans classifier
Unsupervised learning method, it takes a one input folder with 0 deep level folder and takes all the images to separate into n_clusters. Then using the pre-trained model, predict at which cluster is the current image and send to the correspondent cluster folder.

Steps:
- Train model with the specified n_clusters and specified max_files (upper bound if the folder has more than these files). PRE: input folders has 0 deep level folders.
`python3 ./scripts/train_clustering.py input_folder model_path --n_clusters 9 --max_files 1000`
- Compare and move images to their correspondent cluster folder, cluster folders will be created if are not at output_dir. PRE: input_folder has 0 deep level folder.
`python3 ./scripts/move_to_clusters.py input_folder output_folder model_path`


## Misc
- Move images with width > 800 or height > 800 to the target directory.
`python3 ./scripts/move_images.py input_folder destination_folder`
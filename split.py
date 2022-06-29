####
#
# The code bellow splits the dataset into 3 subfolders: train, val and test.
# Train will have 80% of the total data while val and test will both have 10% equaly of the toatal data.
# The program can be run if needed on the console by typing "python split.py"
#
####
import splitfolders

input_folder = "./nih-malaria/"

splitfolders.ratio(input_folder,output="./nih-malaria/cell_images",seed=1337,ratio=(.8,.1,.1),group_prefix=None)


# eye-mouse
Control your PC's mouse by just looking at the screen! (proof of concept)
This program uses multi-class image classifier to predict which part of the screen the user is looking at.
The output is visualized using matplotlib 

The model included only works for PCs with cameras at the top of the screen
Additionally the model has been trained only on one person.

use data-generator.py to generate your own data!
Note: you will have to specify the class before running the generator


# Data Generator:
Remember to modify coordinate values before running the program (Images generated will have these coordinates in their file name)

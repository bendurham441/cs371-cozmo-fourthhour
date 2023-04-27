# CS371 Fourth Hour Project: Anki Cozmo Monte Carlo Localization

Gettysburg College Computer Science
CS 371: Artificial Intelligence

**Authors**: Binh Tran, Owen Gilman, Spencer Hagan, Ben Durham

## Description

In this project, we write a program to help localize the Anki Cozmo robot rotationally. To do this, we first take many pictures of the robot's environment to help it "get its bearings." The first picture it takes denotes it's "home position" Then, after a random rotation, either by external intervention or programatically, the robot should be able to take pictures to try to figure out where it is. From this information, it should then rotate back to its home position.

We apply Monte Carlo localization in the robot's attempt to find where it is. As such, we start with a randomly generated population of possible rotational positions. Then, based on the robot's movement and the image difference with its current location, it makes a new generation of possible positions, with emphasis on the locations that seem most likely, as determined by the image difference between the robot's current position and each of the possible positions.

We have found this method to be successful under most conditions, in multiple different environments. However, there have been some (rare) cases where the method fails and rotates back to a position that is not the home position.

## Setup Instructions:
- TBD

## How to run the files:

### Image Gathering

One should first run the `ImageGathering.ipynb` which creates an `images/` directory and populates it with images. Note that you need to update the `parent_dir` variable in the second code block of this file to reflect the actual file path on your computer. After this, this file can be run top to bottom as a whole. As stated before, the first image taken becomes the robot's home location, so place the robot according to which direction you want to be the home position.

### Sliding Window Localize

After the `images/` folder is populated, one can now run the contents of `SlidingWindowLocalize.ipynb`. Note that this file contains a programatic random rotation of the robot for testing purposes, but is commented out to allow for manual random rotations of the robot. In its current state, the robot will wait for the user to rotate it, then it will start the localization process.

As the robot continues to iterate, this file will print out histograms to the Jupyter Notebook output showing the robot's beliefs about where it is currently facing, which can be useful in seeing the robot zero in on its actual location. These histograms simply show how many locations in the current population are in each "bin."


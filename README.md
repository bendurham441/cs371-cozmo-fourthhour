# CS371 Fourth Hour Project: Anki Cozmo Monte Carlo Localization

Gettysburg College Computer Science
CS 371: Artificial Intelligence

**Authors**: Binh Tran, Owen Gilman, Spencer Hagan, Ben Durham

## Description

In this project, we write a program to help localize the Anki Cozmo robot rotationally. To do this, we first take many pictures of the robot's environment to help it "get its bearings." The first picture it takes denotes it's "home position" Then, after a random rotation, either by external intervention or programatically, the robot should be able to take pictures to try to figure out where it is. From this information, it should then rotate back to its home position.

We apply Monte Carlo localization in the robot's attempt to find where it is. As such, we start with a randomly generated population of possible rotational positions. Then, based on the robot's movement and the image difference with its current location, it makes a new generation of possible positions, with emphasis on the locations that seem most likely, as determined by the image difference between the robot's current position and each of the possible positions.

We have found this method to be successful under most conditions, in multiple different environments. However, there have been some (rare) cases where the method fails and rotates back to a position that is not the home position.

# Setup Instructions:
- TBD
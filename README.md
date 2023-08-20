
# Introduction
This project based on the dataset [ContactPose: A Dataset of Grasps with Object Contact and Hand Pose](https://github.com/facebookresearch/ContactPose). The goal is to levarege the quantity of hand and grasping poses in order to find a lower dimensional interconnection between the involved joints. This could reduce the control effort for robotics hands. To this purpose a kinematic model of a human hand is developed to simulate the grasping movement, instead of just singular poses. The sequqnces of the movement are then analyzed with PCA and ICA algorithms in order to find lower dimensional grasping configurations. The results are than applied on the RBO Hand 3.\
A report of the setup and results of the project can be found here: [Contact Pose Report](https://github.com/Kyrake/Contact-Pose/blob/main/report/Contact%20Pose%20Report.pdf)



## Kinematic Model



https://github.com/Kyrake/Contact-Pose/assets/142335932/1beab1b5-49f8-47b0-b5d6-e18e6b1d56bb




## Results




https://github.com/Kyrake/Contact-Pose/assets/142335932/0eb49e1c-6e56-4317-8b86-7099a65b4255


# Installation

1. First follow the steps to get started with the ContactPose framework: https://github.com/facebookresearch/ContactPose/blob/master/docs/doc.md
2. Copy ContactPose/data/contactpose_data/full1_use and store it in a new directory ContactPose/data/contactpose_data/full51_use . This is important as the embedded open3D simulator just accepts this kind of structure as input in order to visualize our results

## Project Setup


The following image shows the basic pipeline on how the files work together:

![Pipeline](Diag.png)


1. Preprocessing.py:
    A subset of the hand postures provided by the ContactPose framework can be filtered in stored in an array. Types of objects, number of people
    and modi can be adjusted. At the moment both modi are used. Furthermore the resulting array is stored as .npy file for further usage.
    
2. KinematicModel.py:
    This file inhibits all the functions relating to the Kinematic Model. The functions for calculation the angels for each finger, getting link length     of a handposture, and the homogeneous transformations for each finger lie here.
    IMPORTANT: Here lies also the function transformation(). This function orientates all hand postures in such a way, that the indexfinger is the         x-axis and the index and middle finger are spanning the xy plane. This is the basis for calculating the angles, as all hand postures must have         their wrists in (0,0)
    
3. CreatetHand.py:
   This file creates a  hand posture in zero configuration or with own chosen angles with the function createPuppet(). On the other hand modifyPuppet() 
   is used for the angles calculated by KinematicModel.py. It also uses the homogenous transformations to get the cartesian coordinates of the angles 
   average link length and finally makes it possible to visualize the hand posture
   
4. PCA.py, ICA.py:
   Those file ares using PCA or ICA on the stored array from Preprocessing.py respectively. 
   
5. FilterFunctions.py
   Functions that allow easy generation of subsets. For instant all joint coordinates of all people holding an apple, one person using each object etc.
   Those subset aren't stored as file, but just during run time. Also this file inhibits the function that transforms the joint coordinate in a json      format, that is usable for the Open3D simulator
   
5. Handposeanimator.py:
   This class handposeanimator is the foundation for the animation of the grasping movements that results from the pca and ica analysis.  
   
6. Visualization.py:
   Stores hand posture in cartesian coordinates in json file. The embedded open3D simulation environment provided by the ContactPose dataset, is very
   particular about the structure and naming of the input file, therefore creating a new json file should have the same structure as the files in ContactPose/data/contactpose_data. Best copy one of the already provided folders and continue with the numeration e.g: ContactPose/data/contactpose_data/full51_use/apple is the default path for storing hand posture you want to visualize


# Further Improvement :
1. Add joint constraints directly to the kinematic model. Constraint limits are suggested in http://www.ifp.illinois.edu/~yingwu/papers/Humo00.pdf

# Known Bugs :
1. If a json file gets corrupted or the structure gets disrupted, copy & paste the json file of the provided contact_pose data in order to restore the needed structre. Than you can simply overvwrite the joints entries with new hand postures and the function provided by Visualization.py
2. If you use functions of scripts/show_contactmap.py (for animation e.g), sometimes it is required to import import scripts.init_paths in show_contactmap.py instead of import init_paths. But if call the Open 3D visualization from the terminal, it has to be set just to import init_paths.












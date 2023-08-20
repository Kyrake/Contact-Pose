import matplotlib.pyplot as plt
import numpy as np
import transforms3d
import math
import sklearn
import json
from scipy.spatial.transform import Rotation as R
import utilities.init_paths
import utilities.dataset as ds
from utilities.dataset import ContactPose
import FilterFunctions as ff

import utilities.misc as mutils



# Transforms Basis of PersonxObject Relation to Basis,
# where the Metacarpal of the Index Finger is X axis and spans a plane with MC of Middle Finger
# Input Matrix [Observations, 21, 3], Output: [Observations * 21, 3]

def transformation(allObjects):
    biglist = []
    for i in range(allObjects.shape[0]):
        #print(allObjects.shape[0])
        allObjects[i] = allObjects[i] - allObjects[i][0]
        x_axis = allObjects[i][5]/np.linalg.norm(allObjects[i][5])
        z_axis = np.cross(allObjects[i][5], allObjects[i][9])
        z_axis = z_axis/np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, allObjects[i][5])
        y_axis = y_axis/np.linalg.norm(y_axis)
        basis = np.linalg.inv(np.array([x_axis, y_axis, z_axis]).T)
        for j in range(21):
            base = basis @ allObjects[i][j]
            #base = base + np.array([0,0.2, 0.2])
            biglist.append(base)
    arrays = np.asarray(biglist)
    #print(arrays.shape)
    arrays = arrays.reshape(int(arrays.shape[0]/21), 63)
    return arrays

def transformation3(finger):
    biglist = []

    for i in range(0,3):
        cross = np.cross(finger[i], finger[i+1])
        dot = np.dot(finger[i], finger[i+1].transpose())
        angle = math.atan2(np.linalg.norm(cross), dot)
        #rotation_axes = cross/np.linalg.norm(cross)
        cross = cross.reshape(1,-1)
        rotation_axes = sklearn.preprocessing.normalize(cross)
        rotation_m = transforms3d.axangles.axangle2mat(rotation_axes[0], angle, True)
        rotation_angles = transforms3d.euler.mat2euler(rotation_m, 'sxyz')
        biglist.append(np.degrees(rotation_angles))

    return np.asarray(biglist)

def transformation2(allObjects):
    biglist = []
    #transformed = transformation(allObjects)
    for i in range(allObjects.shape[0]):
        #allObjects[i] = allObjects[i][0]

        biglist.append([0, 0, 0])
        for k in range(20):
            h = allObjects[i][k + 1] - allObjects[i][k]
            #h_trans = transformed[i][k + 1] - transformed[i][k]
            link = np.linalg.norm(h)
            T = np.eye(4)
            T[2, 3] = link
            #print("T1:", T)
            T = mutils.rotmat_from_vecs(h, [0, 0, 1])
            #print("T2:", T)
            T[:3, 3] = allObjects[i][k]
            #print("T3:", T)
            r = R.from_matrix(T[:3, :3])
            base = r.as_euler('zyx', degrees=True)
            biglist.append(base)

    return np.asarray(biglist)

def getVectorJoints(joints):
    wrist = np.array([0,0,0])
    joint1 = joints[0]
    joint2 = joints[1] - joints[0]
    joint3 = joints[2] - joints[1]
    joint4 = joints[3] - joints[2]
    finger = np.array([joint1,joint2,joint3,joint4])
    return finger

def getEulerAngles(Finger):

    biglist = [np.arccos(np.dot(np.array([1,0,0]), Finger[0]/np.linalg.norm(Finger[0]))),
               np.arccos(np.dot(np.array([0,0,1]), Finger[0]/np.linalg.norm(Finger[0]))),
               np.arccos(np.dot(np.array([0,1,0]), Finger[0]/np.linalg.norm(Finger[0])))]

    #print("bug", np.degrees(biglist))
    for i in range(0,3):
        x_axis = Finger[i]/np.linalg.norm(Finger[i])
        z_axis = np.cross(Finger[i], Finger[i+1])
        z_axis = z_axis/np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis/np.linalg.norm(y_axis)
        alpha_aroundz = np.arccos(np.dot(Finger[i]/np.linalg.norm(Finger[i]), Finger[i+1]/np.linalg.norm(Finger[i+1])))
        beta_aroundy = np.arccos(np.dot(z_axis, Finger[i+1]/np.linalg.norm(Finger[i+1])))
        gamma_aroundx = np.arccos(np.dot(y_axis, Finger[i+1]/np.linalg.norm(Finger[i+1])))
        # if( i == 0)    :
        biglist.append(np.degrees(alpha_aroundz))
        biglist.append(np.degrees(beta_aroundy))
        biglist.append(np.degrees(gamma_aroundx))
        #biglist.append(gamma_aroundx)
        # else:
        #     biglist.append(alpha_aroundz)

    return np.asarray(biglist)

def CosineRule(joint1, joint2):


    a = np.abs(np.linalg.norm(joint1))
    c = np.abs(np.linalg.norm(joint2))
    b = np.abs(np.linalg.norm(joint2-joint1))

    gamma = np.arccos((a**2 + b**2 - c**2)/(2 * a * b ))


    return gamma





# jointMatrix = setti.reshape(setti.shape[0], 21,3)
# angles = km.getAngles(jointMatrix[0:1])
# hand = Hand.getHands(jointMatrix[0:1])
# links = Hand.getHandVectorJoints(hand[0])
# angles = angles[6:10]
# IndexFinger = HTrafoIndex(angles, links)
#
#
# print(IndexFinger)


# backtrafo = jointMatrix[0].reshape(63)
# json_joint_list = ff.prepareforJson(backtrafo)
# jsonfile_path = 'data/contactpose_data/full53_use/hammer/annotations.json'
# leftOrRightHand = 1
# with open(jsonfile_path, 'r') as json_file:
#     data = json.load(json_file)
# data["hands"][leftOrRightHand]["joints"] = json_joint_list
# data["hands"][leftOrRightHand]["joints"][1] = MCP[0:3].tolist()
# data["hands"][leftOrRightHand]["joints"][2] = Pip[0:3].tolist()
# data["hands"][leftOrRightHand]["joints"][3] = Dip[0:3].tolist()
# data["hands"][leftOrRightHand]["joints"][4] = Tip[0:3].tolist()
# with open(jsonfile_path, "w") as json_file:
#     json.dump(data, json_file)
# # backtrafo = transfom[0]
# json_joint_list = ff.prepareforJson(backtrafo)
# #print(json_joint_list)
#
#
# jsonfile_path = 'data/contactpose_data/full52_use/hammer/annotations.json'
# leftOrRightHand = 1
# with open(jsonfile_path, 'r') as json_file:
#     data = json.load(json_file)
#
# data["hands"][leftOrRightHand]["joints"] = json_joint_list
# with open(jsonfile_path, "w") as json_file:
#     json.dump(data, json_file)
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import utilities.init_paths
import utilities.dataset as ds
from utilities.dataset import ContactPose
import utilities.misc as mutils
from utilities.import_open3d import *
import json

import os
osp = os.path

def listWithAllObjects(intent):
    list = []
    for i in range(1, 51):
        peeps = ds.get_object_names(i, intent)
        list.extend(x for x in peeps if x not in list)
    return list


def getNumberOfOccurences(object):
    people_use = ds.get_p_nums(object, 'use')
    people_handoff = ds.get_p_nums(object, 'handoff')
    number_use = len(people_use)
    number_handoff = len(people_handoff)
    print('Number of' + ' ' + object + 's' + ' ' +'with' + ' ' +'use'+ ':', number_use)
    print('Number of' + ' ' + object + 's' + ' ' +'with' + ' ' +'handoff'+ ':', number_handoff)

    return number_use, number_handoff


def getSubSetWithoutItems(listOfskippedItems, intent):

    objects1 = ds.get_object_names(2, intent)
    objects1.sort()
    object = set(objects1) - set(listOfskippedItems)
    subseti = getSubSetOfItems(object, intent)
    print("Subset with wanted Items:", subseti)
    return subseti


def getSubSetOfItems(listofwantedItems, intent):
    subset = []
    for object in listofwantedItems:
        for i in range(1,51):
            p_id = 'full{:d}_{:s}'.format(i, intent)
            data_dir = osp.join('data', 'contactpose_data', p_id, object)
            if not osp.isdir(data_dir):
                continue
            cp = ContactPose(i, intent, object)
            joints = cp.hand_joints()
            hand_1 = joints[1]
            if (hand_1 is None):
                continue
            if hand_1 is not None:
                hand_1 = hand_1.reshape(21 * 3)
                subset.append(hand_1)
            #print("Subset with wanted Items:", subset_array.shape)
    return np.asarray(subset)


def allCombs(intent):
    allCombs = []
    list = listWithAllObjects()
    allCombs = getSubSetOfItems(list, intent)

    return (np.array(allCombs))

def getMeanOfObject(object, intent):
    people = ds.get_p_nums(object, intent)
    joint_list = []
    for p in people:
        cp = ContactPose(p, intent, object)
        joints = cp.hand_joints()
        hand_1 = joints[1]
        if (hand_1 is None):
            continue

        if hand_1 is not None:
            #hand_1 = hand_1.reshape(21 * 3)
            joint_list.append(hand_1)
    joint_listi = np.asarray(joint_list)
    #joint_listi = joint_listi - np.mean(joint_listi, axis=0)

    return joint_listi

def ObjectsOnePerson(person, intent):
    people = ds.get_object_names(person, intent)
    people.sort()
    print(people)
    #print(people)
    joint_list = []
    for p in people:
        cp = ContactPose(person, intent, p)
        joints = cp.hand_joints()
        hand_1 = joints[1]
        if (hand_1 is None):
            hand_1 = joints[0]
        if hand_1 is not None:

            #hand_1 = hand_1.reshape(21 * 3)
            joint_list.append(hand_1)
    joint_listi = np.asarray(joint_list)
    #joint_listi = joint_listi - np.mean(joint_listi, axis=0)

    return joint_listi





def prepareforJson(objectToSimulate):

    json_joint_list = []
    for i in range(63):
        if (i % 3 == 0):
            new_listi = []

        new_listi.append(objectToSimulate[i])
        # print(new_listi)
        if (i % 3 == 0):
            json_joint_list.append(new_listi)

    return json_joint_list

def storeInJson(jsonfile_path, leftOrRightHand, putinjson):
    with open(jsonfile_path, 'r') as json_file:
        data = json.load(json_file)

    data["hands"][leftOrRightHand]["joints"] = putinjson

    with open(jsonfile_path, "w") as json_file:
        json.dump(data, json_file)


import TransformationFunctions as tf
import KinematicModel as km
import numpy as np
from sklearn.metrics import mean_squared_error
import FilterFunctions as ff
import json

# Function that gives hand posture  in cartesian coordinates for visualization and in a configuration that all flexion angles are zero - angles changeable for debugging, visualization etc...
# Abduction angles still have to be set, as well as the link length. Both are parameter of the function

def createPuppet(angles_fix, avg_link_length):
    joint_array = np.zeros(63)
    joint_array = joint_array.reshape(21,3)
    thumb_angles = 0
    joint_array[1] = np.array([np.cos(np.deg2rad(thumb_angles))*(avg_link_length[1]), np.sin(np.deg2rad(thumb_angles))*(avg_link_length[1]), 0])
    joint_array[2] = joint_array[1] + np.array([np.cos(np.deg2rad(thumb_angles))*(avg_link_length[2]), np.sin(np.deg2rad(thumb_angles))*(avg_link_length[2]), 0])
    joint_array[3] = joint_array[2] + np.array([np.cos(np.deg2rad(thumb_angles))*(avg_link_length[3]), np.sin(np.deg2rad(thumb_angles))*(avg_link_length[3]), 0])
    joint_array[4] = joint_array[3] + np.array([np.cos(np.deg2rad(thumb_angles))*(avg_link_length[4]), np.sin(np.deg2rad(thumb_angles))*(avg_link_length[4]), 0])


    index_angles0 = 0
    index_angles1 = 0

    joint_array[5] = np.array([np.cos(np.deg2rad(index_angles0))*avg_link_length[5], np.sin(np.deg2rad(index_angles0))*avg_link_length[5], 0])
    joint_array[6] = joint_array[5] + np.array([np.cos(index_angles1)*avg_link_length[6],np.sin(index_angles1)*avg_link_length[6], 0])
    joint_array[7] = joint_array[6] + np.array([np.cos(index_angles1)*avg_link_length[7], np.sin(index_angles1)*avg_link_length[7], 0])
    joint_array[8] = joint_array[7] + np.array([np.cos(index_angles1)*avg_link_length[8], np.sin(index_angles1)*avg_link_length[8], 0])


    middle_angle = np.rad2deg(angles_fix[4])
    middle_angles_abd = np.rad2deg(angles_fix[4])#np.rad2deg(angles_fix[5])
    joint_array[9] = np.array([np.cos(np.deg2rad(middle_angle))*avg_link_length[9], np.sin(np.deg2rad(middle_angle))*avg_link_length[9], 0])
    joint_array[10] = joint_array[9] + np.array([np.cos(np.deg2rad(middle_angles_abd))*avg_link_length[10], np.sin(np.deg2rad(middle_angles_abd))*avg_link_length[10], 0])
    joint_array[11] = joint_array[10] + np.array([np.cos(np.deg2rad(middle_angles_abd))*avg_link_length[11], np.sin(np.deg2rad(middle_angles_abd))*avg_link_length[11] , 0])
    joint_array[12] = joint_array[11] + np.array([np.cos(np.deg2rad(middle_angles_abd))*avg_link_length[12], np.sin(np.deg2rad(middle_angles_abd))*avg_link_length[12] , 0])


    Ringangle = np.rad2deg(angles_fix[6])
    ring_angles_abd =  np.rad2deg(angles_fix[6])
    joint_array[13] = np.array([np.cos(np.deg2rad(Ringangle))*avg_link_length[13], np.sin(np.deg2rad(Ringangle))*avg_link_length[13], 0])
    joint_array[14] = joint_array[13] + np.array([np.cos(np.deg2rad(ring_angles_abd))*avg_link_length[14], np.sin(np.deg2rad(ring_angles_abd))*avg_link_length[14] , 0])
    joint_array[15] = joint_array[14] + np.array([np.cos(np.deg2rad(ring_angles_abd))*avg_link_length[15], np.sin(np.deg2rad(ring_angles_abd))*avg_link_length[15], 0])
    joint_array[16] = joint_array[15] + np.array([np.cos(np.deg2rad(ring_angles_abd))*avg_link_length[16], np.sin(np.deg2rad(ring_angles_abd))*avg_link_length[16], 0])

    PinkAngle = np.rad2deg(angles_fix[7])
    pinki_angles_abd = np.rad2deg(angles_fix[7])
    joint_array[17] = np.array([np.cos(np.deg2rad(PinkAngle))*avg_link_length[17], np.sin(np.deg2rad(PinkAngle))*avg_link_length[17], 0])
    joint_array[18] = joint_array[17] + np.array([np.cos(np.deg2rad(pinki_angles_abd))*avg_link_length[18], np.sin(np.deg2rad(pinki_angles_abd))*avg_link_length[18], 0])
    joint_array[19] = joint_array[18] + np.array([np.cos(np.deg2rad(pinki_angles_abd))*avg_link_length[19], np.sin(np.deg2rad(pinki_angles_abd))*avg_link_length[19], 0])
    joint_array[20] = joint_array[19] + np.array([np.cos(np.deg2rad(pinki_angles_abd))*avg_link_length[20], np.sin(np.deg2rad(pinki_angles_abd))*avg_link_length[20] , 0])

    #print(joint_array)
    joint_array = tf.transformation(np.array([joint_array]))
    return joint_array[0]


#Function that gives a handposture in cartesian coordinates for visualization, but not in zero configuration, but instead with a set of controllable angles, additionally
# to the uncontrollable abduction angles and link length
def modifyPuppet( angles,angles_fix,  avg_link_length):
        joint_array = np.zeros(63)
        joint_array = joint_array.reshape(21, 3)

        thumb_abd_base = angles_fix[0]
        thumb_flex_base = angles_fix[1]
        thumb_abd_Pip = angles[0]
        thumb_flex_Pip = angles[1]
        thumb_abd_dip = angles[2]
        thumb_flex_dip =angles[3]
        thumb_tip = angles[4]
        joints = km.HTrafoThumb([thumb_abd_base, thumb_flex_base, thumb_abd_Pip, thumb_flex_Pip,thumb_abd_dip,thumb_flex_dip, thumb_tip], avg_link_length[1:5])
        TMCP = joints[0][0:3]
        TPip = joints[1][0:3]
        TDip = joints[2][0:3]
        TTip = joints[3][0:3]
        joint_array[1] = TMCP
        joint_array[2] = TPip
        joint_array[3] = TDip
        joint_array[4] = TTip

        index_abd = angles[5]
        index_flex = angles[6]
        index_pip = angles[7]
        index_dip = angles[8]
        joints = km.HTrafoIndex([index_abd, index_flex,index_pip,index_dip], avg_link_length[5:9])
        MCP = joints[0][0:3]
        Pip = joints[1][0:3]
        Dip = joints[2][0:3]
        Tip = joints[3][0:3]
        joint_array[5] = MCP
        joint_array[6] = Pip
        joint_array[7] = Dip
        joint_array[8] = Tip

        middle_base_abd = angles_fix[4]
        middle_base_flex = angles_fix[5]
        middle_abd = angles[9]
        middle_flex = angles[10]
        middle_pip = angles[11]
        middle_dip = angles[12]
        Pjoints = km.HTrafoMiddle([middle_base_abd, middle_base_flex, middle_abd, middle_flex,middle_pip, middle_dip],
            avg_link_length[9:13])
        MMCP = Pjoints[0][0:3]
        MPip = Pjoints[1][0:3]
        MDip = Pjoints[2][0:3]
        MTip = Pjoints[3][0:3]
        joint_array[9] = MMCP
        joint_array[10] = MPip
        joint_array[11] = MDip
        joint_array[12] = MTip

        ring_base_abd = angles_fix[6]
        ring_base_flex = angles[13]
        ring_abd = angles[14]
        ring_flex = angles[15]
        ring_pip = angles[16]
        ring_dip = angles[17]
        Pjoints = km.HTrafoRing([ring_base_abd, ring_base_flex, ring_abd ,ring_flex, ring_pip, ring_dip],
            avg_link_length[13:17])
        RMCP = Pjoints[0][0:3]
        RPip = Pjoints[1][0:3]
        RDip = Pjoints[2][0:3]
        RTip = Pjoints[3][0:3]
        joint_array[13] = RMCP
        joint_array[14] = RPip
        joint_array[15] = RDip
        joint_array[16] = RTip


        pinki_base_abd = angles_fix[7]
        pinki_base_flex = angles[18]
        pinki_abd = angles[19]
        pinki_flex = angles[20]
        pinki_pip = angles[21]
        pinki_dip = angles[22]
        Pjoints = km.HTrafoPinki([pinki_base_abd, pinki_base_flex, pinki_abd, pinki_flex, pinki_pip, pinki_dip],avg_link_length[17:21])
        PMCP = Pjoints[0][0:3]
        PPip = Pjoints[1][0:3]
        PDip = Pjoints[2][0:3]
        PTip = Pjoints[3][0:3]
        joint_array[17] = PMCP
        joint_array[18] = PPip
        joint_array[19] = PDip
        joint_array[20] = PTip

        return joint_array

#Function for testing and getting error value between reconstruction and original handposture
def test_and_error():
    with open('matrix.npy', 'rb') as f:
        setti = np.load(f)

    jointMatrix = setti.reshape(setti.shape[0], 21, 3)
    errori = 0
    angeli , angeli_fix = km.getAngles(jointMatrix)
    for i in range(1):
        tempMatrix = jointMatrix[1:2]

        angles, angles_fix = km.getAngles(tempMatrix)
        angles_fix = angles_fix[0]
        angles = angles[0]
        tempMatrix = km.transformation(tempMatrix)
        tempMatrix = tempMatrix.reshape(tempMatrix.shape[0], 21, 3)
        hand = km.getHands(tempMatrix)
        links = km.avgLinkLength(hand)
        links = links.reshape(21)
        #joint_array = createPuppet( angles_fix, links)
        joint_array = modifyPuppet( angles, angles_fix, links)
        #error = mean_squared_error(joint_array, tempMatrix[0], squared=False)
        #errori = errori + error
        #print("error", error)
        #print("i", i)
    #mean = angeli.mean(axis =0)
    #print(np.rad2deg(angeli))
    # print("mean:", np.rad2deg(mean))
    # errori = errori/22
    # print("error", errori)

    backtrafo1 = joint_array.reshape(63)
    backtrafo = tempMatrix.reshape(63)
    json_joint_list1 = ff.prepareforJson(backtrafo1)
    json_joint_list = ff.prepareforJson(backtrafo)

    jsonfile_path = 'data/contactpose_data/full52_use/banana/annotations.json'
    leftOrRightHand = 1
    with open(jsonfile_path, 'r') as json_file:
        data = json.load(json_file)

    data["hands"][leftOrRightHand]["joints"] = json_joint_list1
    data["hands"][0]["joints"] = json_joint_list
    with open(jsonfile_path, "w") as json_file:
        json.dump(data, json_file)
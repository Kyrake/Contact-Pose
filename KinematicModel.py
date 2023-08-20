import numpy as np
import json
import math

#Cosine Rule
def CosineRule(joint1, joint2):

    a = np.abs(np.linalg.norm(joint1))
    b = np.abs(np.linalg.norm(joint2))
    c = np.abs(np.linalg.norm(joint2-joint1))

    gamma = np.arccos((a**2 + b**2 - c**2)/(-2 * a * b ))


    return gamma

#order each observation by finger
def getHands(jointMatrix):

    hand_list = []
    for i in range(jointMatrix.shape[0]):
        Thumb = np.array([0, 0, 0])
        Thumb = np.append(Thumb, jointMatrix[i][1:5])
        Index = np.array([0, 0, 0])
        Index = np.append(Index, jointMatrix[i][5:9])
        Middle = np.array([0, 0, 0])
        Middle = np.append(Middle, jointMatrix[i][9:13])
        Ring = np.array([0, 0, 0])
        Ring = np.append(Ring, jointMatrix[i][13:17])
        Pinki = np.array([0, 0, 0])
        Pinki = np.append(Pinki, jointMatrix[i][17:21])

        fingerlist = np.array([Thumb, Index, Middle, Ring, Pinki])
        fingerlist = fingerlist.reshape(5, 5, 3)
        hand_list.append(fingerlist)
    return np.asarray(hand_list)

#get length of finger bones
def getVectorJoints(finger):
    joint1 = finger[1] - finger[0]
    joint2 = finger[2] - finger[1]
    joint3 = finger[3] - finger[2]
    joint4 = finger[4] - finger[3]
    finger = np.array([np.linalg.norm(joint1),np.linalg.norm(joint2),np.linalg.norm(joint3),np.linalg.norm(joint4)])
    return finger

#get the mean of the  length of finger bones for several people
def avgLinkLength(hands):
    hands_joints_lengths = []
    for hand in hands:
        hand_vector_joints = getHandVectorJoints(hand)
        hands_joints_lengths.append(hand_vector_joints)
    hands_joints_lengths = np.asarray(hands_joints_lengths)
    normed_hands_joints_lengths = np.mean(hands_joints_lengths, axis=0)
    normed_hands_joints_lengths = np.insert(normed_hands_joints_lengths, 0, 0)
    normed_hands_joints_lengths = normed_hands_joints_lengths.reshape(21)
    return normed_hands_joints_lengths


#get length of finger bones for several people
def getHandVectorJoints(hand):
    hand_vector_joints = []
    for finger in hand:
        finger_vector_joints = getVectorJoints(finger)
        hand_vector_joints.append(finger_vector_joints)
    return np.asarray(hand_vector_joints)

# IMPORTANT: makes sure all hand postures are orientated similar in space. Indexfinger is the x-axis, Indexfinger and Middle finger
# are spanning the x-y plane
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

#Wrapper Function to get mean  of non controllable angles and links
def avgAngleAndLinks():
    with open('matrix.npy', 'rb') as f:
        setti = np.load(f)
    jointMatrix = setti.reshape(setti.shape[0], 21,3)
    angles, angles_fix = getAngles(jointMatrix)
    angles_fix = angles_fix.mean(axis=0)
    jointMatrix = transformation(jointMatrix)
    jointMatrix = jointMatrix.reshape(jointMatrix.shape[0],21,3)
    hands = getHands(jointMatrix)
    avg_link_length = avgLinkLength(hands)
    avg_link_length = avg_link_length.reshape(20)
    avg_link_length = np.insert(avg_link_length,0, 0)

    return angles,angles_fix, avg_link_length

#Dot Product Funciton
def AngleDotProd(finger1, finger2):
    rho = np.arccos(np.dot(finger1/np.linalg.norm(finger1), finger2/np.linalg.norm(finger2)))
    return rho

#Rotation around Z-axis
def rotZ(angle, length):
    R_z = np.array([[math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle), math.cos(angle), 0],
                    [0, 0, 1]
                    ])
    Rota = np.eye(4)
    Rota[:3,:3] = R_z
    # if np.all(length == 0):
    #     Rota[0][3] = 0
    # else:
    Rota[0][3] = math.cos(angle)*np.linalg.norm(length)
    Rota[1][3] = math.sin(angle)*np.linalg.norm(length)
    Rota[2][3] = 0
    return Rota


#Rotation around X-axis
def rotX(angle, length):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(angle), -math.sin(angle)],
                    [0, math.sin(angle), math.cos(angle)]
                    ])

    Rota = np.eye(4)
    Rota[:3,:3] = R_x
    Rota[0][3] = math.cos(angle) * np.linalg.norm(length)
    Rota[1][3] = math.sin(angle) * np.linalg.norm(length)
    Rota[2][3] = 0
    return Rota

#Rotation around Y-axis
def rotY(angle, length):

    R_y = np.array([[math.cos(angle), 0, math.sin(angle)],
                    [0, 1, 0],
                    [-math.sin(angle), 0, math.cos(angle)]
                    ])

    Rota = np.eye(4)
    Rota[:3,:3] = R_y
    Rota[0][3]= math.cos(angle) * np.linalg.norm(length)
    Rota[1][3] = math.sin(angle) * np.linalg.norm(length)
    Rota[2][3] = 0
    return Rota

#Get 7 angles for thumb. 5 controllable, 2 non controllable
def getThumbAngles(finger0):
    Thumb1 = finger0[1]
    Thumb2 = finger0[2] - finger0[1]
    Thumb3 = finger0[3] - finger0[2]
    Thumb4 = finger0[4] - finger0[3]

    cmc_flexion = math.atan(Thumb1[2]/ Thumb1[0])
    cmc_abduction = np.deg2rad(90) + math.atan(Thumb1[1]/ Thumb1[0])

    mcp_abduction = np.deg2rad(90) + math.atan(Thumb2[1]/ Thumb2[0]) - cmc_abduction
    mcp_flexion = math.atan(Thumb2[2]/ Thumb2[0]) - cmc_flexion

    pip_abduction = np.deg2rad(90) + math.atan(Thumb3[1]/ Thumb3[0])- (np.deg2rad(90) + math.atan(Thumb2[1]/ Thumb2[0]))
    pip_flexion = math.atan(Thumb3[2]/ Thumb3[0]) - math.atan(Thumb2[2]/ Thumb2[0])

    theta_dip = np.deg2rad(90) + math.atan(Thumb4[1]/ Thumb4[0])-(np.deg2rad(90) + math.atan(Thumb3[1]/ Thumb3[0]))

    angles = np.array([ mcp_abduction, mcp_flexion,pip_abduction,pip_flexion, theta_dip])
    angles_fix = np.array([cmc_abduction, cmc_flexion])

    return angles, angles_fix

#Get 6 angles for index finger 4 controllable, 2 non controllable
def getIndexAngles(finger0):
    Index1 = finger0[1]
    Index2 = finger0[2] - finger0[1]
    Index3 = finger0[3] - finger0[2]
    Index4 = finger0[4] - finger0[3]


    mcp_flexion = math.atan2(Index2[2] , Index2[0]) - math.atan2( Index1[2],  Index1[0])
    mcp_abdcution = np.arctan(Index2[1]/Index2[0])

    theta_pip= math.atan2(Index3[2],Index3[0]) - math.atan2(Index2[2],Index2[0])
    theta_dip = math.atan2(Index4[2],Index4[0]) - math.atan2(Index3[2],Index3[0])

    return np.array([mcp_abdcution, mcp_flexion, theta_pip, theta_dip]), np.array([0,0])

#Get 6 angles for middle finger 4 controllable, 2 non controllable
def getMiddleAngles( finger1):
    MiddleFinger1 = finger1[1]
    MiddleFinger2 = finger1[2] - finger1[1]
    MiddleFinger3 = finger1[3] - finger1[2]
    MiddleFinger4 = finger1[4] - finger1[3]

    cmc_abduction = math.atan(MiddleFinger1[1]/MiddleFinger1[0])

    mcp_flexion = math.atan2(MiddleFinger2[2] , MiddleFinger2[0]) - math.atan2(MiddleFinger1[2] , MiddleFinger1[0])
    mcp_abdcution = math.atan2(MiddleFinger2[1] , MiddleFinger2[0]) - cmc_abduction

    theta_pip = math.atan2(MiddleFinger3[2] , MiddleFinger3[0]) - math.atan2(MiddleFinger2[2] , MiddleFinger2[0])
    theta_dip = math.atan2(MiddleFinger4[2] , MiddleFinger4[0]) - math.atan2(MiddleFinger3[2] , MiddleFinger3[0])
    if (math.atan2(MiddleFinger4[2], MiddleFinger4[0]) < np.deg2rad(-90)):
        theta_dip = np.deg2rad(360) + theta_dip

    return np.array([ mcp_abdcution, mcp_flexion, theta_pip, theta_dip]),  np.array([cmc_abduction,0])


#Get 6 angles for ring finger 5 controllable, 1 non controllable
def getRingAngles(finger1):
    RingFinger1 = finger1[1]
    RingFinger2 = finger1[2] - finger1[1]
    RingFinger3 = finger1[3] - finger1[2]
    RingFinger4 = finger1[4] - finger1[3]

    cmc_abduction = math.atan2(RingFinger1[1],RingFinger1[0])
    cmc_flexion = np.deg2rad(90) - AngleDotProd(RingFinger1, np.array([0, 0, 1]))

    mcp_flexion = math.atan2(RingFinger2[2] , RingFinger2[0]) - math.atan2(RingFinger1[2],RingFinger1[0])
    mcp_abdcution = math.atan2(RingFinger2[1] , RingFinger2[0]) - cmc_abduction

    theta_pip = AngleDotProd(RingFinger2,RingFinger3)
    theta_dip = AngleDotProd(RingFinger3,RingFinger4)

    return np.array([cmc_flexion, mcp_abdcution, mcp_flexion, theta_pip, theta_dip]),  np.array([cmc_abduction])


#Get 6 angles for pinky 5 controllable, 1 non controllable
def getPinkiAngles(finger1):
    Pinki1 = finger1[1]
    Pinki2 = finger1[2] - finger1[1]
    Pinki3 = finger1[3] - finger1[2]
    Pinki4 = finger1[4] - finger1[3]

    cmc_abduction = math.atan2(Pinki1[1] , Pinki1[0])
    cmc_flexion = np.deg2rad(90) - AngleDotProd(Pinki1, np.array([0, 0, 1]))

    mcp_flexion = math.atan2(Pinki2[2] , Pinki2[0]) - math.atan2(Pinki1[2] , Pinki1[0])
    mcp_abdcution = math.atan2(Pinki2[1] ,Pinki2[0]) - cmc_abduction

    theta_pip = AngleDotProd(Pinki2, Pinki3)

    theta_dip = AngleDotProd(Pinki3, Pinki4)

    return np.array([ cmc_flexion,mcp_abdcution, mcp_flexion, theta_pip, theta_dip]), np.array([cmc_abduction])

#Wrapper function to get angles over several observations
def getAngles(jointMatrix):
    jointMatrix = transformation(jointMatrix)
    jointMatrix = jointMatrix.reshape(jointMatrix.shape[0], 21, 3)
    hands = getHands(jointMatrix)
    bigAngleList = []
    bigAngleList_fix = []
    for j in range(hands.shape[0]):
        finger = hands[j]
        fingeranglelist = []
        fingeranglelist_fix = []
        for i in range(finger.shape[0]):
            angle = []
            angle_fix = []

            if i == 0:
                angle, angle_fix = getThumbAngles(finger[0])
            if i == 1:
                angle, angle_fix = getIndexAngles(finger[1])
            if i == 2:
                angle, angle_fix = getMiddleAngles(finger[2])
            if i == 3:
                angle, angle_fix = getRingAngles(finger[3])
            if i == 4:
                angle, angle_fix = getPinkiAngles(finger[4])

            for fingerangle in angle:
                fingeranglelist.append(fingerangle)
            for fingerangle_fix in angle_fix:
                fingeranglelist_fix.append(fingerangle_fix)
        bigAngleList.append(np.asarray(fingeranglelist))
        bigAngleList_fix.append(np.asarray(fingeranglelist_fix))
    bigAngleList = np.asarray(bigAngleList)
    bigAngleList_fix = np.asarray(bigAngleList_fix)
    return bigAngleList, bigAngleList_fix

#Homogeneous Transformation Thumb
def HTrafoThumb(angles, links):
    IndexFinger1 = links[0]
    IndexFinger2 = links[1]
    IndexFinger3 = links[2]
    IndexFinger4 = links[3]
    ring_base_abduction = angles[0]
    ring_base_flexion = angles[1]
    theta_abdcution = angles[2]
    theta_flexion = angles[3]
    theta_pip = angles[4]
    theta_dip = angles[5]
    theta_tip = angles[6]
    # Calculate Homogeneous Transformations

    Txx = np.array([[0, 1,0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T00 = rotZ(ring_base_abduction, np.array([0, 0, 0]))
    T011 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ rotZ(ring_base_flexion,
                                                                                         IndexFinger1)
    T01 = Txx@T00 @ T011
    T12 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) @ rotZ(theta_abdcution,
                                                                                        np.array([0, 0, 0]))
    T231 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    T232 = rotZ(theta_flexion, IndexFinger2)
    T23 = T231 @ T232
    T31 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) @ rotZ(theta_pip, np.array([0, 0, 0]))
    T34 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @rotZ(theta_dip, IndexFinger3)
    T34  = T31 @ T34
    T45 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) @rotZ(theta_tip, IndexFinger4)
    MCP = T01 @ np.array([0, 0, 0, 1])
    Pip = T01 @ T12@T23  @ np.array([0, 0, 0, 1])
    Dip = T01 @ T12  @T23@ T34 @ np.array([0, 0, 0, 1])
    Tip = T01 @ T12 @T23 @ T34 @ T45 @ np.array([0, 0, 0, 1])
    return np.array([MCP, Pip, Dip, Tip])

#Homogeneous Transformation Index
def HTrafoIndex(angles, links):
    IndexFinger1 = links[0]
    IndexFinger2 = links[1]
    IndexFinger3 = links[2]
    IndexFinger4 = links[3]
    ring_base_abduction = 0
    ring_base_flexion = 0
    theta_abdcution = angles[0]
    theta_flexion = angles[1]
    theta_pip = angles[2]
    theta_dip = angles[3]
    # Calculate Homogeneous Transformations
    T00 = rotZ(ring_base_abduction, np.array([0, 0, 0]))
    T011 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ rotZ(ring_base_flexion,
                                                                                         IndexFinger1)
    T01 = T00 @ T011
    T12 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) @ rotZ(theta_abdcution,
                                                                                        np.array([0, 0, 0]))
    T231 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    T232 = rotZ(theta_flexion, IndexFinger2)
    T23 = T231 @ T232
    T34 = rotZ(theta_pip, IndexFinger3)
    T45 = rotX(theta_dip, IndexFinger4)
    MCP = T01 @ np.array([0, 0, 0, 1])
    Pip = T01 @ T12 @ T23 @ np.array([0, 0, 0, 1])
    Dip = T01 @ T12 @ T23 @ T34 @ np.array([0, 0, 0, 1])
    Tip = T01 @ T12 @ T23 @ T34 @ T45 @ np.array([0, 0, 0, 1])
    return np.array([MCP, Pip, Dip, Tip])

#Homogeneous Transformation Middle Finger
def HTrafoMiddle(angles, links):
    MiddleFinger1 = links[0]
    MiddleFinger2 = links[1]
    MiddleFinger3 = links[2]
    MiddleFinger4 = links[3]
    ring_base_abduction = angles[0]
    ring_base_flexion = angles[1]
    theta_abdcution = angles[2]
    theta_flexion = angles[3]
    theta_pip = angles[4]
    theta_dip = angles[5]
    # Calculate Homogeneous Transformations
    T00 = rotZ(ring_base_abduction, np.array([0, 0, 0]))
    T011 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ rotZ(ring_base_flexion, MiddleFinger1)
    T01 = T00 @ T011
    T12 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) @ rotZ(theta_abdcution, np.array([0, 0, 0]))
    T231 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    T232 = rotZ(theta_flexion, MiddleFinger2)
    T23 = T231 @ T232
    T34 = rotZ(theta_pip, MiddleFinger3)
    T45 = rotZ(theta_dip, MiddleFinger4)
    MCP = T01 @ np.array([0, 0, 0, 1])
    Pip = T01 @ T12 @ T23 @ np.array([0, 0, 0, 1])
    Dip = T01 @ T12 @ T23 @ T34 @ np.array([0, 0, 0, 1])
    Tip = T01 @ T12 @ T23 @ T34 @ T45 @ np.array([0, 0, 0, 1])
    np.array([MCP, Pip, Dip, Tip])
    return np.array([MCP, Pip, Dip, Tip])

#Homogeneous Transformation Ring Finger
def HTrafoRing(angles, links):
    RingFinger1 = links[0]
    RingFinger2 = links[1]
    RingFinger3 = links[2]
    RingFinger4 = links[3]

    ring_base_abduction = angles[0]
    ring_base_flexion = angles[1]
    theta_abdcution = angles[2]
    theta_flexion = angles[3]
    theta_pip = angles[4]
    theta_dip = angles[5]
    # Calculate Homogeneous Transformations
    T00 = rotZ(ring_base_abduction, np.array([0, 0, 0]))
    T011 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ rotZ(ring_base_flexion, RingFinger1)
    T01 = T00 @ T011
    T12 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) @ rotZ(theta_abdcution,np.array([0, 0, 0]))
    T231 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    T232 = rotZ(theta_flexion, RingFinger2)
    T23 = T231 @ T232
    T34 = rotZ(theta_pip, RingFinger3)
    T45 = rotZ(theta_dip, RingFinger4)

    MCP = T01 @ np.array([0, 0, 0, 1])
    Pip = T01 @ T12 @ T23 @ np.array([0, 0, 0, 1])
    Dip = T01 @ T12 @ T23 @ T34 @ np.array([0, 0, 0, 1])
    Tip = T01 @ T12 @ T23 @ T34 @ T45 @ np.array([0, 0, 0, 1])

    return np.array([MCP, Pip, Dip, Tip])

#Homogeneous Transformation Pinky
def HTrafoPinki(angles, links):
    Pinki1 = links[0]
    Pinki2 = links[1]
    Pinki3 = links[2]
    Pinki4 = links[3]


    ring_base_abduction = angles[0]
    ring_base_flexion = angles[1]
    theta_abdcution = angles[2]
    theta_flexion = angles[3]
    theta_pip = angles[4]
    theta_dip = angles[5]
    T00 = rotZ(ring_base_abduction,  np.array([0, 0, 0]))
    T011 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ rotZ(ring_base_flexion,Pinki1)
    T01 = T00 @ T011
    T12 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) @ rotZ(theta_abdcution,np.array([0, 0, 0]))
    T231 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    T232 = rotZ(theta_flexion, Pinki2)
    T23 = T231 @ T232
    T34 = rotZ(theta_pip, Pinki3)
    T45 = rotZ(theta_dip, Pinki4)
    # Use Homogenoues Transformation to transform origin of respective frame ot point w.r.t base frame
    MCP = T01 @ np.array([0, 0, 0, 1])
    Pip = T01 @ T12 @ T23 @ np.array([0, 0, 0, 1])
    Dip = T01 @ T12 @ T23 @ T34 @ np.array([0, 0, 0, 1])
    Tip = T01 @ T12 @ T23 @ T34 @ T45 @ np.array([0, 0, 0, 1])

    return np.array([MCP, Pip, Dip, Tip])



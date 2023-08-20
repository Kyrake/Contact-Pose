import numpy as np
import utilities.dataset as ds
from utilities.dataset import ContactPose


import os

#Function to store hand postures of ContactPose data set in array
def getJointMatrix():

    #acquire list with all objects or create own list with possible objects
    osp = os.path
    objects1 = ds.get_object_names(1, 'use')
    #objects1 = ['apple']
    objects1.sort()

    # get all the combinations of participants, objects and both modi "use" and "handoff"
    # has to start with 1, because of the numerations of the files the data is stored in
    joint_person = []
    count = 0
    for index in range(1,50):
        print(index)
        for object_name in objects1:

            #access data from their directories
            p_id = 'full{:d}_{:s}'.format(index, 'handoff')
            p_id2 = 'full{:d}_{:s}'.format(index, 'use')
            data_dir = osp.join('data', 'contactpose_data', p_id, object_name)
            data_dir2 = osp.join('data', 'contactpose_data', p_id2, object_name)
            if not osp.isdir(data_dir):
                continue
            if not osp.isdir(data_dir2):
                continue
            # store the accessed data from their directories
            cp = ContactPose(index, "handoff", object_name)
            cp2 = ContactPose(index, "use", object_name)

            # we just want the coordinates of the joints
            joints = cp.hand_joints()
            joints2 = cp2.hand_joints()
            hand_1 = joints[1]
            hand_2 = joints2[1]
            if(hand_1 is None):
                hand_1 = joints[0]
            if (hand_2 is None):
                hand_2 = joints[0]

            if hand_1 is not None:
                hand_1 = hand_1.reshape(21*3)
                joint_person.append(hand_1)
            if hand_2 is not None:
                hand_2 = hand_2.reshape(21 * 3)
                joint_person.append(hand_2)
            count = count +1
            print("Count:,", str(count) +" " + " " + str(object_name) + " "+  " "+ str(index))
    return np.asarray(joint_person)

matrix= getJointMatrix()
print(matrix.shape)
with open('matrix.npy', 'wb') as f:
    np.save(f, matrix)








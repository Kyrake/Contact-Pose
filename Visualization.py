import FilterFunctions as ff
import json

def Visualize(joints, path = 'data/contactpose_data/full51_use/apple/annotations.json'):
    objectToSimulate = joints
    objectToSimulate = objectToSimulate.reshape(63)

    json_joint_list = ff.prepareforJson(objectToSimulate)
    jsonfile_path = path
    leftOrRightHand = 1
    with open(jsonfile_path, 'r') as json_file:
        data = json.load(json_file)

    data["hands"][leftOrRightHand]["joints"] = json_joint_list
    with open(jsonfile_path, "w") as json_file:
        json.dump(data, json_file)


def VisualizeTwo(joints, jointMatrix,path = 'data/contactpose_data/full51_use/apple/annotations.json'):
    objectToSimulate = joints
    objectToSimulate2 = jointMatrix

    objectToSimulate = objectToSimulate.reshape(63)
    objectToSimulate2 = objectToSimulate2.reshape(63)

    json_joint_list = ff.prepareforJson(objectToSimulate)
    json_joint_list2 = ff.prepareforJson(objectToSimulate2)
    jsonfile_path = 'data/contactpose_data/full51_use/apple/annotations.json'
    leftOrRightHand = 1
    with open(jsonfile_path, 'r') as json_file:
        data = json.load(json_file)

    #data["hands"][0]["joints"] = json_joint_list
    data["hands"][1]["joints"] = json_joint_list
    with open(jsonfile_path, "w") as json_file:
        json.dump(data, json_file)
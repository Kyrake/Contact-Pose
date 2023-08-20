import Handposeanimator
import scripts.show_contactmap
import CreateHand as cph
from pylab import *
from sklearn.metrics import mean_squared_error
import KinematicModel as km
from sklearn import decomposition
import Visualization as vis

# Open subset of the contactpose dataset, that you wish to observe
with open('matrix_apple.npy', 'rb') as f:
    joints = np.load(f)


#get angles for ICA and also non controllable angles, and avg link lengths for visual reconstruction
jointMatrix = joints.reshape(joints.shape[0], 21, 3)
angles, angles_fix = km.getAngles(jointMatrix)
hands = km.getHands(jointMatrix)
links = km.avgLinkLength(hands)


#applying ICA on data matrix. Beware: each time ICA is called on the same data set, it gives a different ordering of the
#components. Therefore the components will be stored in a .npy file for further usage.

def icaPrep(compoenents):

    ica = decomposition.FastICA(n_components=compoenents,  whiten=True, max_iter=200)
    source = ica.fit_transform(angles)
    components = ica.components_
    original_signals = ica.inverse_transform(source)

    np.set_printoptions(precision=1, suppress=True)
    with open('ICA_new.npy', 'wb') as f:
        np.save(f, components)

with open('ICA.npy', 'rb') as f:
     components = np.load(f)
#number_of_components = 23
#icaPrep(number_of_components)


#Constraints for batteling the ambiguity of the amplitude and sign of the ICA results
def constraints(posture):
    for i in range(posture.shape[0]):
        if i ==5  or i == 14 or i ==  9  or i ==  19:
            if posture[i] >  np.deg2rad(15) or  posture[i] < np.deg2rad(-15):
                posture[i] = np.deg2rad(0)
        if  posture[i] < np.deg2rad(-7):
            posture[i] = - posture[i]
    return posture

#Chosen hand postures after manually deciding, because all compoents are equally important
posture2 = components[1]
posture5 = components[4]
posture11 = components[10]
mean_grasp = angles.mean(axis = 0)

#Prepare hand posture for storing in Json and therefore for visualization
objectToSimulate = cph.modifyPuppet(constraints(posture5), angles_fix[0], links)
vis.Visualize(objectToSimulate)

#Animation of chosen components
coefficients = np.linspace(-0.2,0.2,50)
coefficients = np.append(coefficients, coefficients[::-1])
geomslist = []
for c in coefficients:
    mean_grasp[19] = np.deg2rad(-15) #upholding constraints for abduction -15 deg< angle < 15 deg
    objectToSimulate =  mean_grasp +  c * (constraints(posture11)) + c* (constraints(posture5)) + c * (constraints(posture2))

    objectToSimulate = cph.modifyPuppet(objectToSimulate, angles_fix[0], links)
    vis.Visualize(objectToSimulate)

    geoms = scripts.show_contactmap.create_contactmap(51, 'use', 'apple', 'simple_hands')
    geomslist.append(geoms)

hand_pose_animator = Handposeanimator.HandPoseAnimator(geomslist, 0.1)
hand_pose_animator.start_animation()


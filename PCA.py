import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import Handposeanimator
plt.style.use('seaborn-paper')
import CreateHand as cph
import KinematicModel as km
import scripts.show_contactmap
import Visualization as vis

np.set_printoptions(precision=5, suppress=True)

#load preprocessed data
with open('matrix.npy', 'rb') as f:
    joints = np.load(f)

# get angles and average links of loaded data
jointMatrix = joints.reshape(joints.shape[0], 21, 3)
angles, angles_fix = km.getAngles(jointMatrix)
angles_fix = angles_fix.mean(axis=0)
mean = angles.mean(axis=0)
angles = angles - mean
hand = km.getHands(jointMatrix)
links = km.avgLinkLength(hand)

#apply PCA function
numberOfComponents = 13
def pca(angles):
    pca = PCA(n_components=numberOfComponents)
    result = pca.fit_transform(angles)
    components = pca.components_
    mean = pca.mean_
    ratio = pca.explained_variance_ratio_

    return ratio, components, mean, result


ratio, components, meanpc, result = pca(angles)


#Visualize in Open3Dwindow
mean[19] = mean[19] + np.deg2rad(15) #upholding constraints for abduction -15 deg< angle < 15 deg
objectToSimulate = mean
objectToSimulate = cph.modifyPuppet(objectToSimulate, angles_fix, links)
vis.Visualize(objectToSimulate)

#CreateCoefficients
coefficients = np.linspace(-1,1, 20)
coefficients = np.append(coefficients, coefficients[::-1])

# Animate handposture and coefficiants
geomslist = []
component1 = components[0]
component2 = components[1]

for c in coefficients:
    component1[6] = -component1[6]
    component1[12] = 0.2 * component1[12]
    objectToSimulate =  mean + c*component1+c*component2
    objectToSimulate = cph.modifyPuppet(objectToSimulate, angles_fix, links)
    vis.Visualize(objectToSimulate)

    geoms = scripts.show_contactmap.create_contactmap(51, 'use', 'apple', 'simple_hands')
    geomslist.append(geoms)

hand_pose_animator = Handposeanimator.HandPoseAnimator(geomslist, 0.1)
hand_pose_animator.start_animation()

#Plot Cumulative Variance
print("Percentage of variance for each component", ratio)
endline = numberOfComponents + 1
x = np.arange(0, endline)
ax = plt.axes(facecolor='#E6E6E6')
ax.set_axisbelow(True)
plt.title("PC for 1947 observations over all participants, objects and modi")
plt.xlim(0, numberOfComponents)
plt.ylim(0, 1)
plt.plot(x, np.cumsum(np.insert(ratio, 0, 0)), label="All participants")
plt.xlabel('Number of Principle Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend(loc="lower right", prop={'size': 10})
plt.grid(color='w', linestyle='solid')
plt.show()

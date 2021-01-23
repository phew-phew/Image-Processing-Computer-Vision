import cv2
import numpy as np
from scipy.spatial.distance import cdist
import random as rand
import matplotlib.pyplot as plt
import os
import glob

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def detect_and_compute_keypoints(image):

    sift = cv2.xfeatures2d.SIFT_create()
    kps, desc = sift.detectAndCompute(image,None) 
    
    return kps, desc

#Keypoint Matching using Squared Euclidian Distance

def keypointMatchingSquelidian(kp1,kp2,des1,des2):
  #--write your own function to calulate euclidian distance
  pairwiseDistances = cdist(des1, des2, 'sqeuclidean')
  threshold   = 7000
  
  #--write your funcion to find points in image where threhold is required
  points_in_img1 = np.where(pairwiseDistances < threshold)[0]
  points_in_img2 = np.where(pairwiseDistances < threshold)[1]
  
  #coordinates_in_img1 = np.array([kp1[point].pt for point in points_in_img1])
  #coordinates_in_img2 = np.array([kp2[point].pt for point in points_in_img2])

  coordinates_in_img1=np.ones((len(points_in_img1),2))
  coordinates_in_img2=np.ones((len(points_in_img2),2))
    
  i=0
  for point in points_in_img1:
    coordinates_in_img1[i]=kp1[point].pt
    i=i+1

  j=0
  for point in points_in_img2:
    coordinates_in_img2[j]=kp2[point].pt
    j=j+1

  return np.concatenate( (coordinates_in_img1, coordinates_in_img2) , axis=1 )

#Ransac Algorithm

def ransac_algo(matchingPoints,totalIteration):
    
    # Ransac parameters
    highest_inlier_count = 0
    best_H = []
    
    # Loop parameters
    counter = 0
    while counter < totalIteration:
        counter = counter + 1
        # Select 4 points randomly
        secure_random  = rand.SystemRandom()
        
        matachingPair1 = secure_random.choice(matchingPoints)
        matachingPair2 = secure_random.choice(matchingPoints)
        matachingPair3 = secure_random.choice(matchingPoints)
        matachingPair4 = secure_random.choice(matchingPoints)
        
        fourMatchingPairs=np.concatenate(([matachingPair1],[matachingPair2],[matachingPair3],[matachingPair4]),axis=0)
        
        # Finding homography matrix for this 4 matching pairs
        # H = get_homography(fourMatchingPairs)

        points_in_image_1 = np.float32(fourMatchingPairs[:,0:2])
        points_in_image_2 = np.float32(fourMatchingPairs[:,2:4])
        
        H = cv2.getPerspectiveTransform(points_in_image_1, points_in_image_2)
        
        rank_H = np.linalg.matrix_rank(H)
        
        # Avoid degenrate H
        if rank_H < 3:
            continue
        
        # Calculate error for each point using the current homographic matrix H
        total_points = len(matchingPoints)
        
        points_img1 = np.concatenate( (matchingPoints[:, 0:2], np.ones((total_points, 1))), axis=1)
        points_img2 = matchingPoints[:, 2:4]
        
        correspondingPoints = np.zeros((total_points, 2))
        
        for i in range(total_points):
            t = np.matmul(H, points_img1[i])
            correspondingPoints[i] = (t/t[2])[0:2]

        error_for_every_point = np.linalg.norm(points_img2 - correspondingPoints, axis=1) ** 2

        inlier_indices = np.where(error_for_every_point < 0.5)[0]
        inliers        = matchingPoints[inlier_indices]
    
        curr_inlier_count = len(inliers)
      
        if curr_inlier_count > highest_inlier_count:
            highest_inlier_count = curr_inlier_count
            best_H = H.copy()

    return best_H

def func(kp1,kp2, des1,des2,img1,img2):
  numberOfIteration=1000
  totalMatchingPoints=keypointMatchingSquelidian(kp1,kp2, des1,des2)
  homography=ransac_algo(totalMatchingPoints, numberOfIteration)
  result = cv2.warpPerspective(img1, homography ,( int(img1.shape[1] + img2.shape[1]*0.8),int(img1.shape[0] + img2.shape[0]*0.4) ))
  result[0:img2.shape[0], 0:img2.shape[1]] = img2

  return result,img2

def final_panaroma_image(result):
     # Resizing the final panorama
        black = np.zeros(3)
        color123 = result #cv2.imread(directory + '/panorama123.jpg', 1)
        color321 = result #cv2.imread(directory + '/panorama321.jpg', 1)
        
        count123 = 0
        count321 = 0
        
        for i in range(color123.shape[0]):
            for j in range(color123.shape[1]):
                pixel_value = color123[i, j, :]
                if np.array_equal(pixel_value, black):
                    count123 = count123 + 1

        for i in range(color321.shape[0]):
            for j in range(color321.shape[1]):
                pixel_value = color321[i, j, :]
                if np.array_equal(pixel_value, black):
                    count321 = count321 + 1
                    
        # Resizing the final panorama
        colorPan = result #cv2.imread(directory + '/panorama.jpg', 1)
        x_max = 0
        y_max = 0

        for i in range(colorPan.shape[0]):
            for j in range(colorPan.shape[1]):
                pixel_value = colorPan[i, j, :]
                if not np.array_equal(pixel_value, black):
                    if j > x_max:
                       x_max = j
                    if i > y_max:
                       y_max = i
    
        crop_img = colorPan[0:y_max,0:x_max, :]
        cv2.imwrite(os.path.join('images' , 'final_result.jpg'), crop_img)

# this method can be used for any number of input images
def optimized_perform_forward_and_backword_stitching(sift, images, colorImages):
    image_length = len(images)
    
    intial_f_image = intial_f_colorImage= images[0]
    #loop for forward propagation
    for i in range(0, (image_length-1)):
        kps1, desc1 = sift.detectAndCompute(intial_f_image,None)
        kps2, desc2 = sift.detectAndCompute(images[i+1],None)
        
        result, colorImages[i+1] = func(kps1, kps2, desc1, desc2, intial_f_colorImage, colorImages[i+1])
        intial_f_image = intial_f_colorImage = result
        
    #loop for backword propagation
    # taking the last image and storing in intial_b_image for the backword stitching
    intial_b_image = intial_b_colorImage= images[-1]
    for i in range((image_length-1), 0, -1):
        kps1, desc1 = sift.detectAndCompute(intial_b_image,None)
        kps2, desc2 = sift.detectAndCompute(images[i-1],None)
        
        result, colorImages[i-1] = func(kps1, kps2, desc1, desc2, intial_b_colorImage, colorImages[i-1])
        intial_b_image = intial_b_colorImage = result
        
        
    return result

sift = cv2.xfeatures2d.SIFT_create()
images = colorImages = load_images_from_folder("images") # get all the images from a directory and store it in a list
print(len(images))
result = optimized_perform_forward_and_backword_stitching(sift, images, colorImages)
final_panaroma_image(result)


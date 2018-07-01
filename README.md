# Vehicle Detection

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/nonVehicleImage.png
[image2]: ./output_images/vehicleImage.png
[image3]: ./output_images/featureExtration.png
[image4]: ./output_images/testImage.png
[image5]: ./output_images/window.png
[image6]: ./output_images/applyWindows.png
[image7]: ./output_images/falsePositive.png
[video1]: ./project_video.mp4
[video2]: ./result_project_video.mp4

## A. Histogram of Oriented Gradients (HOG)

### 1. The Training Images.

The Examples to train classifier are given as the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip).  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/).

![alt text][image1]

![alt text][image2]

### 2. HOG Features from the Training Images.

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()`.

|        HOG Features        |     Option    | 
|:--------------------------:|:-------------:| 
|         Color Space        |      LUV      | 
|      HOG Orientations      |       16      |
|     HOG pixels per cell    |        2      |
|         HOG Channel        |       ALL     |
| Spatial Binning Dimensions |      32,32    |
|   Num of Histogram Bins    |        32     |


#### 3. final choice of HOG parameters.

* Given Hint 
	* In one of the previous exercises you extracted HOG features from each individual window as you searched across the image, but it turns out this is rather inefficient.
	* To speed things up, extract HOG features just once for the entire region of interest (i.e. lower half of each frame of video) and subsample that array for each sliding window.
	* To do this, apply skimage.feature.hog() with the flag feature_vec=False

As the hog algorithm is primarily focused on grey images, I initially used the YCrCB colour space with the Y channel (used to represent a gray images). However I found that it was not selective enough during the detection phase. I thus used all 3 colour channels. To reduce the number of features, I increased the number of HOG pixels per cell. I used an interactive feature in my notebook to find an orient setting of 16 that showed distinctive features of vehicle.


#### 4. I trained a linear SVM using...

Once HOG features (no Colour Hist or Bin Spatial) were extracted from car (GTI Vehicle Image Database and Udacity Extras) and not_car (GTI, KITTI) image sets. They were then stacked and converted to float in the vehicle detection notebook.

Features were then scaled using the Sklearn RobustScaler sample result follows. 

![alt text][image3]

Experimentation occurred in the Classifier Experimentation Notebook between LinearSVC (Support Vector Machine Classifier), Random Forest and Extra Trees classifiers. LinearSVC was chosen as the prediction time was 0.028 seconds for 10 labels compared to ~0.10 seconds for the other two. The test accuracy of SVC was 99.07% and it tool 10.2 seconds to train SVC

## B. Sliding Window Search

### 1. Test images while will apply the sliding window search

![alt text][image4]

### 2. Selected three different windows size and the location as following. 

|   Windows Location   |     Size    |    Overlap   |   X Location   |  Y Location   |
|:--------------------:|:-----------:|:------------:|:--------------:|:-------------:| 
|    Far Location      |   40 x 40   |     90%      |   550 to 1000  |   410 to 470  |
|   Medium Location    |   72 x 72   |     80%      |   450 to None  |   410 to 600  |
|    Near Location     |  128 x 128  |     75%      |   300 to None  |   400 to 700  |

Initially, I only created 2 windows. But while I was tresting pipeline, I found while front car was moving away from the car, it wasn't able to detect properly. So I increased to 3 different windows size for search the car, which was more efficient depend on the distance from the driver. As further away from the car, I chosed small window size to search and grater overlap, but as closer to camera, I chosed bigger windows size to search and less overlaop. And Since the driver was driving on the 3rd lane, I chose the windows search location as acordingly as following test images.

![alt text][image5]

### 3. The result after applying the sliding window search

    hot_windows = (search_windows(test_Image, windows1, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat))           
    
    hot_windows += (search_windows(test_Image, windows2, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat))  
    
    hot_windows += (search_windows(test_Image, windows3, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat))  

    window_img = draw_boxes(rgb_image, hot_windows, color=(0, 0, 255), thick=6)   

![alt text][image6]

## C. Remove multiple detection and False positive

### 1. def add_heat(heatmap, bbox_list):

    # Iterate through list of bboxes  
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

### 2. def apply_threshold(heatmap, threshold): 
  
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    
    # Return thresholded map
    return heatmap
	
### 3. def draw_labeled_bboxes(img, labels):  
   
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()        
      
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        
    # Return the image
    return img	

### 4. Implementation for multiple detection and false positives.

From sliding window search, I was able to detect the window of vehicle image. From image of pixels, add one wherever vehicle was detected and zero out everywhere else. Once loop though all the box, using the vector of pixels draw the image, which is call heatmap.   

### Here are six test image and their corresponding heatmaps:

![alt text][image7]

## D. Design the Pipeline which is based on HOG, Sliding Window Search, and heatmap

### 1. def Vehicle_Dection( image ):
    global count, frame_count

    resultImage = np.copy(image)
       
    if (frame_count % 3 == 0) : # Skip every second video frame
    
        heat = []
        slideWindow = []
        
        test_Image = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2RGB )
        
        slideWindow = (search_windows(test_Image, windows1, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat))           
    
        slideWindow += (search_windows(test_Image, windows2, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat))      

        slideWindow += (search_windows(test_Image, windows3, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)) 

        slideWindowImage = draw_boxes(test_Image, slideWindow, color=(0, 0, 255), thick=6)         
    
        heat = np.zeros_like(slideWindowImage[:,:,0]).astype(np.float)

        # Add heat to each box in box list
        heat = add_heat(heat, slideWindow)
    
        # Apply threshold to help remove false positives            
        heat = apply_threshold(heat, 4 )
    
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)      

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        resultImage = draw_labeled_bboxes(np.copy(image), labels)        
   
    frame_count += 1
    
    return resultImage

### 2. Pipeline Implementation 
I combined HOG, Sliding Window Search, and heatmap through seperate function call as one. I had to test through step by step from HOG all the way through heatmap. Due to difference of color space between the image and the image which was extracted from Video, I debug step by step to match the color space to meet with pre pipeline work. I also found the threshold for pre pipeline heatmap wasn't able to filter out all the false positive, so I also increased from 1 to 4.  

---

### Video Implementation

#### [link to project video](./project_video.mp4)     

#### [link to my video result](./result_project_video.mp4)

---

# Discussion

## 1. Designed the windows. 
Currently I limited the windows searching, as the my car is driving in third lain. So if my car move to second or first lain, the sliding windows searching wouldn't work properly

## 2. False positives detection.
In this project we used the simple false positiive detection algothem. While window was detected, I keep the detected windows's pixel as vector. By given the value for each detected pixel, I was able to find where the car is located. And I eliminated the false positve by zero out the pixel where the fiven value is less then threshold. I found out whenever there was shodow, vehicle was detected fausely. So My false positive detection wasn't able to eliminated completely.     








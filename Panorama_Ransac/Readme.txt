~ Phew_Phew~

# This Program is written to stitch any given images into a panorama image

# The images that does not overlap will be ignored

RANSAC: Used for Feature Matching

https://en.wikipedia.org/wiki/Random_sample_consensus

==================================================================================

The latest version of OpenCV does not have the required APIs (xfeature2d).  So install the given version

$ pip install opencv−python ==3.4.2.17

OR

$ pip install opencv−contrib−python ==3.4.2.17

==================================================================================

Execution:

Place the implementation.py python file outside the folder containing the images to be stitched. (The name of this folder should be images or can be modified within the code)

Run Implementation.py  and a fianl_result.jpg file will be created which is the stitched image

==================================================================================

Reference:

https://opencv.org/

https://programmer.group/panoramic-stitching-using-ransac-algorithm.html#:~:text=Principle%20of%20panoramic%20stitching&text=RANSAC%2C%20Random%20Sample%20Consensus%2C%20is,data%20points%20without%20noise%20points.

==================================================================================
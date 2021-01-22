More information on Prokudin-Gorskii Collection can be found here:

https://www.loc.gov/collections/prokudin-gorskii/about-this-collection/
https://www.loc.gov/pictures/collection/prok/
=======================================================================================

Execution:

Place the main.py python file inside the folder containing images (.jpg) and execute.

# new images will be created (3 each for every input image) in PNG format

	image*-color.png - simple overlay without alignment (Blurred)
	image*-ssd.png - Aligned using L2 norm (SSD)
	image*-ncc.png - Aligned using NCC

# Displacements will be printed on the terminal as shown in the example below

# The window size used is -15 to 15

=======================================================================================

Reference
https://pillow.readthedocs.io/en/stable/

=======================================================================================

Output Displacements for the given example images :

image1 SSD Alignment
[-14, -5] SSD Red-Blue
[-7, -2] SSD Red-Green
------------------------------------------
image1 NCC Alignment
[-14, -5] NCC Red-Blue
[-7, -2] NCC Red-Green
==========================================
image2 SSD Alignment
[-11, -4] SSD Red-Blue
[-6, -1] SSD Red-Green
------------------------------------------
image2 NCC Alignment
[-11, -4] NCC Red-Blue
[-6, -1] NCC Red-Green
==========================================
image3 SSD Alignment
[-11, -6] SSD Red-Blue
[-7, -3] SSD Red-Green
------------------------------------------
image3 NCC Alignment
[-11, -6] NCC Red-Blue
[-7, -3] NCC Red-Green
==========================================

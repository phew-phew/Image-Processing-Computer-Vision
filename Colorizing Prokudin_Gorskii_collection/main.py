# ~ Phew_Phew ~

from PIL import Image
import numpy as np
import os 
ls=[]
file_list = os.listdir(".")
pattern = ".jpg"

for files in file_list:
    if pattern in files:
        ls.append(files)

def ncc(im_1,im_2):
    no_cc = np.sum(((im_1/np.linalg.norm(im_1)) * (im_2/np.linalg.norm(im_2))))
    return  no_cc

# Calculating the NCC Displacements

def score_NCC(im_1, im_2, window):
    thr = -999999
    for i in window:
        for j in window:
            ncc_val = ncc(im_1,np.roll(im_2,[i,j],axis=(0,1)))
            if ncc_val > thr:
                thr = ncc_val
                n_disp = [i,j]
    return n_disp

def ssd(img_1, img_2):
    img2 = np.array(img_2, dtype=np.float32)
    ssd = np.sum((img_1 - img2)**2)
    return ssd

# Calculating the SSD Displacements

def score_SSD(im_1, im_2, window):
    ssd_max = 9999999999
    for i in window:
        for j in window:
            Diffssd = ssd(np.array(im_1, dtype=np.float32),np.roll(im_2,[i,j],axis=(0,1)))
            if Diffssd < ssd_max:
                disp_vec = [i,j]
                ssd_max = Diffssd
    return disp_vec

# To loop through all images (.jpg) in the folder
for i in ls:
    ii = (i.replace('.jpg',''))
    im = em = (Image.open(i))
    im = im.convert('RGB')
    width, height = im.size 
    h = np.floor(height/3)
    gr = height - h

# Cropping the images into 3 different images
    im_bl = im.crop((0, 0, width, h))
    im_gr = im.crop((0, h, width, 2*h))
    im_rd = im.crop((0,2*h,width,3*h))

    red1, green1, blue = im_bl.split()
    red1, green, blue1 = im_gr.split()
    red, green1, blue1 = im_rd.split()

    img1 = Image.merge('RGB', (red,green,blue))
    img1.save(ii+'-color.png')# saving the merged color image (Blurred) without adjusting the displacements 


    re  = np.array(red)
    bl = np.array(blue)
    gre = np.array(green)

    #cropping the images to remove the borders
    n_im_bl = im.crop((25, 25, width-25, h-25))
    n_im_gr = im.crop((25, h+25, width-25, (2*h)-25))
    n_im_rd = im.crop((25,(2*h)+25,width-25,(3*h)-25))
 
    n_re  = np.array(n_im_rd)
    n_bl = np.array(n_im_bl)
    n_gre = np.array(n_im_gr)

    rng = 15
    b_list =  [i for i in range(-rng,rng)]

    ###  SSD
    print(ii + ' SSD Alignment')
    ss_r2b = score_SSD(n_re,n_bl,b_list)
    print(ss_r2b,'SSD Red-Blue')
    ss_r2g = score_SSD(n_re,n_gre,b_list)
    print(ss_r2g,'SSD Red-Green')
    ssd_b = np.roll(bl,ss_r2b,axis=(0,1))
    ssd_g=np.roll(gre,ss_r2g,axis=(0,1))
    ss_b = Image.fromarray(ssd_b)
    ss_g = Image.fromarray(ssd_g)

    img2 = Image.merge('RGB', (red,ss_g,ss_b))
    img2.save(ii+'-ssd.png')
    print("------------------------------------------")

    ### NCC
    print(ii + ' NCC Alignment')
    n_r2b = score_NCC(n_re,n_bl,b_list)
    print(n_r2b,'NCC Red-Blue')
    n_g2b = score_NCC(n_re,n_gre,b_list)
    print(n_g2b,'NCC Red-Green')
    print("==========================================")
    ncc_r = np.roll(bl,n_r2b,axis=(0,1))
    ncc_g = np.roll(gre,n_g2b,axis=(0,1))
    nc_r = Image.fromarray(ncc_r)
    nc_g = Image.fromarray(ncc_g)

    img3 = Image.merge('RGB', (red,nc_g,nc_r))
    img3.save(ii+'-ncc.png')

 


#Reference
# https://pillow.readthedocs.io/en/stable/

#import os
import numpy as np
#import matplotlib.pyplot as plt
import cv2

def get_masks_ipac(img,noise_level,filter_len, len_min, len_max, filter_area, area_min, area_max, filter_n, nr_contours):
    """
    Perform contour finding to get a masks of the (possibly) multiple cells
    in the image

    Parameters
    ----------
    img : numpy array of dimensions (width,height)
        Grayscale image.
    sorting : bool
        Insert True if you use that script during sorting. That has the effect that
        the function returs -1 if multiple contours were found.
        Insert False if you want to use that script for preparing the dataset for AIDeveloper. In this case,
        if there is an image with multiple contours, the script will iterate over the contours and only keep
        the object if it is not at the edge of the image
    Returns
    -------
    contour: largest contour found in image
    mask : numpy array of dimensions (width,height)
        True=Cell, False=Backgound.
    """
    #img = img[10:57] #remove top[0:10] and bottom[57:67] , not for findcontours
    img = cv2.medianBlur(img,3)
    _, thresh = cv2.threshold(img, noise_level, 255, cv2.THRESH_BINARY)
    
    
    kern_size = 2*int(3)+1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_size,kern_size))
    dilate = cv2.dilate(thresh,kernel)
    thresh_img = cv2.erode(dilate,kernel)
    
    
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key = len,reverse=True)
    
    ##############  ind  ###############
    cnt_lenghts = np.array([len(cnt) for cnt in contours])
    cnt_area = np.array([cv2.contourArea(cnt) for cnt in contours])
    #cnt_area = sorted(cnt_area,reverse=True)
    
    if filter_len and not filter_area:
        ind = np.where( (cnt_lenghts>len_min) & (cnt_lenghts<len_max) )[0]
        
    if filter_area and not filter_len :
        ind = np.where( (cnt_area>area_min) & (cnt_area<area_max) )[0]        
        #print("ind:", ind)
        
    if filter_len and filter_area:
        ind = np.where( (cnt_area>area_min) & (cnt_area<area_max) 
                       & (cnt_lenghts>len_min) & (cnt_lenghts<len_max))[0] 
    
    ##############  ind  ###############
    
    if filter_n:
        contours = list(np.array(contours)[ind])
        contours.sort(key=len,reverse=True)
        iterations = min(len(contours),nr_contours)
        
    else:
        iterations = len(contours)
    
    contours = contours[0:iterations]

    if len(ind) == 0: #if no conotur was found
        return [],[np.nan] #return nan -> omit sorting

    masks = []
    for it in range(iterations):
        mask = np.zeros_like(img)
        #mask = np.zeros(shape=(67,67),dtype="uint8")
        cv2.drawContours(mask,contours,it,255, cv2.FILLED)#,offset=(0,10) )# produce a contour with filled inside
        masks.append(mask)
    
    return contours,masks


def get_masks_iacs(img, filter_len, len_min, len_max, filter_area, area_min, area_max, filter_n, nr_contours):
    img = cv2.medianBlur(img,3)
    _, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #contours = contours.sort(key=len,reverse=True) ####only return None
    contours = sorted(contours, key = len,reverse=True)
    
    if filter_len and not filter_area:
        cnt_lenghts = np.array([len(cnt) for cnt in contours])
        ind = np.where( (cnt_lenghts>len_min) & (cnt_lenghts<len_max) )[0]
        contours = list(np.array(contours)[ind])
        
    if filter_area and not filter_len :
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        ind = np.where( (cnt_area[0]>area_min) & (cnt_area[0]<area_max) )[0]        
        #print("ind:", ind)
        contours = list(np.array(contours)[ind])
        
    if filter_len and filter_area:
        cnt_lenghts = np.array([len(cnt) for cnt in contours])
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        ind = np.where( (cnt_area[0]>area_min) & (cnt_area[0]<area_max) 
                       & (cnt_lenghts>len_min) & (cnt_lenghts<len_max))[0]  
        contours = list(np.array(contours)[ind])
    
    if len(contours) == 0: #if no conotur was found
        return [],[np.nan] #return nan -> omit sorting
    
    
    if filter_n:
        # contours = list(np.array(contours)[ind])
        iterations = min(len(contours),nr_contours)
        contours = contours[0:iterations]
    else:
        iterations = len(contours)
        
        
    masks = []
    for it in range(iterations):
        mask = np.zeros_like(img)
        cv2.drawContours(mask,contours,it,1,cv2.FILLED)# produce a contour with filled inside
        masks.append(mask)
    
    return contours,masks  


def tiled_2_list(tiled_image):
    img_list = []
    for r in range(0, tiled_image.shape[0], 100):
        for c in range(0, tiled_image.shape[1], 88): 
            img_list.append(tiled_image[r:r +100,c:c +88]) #this was wrong before. Hence, you had images of 88x88 sometimes. But still, the resulting images look almost identical for both cases.
    return img_list


#get the approximate position of the oject in an image
def get_boundingbox_features( img,contour,pixel_size):
    """
    Get the position of the cell by calculating the middle of the bounding box.
    This is actually only an approximation of "the middle of a cell". More accurate 
    would be to compute the centroid but that takes a little longer to compute.
    I guess the bounding box is good enough.

    Parameters
    ----------
    img : numpy array of dimensions (width,height)
        Grayscale image.
    sorting : bool
        Insert True if you use that script during sorting. That has the effect that
        the function returs -1 if multiple contours were found.
        Insert False if you want to use that script for preparing the dataset for AIDeveloper. In this case,
        if there is an image with multiple contours, the script will iterate over the contours and only keep
        the object if it is not at the edge of the image
    Returns
    -------
    pos_x : float
        x-position of the middle of the bounding box.
    pos_y : flaot
        y-position of the middle of the bounding box.

    """
    if type(contour)!=np.ndarray:#get_mask_sorting returned nan
        return np.nan,np.nan

    #Bounding box: fast to compute and gives the approximate location of cell
    x,y,w,h = cv2.boundingRect(contour) #get a bounding box around the object
    # There should be at least 2 pixels between bounding box and the edge of the image
    if x>2 and y>2 and y+h+2<img.shape[0] and x+w+2<img.shape[1]:
        pos_x,pos_y = x+w/2,y+h/2
    else:#Omit sorting if the object is too close to the edge of the image
        pos_x,pos_y = np.nan,np.nan
    
    return pos_x*pixel_size,pos_y*pixel_size,w*pixel_size,h*pixel_size


def get_brightness(img,mask):
    bright = cv2.meanStdDev(img, mask=mask)
    return {"bright_avg":bright[0][0,0],"bright_sd":bright[1][0,0]}


def get_contourfeatures(contour,pixel_size):
    
    
    hull = cv2.convexHull(contour,returnPoints=True)

    mu_orig = cv2.moments(contour)
    mu_hull = cv2.moments(hull)
    
    area_orig = mu_orig["m00"]
    area_hull = mu_hull["m00"]
    area_um = area_hull*pixel_size**2
    if area_orig>0:
        area_ratio = area_hull/area_orig
    else:
        area_ratio = np.nan

    arc = cv2.arcLength(hull, True)    
    circularity = 2.0 * np.sqrt(np.pi * mu_orig["m00"]) / arc
    
    if mu_orig["mu02"]>0:
        inert_ratio_raw =  np.sqrt(mu_orig["mu20"] / mu_orig["mu02"])
    else:
        inert_ratio_raw = np.nan
    
    dic = {"area_um":area_um,"area_orig":area_orig,"area_hull":area_hull,\
           "area_ratio":area_ratio,"circularity":circularity,\
           "inert_ratio_raw":inert_ratio_raw}
    return dic


def add_colormap(img, color):
    """
    

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    color : str
        "Red","Green","Aqua"

    Returns
    -------
    TYPE
        image with colormap

    """
    if len(img.shape) == 2: #grayscale or RGB
        img = cv2.merge((img.copy(),img.copy(),img.copy()))
    
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    if color == "Red":
        lut[:,0,0] = np.arange(256, dtype = np.dtype('uint8')) # 0:red,1:green,3:blue
    if color == "Green": 
        lut[:,0,1] = np.arange(256, dtype = np.dtype('uint8')) # 0:red,1:green,3:blue
    if color == "Aqua":
        lut[:,0,1] = np.arange(256, dtype = np.dtype('uint8')) # 0:red,1:green,3:blue
        lut[:,0,2] = np.arange(256, dtype = np.dtype('uint8')) # 0:red,1:green,3:blue
    return cv2.LUT(img,lut)


def get_color(color_index):
    dic = {"Aqua":(0,255,255,255),"Green":(0,255,0,255),"Red":(255,0,0,255),"White":(255,255,255,255),"Black":(0,0,0,255)}
    return dic[color_index]


def uint16_2_unit8(image):
    """
    This function converts an image from unit16 into uint8
    The function cv2.convertScaleAbs is NOT used as the result did not look 
    good when testing on some images
    Instead, the iamge is divided by the maximum and *255 

    Parameters
    ----------
    image : np.array of dimension (w,h)
        the image in unit16.

    Returns
    -------
    image, converted into unit8.

    """
    image = image.astype(np.float32) #for conversion to unit8, first make it float (float16 is sufficient)
    factor = 255/image.max()
    image = np.multiply(image, factor)

    return image.astype(np.uint8),factor


def vstripes_removal(image):

    bg = np.r_[image[0:5,:],image[-5:,:]]
    bg = cv2.reduce(bg, 0, cv2.REDUCE_AVG)
    bg = np.tile(bg,(100,1))
    
    return cv2.subtract(image,bg)


def hstripes_removal(image):
    """
    Backgound shows vertical stripes
    Get this pattern using top and bottom 5 pixels
    and remove that from original image
    """
    ##Background finding
    #get a slice of 88x5pix at top and bottom of image
    #and put both stripes in one array
    bg = np.c_[image[:,0:10],image[:,-10:]]

    #vertical mean
    bg = cv2.reduce(bg, 1, cv2.REDUCE_AVG)
    #stack to get it back to 100x88 pixel image
    bg = np.tile(bg,(1,image.shape[1]))
    #remove the background
    subtr = image.astype(np.int)-bg.astype(np.int) #after that, the background is approx 0
    
    subtr_abs = abs(subtr).astype(np.uint8) #absolute. Neccessary for contour finding
    
    subtr = subtr+128 #add 255/2 to the whole image. Now, the background has that brightness    
    subtr = subtr.astype(np.uint8)
    return cv2.subtract(image,bg) #subtr,subtr_abs
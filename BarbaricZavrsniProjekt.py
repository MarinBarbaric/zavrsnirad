from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
    help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
    help="path to the output image")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    images.append(image)


print("1. Thresholding")
print("2. Otsu binarization")
print("3. Canny edge detection")
print("4. Sobel edge detection")
print("5. Watershed segmentation")
print("6. Shi-Tomasi corner detection")
print("7. Harris corner detection")
print("8. BF matcher")
print("9. FLANN matcher")
print("10. Panorama")

odabir = input("Odaberite funkciju: ")


def Thresholding(images):
    img = images[0]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,5)
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)

    titles = ['Original Image', 'Global Thresholding (v = 127)',
                'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images2 = [img, th1, th2, th3]

    cv2.imwrite('threshold.png',th3)

    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images2[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    
def Otsu(images):
    img = images[0]
    img = img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # global thresholding
    ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)


    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # plot all the images and their histograms
    images3 = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
              'Original Image','Histogram',"Otsu's Thresholding",
              'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

    for i in range(3):
        plt.subplot(3,3,i*3+1),plt.imshow(images3[i*3],'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+2),plt.hist(images3[i*3].ravel(),256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+3),plt.imshow(images3[i*3+2],'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()
    

def CannyEdge(images):
    img = images[0]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img,100,200)

    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Canny edge detection'), plt.xticks([]), plt.yticks([])

    plt.show()

def SobelEdge(images):
    img = images[0]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    
    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel'), plt.xticks([]), plt.yticks([])
    plt.show()


def WathershedSeg(images):
    img = images[0]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)


    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    cv2.imshow('Dilatation',sure_bg)


    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    cv2.imshow('Image',sure_fg)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Označimo marker
    ret, markers = cv2.connectedComponents(sure_fg)

    # Marker povečavamo za 1
    markers = markers+1

    # Označavamo nepoznati dio slike
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    #Prikaz slike
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def ShiTomasiCorner(images):
    img = images[0]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)

    plt.imshow(img),plt.show()


def HarrisCorner(images):
    img = images[0]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    
    dst = cv2.dilate(dst,None)

    
    img[dst>0.01*dst.max()]=[0,0,255]

    cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
    cv2.imshow('Image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def BFmatcher(images):
    img1 = images[0]
    img2 = images[1]
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100],None, flags=2)

    plt.imshow(img3),plt.show()

def FLANNmatcher(images):
    img1 = images[0]
    img2 = images[1]
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    plt.imshow(img3,),plt.show()

def Panorama(images):
    print("[INFO] stitching images...")
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)

    
    if status == 0:
        
        cv2.imwrite(args["output"], stitched)

        
        cv2.imshow("Stitched", stitched)
        cv2.waitKey(0)

    
    else:
        print("[INFO] image stitching failed ({})".format(status))


if odabir == '1':
    Thresholding(images)
elif odabir == '2':
    Otsu(images)
elif odabir == '3':
    CannyEdge(images)
elif odabir == '4':
    SobelEdge(images)
elif odabir == '5':
    WathershedSeg(images)
elif odabir == '6':
    ShiTomasiCorner(images)
elif odabir == '7':
    HarrisCorner(images)
elif odabir == '8':
    BFmatcher(images)
elif odabir == '9':
    FLANNmatcher(images)
elif odabir == '10':
    Panorama(images)
else:
    print("Krivi unos...")

from libs import *

##functions##

def crop(input_image):
    _, mask = cv.threshold(cv.cvtColor(input_image, cv.COLOR_RGB2GRAY), 10, 1, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    areas = list(cv.contourArea(contour) for contour in contours)
    index = np.where(areas == np.max(areas))[0][0]
    x0, y0, w, h = cv.boundingRect(contours[index])
    mask = cv.fillPoly(img=np.zeros(shape=mask.shape,dtype=np.uint8), pts=[contours[index]], color=1)
    res_img = np.empty_like(input_image)
    for j in range(mask.shape[0]):
        for i in range(mask.shape[1]):
            for k in range(input_image.shape[2]): 
                res_img[j, i, k] = input_image[j, i, k] * mask[j, i]
    
    return res_img[y0:y0+h, x0:x0+w, :], mask[y0:y0+h, x0:x0+w], x0, y0, w, h

def CLAHE(input_img, tile_dim = 8, cliplim = 0.01, num_bins = 256):
    #Histogram equalization via CLAHE
    min_cliplim = (tile_dim**2)/num_bins
    actual_cliplim = min_cliplim + round(cliplim*(tile_dim**2 - min_cliplim))
    clahe = cv.createCLAHE(clipLimit = actual_cliplim, tileGridSize = (tile_dim,  tile_dim))

    return clahe.apply(input_img)

def Half_wave(input_img, thresh = 0.1):
    #Halfwave rectification with a defined threshold value
    result_img = np.empty_like(input_img)
    max_intensity = np.max(input_img)
    perVal = max_intensity*(thresh)
    for j in range(input_img.shape[0]):
        for i in range(input_img.shape[1]):
            result_img[j, i] = (max_intensity if input_img[j, i] >= perVal else 0)

    return result_img

def Gabor_filt(img, wavelength = 9, theta = 0, phase = 90, gamma = 0.5, bandwidth = 1, Orient_num = 24):
    #Gabor filtering with a 360° sweep
    filt_img = np.zeros((img.shape[0], img.shape[1], Orient_num))
    result_img = np.zeros(img.shape)
    slratio = np.sqrt((np.log(2)*(2**bandwidth + 1))/(2*(2**bandwidth - 1)))/np.pi
    sigma = slratio*wavelength
    size = int(2*np.floor(2.5*(sigma/gamma)) + 1)
    for n in range(Orient_num):
        theta = np.deg2rad(theta) + n*np.deg2rad(360/Orient_num)
        phase = np.deg2rad(phase)
        kernel = cv.getGaborKernel(ksize = (size, size), sigma = sigma, theta = theta, lambd = wavelength, gamma = gamma, psi = phase)
        filt_img[:, :, n] = ~cv.filter2D(img, cv.CV_8UC1, kernel)
        max_val = np.max(filt_img[:,:, n])
        for j in range(filt_img.shape[0]):
            for i in range(filt_img.shape[1]):
                if filt_img[j, i, n] == max_val: result_img[j, i] = filt_img[j, i, n]

    return np.uint8(result_img)

def BW_open(bw_img, SE, iterations = 1):
    #Binary image area opening
    res_img = cv.erode(src = bw_img, kernel = SE, iterations = iterations)
    res_img = cv.dilate(src = res_img, kernel = SE, iterations = iterations)

    return res_img

def BW_close(bw_img, SE, iterations = 1):
    #Binary image area closing
    res_img = cv.dilate(src = bw_img, kernel = SE, iterations = iterations)
    res_img = cv.erode(src = res_img, kernel = SE, iterations = iterations)

    return res_img

def get_dynamic_SE(kernel_area, bh_ratio = 1, cv_shape = cv.MORPH_RECT):
    """
        Structuring Element generation based on dynamic input data
        kernel_area (int): Dynamic desired area for Structuring element.
        bh_ratio (int): Base vs. Height ratio for the kernel based on the kernel area
        cv_shape (cv object): Kernel shape using OpenCV format
    """
    base = int(np.ceil(np.sqrt(kernel_area)*bh_ratio))
    height = int(np.ceil(np.sqrt(kernel_area)/bh_ratio))

    return cv.getStructuringElement(shape = cv_shape, ksize = (base, height))

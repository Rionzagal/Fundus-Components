from libs import *
from ufuncs import *

def retrieve_macula(image, mask):
    ###Color segmentation stage
    g_channel = image[:,:, 1]
    range_i = np.max(g_channel) - np.min(g_channel)
    cliplimit = (np.mean(g_channel))/(range_i)
    clahe_g = (~CLAHE(g_channel, tile_dim = 8, cliplim = cliplimit, num_bins = 256))*mask
    _, bwG = cv.threshold(clahe_g, (np.mean(clahe_g) + np.std(clahe_g)), 0xFF, cv.THRESH_BINARY)

    ###Shape segmentation stage
    SE = np.ones((3, 3), np.uint8)
    bw_clean = cv.erode(src = bwG, kernel = SE, iterations = 2)
    areas = list(cv.contourArea(contour) for contour in cv.findContours(bw_clean, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[0])
    SE = get_dynamic_SE(kernel_area = np.mean(areas), bh_ratio = 1, cv_shape = cv.MORPH_ELLIPSE)
    bw_clean = BW_close(bw_img = bw_clean, SE = SE, iterations = 2)
    contours, _ = cv.findContours(bw_clean, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    areas = list(cv.contourArea(contour) for contour in contours)
    #circularities = list((areas[index]/(np.pi*cv.minEnclosingCircle(contours[index])[1])) for index in range(areas.__len__()))
    ref_img = np.zeros(shape = image.shape[:2], dtype = np.uint8)
    for index in range(len(areas)):
        if np.max(areas) == areas[index]:
            Macula = contours[index]
            center, Macula_radius = cv.minEnclosingCircle(Macula)
    center = (round(center[0]), round(center[1]))
    radius = round(Macula_radius)
    return cv.circle(ref_img, center, radius, 0xFF, -1), center, radius

if __name__ == '__main__':
    image = plt.imread(f"{os.getcwd()}\\datasets\\Drive datasets\\images\\21_training.tif")
    input_image, mask = crop(image)[:2]

    macula, m_center, m_radius = retrieve_macula(input_image, mask)

    plt.figure()
    plt.imshow(cv.circle(input_image, m_center, m_radius, (0, 0xFF, 0), 3))

    plt.show()

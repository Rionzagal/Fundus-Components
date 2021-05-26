from matplotlib.figure import Figure
from matplotlib.pyplot import figure
from libs import *
from ufuncs import *

def retrieve_optic_disc(image, mask):
    frame = cv.morphologyEx(src=mask, op=cv.MORPH_GRADIENT, kernel=np.ones(shape=(7,7), dtype=np.uint8))
    ###Maximum Difference Method
    size = 2*round((np.sqrt(image.shape[0]*image.shape[1])/40)) + 1
    I_med = cv.medianBlur(src = image[:,:, 1], ksize = size)
    for j in range(I_med.shape[0]):
        for i in range(I_med.shape[1]):
            window = I_med[j: j + size, i: i + size]
            I_med[j, i] = np.max(window) - np.min(window)
    I_med *= (mask)
    index = np.where(I_med == np.max(I_med))
    MDP = (index[0][0] + round((size/2)), index[1][0] + round((size/2))) #Maximum difference method candidate pixel

    ###Maximum Variance Method
    size = 4*round((np.sqrt(image.shape[0]*image.shape[1])/40)) + 1
    otsu_thresh, _ = cv.threshold(src = image[:,:, 2], thresh = 0, maxval = 0xFF, type = cv.THRESH_TOZERO + cv.THRESH_OTSU)
    _, I_otsu = cv.threshold(src = image[:,:, 1], thresh = otsu_thresh, maxval = 0xFF, type = cv.THRESH_TOZERO)
    I_otsu *= (mask - frame)
    var_I = np.zeros(shape=I_otsu.shape)
    for j in range(I_otsu.shape[0]):
        for i in range(I_otsu.shape[1]):
            window = I_otsu[j:j+size, i:i+size]
            bright_pixels = np.where(window.reshape(-1) > (np.mean(I_otsu) + np.std(I_otsu)))[0]
            var_I[j,i] = round(np.var(window) if 20 <= bright_pixels.size else 0)
    index = np.where(var_I == np.max(var_I))
    MVP = (index[0][0]+round((size/2)), index[1][0]+round((size/2))) #Maximum variance method candidate pixel

    ###Low-pass filter method
    size = 2*round((np.sqrt(image.shape[0]*image.shape[1])/80)) + 1
    gauss_img = cv.GaussianBlur(src = image[:,:, 1], ksize = (size, size), sigmaX = 25)
    gauss_img *= (mask)
    index = np.where(gauss_img == np.max(gauss_img))
    LFP = (index[0][0] + round(size/2), index[1][0] + round(size/2)) #Low-Pass filter method candidate pixel

    ###Voting procedure
    CP = (round((MDP[0] + MVP[0] + LFP[0])/3), round((MDP[1] + MVP[1] + LFP[1])/3)) #centroid pixel
    threshold_distance = np.sqrt(image.shape[0]*image.shape[1])/5
    candidate_pixels = [MDP, MVP, LFP]
    distances = list()
    for c in range(len(candidate_pixels)):
        distances.append(np.sqrt((candidate_pixels[c][0] - CP[0])**2 + (candidate_pixels[c][1] - CP[1])**2))
    NCP = np.where(distances <= threshold_distance)[0]
    if 3 == NCP.size:
        ODP = CP
    elif 2 == NCP.size:
        ODP = (
            round((candidate_pixels[NCP[0]][0]+candidate_pixels[NCP[1]][0])/2),
            round((candidate_pixels[NCP[0]][1]+candidate_pixels[NCP[1]][1])/2)
            )
    else:
        ODP = candidate_pixels[2]
    ###Optic Disc area zoom
    OD_area_size = 2*round((np.sqrt(image.shape[0]*image.shape[1])/5)) + 1
    if (ODP[0] > (OD_area_size - 1)/2) and (ODP[1] > (OD_area_size - 1)/2):
        rect_loc = (ODP[0] - round((OD_area_size - 1)/2), (ODP[1] - round((OD_area_size - 1)/2)))
    elif (ODP[0] < (OD_area_size - 1)/2) and (ODP[1] > (OD_area_size - 1)/2):
        rect_loc = (0, (ODP[1] - round((OD_area_size - 1)/2)))
    elif (ODP[0] > (OD_area_size - 1)/2) and (ODP[1] < (OD_area_size - 1)/2):
        rect_loc = (ODP[0] - round((OD_area_size - 1)/2), 0)
    else:
        rect_loc = (0, 0)
    OD_area = image[rect_loc[0]: rect_loc[0] + OD_area_size, rect_loc[1]: rect_loc[1] + OD_area_size, :]
    OD_gray = cv.cvtColor(OD_area, code = cv.COLOR_RGB2GRAY)
    _, OD_gray = cv.threshold(OD_gray, np.mean(OD_gray) + np.std(OD_gray), maxval = 0xFF, type = cv.THRESH_BINARY)
    areas = list(cv.contourArea(contour) for contour in cv.findContours(OD_gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[0])
    r = 1*round((np.sqrt(np.mean(areas)/np.pi)))
    OD_aG = CLAHE(input_img = OD_area[:, :, 1])
    SE = cv.getStructuringElement(shape = cv.MORPH_ELLIPSE, ksize = (r, r))
    cv.blur(src = OD_aG, ksize = (3, 3), dst = OD_aG)
    _, OD_borders = cv.threshold(src = OD_aG, thresh = 0.9*np.max(OD_aG), maxval = 0xFF, type = cv.THRESH_BINARY)
    cv.dilate(src = OD_borders, kernel = SE, dst = OD_borders)
    cv.erode(
        src = OD_borders, 
        kernel = cv.getStructuringElement(shape = cv.MORPH_ELLIPSE, ksize = (int(r/4), int(r/4))), 
        dst = OD_borders
    )
    ##Optic Disc complete insertion
    od_image = np.zeros(shape = image.shape[:2], dtype = np.uint8)
    od_image[rect_loc[0]: rect_loc[0] + OD_area_size, rect_loc[1]: rect_loc[1] + OD_area_size] = OD_borders
    od_contour = cv.findContours(image = od_image, mode = cv.RETR_TREE, method = cv.CHAIN_APPROX_NONE)[0][0]
    center, radius = cv.minEnclosingCircle(od_contour)
    center = (round(center[0]), round(center[1]))
    radius = round(radius)

    plt.figure()
    plt.imshow(I_med, cmap='gray')
    plt.plot(MDP[1], MDP[0], 'o')
    plt.title("Maximum Difference Method")
    plt.axis(False)

    plt.figure()
    plt.imshow(I_otsu, cmap='gray')
    plt.plot(MVP[1], MVP[0], 'o')
    plt.title("Maximum Variance method")
    plt.axis(False)

    plt.figure()
    plt.imshow(gauss_img, cmap='gray')
    plt.plot(LFP[1], LFP[0], 'o')
    plt.title("Low pass Filter method")
    plt.axis(False)

    plt.figure()
    plt.imshow(image)
    plt.plot(MDP[1], MDP[0], 'or', label = 'MDP')
    plt.plot(MVP[1], MVP[0], 'og', label = 'MVP')
    plt.plot(LFP[1], LFP[0], 'ob', label = 'LFP')
    plt.plot(ODP[1], ODP[0], 'om', label = 'ODP')
    plt.title("Candidate pixels and ODP")
    plt.axis(False)
    plt.legend()

    plt.figure()
    plt.imshow(od_image, cmap='gray')
    plt.plot(center[0], center[1], 'o', label = 'OD center')
    plt.title("Optic Disc binary area")
    plt.axis(False)
    plt.legend()

    return cv.circle(od_image, center, radius, 0xFF, -1), center, radius

if __name__ == '__main__':
    image = plt.imread(f"{os.getcwd()}\\datasets\\Drive datasets\\images\\21_training.tif")
    input_image, mask = crop(image)[:2]

    OD_mask, od_center, od_radius = retrieve_optic_disc(input_image, mask)

    plt.figure()
    plt.imshow(cv.circle(input_image, od_center, od_radius, (0, 0xFF, 0), 3))
    plt.title("Detected Optic Disc")
    plt.axis(False)

    plt.show()
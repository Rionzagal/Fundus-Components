from libs import *
from ufuncs import *

def preprocess(image, mask):
    ###color channel separations
        g1_channel = np.zeros(shape = image.shape[:2], dtype = np.uint8)
        for j in range(g1_channel.shape[0]):
            for i in range(g1_channel.shape[1]):
                g1_channel[j, i] = 0.06*image[j, i, 0] + 0.63*image[j, i, 1] + 0.27*image[j, i, 2]
        color_imgs = list()
        color_imgs.append(image[:, :, 1])
        color_imgs.append(cv.cvtColor(image, cv.COLOR_RGB2YCR_CB)[:, :, 0])
        color_imgs.append(cv.cvtColor(image, cv.COLOR_RGB2LAB)[:, :, 0])
        color_imgs.append(g1_channel)
        ###Contrast enhancement using CLAHE
        contrast_imgs = list()
        for image in color_imgs:
            cliplimit = 0.6*(np.mean(image))/(np.max(image) - np.min(image))
            contrast_imgs.append((CLAHE(input_img = image, tile_dim = 8, cliplim = cliplimit, num_bins = 256))*mask)
        ###Gabor filtering
        frame = cv.morphologyEx(src=mask, op=cv.MORPH_GRADIENT, kernel=np.ones((7,7), dtype=np.uint8))
        gabor_filter9 = list()
        gabor_filter10 = list()
        gabor_filter11 = list()
        for image in contrast_imgs:
            #wl9 gabor filter
            g9 = Gabor_filt(img = image, wavelength = 9, theta = 0, phase = 0, gamma = 0.5, bandwidth = 1, Orient_num = 24)
            cv.threshold(src = g9, thresh = np.mean(g9), maxval = 0xFF, dst = g9, type = cv.THRESH_BINARY)
            gabor_filter9.append(g9*(mask - frame))
            #wl10 gabor filter
            g10 = Gabor_filt(img = image, wavelength = 10, theta = 0, phase = 0, gamma = 0.5, bandwidth = 1, Orient_num = 24)
            cv.threshold(src = g10, thresh = np.mean(g10), maxval = 0xFF, dst = g10, type = cv.THRESH_BINARY)
            gabor_filter10.append(g10*(mask - frame))
            #wl11 gabor filter
            g11 = Gabor_filt(img = image, wavelength = 11, theta = 0, phase = 0, gamma = 0.5, bandwidth = 1, Orient_num = 24)
            cv.threshold(src = g11, thresh = np.mean(g11), maxval = 0xFF, dst = g11, type = cv.THRESH_BINARY)
            gabor_filter11.append(g11*(mask - frame))
        ###feature extraction
        feature_list = list()
        features = pd.DataFrame()
        feature_list.append(contrast_imgs[0].reshape(-1))
        for image in gabor_filter9: feature_list.append(image.reshape(-1))
        for image in gabor_filter10: feature_list.append(image.reshape(-1))
        for image in gabor_filter11: feature_list.append(image.reshape(-1))
        for column in range(len(feature_list)): features[str(column + 1)] = feature_list[column]

        return features

def train_classifier(features, ground_truth):
    model = RandomForestClassifier(n_estimators= 10000, random_state=0)
    X_train, X_test, Y_train, Y_test = train_test_split(features, ground_truth.reshape(-1), test_size = 0.4, random_state = 0)
    model.fit(X_train, Y_train)
    pred_test = model.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test, pred_test)
    print("Trained model accuracy: {:.4f}%".format(100*accuracy))

    return model

if __name__ == '__main__':
    image = plt.imread(f"{os.getcwd()}\\datasets\\Drive datasets\\images\\21_training.tif")
    g_truth = plt.imread(f"{os.getcwd()}\\datasets\\Drive datasets\\g_truth\\21_manual1.gif")
    input_image, mask, x0, y0, w, h = crop(image)
    g_truth = g_truth[y0:y0+h, x0:x0+w]

    data_features = preprocess(input_image, mask)
    trained_model = train_classifier(data_features, g_truth)

    test_img = plt.imread(f"{os.getcwd()}\\datasets\\Drive datasets\\images\\33_training.tif")
    test_truth = plt.imread(f"{os.getcwd()}\\datasets\\Drive datasets\\g_truth\\33_manual1.gif")

    test_img, test_mask, x0, y0, w, h = crop(test_img)
    test_truth = test_truth[y0:y0+h, x0:x0+w]

    prediction = trained_model.predict(preprocess(test_img, test_mask))
    accuracy = metrics.accuracy_score(test_truth.reshape(-1), prediction)
    cv.imshow("Ground truth", test_truth)
    cv.imshow("Training and testing results", prediction.reshape(test_mask.shape))
    print(f"Test accuracy: {(100*accuracy):.4f}%")
    cv.waitKey(0)
    cv.destroyAllWindows

    joblib.dump(trained_model, 'classifier.sav') #Model saving

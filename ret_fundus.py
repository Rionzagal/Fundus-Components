from libs import *
from ufuncs import *
from ML_training import *
from optic_disc import *
from macula_color import *

class retinal_fundus(object):
    retina = None
    mask = None
    frame = None
    shape = None
    Optic_Disc = {
        'Center': None,
        'Radius': None,
        'Mask': None
    }
    Macula = {
        'Center': None,
        'Radius': None,
        'Mask': None
    }
    Vascular_tree = {
        'Mask': None
    }

    def vascular_tree(self, model_dir):
        ##ML model evaluation
        if os.path.exists(model_dir):
            model = joblib.load(model_dir)
        else:
            model = None
            warnings.warn("Model path incorrect or non-existent! Cannot obtain requested component!")
            return None
        ##Feature dataframe retrieval
        features = preprocess(self.retina, self.mask)
        ###ML model inference
        vessel_inference = model.predict(features)
        vessel_inference = 0xFF*vessel_inference.reshape(self.shape[:2])*(self.mask - self.frame)
        SE = np.ones(shape = (3, 3), dtype = np.uint8)
        vessel_inference = BW_open(vessel_inference, np.ones(shape = (1, 1), dtype = np.uint8), iterations = 5)
        self.Vascular_tree['Mask'] = BW_close(vessel_inference, SE, iterations = 1)

    def optic_disc(self):
        self.Optic_Disc['Mask'], self.Optic_Disc['Center'], self.Optic_Disc['Radius'] = retrieve_optic_disc(self.retina, self.mask)
    
    def macula(self):
        self.Macula['Mask'], self.Macula['Center'], self.Macula['Radius'] = retrieve_macula(self.retina, self.mask)
        
    def __init__(self, image_dir, height = None, width = None):
        """
            Retinal fundus first definition.
            image_dir: fundus image directory
            height: desired image height
            width: desired image width
        """
        I = plt.imread(image_dir)
        if (height is not None) and (width is not None):
            I = cv.resize(src=I, dsize=(height,width))
        elif height is not None:
            ratio = I.shape[0]/I.shape[1]
            I = cv.resize(src=I, dsize=(height, int(height*ratio)))
        elif width is not None:
            ratio = I.shape[1]/I.shape[0]
            I = cv.resize(src=I, dsize=(int(width*ratio), width))
        self.retina, self.mask = crop(I)[:2]
        self.frame = cv.morphologyEx(src = self.mask, op = cv.MORPH_GRADIENT, kernel = np.ones(shape = (7, 7), dtype = np.uint8))
        self.shape = self.retina.shape

        self.vascular_tree(model_dir=f"{os.getcwd()}\\classifier.sav")
        self.optic_disc()
        self.macula()


if __name__ == '__main__':
    retina = retinal_fundus(f"{os.getcwd()}\\datasets\\Accepted images\\01_test.tif")

    plt.figure()
    plt.imshow(retina.retina)
    plt.title("Orignal fundus image")
    plt.axis(False)

    plt.figure()
    plt.imshow(retina.Vascular_tree['Mask'], cmap='gray')
    plt.title("Vascular tree")
    plt.axis(False)

    plt.figure()
    plt.imshow(retina.retina + retina.Optic_Disc['Mask'])
    plt.title("Retinal Optic Disc")
    plt.axis(False)

    plt.figure()
    plt.imshow(cv.circle(retina.retina, retina.Macula['Center'], retina.Macula['Radius'], (0, 0xFF, 0), 3))
    plt.title("Retinal Macula")
    plt.axis(False)

    plt.show()
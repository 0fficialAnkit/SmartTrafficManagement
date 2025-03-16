import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_local
import tensorflow as tf
from skimage import measure
import imutils
import os

# Define the segment_chars function here
def segment_chars(plate, threshold_value=400):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    character_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 10:
            char_image = plate[y:y+h, x:x+w]
            character_images.append(char_image)

    return character_images if len(character_images) > 0 else None


class PlateFinder:
    def __init__(self, minPlateArea, maxPlateArea):
        self.min_area = minPlateArea
        self.max_area = maxPlateArea
        self.element_structure = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(22, 3))

    def preprocess(self, input_img):
        imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0)
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        ret2, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        element = self.element_structure
        morph_n_thresholded_img = threshold_img.copy()
        cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_n_thresholded_img)

        return morph_n_thresholded_img

    def extract_contours(self, after_preprocess):
        contours, _ = cv2.findContours(after_preprocess, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        return contours

    def clean_plate(self, plate):
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            max_cnt = contours[max_index]
            max_cntArea = areas[max_index]
            x, y, w, h = cv2.boundingRect(max_cnt)
            rect = cv2.minAreaRect(max_cnt)
            if not self.ratioCheck(max_cntArea, plate.shape[1], plate.shape[0]):
                return plate, False, None
            return plate, True, [x, y, w, h]
        else:
            return plate, False, None

    def check_plate(self, input_img, contour):
        min_rect = cv2.minAreaRect(contour)
        if self.validateRatio(min_rect):
            x, y, w, h = cv2.boundingRect(contour)
            after_validation_img = input_img[y:y + h, x:x + w]
            after_clean_plate_img, plateFound, coordinates = self.clean_plate(after_validation_img)
            if plateFound:
                characters_on_plate = self.find_characters_on_plate(after_clean_plate_img)
                if characters_on_plate is not None and len(characters_on_plate) == 8:
                    x1, y1, w1, h1 = coordinates
                    coordinates = x1 + x, y1 + y
                    return after_clean_plate_img, characters_on_plate, coordinates
        return None, None, None

    def find_possible_plates(self, input_img):
        plates = []
        self.char_on_plate = []
        self.corresponding_area = []
        self.after_preprocess = self.preprocess(input_img)
        possible_plate_contours = self.extract_contours(self.after_preprocess)

        for cnts in possible_plate_contours:
            plate, characters_on_plate, coordinates = self.check_plate(input_img, cnts)
            if plate is not None:
                plates.append(plate)
                self.char_on_plate.append(characters_on_plate)
                self.corresponding_area.append(coordinates)

        if len(plates) > 0:
            return plates
        else:
            return None

    def find_characters_on_plate(self, plate):
        charactersFound = segment_chars(plate, 400)
        if charactersFound:
            return charactersFound

    def ratioCheck(self, area, width, height):
        min = self.min_area
        max = self.max_area
        ratioMin = 3
        ratioMax = 6
        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio
        if area < min or area > max or ratio < ratioMin or ratio > ratioMax:
            return False
        return True

    def validateRatio(self, rect):
        (x, y), (width, height), rect_angle = rect
        if width > height:
            angle = -rect_angle
        else:
            angle = 90 + rect_angle
        if angle > 15:
            return False
        if height == 0 or width == 0:
            return False
        area = width * height
        if not self.ratioCheck(area, width, height):
            return False
        else:
            return True

class OCR:
    def __init__(self, modelFile, labelFile):
        self.model_file = modelFile
        self.label_file = labelFile
        self.label = self.load_label(self.label_file)
        self.graph = self.load_graph(self.model_file)
        self.sess = tf.compat.v1.Session(graph=self.graph, config=tf.compat.v1.ConfigProto())

    def load_graph(self, modelFile):
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with open(modelFile, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    def load_label(self, labelFile):
        label = []
        proto_as_ascii_lines = tf.io.gfile.GFile(labelFile).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def convert_tensor(self, image, imageSizeOuput):
        image = cv2.resize(image, dsize=(imageSizeOuput, imageSizeOuput), interpolation=cv2.INTER_CUBIC)
        np_image_data = np.asarray(image)
        np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        np_final = np.expand_dims(np_image_data, axis=0)
        return np_final

    def label_image(self, tensor):
        input_name = "import/input"
        output_name = "import/final_result"
        input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)
        results = self.sess.run(output_operation.outputs[0], {input_operation.outputs[0]: tensor})
        results = np.squeeze(results)
        labels = self.label
        top = results.argsort()[-1:][::-1]
        return labels[top[0]]

    def label_image_list(self, listImages, imageSizeOuput):
        plate = ""
        for img in listImages:
            plate = plate + self.label_image(self.convert_tensor(img, imageSizeOuput))
        return plate, len(plate)

if __name__ == "__main__":
    findPlate = PlateFinder(minPlateArea=4100, maxPlateArea=15000)
    model = OCR(modelFile=r"D:\Downloads\binary_128_0.50_ver3.pb", labelFile=r"D:\Downloads\binary_128_0.50_labels_ver2.txt")

    cap = cv2.VideoCapture(r"D:\Downloads\output_video_car.mov")

    while cap.isOpened():
        ret, img = cap.read()

        if ret:
            cv2.imshow('Original Video', img)
            possible_plates = findPlate.find_possible_plates(img)

            if possible_plates is not None:
                for i, p in enumerate(possible_plates):
                    chars_on_plate = findPlate.char_on_plate[i]
                    recognized_plate, _ = model.label_image_list(chars_on_plate, imageSizeOuput=128)

                    print(f"Detected Plate Number: {recognized_plate}")

                    cv2.imshow('Detected Plate', p)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

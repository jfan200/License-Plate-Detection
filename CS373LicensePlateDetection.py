import sys
from pathlib import Path
from PIL import Image
import numpy as np
from easyocr import easyocr
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import re
import requests

url = "https://api.mindee.net/v1/products/Jinhua/template_ocr/v1/predict"

# import our basic, light-weight png reader library
import imageIO.png


# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue=0):
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# Conversion to Greyscale
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for row in range(image_height):
        for col in range(image_width):
            r = pixel_array_r[row][col]
            g = pixel_array_g[row][col]
            b = pixel_array_b[row][col]

            greyscale_pixel_array[row][col] = round(0.299 * r + 0.587 * g + 0.114 * b)

    return greyscale_pixel_array


# Contrast Stretching
def computeMinAndMaxValues(pixel_array, image_width, image_height):
    min_v = pixel_array[0][0]
    max_v = 0
    for i in pixel_array:
        for j in i:
            if j < min_v:
                min_v = j
            if j > max_v and j != min_v:
                max_v = j
    return [min_v, max_v]


def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    contrast_stretched_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    min_v, max_v = computeMinAndMaxValues(pixel_array, image_width, image_height)

    for row in range(len(pixel_array)):
        for col in range(len(pixel_array[row])):
            if pixel_array[row][col] == min_v:
                contrast_stretched_pixel_array[row][col] = 0
            if pixel_array[row][col] == max_v:
                contrast_stretched_pixel_array[row][col] = 255
            else:
                contrast_stretched_pixel_array[row][col] = round(
                    (255 / (max_v - min_v)) * (pixel_array[row][col] - min_v))
    return contrast_stretched_pixel_array


# Thresholding for Segmentation
def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    thresholded = []
    for i in range(image_height):
        thresholded.append([0] * image_width)

    for row in range(len(pixel_array)):
        for col in range(len(pixel_array[row])):
            if pixel_array[row][col] >= threshold_value:
                thresholded[row][col] = 255
    return thresholded


#
def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    pixel = createInitializedGreyscalePixelArray(image_width, image_height)

    for row in range(2, image_height - 2):
        for col in range(2, image_width - 2):
            temp = []
            temp += pixel_array[row - 2][col - 2: col + 3]
            temp += pixel_array[row - 1][col - 2: col + 3]
            temp += pixel_array[row][col - 2: col + 3]
            temp += pixel_array[row + 1][col - 2: col + 3]
            temp += pixel_array[row + 2][col - 2: col + 3]

            sum_ = 0
            for num in temp:
                sum_ += (num - sum(temp) / 25) ** 2
            sd = (sum_ / 25) ** 0.5
            pixel[row][col] = sd

    return pixel


#
def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    array1 = [[0] * image_width for i in range(image_height)]

    for row in range(1, image_height - 1):
        for col in range(1, image_width - 1):
            result = []
            result += (pixel_array[row - 1][col - 1: col + 2])
            result += (pixel_array[row][col - 1: col + 2])
            result += (pixel_array[row + 1][col - 1: col + 2])

            if 0 not in result:
                array1[row][col] = 1
            else:
                array1[row][col] = 0
    return array1


def do_erosion3x3(pixel_array, num, image_width, image_height):
    for i in range(num):
        pixel_array = computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height)
    return pixel_array


def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    array1 = [[0] * image_width for i in range(image_height)]

    for row in range(0, image_height):
        for col in range(0, image_width):
            num = pixel_array[row][col]
            if num != 0:
                for row_index in [row - 1, row, row + 1]:
                    for col_index in [col - 1, col, col + 1]:
                        if row_index != -1 and col_index != -1:
                            try:
                                array1[row_index][col_index] = 1
                            except:
                                pass
    return array1


def do_dilation3x3(pixel_array, num, image_width, image_height):
    for i in range(num):
        pixel_array = computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height)
    return pixel_array


# DeepCopy
def DeepCopy(pixel_array):
    result = []
    for row in pixel_array:
        temp = []
        for col in row:
            temp.append(col)
        result.append(temp)
    return result


#
def computeGaussianAveraging3x3RepeatBorder(pixel_array, image_width, image_height):
    pixel = (createInitializedGreyscalePixelArray(image_width, image_height))
    core = [[0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]]

    pixel_array = [6 * [0]] + pixel_array + [6 * [0]]

    for i in range(len(pixel_array)):
        pixel_array[i] = [0] + pixel_array[i] + [0]

    pixel_array[0], pixel_array[-1] = pixel_array[1], pixel_array[-2]

    for i in range(image_height + 2):
        pixel_array[i][0] = pixel_array[i][1]
        pixel_array[i][-1] = pixel_array[i][-2]

    for i in range(1, image_height + 1):
        for j in range(1, image_width + 1):
            pixel[i - 1][j - 1] = abs(
                pixel_array[i - 1][j - 1] * core[0][0] + pixel_array[i - 1][j] * core[0][1] + pixel_array[i - 1][
                    j + 1] * core[0][2] + pixel_array[i][j - 1] * core[1][0] + pixel_array[i][j] * core[1][1] +
                pixel_array[i][j + 1] * core[1][2] + pixel_array[i + 1][j - 1] * core[2][0] + pixel_array[i + 1][j] *
                core[2][1] + pixel_array[i + 1][j + 1] * core[2][2])
    return pixel


# Vertical
def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    pixel = createInitializedGreyscalePixelArray(image_width, image_height)
    core = [[0.125, 0, -0.125],
            [0.25, 0, -0.25],
            [0.125, 0, -0.125]]
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            pixel[i][j] = abs(
                pixel_array[i - 1][j - 1] * core[0][0] + pixel_array[i - 1][j] * core[0][1] + pixel_array[i - 1][
                    j + 1] * core[0][2] + pixel_array[i][j - 1] * core[1][0] + pixel_array[i][j] * core[1][1] +
                pixel_array[i][j + 1] * core[1][2] + pixel_array[i + 1][j - 1] * core[2][0] + pixel_array[i + 1][j] *
                core[2][1] + pixel_array[i + 1][j + 1] * core[2][2])
    return pixel


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    pixel = [[0] * image_width for i in range(image_height)]

    pixel_array = [[0] * image_width] + pixel_array + [[0] * image_width]
    for index in range(len(pixel_array)):
        pixel_array[index] = [0] + pixel_array[index] + [0]

    num = 1
    dict1 = {}

    for row in range(1, image_height + 1):
        for col in range(1, image_width + 1):
            if pixel_array[row][col] != 0 and pixel_array[row][col] != "visited":
                q = Queue()
                q.enqueue((row, col,))
                pixel_array[row][col] = "visited"
                while not q.isEmpty():
                    (a, b,) = q.dequeue()
                    pixel[a - 1][b - 1] = num
                    if num not in dict1.keys():
                        dict1[num] = [(a, b,)]
                    else:
                        dict1[num].append((a, b,))

                    if pixel_array[a - 1][b] != 0 and pixel_array[a - 1][b] != "visited":
                        q.enqueue((a - 1, b,))
                        pixel_array[a - 1][b] = "visited"
                    if pixel_array[a + 1][b] != 0 and pixel_array[a + 1][b] != "visited":
                        q.enqueue((a + 1, b,))
                        pixel_array[a + 1][b] = "visited"
                    if pixel_array[a][b + 1] != 0 and pixel_array[a][b + 1] != "visited":
                        q.enqueue((a, b + 1,))
                        pixel_array[a][b + 1] = "visited"
                    if pixel_array[a][b - 1] != 0 and pixel_array[a][b - 1] != "visited":
                        q.enqueue((a, b - 1,))
                        pixel_array[a][b - 1] = "visited"
                num += 1
    return (pixel, dict1)


def detect_the_region(my_tuple):
    # print()
    # print("Start detect the region from the image...")
    my_dict = {}
    area_list = [0, ]

    for i in my_tuple[1].keys():
        y_cor = [j[0] for j in my_tuple[1][i]]
        x_cor = [j[1] for j in my_tuple[1][i]]

        min_x_cor = min(x_cor)
        max_x_cor = max(x_cor)
        min_y_cor = min(y_cor)
        max_y_cor = max(y_cor)
        my_dict[i] = (min_x_cor, max_x_cor, min_y_cor, max_y_cor,)

        area_list.append(len(my_tuple[1][i]))

    suit_position = []
    for key in my_dict.keys():
        if 5 > (my_dict[key][1] - my_dict[key][0]) / (my_dict[key][3] - my_dict[key][2]) > 2:
            suit_position.append(key)

    temp_area = []
    for key in suit_position:
        temp_area.append(len(my_tuple[1][key]))
    index = area_list.index(max(temp_area))

    # print()
    # print("The region has been detected from the image! ")

    return my_dict[index]


# save ocr image
def save_OCR_image(pixel_array, input_filename):
    # print()
    OCR_output_path = Path("OCR_output_images")
    if not OCR_output_path.exists():
        # create output directory
        OCR_output_path.mkdir(parents=True, exist_ok=True)
        # print(f"Path of OCR_output_images has been created.")

    OCR_output_filename = OCR_output_path / Path(input_filename.replace(".png", "_OCR_output.png"))

    img = np.array(pixel_array)
    im = Image.fromarray(img.astype('uint8'))
    im.save(OCR_output_filename)
    # print(f'The image of {input_filename.replace(".png", "_OCR_output.png")} has been saved to the dir: {OCR_output_filename}.')
    # print()
    return OCR_output_filename


def get_OCR_image(pixel_array, position, input_filename):
    OCR_px = []
    for row in range(len(pixel_array)):
        temp = []
        for col in range(len(pixel_array[row])):
            if row > position[2] and row < position[3]:
                if col > position[0] and col < position[1]:
                    temp.append(pixel_array[row][col])
        if temp != []:
            OCR_px.append(temp)

    return save_OCR_image(OCR_px, input_filename)


def read_img_by_api(filename, input_filename):
    try:
        print("\nStarting read the numbers and letters from the image with API...\n")
        with open(filename, "rb") as myfile:
            files = {"document": myfile}
            headers = {"Authorization": "Token fc1b51c52af7f507ce549b50df41a61e"}
            response = requests.post(url, files=files, headers=headers)
            find_content = re.compile(r'"content": "(.*?)",')
            result = re.findall(find_content, response.text)
            print(f'The letters and numbers in the number plate is: {" ".join(result[:len(result) // 2])}\n')
    except Exception as e:
        print("There is an issue occurred in during the read image by API")



def read_img_by_Easyocr(input_filename):
    print("Starting read the numbers and letters from the image with Easyocr...\n")
    reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

    file = "OCR_output_images/" + input_filename.replace(".png", "_OCR_output.png")
    print(f'The letters and numbers in the number plate is: {" ".join(reader.readtext(file, detail=0))}')


# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():
    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate6.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])


    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')


    # page 4
    px_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_g, image_width, image_height)

    # STUDENT IMPLEMENTATION here
    org_px_array = DeepCopy(px_array)

    # page 5
    px_array = computeStandardDeviationImage5x5(px_array, image_width, image_height)
    # px_array = computeGaussianAveraging3x3RepeatBorder(px_array, image_width, image_height)
    px_array = scaleTo0And255AndQuantize(px_array, image_width, image_height)

    # page 6
    px_array = computeVerticalEdgesSobelAbsolute(px_array, image_width, image_height)
    px_array = computeThresholdGE(px_array, 50, image_width, image_height)
    px_array = do_dilation3x3(px_array, 17, image_width, image_height)
    px_array = do_erosion3x3(px_array, 9, image_width, image_height)

    # page 7
    my_tuple = computeConnectedComponentLabeling(px_array, image_width, image_height)
    position = detect_the_region(my_tuple)


    # get ocr image
    output_OCR_images = get_OCR_image(org_px_array, position, input_filename)
    read_img_by_api(output_OCR_images, input_filename)
    read_img_by_Easyocr(input_filename)


    # compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    # center_x = image_width / 2.0
    # center_y = image_height / 2.0
    # bbox_min_x = center_x + image_width / 4.0
    # bbox_max_x = center_x + image_width / 4.0
    # bbox_min_y = center_y - image_height / 4.0
    # bbox_max_y = center_y + image_height / 4.0
    bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y = position

    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(org_px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)


    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()

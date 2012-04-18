from PIL import Image
import leargist


def grayscale_gist(image_filename):
    im = Image.open(image_filename)
    im = im.convert('L')  # Convert to luminosity only, i.e. grayscale
    descriptors = leargist.color_gist(im)

    return descriptors[:descriptors.size / 3]


def color_gist(image_filename):
    im = Image.open(image_filename)
    return leargist.color_gist(im)

from PIL import Image

def load_image(path):
    """
    Loads an image from a path.
    """
    return Image.open(path)

def thumbnail(image, size=(400, 400)):
    """
    Generates a thumbnail of an image.
    """
    image = image.copy()
    image.thumbnail(size, Image.ANTIALIAS)
    return image
import matplotlib.pyplot as plt
import numpy as np
import cv2

def img_is_color(img):
  if len(img.shape) == 3:
    # Check the color channels to see if they're all the same.
    c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
    if (c1 == c2).all() and (c2 == c3).all():
      return True

  return False

def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10), title_fontsize=20):
  '''
  Shows a grid of images, where each image is a Numpy array. The images can be either
  RGB or grayscale.

  Parameters:
  ----------
  images: list
      List of the images to be displayed.
  list_titles: list or None
      Optional list of titles to be shown for each image.
  list_cmaps: list or None
      Optional list of cmap values for each image. If None, then cmap will be
      automatically inferred.
  grid: boolean
      If True, show a grid over each image
  num_cols: int
      Number of columns to show.
  figsize: tuple of width, height
      Value to be passed to pyplot.figure()
  title_fontsize: int
      Value to be passed to set_title().
  '''

  assert isinstance(list_images, list)
  assert len(list_images) > 0
  assert isinstance(list_images[0], np.ndarray)

  if list_titles is not None:
    assert isinstance(list_titles, list)
    assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

  if list_cmaps is not None:
    assert isinstance(list_cmaps, list)
    assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

  num_images  = len(list_images)
  num_cols    = min(num_images, num_cols)
  num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

  # Create a grid of subplots.
  fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

  # Create list of axes for easy iteration.
  if isinstance(axes, np.ndarray):
    list_axes = list(axes.flat)
  else:
    list_axes = [axes]

  for i in range(num_images):
    img    = list_images[i]
    title  = list_titles[i] if list_titles is not None else 'Image %d' % (i)
    cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')
    list_axes[i].imshow(img, cmap=cmap)
    list_axes[i].set_title(title, fontsize=title_fontsize) 
    list_axes[i].grid(grid)

  for i in range(num_images, len(list_axes)):
    list_axes[i].set_visible(False)

  fig.tight_layout()
  _ = plt.show()

def PM_anisotropic_diffusion(image, iterations, delta, kappa):
  # Convert the input image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # Initialize the diffusion tensor
  D = np.ones(gray.shape, dtype=np.float32)

  for i in range(iterations):
    # Compute the gradient of the grayscale image
    grad = cv2.Sobel(gray, cv2.CV_32F, 1, 1)

    # Compute the magnitude and direction of the gradient
    mag = np.abs(grad)
    direction = np.sign(grad)

    # Compute the diffusion tensor
    Dxx = cv2.Sobel(D, cv2.CV_32F, 2, 0)
    Dyy = cv2.Sobel(D, cv2.CV_32F, 0, 2)
    Dxy = cv2.Sobel(D, cv2.CV_32F, 1, 1)
    D = (1 / (1 + ((mag * kappa) / delta)**2)) * \
        (D + delta**2 * (Dxx*direction[0]**2 + 2*Dxy*direction[0]*direction[1] + Dyy*direction[1]**2))

    # Apply the diffusion tensor to the image
    gray = gray + 0.25 * D * (cv2.Laplacian(gray, cv2.CV_32F))

  # Convert the output image back to color
  output = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)

  return output

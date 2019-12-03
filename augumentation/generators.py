"""
Module docs
"""
import struct
import numpy as np


# Load load_MNST_data
filename = {
    'images': 'augumentation/data/train-images.idx3-ubyte',
    'labels': 'augumentation/data/train-labels.idx1-ubyte'
}
# Images
with open(filename['images'], 'rb') as train_imagesfile:
    info_bytes = train_imagesfile.read(16)
    # TODO: change to tuple comprehension
    (n_imgs, n_rows, n_cols) = (
        struct.unpack('>I', info_bytes[4:8])[0],
        struct.unpack('>I', info_bytes[8:12])[0],
        struct.unpack('>I', info_bytes[12:16])[0]
    )
    pixel_bytes = train_imagesfile.read(n_imgs * n_rows * n_cols)
    images = np.asarray(
        struct.unpack('>' + 'B' * (n_imgs * n_rows * n_cols), pixel_bytes)
    ).reshape((n_imgs, n_rows, n_cols))
# Labelss
with open(filename['labels'], 'rb') as train_labels:
    info_bytes = train_labels.read(8)
    n_imgs_lbls = struct.unpack('>I', info_bytes[4:8])[0]
    labels = np.array(struct.unpack('>' + 'B' * n_imgs, train_labels.read(n_imgs_lbls)))


# @lru_cache(maxsize=10000)
def permutations_w_constraints(n_perm_elements, sum_total, min_value, max_value):
    # base case
    if n_perm_elements == 1:
        if (sum_total <= max_value) & (sum_total >= min_value):
            yield (sum_total,)
    else:
        for value in range(min_value, max_value + 1):
            for permutation in permutations_w_constraints(
                n_perm_elements - 1, sum_total - value, min_value, max_value
            ):
                yield (value,) + permutation


def generate_numbers_sequence(digits, spacing_range, image_width):
    """
    Generate an image that contains the sequence of given numbers, spaced
    randomly using an uniform distribution.

    Parameters
    ----------
    digits:
        A list-like containing the numerical values of the digits from which
        the sequence will be generated (for example [3, 5, 0]).
    spacing_range:
        A (minimum, maximum) pair (tuple), representing the min and max spacing
        between digits. Unit should be pixel.
    image_width:
        specifies the width of the image in pixels.

    Returns
    -------
    The image containing the sequence of numbers. Images should be represented
    as floating point 32bits numpy arrays with a scale ranging from 0 (black) to
    1 (white), the first dimension corresponding to the height and the second
    dimension to the width.
    """
    # digits
    # for each digit: choose a random representation in labels and get its index ---> image representation
    digit_selection_ixs = [np.random.choice(np.where(labels == x)[0], 1)[0] for x in digits]
    candidate_imgs = images[digit_selection_ixs]
    rescaled_candidate_imgs = candidate_imgs / 255
    n_digits = len(digits)
    available_space = (image_width - n_digits * 28)
    # search for spacing options
    spacing_options = list(
        permutations_w_constraints(
            n_digits - 1,
            available_space,
            spacing_range[0],
            spacing_range[1]
        )
    )
    spaces = spacing_options[np.random.choice(len(spacing_options), 1)[0]]

    spaces_matrix = []
    for i in range(len(spaces)):
        spaces_matrix.append(np.zeros(28 * spaces[i], dtype='float32').reshape(28, spaces[i]))

    stacked_images = rescaled_candidate_imgs[0]
    for i in range(1, n_digits):
        stacked_images = np.hstack([stacked_images, spaces_matrix[i - 1], rescaled_candidate_imgs[i]])

    return stacked_images

    # Notes
    # training process needs fixed width
    # ideally space between digits would vary with some distribution
    # if all values are max but sum < tot_sum then raise error.
    # if all values are min but sum > tot_sum then raise error
    # min and max have to be more and less than tot_sum/n

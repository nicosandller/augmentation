"""
Sequence Generators

This module hosts sequence generators with the purpose of generating images
representing sequences (eg: numbers, others), for data augmentation purposes.
"""
import uuid
import struct
import argparse
import numpy as np
from numpy.random import dirichlet
import matplotlib.pyplot as plt


class NumberSequenceGenerator():
    """
    Generator class for digit sequences.

    Parameters
    ----------
    input_filespec:
        An dict arrangement of 2 elements: {'images':[path], 'labels':[path]}
        referencing idx fileformat containing digit images and labels.
    spacing_method:
        A string specifying the type of spacing method selected. Options are:
        ['equidistant', 'random_selection', 'dirichlet'].
        Default is 'dirichlet'
    """
    def __init__(self, input_filespec=None, spacing_method='dirichlet'):
        if input_filespec is None:
            input_filespec = {
                'images': 'augmentation/data/train-images.idx3-ubyte',
                'labels': 'augmentation/data/train-labels.idx1-ubyte'
            }
        data = self._load_idx_data(input_filespec)
        self._images = data[0]
        self._labels = data[1]
        self.n_imgs = data[2]
        self._single_img_height = data[3]
        self._single_img_width = data[4]
        valid_spacing_methods = [
            'equidistant', 'random_selection', 'dirichlet'
        ]
        if spacing_method not in valid_spacing_methods:
            raise Exception(
                (
                    'Error: Invalid <spacing_method>;'
                    ' must be one of the following: {methods}'
                ).format(methods=valid_spacing_methods)
            )
        self.method = spacing_method

    def _load_idx_data(self, filename):
        """
        Loads idx fileformat data to class instance. Idx is described and used
        on the MNIST project.
        """
        try:
            (img_file, lbl_file) = (filename['images'], filename['labels'])
        except (KeyError, TypeError):
            raise Exception(
                'Error: Wrong filename input <input_filespec>.'
                ' It must match what specified in docstrings.'
            )

        with open(img_file, 'rb') as images_binary:
            info_bytes = images_binary.read(16)
            (n_imgs, n_rows, n_cols) = (
                struct.unpack('>I', info_bytes[x:(x + 4)])[0]
                for x
                in range(4, 13, 4)  # 5 through 12 have the information needed.
            )
            pixel_bytes = images_binary.read(n_imgs * n_rows * n_cols)
            images = np.asarray(
                struct.unpack(
                    '>' + 'B' * (n_imgs * n_rows * n_cols),
                    pixel_bytes
                )
            ).reshape(n_imgs, n_rows, n_cols)
        with open(lbl_file, 'rb') as label_binary:
            info_bytes = label_binary.read(8)
            n_imgs_lbls = struct.unpack('>I', info_bytes[4:8])[0]
            if (n_imgs != n_imgs_lbls):
                raise Exception(
                    'Error: Number of images does not match the number of '
                    'labels. \n n_images: {n_imgs} \n n_labels: {n_imgs_lbls}'
                    '\n Check the input file specifications.'
                    .format(n_imgs=n_imgs, n_imgs_lbls=n_imgs_lbls)
                )
            labels = np.array(
                struct.unpack(
                    '>' + 'B' * n_imgs_lbls,
                    label_binary.read(n_imgs_lbls)
                )
            )

        return (images, labels, n_imgs, n_rows, n_cols)

    def _select_image_representations(self, digits):
        """
        Randomnly selects image representations for each digit.
        """
        # if digits is empty list or not list
        if (not digits or not isinstance(digits, list)):
            raise Exception(
                'Wrong digit input. Expected a number sequence. e.g: [1,2,3]'
            )
        if not all((x >= 0 and x < 10) for x in digits):
            raise Exception(
                'Error: Wrong digit input. All elements in '
                'sequence must be within the [0-9] range.'
            )

        # TODO: add exception to handle case where digit not in self._labels
        digit_selection_ixs = [
            np.random.choice(np.where(self._labels == x)[0], 1)[0]
            for x
            in digits
        ]
        candidate_imgs = self._images[digit_selection_ixs]
        rescaled_candidate_imgs = (candidate_imgs / 255)

        return rescaled_candidate_imgs

    def _calculate_available_space(self, spacing_range, image_width, n_digits):
        """
        Calculates space available to distribute amongst the digits in
        compiled image. Checks if th available space can be filled or is
        enough to place the digit images with constrained spacing in between.
        """
        if not isinstance(image_width, int):
            raise Exception(
                'Error: Wrong <image_width> input: expected <int>, got {input}'
                .format(input=type(image_width))
            )
        digit_space_req = n_digits * self._single_img_width
        n_spaces = (n_digits - 1)
        min_space = (digit_space_req + (n_spaces * spacing_range[0]))
        max_space = (digit_space_req + (n_spaces * spacing_range[1]))
        if (
            (image_width < digit_space_req)
            or (image_width < min_space)
            or (image_width > max_space)
        ):
            raise Exception(
                'Error: Input <image_width>: {image_width} is not enough or'
                ' cannot be filled by the specified <spacing_range>: '
                '{spacing_range}, <image_width> must be within the following '
                ' limits: (min:{min_width}, max:{max_width})'
                .format(
                    image_width=image_width,
                    spacing_range=spacing_range,
                    min_width=max(digit_space_req, min_space),
                    max_width=max_space
                )
            )
        available_space = (image_width - digit_space_req)

        return available_space

    def _calculate_digit_spacing(self, n_digits, free_space, spacing_range):
        """
        Calculates spacing between digits based on the selected calculation
        method. Returns a list of matrices of [space x image_height].
        """
        spacing_exception = (
            'Error: Wrong <spacing_range> input: expected <tuple> of size 2 '
            'with each element of <int>, got {input}'.format(
                input=spacing_range
            )
        )
        if not isinstance(spacing_range, tuple):
            raise Exception(spacing_exception)
        elif (
                (len(spacing_range) != 2)
                or not all([isinstance(x, int) for x in spacing_range])
        ):
            raise Exception(spacing_exception)

        n_spaces = (n_digits - 1)
        if n_digits == 1:  # if only 1 digit in sequence: append all free space
            selected_spaces = [free_space]
        elif self.method == 'equidistant':
            equidistant_space = free_space / n_spaces
            if (equidistant_space % 1):
                raise Exception(
                    'Error: There is no integer split for digit spacing with '
                    'the specified <image_width>.'
                )
            selected_spaces = [int(equidistant_space) for x in range(n_spaces)]
        elif self.method == 'random_selection':
            # TODO: add timeout
            spacing_options = list(
                self._permutations_w_constraints(
                    n_spaces, free_space, spacing_range[0], spacing_range[1]
                )
            )
            selected_spaces = spacing_options[
                np.random.choice(len(spacing_options), 1)[0]
            ]
        elif self.method == 'dirichlet':
            alphas = [1 for x in range(n_digits - 1)]

            while True:
                dirichlet_candidates = (
                    dirichlet(alphas, 1).flatten() * free_space
                )
                candidate_for_remainder = np.random.choice(range(n_spaces))

                selected_spaces = [
                    int(np.floor(x)) for x in dirichlet_candidates
                ]
                remainder = round(sum([x % 1 for x in dirichlet_candidates]))
                selected_spaces[candidate_for_remainder] += int(remainder)

                if all(
                    [
                        (x >= spacing_range[0] and x <= spacing_range[1])
                        for x in selected_spaces
                    ]
                ):
                    break

        spacing = []
        for i in range(len(selected_spaces)):
            spacing.append(
                np.zeros(
                    (self._single_img_height * selected_spaces[i]),
                    dtype='float32'
                )
                .reshape(self._single_img_height, selected_spaces[i])
            )

        return spacing

    def _permutations_w_constraints(self, n_elements,
                                    sum_total, min_value, max_value):
        """
        Calculates all permutations for a set of <n_elements> within the range
        <min_value,max_value> that add up to <sum_total>. Returns generator.
        """
        if n_elements == 1:  # base case
            if (sum_total <= max_value) & (sum_total >= min_value):
                yield (sum_total,)
        else:
            for value in range(min_value, max_value + 1):
                for permutation in self._permutations_w_constraints(
                    n_elements - 1, sum_total - value, min_value, max_value
                ):
                    yield (value,) + permutation

    def generate_numbers_sequence(self, digits, spacing_range, image_width):
        """
        Generate an image that contains the sequence of given numbers, spaced
        randomly using an uniform distribution.

        Parameters
        ----------
        digits:
            A list-like containing the numerical values of the digits from
            which the sequence will be generated (for example [3, 5, 0]).
        spacing_range:
            A (minimum, maximum) pair (tuple), representing the min and max
            spacing between digits. Unit should be pixel.
        image_width:
            specifies the width of the image in pixels.

        Returns
        -------
        The image containing the sequence of numbers. Images should be
        represented as floating point 32bits numpy arrays with a scale ranging
        from 0 (black) to 1 (white), the first dimension corresponding to the
        height and the second dimension to the width.
        """
        n_digits = len(digits)

        image_representations = self._select_image_representations(digits)

        available_space = self._calculate_available_space(
            spacing_range, image_width, n_digits
        )

        digit_spacing = self._calculate_digit_spacing(
            n_digits, available_space, spacing_range
        )

        stacked_images = image_representations[0]
        for i in range(1, n_digits):
            stacked_images = np.hstack(
                [
                    stacked_images, digit_spacing[i - 1],
                    image_representations[i]
                ]
            )

        return stacked_images.astype('float32')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'digits',
        help='number sequence to generate range:[0-9]. eg: 1,4,7', type=str,
    )
    parser.add_argument(
        'min_spacing', help='minimum spacing range. -int', type=int,
    )
    parser.add_argument(
        'max_spacing', help='maximum spacing range. -int', type=int,
    )
    parser.add_argument(
        'image_width', type=int, help='total image width -int',
    )
    parser.add_argument(
        '-m', type=str,
        help=(
            'spacing calculation method. Options:'
            '["equidistant", "random_selection"]'
        ),
        required=False, metavar='spacing_method', default='equidistant'

    )
    parser.add_argument(
        '-n', type=int,
        help=('number of sequence images to generate. -int'),
        required=False, metavar='n_sequence_images', default=1
    )
    args = parser.parse_args()
    digits = [int(item)for item in args.digits.split(',')]
    spacing_method = args.m
    n_sequence_images = args.n

    sg = NumberSequenceGenerator(spacing_method=spacing_method)
    for i in range(n_sequence_images):
        filename = (str(uuid.uuid4()) + '.png')
        stacked_images = sg.generate_numbers_sequence(
                digits,
                (args.min_spacing, args.max_spacing),
                args.image_width
        )
        plt.imsave(filename, stacked_images, cmap='Greys')

    print(
        'Successfully created {n_sequence_images} digit sequence and saved on '
        'current directory'.format(n_sequence_images=n_sequence_images)
    )

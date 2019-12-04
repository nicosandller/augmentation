"""
Sequence Generators
[In progress]
"""
import sys
import struct
import numpy as np
import matplotlib.pyplot as plt


class NumberSequenceGenerator():
    def __init__(self, input_data_filespec=None):
        if input_data_filespec is None:
            input_data_filespec = {
                'images': 'augumentation/data/train-images.idx3-ubyte',
                'labels': 'augumentation/data/train-labels.idx1-ubyte'
            }
        data = self._load_idx_data(input_data_filespec)
        self.images = data[0]
        self.labels = data[1]
        self.n_imgs = data[2]
        self.single_img_heigh = data[3]
        self.single_img_width = data[4]

    def _load_idx_data(self, filename):
        """
        [In progress]
        """
        try:
            (img_file, lbl_file) = (filename['images'], filename['labels'])
        except (KeyError, TypeError):
            raise Exception("Wrong filename input [input_data_filespec]. It must match what specified in docstrings.")

        with open(img_file, 'rb') as images_binary:
            info_bytes = images_binary.read(16)
            (n_imgs, n_rows, n_cols) = (struct.unpack('>I', info_bytes[x:x + 4])[0] for x in range(4, 13, 4))
            pixel_bytes = images_binary.read(n_imgs * n_rows * n_cols)
            images = np.asarray(
                struct.unpack('>' + 'B' * (n_imgs * n_rows * n_cols), pixel_bytes)
            ).reshape((n_imgs, n_rows, n_cols))
        with open(lbl_file, 'rb') as label_binary:
            info_bytes = label_binary.read(8)
            n_imgs_lbls = struct.unpack('>I', info_bytes[4:8])[0]
            if n_imgs != n_imgs_lbls:
                raise Exception(
                    """
                        Number of images does not match the number of labels in file.
                        n_images: {n_imgs}
                        n_labels: {n_imgs_lbls}
                        Check the input file specifications.
                    """.format(n_imgs=n_imgs, n_imgs_lbls=n_imgs_lbls)
                )
            labels = np.array(struct.unpack('>' + 'B' * n_imgs_lbls, label_binary.read(n_imgs_lbls)))

        return (images, labels, n_imgs, n_rows, n_cols)

    def _select_image_representations(self, digits):
        """
        [In progress]
        """
        if (not digits or not isinstance(digits, list)):
            raise Exception("Wrong digit input. Expected a number sequence. e.g: [1,2,3]")
        if not all((x >= 0 and x < 10) for x in digits):
            raise Exception("Wrong digit input. All elements in sequence must be within the [0-9] range.")

        digit_selection_ixs = [np.random.choice(np.where(self.labels == x)[0], 1)[0] for x in digits]
        candidate_imgs = self.images[digit_selection_ixs]
        rescaled_candidate_imgs = candidate_imgs / 255

        return rescaled_candidate_imgs

    def _calculate_available_space(self, spacing_range, image_width, n_digits):
        """
        [In progress]
        Calculates space available to distribute amongst the digits in compiled image.
        Checks available space can be filled or is enough to place the digit images with
        constrained spacing in between.
        """
        if not isinstance(image_width, int):
            raise Exception(
                "Wrong <image_width> input: expected <int>, got {input}".format(input=type(image_width))
            )
        digit_space_req = n_digits * self.single_img_width
        # Note: spaces between digits is n_digits - 1
        min_total_space = (digit_space_req + ((n_digits - 1) * spacing_range[0]))
        max_total_space = (digit_space_req + ((n_digits - 1) * spacing_range[1]))
        if (image_width < digit_space_req) or (image_width < min_total_space) or (image_width > max_total_space):
            raise Exception(
                """
                Input <image_width>: {image_width} is not enough or cannot be filled by the specified
                <spacing_range>: {spacing_range}, <image_width> must be within the following limits:
                (min:{min_width}, max:{max_width})
                """.format(
                    image_width=image_width,
                    spacing_range=spacing_range,
                    min_width=max(digit_space_req, min_total_space),
                    max_width=max_total_space
                )
            )
        available_space = (image_width - digit_space_req)

        return available_space

    def _calculate_digit_spacing(self, n_digits, available_space, spacing_range):
        """
        [In progress]
        Calculates spacing between digits.
        """
        try:
            if (not isinstance(spacing_range, tuple)) or (len(spacing_range) != 2):
                raise Exception()
        except (TypeError, Exception):
            raise Exception(
                "Wrong <spacing_range> input: expected <tuple> of size 2, got {input}".format(input=spacing_range)
            )
        spacing_options = list(
            self.permutations_w_constraints(n_digits - 1, available_space, spacing_range[0], spacing_range[1])
        )
        selected_spaces = spacing_options[np.random.choice(len(spacing_options), 1)[0]]
        spacing = []
        for i in range(len(selected_spaces)):
            spacing.append(
                np.zeros(self.single_img_width * selected_spaces[i], dtype='float32')
                .reshape(self.single_img_width, selected_spaces[i])
            )

        return spacing

    # @lru_cache(maxsize=10000)
    def permutations_w_constraints(self, n_elements, sum_total, min_value, max_value):
        """
        [In progress]
        Calculates all permutations for a set of <n_elements> within the range <min_value,max_value>
        that add up to <sum_total>.
        """
        if n_elements == 1:  # base case
            if (sum_total <= max_value) & (sum_total >= min_value):
                yield (sum_total,)
        else:
            for value in range(min_value, max_value + 1):
                for permutation in self.permutations_w_constraints(
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
        n_digits = len(digits)

        image_representations = self._select_image_representations(digits)

        available_space = self._calculate_available_space(spacing_range, image_width, n_digits)

        digit_spacing = self._calculate_digit_spacing(n_digits, available_space, spacing_range)

        stacked_images = image_representations[0]
        for i in range(1, n_digits):
            stacked_images = np.hstack([stacked_images, digit_spacing[i - 1], image_representations[i]])

        return stacked_images.astype('float32')


if __name__ == "__main__":
    # TODO: add try statement and exception
    sequence_to_gen = sys.argv[1]
    min_spacing = sys.argv[2]
    max_spacing = sys.argv[3]
    image_width = sys.argv[4]

    stacked_images = NumberSequenceGenerator().generate_numbers_sequence(
        sequence_to_gen,
        (min_spacing, max_spacing),
        image_width
    )
    plt.imsave('stacked_images.png', stacked_images, cmap='Greys')
    print("Successfully created a digit sequence and saved as 'stacked_images.png'.")

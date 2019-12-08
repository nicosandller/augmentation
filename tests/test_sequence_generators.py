import pickle
import unittest
import numpy as np
from scipy.stats import kstest

from augmentation.sequence_generators import NumberSequenceGenerator


class TestNumberSequenceGeneration(unittest.TestCase):
    def setUp(self):
        self.seed = 0  # fixed randomness
        self.MNIST_filepath = {
            'images': 'tests/test_data/test-images.idx3-ubyte_A',
            'labels': 'tests/test_data/test-labels.idx3-ubyte_A'
        }
        self.number_sequence = [3, 7, 8, 6]
        self.spacing_range = (1, 4)
        self.image_width = 118
        self.nsg_eq = NumberSequenceGenerator(
            self.MNIST_filepath, 'equidistant'
        )
        self.nsg_rs = NumberSequenceGenerator(
            self.MNIST_filepath, 'random_selection'
        )
        self.equidistant_number_sequence_output = (
            self.nsg_rs.generate_numbers_sequence(
                self.number_sequence,
                self.spacing_range,
                self.image_width
            )
        )

    def test_invalid_generator_method(self):
        unsupported_method = 'dirilecht'
        with self.assertRaisesRegex(Exception, "Invalid <spacing_method>"):
            NumberSequenceGenerator(self.MNIST_filepath, unsupported_method)

    def test_load_idx_data_matrix_size(self):
        expected = (100, 28, 28)
        actual = self.nsg_eq._load_idx_data(self.MNIST_filepath)[2:5]
        self.assertTupleEqual(expected, actual)

    def test_load_idx_data_wrong_input(self):
        bad_filename = 245
        with self.assertRaisesRegex(Exception, "filename input"):
            self.nsg_eq._load_idx_data(bad_filename)

    def test_load_idx_data_input_mismatch(self):
        bad_filename = {
            'images': 'tests/test_data/test-images.idx3-ubyte_A',
            'labels': 'tests/test_data/test-labels.idx3-ubyte_B'
        }
        with self.assertRaisesRegex(Exception, "n_labels: 20"):
            self.nsg_eq._load_idx_data(bad_filename)

    def test_digits_empty_list(self):
        with self.assertRaisesRegex(Exception, "Expected a number sequence"):
            self.nsg_eq._select_image_representations([])

    def test_digits_not_list(self):
        with self.assertRaisesRegex(Exception, "Expected a number sequence"):
            self.nsg_eq._select_image_representations("some string")

    def test_digits_outside_range(self):
        with self.assertRaisesRegex(Exception, "must be within"):
            self.nsg_eq._select_image_representations([1, 10])

    def test_select_image_representation(self):
        np.random.seed(self.seed)
        expected = np.load('tests/test_data/raw_image_representations.npy')
        actual = self.nsg_eq._select_image_representations(
            self.number_sequence
        )
        np.testing.assert_array_equal(expected, actual)

    def test_width_not_number(self):
        wrong_input = (34, 506)
        n_digits = len(self.number_sequence)
        with self.assertRaisesRegex(Exception, "expected <int>"):
            self.nsg_eq._calculate_available_space(
                self.spacing_range, wrong_input, n_digits
            )

    def test_width_below_min_bound(self):
        image_width = 112
        n_digits = len(self.number_sequence)
        with self.assertRaisesRegex(Exception, r"min:115, max:124"):
            self.nsg_eq._calculate_available_space(
                self.spacing_range, image_width, n_digits
            )

    def test_width_above_max_bound(self):
        image_width = 130
        n_digits = len(self.number_sequence)
        with self.assertRaisesRegex(Exception, r"min:115, max:124"):
            self.nsg_eq._calculate_available_space(
                self.spacing_range, image_width, n_digits
            )

    def test_width(self):
        image_width = 120
        n_digits = len(self.number_sequence)
        expected = 8
        actual = self.nsg_eq._calculate_available_space(
            self.spacing_range, image_width, n_digits
        )
        self.assertEqual(expected, actual)

    def test_spacing_range_not_tuple(self):
        wrong_input = [23, 42]
        n_digits = len(self.number_sequence)
        with self.assertRaisesRegex(Exception, "expected <tuple>"):
            self.nsg_eq._calculate_digit_spacing(n_digits, 8, wrong_input)

    def test_spacing_range_bad_tuple(self):
        wrong_input = (2, 5, 2)
        n_digits = len(self.number_sequence)
        with self.assertRaisesRegex(Exception, "expected <tuple> of size 2"):
            self.nsg_eq._calculate_digit_spacing(n_digits, 8, wrong_input)

    def test_spacing_range_tuple_elements_not_int(self):
        wrong_input = (2, 10.5)
        n_digits = len(self.number_sequence)
        with self.assertRaisesRegex(Exception, "expected <tuple> of size 2"):
            self.nsg_eq._calculate_digit_spacing(n_digits, 8, wrong_input)

    def test_one_digit(self):
        n_digits = 1  # if only one digit, assigns all free space after digit
        expected = 3  # width of the space to be attatched after digit
        actual = self.nsg_eq._calculate_digit_spacing(
            n_digits, 3, self.spacing_range
        )[0].shape[1]
        self.assertEqual(expected, actual)

    def test_digit_spacing_random_selection(self):
        np.random.seed(self.seed)
        n_digits = len(self.number_sequence)
        with open(
            'tests/test_data/test_digit_spacing_random_selection.npy', 'rb'
        ) as file:
            expected = pickle.load(file)
        actual = self.nsg_rs._calculate_digit_spacing(
            n_digits, 8, self.spacing_range
        )
        for i in range(len(expected)):
            np.testing.assert_array_equal(expected[i], actual[i])

    def test_digit_spacing_equidistant_selection(self):
        n_digits = len(self.number_sequence)
        with open(
            'tests/test_data/test_digit_spacing_equidistant.npy', 'rb'
        ) as file:
            expected = pickle.load(file)
        actual = self.nsg_eq._calculate_digit_spacing(
            n_digits, 6, self.spacing_range
        )
        for i in range(len(expected)):
            np.testing.assert_array_equal(expected[i], actual[i])

    @unittest.skip("Not implemented: permutation calc timeout")
    def test_digit_spacing_random_selection_search_timeout(self):
        n_digits = 3
        spacing_range = (1, 4)
        available_space = 5
        with self.assertRaisesRegex(Exception, "timeout"):
            self.nsg_rs._calculate_digit_spacing(
                n_digits, available_space, spacing_range
            )

    @unittest.skip("Not implemented: uniformity test")
    def test_digit_spacing_uniformity(self):
        p_val_thresh = 0.2
        np.random.seed(self.seed)
        n_digits = 3
        spacing_range = (1, 4)
        available_space = 5
        spacing_samples = []
        for i in range(3000):
            for space in self.nsg_rs._calculate_digit_spacing(
                    n_digits, available_space, spacing_range):
                spacing_samples.append(space.shape[1])

        normed_samples = [x / max(spacing_samples) for x in spacing_samples]
        self.assertTrue(kstest(normed_samples, 'uniform') > p_val_thresh)

    def test_digit_spacing_equidistant_selection_no_integer_split(self):
        n_digits = 3
        spacing_range = (1, 4)
        available_space = 5
        with self.assertRaisesRegex(Exception, "no integer split"):
            self.nsg_eq._calculate_digit_spacing(
                n_digits, available_space, spacing_range
            )

    def test_image_generation_rs(self):
        np.random.seed(self.seed)
        expected = np.load(
            'tests/test_data/stacked_digits_random_selection.npy'
        )
        actual = self.nsg_rs.generate_numbers_sequence(
            self.number_sequence,
            self.spacing_range,
            self.image_width
        )
        np.testing.assert_array_equal(expected, actual)

    def test_image_generation_eq(self):
        np.random.seed(self.seed)
        expected = np.load('tests/test_data/stacked_digits_equidistant.npy')
        actual = self.nsg_eq.generate_numbers_sequence(
            self.number_sequence,
            self.spacing_range,
            self.image_width
        )
        np.testing.assert_array_equal(expected, actual)

    def test_pixels_within_range(self):
        number_sequence_output = self.equidistant_number_sequence_output
        outside_range_check_list = [
            True if (x > 1) or (x < 0) else False
            for x in number_sequence_output.flat
        ]
        # true if any pixel is outside range.
        self.assertFalse(any(outside_range_check_list))

    def test_image_height(self):
        expected = 28
        actual = self.equidistant_number_sequence_output.shape[0]
        self.assertEqual(expected, actual)

    def test_pixel_dtype(self):
        expected = np.float32().dtype
        actual = self.equidistant_number_sequence_output[0][0].dtype
        self.assertIs(actual, expected)


if __name__ == '__main__':
    unittest.main()

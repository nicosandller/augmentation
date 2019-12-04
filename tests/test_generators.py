import pickle
import unittest
import numpy as np

from augumentation.generators import NumberSequenceGenerator


class TestNumberSequenceGeneration(unittest.TestCase):
    def setUp(self):
        self.seed = 0  # fixed randomness
        self.MNIST_filepath = {
            'images': 'tests/test_data/test-images.idx3-ubyte_A',
            'labels': 'tests/test_data/test-labels.idx3-ubyte_A'
        }
        self.working_nsg = NumberSequenceGenerator(self.MNIST_filepath)
        self.number_sequence = [3, 7, 8, 6]
        self.spacing_range = (1, 4)
        self.image_width = 118
        self.testing_raw_image_representations = np.load('tests/test_data/raw_image_representations.npy')
        self.testing_stacked_digit_sequence = np.load('tests/test_data/stacked_digits.npy')
        self.number_sequence_output = self.working_nsg.generate_numbers_sequence(
            self.number_sequence,
            self.spacing_range,
            self.image_width
        )

    def test_image_generation(self):
        np.random.seed(self.seed)
        expected = self.testing_stacked_digit_sequence
        actual = self.working_nsg.generate_numbers_sequence(
            self.number_sequence,
            self.spacing_range,
            self.image_width
        )

        np.testing.assert_array_equal(expected, actual)

    def test_load_idx_data_matrix_size(self):
        expected = (100, 28, 28)
        actual = self.working_nsg._load_idx_data(self.MNIST_filepath)[2:5]
        self.assertTupleEqual(expected, actual)

    def test_load_idx_data_wrong_input(self):
        bad_filename = 245
        with self.assertRaisesRegex(Exception, "filename input"):
            self.working_nsg._load_idx_data(bad_filename)

    def test_load_idx_data_input_mismatch(self):
        bad_filename = {
            'images': 'tests/test_data/test-images.idx3-ubyte_A',
            'labels': 'tests/test_data/test-labels.idx3-ubyte_B'
        }
        with self.assertRaisesRegex(Exception, "n_labels: 20"):
            self.working_nsg._load_idx_data(bad_filename)

    def test_digits_empty_list(self):
        with self.assertRaisesRegex(Exception, "Expected a number sequence"):
            self.working_nsg._select_image_representations([])

    def test_digits_not_list(self):
        with self.assertRaisesRegex(Exception, "Expected a number sequence"):
            self.working_nsg._select_image_representations("some string")

    def test_digits_outside_range(self):
        with self.assertRaisesRegex(Exception, "must be within"):
            self.working_nsg._select_image_representations([1, 10])

    def test_number_sequence(self):
        np.random.seed(self.seed)
        expected = self.testing_raw_image_representations
        actual = self.working_nsg._select_image_representations(self.number_sequence)
        np.testing.assert_array_equal(expected, actual)

    def test_width_not_number(self):
        wrong_input = (34, 506)
        with self.assertRaisesRegex(Exception, "expected <int>"):
            self.working_nsg.generate_numbers_sequence(self.number_sequence, self.spacing_range, wrong_input)

    def test_width_below_min_bound(self):
        with self.assertRaisesRegex(Exception, r"min:115, max:124"):
            self.working_nsg._calculate_available_space(
                self.spacing_range,
                112,
                len(self.number_sequence)
            )

    def test_width_above_max_bound(self):
        with self.assertRaisesRegex(Exception, r"min:115, max:124"):
            self.working_nsg._calculate_available_space(
                self.spacing_range,
                130,
                len(self.number_sequence)
            )

    def test_width(self):
        expected = 8
        actual = self.working_nsg._calculate_available_space(
            self.spacing_range,
            120,
            len(self.number_sequence)
        )
        self.assertEqual(expected, actual)

    def test_spacing_range_not_tuple(self):
        wrong_input = [23, 42]
        with self.assertRaisesRegex(Exception, "expected <tuple>"):
            self.working_nsg._calculate_digit_spacing(len(self.number_sequence), 8, wrong_input)

    def test_spacing_range_bad_tuple(self):
        wrong_input = (2, 5, 2)
        with self.assertRaisesRegex(Exception, "expected <tuple> of size 2"):
            self.working_nsg._calculate_digit_spacing(len(self.number_sequence), 8, wrong_input)

    def test_spacing_range(self):
        np.random.seed(self.seed)
        with open('tests/test_data/test_digit_spacing.npy', 'rb') as file:
            expected = pickle.load(file)
        actual = self.working_nsg._calculate_digit_spacing(len(self.number_sequence), 8, self.spacing_range)
        for i in range(len(expected)):
            np.testing.assert_array_equal(expected[i], actual[i])

    # TODO: add timeout tests for digit spacing range generator function

    def test_pixels_within_range(self):
        number_sequence_output = self.number_sequence_output
        outside_range_check_list = [True if (x > 1) or (x < 0) else False for x in number_sequence_output.flat]
        self.assertFalse(any(outside_range_check_list))  # true if any pixel is outside range.

    def test_image_height(self):
        expected = 28
        actual = self.number_sequence_output.shape[0]
        self.assertEqual(expected, actual)

    def test_pixel_dtype(self):
        expected = np.float32().dtype
        actual = self.number_sequence_output[0][0].dtype
        self.assertIs(actual, expected)

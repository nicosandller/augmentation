import unittest
import numpy as np

from augumentation import generators


class TestNumberSequenceGeneration(unittest.TestCase):
    def setUp(self):
        # Set seed for randomness
        np.random.seed(0)
        self.number_sequence = [3, 7, 8, 6]
        self.spacing_range = (1, 4)
        self.image_width = 118
        self.generated_stacked_digit_sequence = generators.generate_numbers_sequence(
            self.number_sequence,
            self.spacing_range,
            self.image_width
        )
        self.testing_stacked_digit_sequence = np.load('tests/test_data/stacked_digits.npy')

    def test_image_generation(self):
        expected = self.testing_stacked_digit_sequence
        actual = self.generated_stacked_digit_sequence
        np.testing.assert_array_equal(expected, actual)

    def test_image_generation_timeout_error(self):
        with self.assertRaisesRegexp(Exception, "empty list"):
            generators.generate_numbers_sequence(
                self.number_sequence,
                self.spacing_range,
                self.image_width
            )

    def test_number_sequence_empty_list(self):
        # Empty list is provided as numbers to generate. test right assertion is thrown
        # "Expected number sequence, received []"
        with self.assertRaisesRegexp(Exception, "empty list"):
            generators.generate_numbers_sequence([], self.spacing_range, self.image_width)

    def test_number_sequence_not_list(self):
        # String as numbers to generate. test right assertion is thrown
        with self.assertRaisesRegexp(Exception, "got <some string>"):
            generators.generate_numbers_sequence("some string", self.spacing_range, self.image_width)

    def test_number_sequence_outside_range(self):
        # one of the generator numbers is outside the [0-9] range
        with self.assertRaisesRegexp(Exception, r"generator number outside range \[0-9\]"):
            generators.generate_numbers_sequence([1, 77], self.spacing_range, self.image_width)

    @unittest.skip("not implemented")
    def test_number_sequence(self):
        expected = self.number_sequence
        # breakout function to test
        # actual = somefuction(self.number_sequence)
        self.asserEqual(expected, actual)

    def test_width_not_number(self):
        # wrong dtype provided
        with self.assertRaisesRegexp(Exception, r"got <(34,506)>"):
            generators.generate_numbers_sequence(self.number_sequence, self.spacing_range, (34, 506))

    def test_width_outside_min_bound(self):
        # min required width = (n * 28) + (min_range * (n-1))
        # here ==> (10 * 28) + (1*9) = 289
        with self.assertRaisesRegexp(Exception, r"width expected in ranges: \[289-334\]"):
            generators.generate_numbers_sequence(
                self.number_sequence,  # n = 4
                self.spacing_range,  #
                230
            )

    def test_width_outside_max_bound(self):
        with self.assertRaisesRegexp(Exception, r"width expected in ranges: \[289-334\]"):
            generators.generate_numbers_sequence(
                self.number_sequence,  # n = 4
                self.spacing_range,  #
                500
            )

    @unittest.skip("not implemented")
    def test_image_width(self):
        # distance of resulting image is same as prescrived
        expected = 300
        sequence_image = generators.generate_numbers_sequence(
            self.number_sequence,  # n = 4
            self.spacing_range,  #
            expected
        )
        actual = sequence_image.shape[0]
        self.asserEqual(expected, actual)

    def test_spacing_range_not_tuple(self):
        # test exception
        with self.assertRaisesRegexp(Exception, r"got <\{1\}>"):
            generators.generate_numbers_sequence(
                self.number_sequence,
                {1},
                self.image_width
            )

    def test_spacing_range_empty_tuple(self):
        # test exception
        with self.assertRaisesRegexp(Exception, "empty tuple"):
            generators.generate_numbers_sequence(
                self.number_sequence,
                (),
                self.image_width
            )

    # TODO: expand this to test space_search more extensively
    @unittest.skip("not implemented")
    def test_spacing_range(self):
        pass

    def test_pixel_range(self):
        # TODO: re-write to have a more descriptive fail message.
        # each pixel is within 0 an 1 range (scaled down from 0 to 255).
        sequence_image = self.test_sequence_image
        outside_range_check_list = [True if (x > 1) or (x < 0) else False for x in sequence_image.flat]
        # true if any pixel is outside range.
        any_outside_range = any(outside_range_check_list)
        self.assertFalse(any_outside_range)

    def test_image_height(self):
        # MNST height is 28 pixels
        expected = 28
        sequence_image = self.test_sequence_image
        actual = sequence_image.shape[0]
        self.assertEqual(expected, actual)

    def test_pixel_dtype(self):
        expected = np.float32().dtype
        actual = self.test_sequence_image[10][50].dtype
        self.assertEqual(expected, actual)

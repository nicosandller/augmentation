# Augmentation

Image recognition models sometimes need large quantities of training data to learn all the nuances in the process of image creation.

This package serves as a container for tools that implement data augmentation techniques and help improve model training processes.

### Installation

Developed and tested with Python 3.7.5

Clone package and then from a shell on the package root directory run:
  - with pipenv:
  ```shell
  pipenv install
  ```
  - pip:
  ```shell
  pip install -r requirements.txt
  ```

## Sequence Generators
`sequence_generators.py`

This module focuses on generating images of number sequences based
of MNIST digit database. It does so by:
- parsing MNIST idx (or any other idx fileformat with similar specs) into a readable format.
- selecting digits that match the desired sequence.
- finding suitable spacing between them.
- assembling them to a specified total image width.

The main challenges presented when building composite images is the spacing between the elements of the composite image needs to be arranged in some form to comply with the expected pixel size of the underlying statistical model for which it is being used as training data.

To tackle this challenge this implementation builds 2 spacing calculation methods:
- `equidistant`:

Equally splits total required image width amongst the spaces to fill (n_digits - 1). Because pixels are non divisible, this method only works if the equal split is an integer. **This method can be tedious and might create less natural distribution of spaces for the model to train on.**

- `random_selection`:

Randomly selects a combination of spaces between digits that together with the individual digit width adds up to the required image width. **This method creates a more natural distribution of space amongst digits, but can take longer than needed to calculate all space combinations possible and select one.**

### Usage
_Via package import_

On a python interpreter at the package's root directory:
```python
from augmentation.sequence_generators import NumberSequenceGenerator


digits = [1,3,7,6,2]
spacing_range = (3, 7)
image_width = 150
nsg = NumberSequenceGenerator(spacing_method='random_selection')
mnist_digit_sequence = nsg.generate_numbers_sequence(digits, spacing_range, image_width)
```
```python
>>> mnist_digit_sequence
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)
```

_Via CLI_

From the package root directory, on a shell:
```shell
> python augmentation/sequence_generators.py -h
```
```shell
>>> usage: sequence_generators.py [-h] [-m spacing_method] [-n n_sequence_images]
                            digits min_spacing max_spacing image_width

positional arguments:
digits                number sequence to generate range:[0-9]. eg: 1,4,7
min_spacing           minimum spacing range. -int
max_spacing           maximum spacing range. -int
image_width           total image width -int

optional arguments:
-h, --help            show this help message and exit
-m spacing_method     spacing calculation method. Options:["equidistant",
                      "random_selection"]
-n n_sequence_images  number of sequence images to generate. -int
```

 ```shell
> python augmentation/sequence_generators.py 1,2,5 1 9 90 -m "random_selection" -n 5
 ```
 ```shell
>>> Successfully created 5 digit sequence and saved on current directory
 ```

_Running tests_

On a shell interpreter on the package root directory run:
```shell
> python -m unittest discover -v
```
```shell
... test_width_above_max_bound (tests.test_sequence_generators.TestNumberSequenceGeneration) ... ok
test_width_below_min_bound (tests.test_sequence_generators.TestNumberSequenceGeneration) ... ok
test_width_not_number (tests.test_sequence_generators.TestNumberSequenceGeneration) ... ok

----------------------------------------------------------------------
Ran 25 tests in 0.426s
```

### Further improvements
- Add timeout to random_selection spacing method with a suggestion to reduce the min/max spacing range and decrease the computational complexity.
- Parse image bytes for selected digits.
- Create Exceptions suite with custom exceptions.
- Add logging and log creation logic to the module.

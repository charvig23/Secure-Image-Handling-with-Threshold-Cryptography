# Secure Image Handling with Threshold Cryptography


## Features

- Secure encryption and decryption of images using the Shamir scheme.
- Supports both grayscale and color images.
- Zero loss of pixel data (lossless recovery), including edge cases with pixel values of 256.
- Efficient metadata storage for necessary reconstruction information.
- Detailed progress indicators and performance metrics display.

## Installation

To use this secret sharing algorithm, you need to install Python and several dependencies on your system.

- Python 3.x
- NumPy
- Pillow
- PyCryptodome
- pypng

You can install all required packages with the following command:

```shell
pip install -r requirements.txt
```

## How to Use

To encrypt (share) an image:

```shell
python finalShamir.py -e <image-path> -n <number-of-shares> -r <threshold>
```

To decrypt (reconstruct) an image using shares:

```shell
python finalShamir.py -d <output-path> -r <threshold> -i <share-indexes>
```

To compare the original and reconstructed images:

```shell
python finalShamir.py -c <original-image-path> <reconstructed-image-path>
```

For a full list of options, use the `-h` flag:

```shell
python finalShamir.py -h
```

## Example

Here is an example of how to encode and decode an image:

```shell
# Encode the image into 5 shares with a threshold of 3
python finalShamir.py -e real.jpg -n 5 -r 3

# Decode the image using shares 1, 4, and 5
python finalShamir.py -d decoded.png -r 3 -i 1 4 5

# Compare the original and recovered images
python finalShamir.py -c real.jpg decoded.png
```


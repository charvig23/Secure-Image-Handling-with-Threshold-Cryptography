import time
import numpy as np
import argparse
import png
import sys
import os
import math
from PIL import Image
from Crypto.Util.number import *

# Preprocessing to flatten the image and store its shape
def preprocessing(path):
    img = Image.open(path)
    data = np.asarray(img)
    return data.flatten(), data.shape

# Insert custom text into the PNG chunk (for watermark)
def insert_text_chunk(src_png, dst_png, text):
    reader = png.Reader(filename=src_png)
    chunks = reader.chunks()  
    chunk_list = list(chunks)
    if isinstance(text, str):
        text = text.encode('utf-8')  
    chunk_item = tuple([b'tEXt', text]) 

    index = 1
    chunk_list.insert(index, chunk_item)

    with open(dst_png, 'wb') as dst_file:
        png.write_chunks(dst_file, chunk_list)

# Read custom text chunk from PNG
def read_text_chunk(src_png, index=1):
    try:
        reader = png.Reader(filename=src_png)
        chunks = reader.chunks()
        chunk_list = list(chunks)
        if len(chunk_list) > index:
            img_extra = chunk_list[index][1]
            try:
                img_extra = img_extra.decode('utf-8')  
            except UnicodeDecodeError:
                print(f"Warning: Could not decode watermark at index {index}. Returning raw byte data.")
                return img_extra  
            print(f"Read watermark content: {img_extra}")
            return img_extra
        else:
            print(f"Error: No chunk found at index {index}")
            return None
    except Exception as e:
        print(f"Error reading text chunk: {e}")
        return None
    

# Polynomial generation for secret sharing
def polynomial(img, n, r):
    num_pixels = img.shape[0]
    # Generate polynomial coefficients
    coefficients = np.random.randint(low=0, high=257, size=(num_pixels, r - 1))
    secret_imgs = []
    imgs_extra = []
    for i in range(1, n + 1):
        # Construct the (r-1) degree polynomial
        base = np.array([i ** j for j in range(1, r)])
        base = np.matmul(coefficients, base)

        secret_img = (img + base) % 257

        indices = np.where(secret_img == 256)[0]
        img_extra = indices.tolist()
        secret_img[indices] = 0

        secret_imgs.append(secret_img)
        imgs_extra.append(img_extra)
    return np.array(secret_imgs), imgs_extra

# Format the byte size for output
def format_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_names = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

# Get the file size for output
def get_file_size(file_path):
    try:
        size = os.path.getsize(file_path)
        return format_size(size)
    except OSError as e:
        return f"Error: {e}"

# Lagrange interpolation for reconstruction
def lagrange(x, y, num_points, x_test):
    l = np.zeros(shape=(num_points,))
    for k in range(num_points):
        l[k] = 1
        for k_ in range(num_points):
            if k != k_:
                d = int(x[k] - x[k_])
                inv_d = inverse(d, 257)
                l[k] = l[k] * (x_test - x[k_]) * inv_d % 257
    L = 0
    for i in range(num_points):
        L += y[i] * l[i]
    return L

# Decoding with watermark authentication
def decode_with_authentication(imgs, imgs_extra, index, r, expected_watermark):
    """
    Decodes the shares and verifies the watermark to authenticate the shares.
    """
    # Flag to check if all shares are authentic
    all_authentic = True

    # Extract watermark from each share to verify authenticity
    for i in range(r):
        secret_img_path = f"secret_{index[i]}.png"
        try:
            watermark = read_text_chunk(secret_img_path)  # Read the watermark
            if watermark != expected_watermark:
                print(f"Error: Share {index[i]} is not authentic!")
                all_authentic = False  # Set flag to False if any share fails
            else:
                print(f"Share {index[i]} is authentic.")
        except Exception as e:
            print(f"Error evaluating watermark for share {index[i]}: {e}")
            all_authentic = False

    # Proceed with decoding only if all shares are authentic
    if all_authentic:
        return decode(imgs, imgs_extra, index, r)
    else:
        print("Decoding aborted: One or more shares failed authentication.")
        return None  # Return None to indicate decoding was not successful

# Decoding the shares and reconstructing the image

def decode(imgs, imgs_extra, index, r):
    assert imgs.shape[0] >= r 
    x = np.array(index)  
    dim = imgs.shape[1] 
    img = [] 

    print("Decoding:")
    last_percent_reported = None
    imgs_add = np.zeros_like(imgs, dtype=np.int32)  
    for i in range(r):
        for indices in imgs_extra[i]: 
            if isinstance(indices, str):  
                try:
                    if indices.isdigit():
                        indices = list(map(int, indices.split())) 
                    else:
                        # print(f"Skipping invalid indices: {indices} (non-numeric characters found)")
                        continue  
                except ValueError as e:
                    print(f"Error parsing indices '{indices}': {e}")
                    continue 
            elif isinstance(indices, (list, np.ndarray)):  
                indices = np.array(indices) 
            else:
                raise ValueError(f"Unexpected type for indices: {type(indices)}")

            if indices.ndim == 1: 
                imgs_add[i][indices] = 256  
            else:
                raise ValueError(f"Expected 1D indices array, got {indices.ndim}D.")

    # Perform Lagrange interpolation for each pixel
    for i in range(dim):
        y = imgs[:, i]  
        ex_y = imgs_add[:, i] 
        y = y + ex_y 
        pixel = lagrange(x, y, r, 0) % 257  
        img.append(pixel)  

        # Show progress in percentage
        percent_done = (i + 1) * 100 // dim
        if last_percent_reported != percent_done:
            if percent_done % 1 == 0:
                last_percent_reported = percent_done
                bar_length = 50  
                block = int(bar_length * percent_done / 100)  
                text = "\r[{}{}] {:.2f}%".format("█" * block, " " * (bar_length - block), percent_done)
                sys.stdout.write(text)
                sys.stdout.flush()

    print()
    return np.array(img)  

# Compare two images
def compare_images(image1_path, image2_path):
    image1 = np.array(Image.open(image1_path))
    image2 = np.array(Image.open(image2_path))
    diff = np.abs(image1 - image2)
    diff_value = round(np.mean(diff), 4)
    print("Mean difference:", diff_value)
    print("Max difference:", round(np.max(diff), 4))
    print("Min difference:", round(np.min(diff), 4))
    print("Standard deviation of difference:", round(np.std(diff), 4))

# Main function to handle encoding and decoding
def main():
    parser = argparse.ArgumentParser(description='Shamir Secret Image Sharing')
    parser.add_argument('-e', '--encode', help='Path to the image to be encoded')
    parser.add_argument('-d', '--decode', help='Path for the origin image to be saved')
    parser.add_argument('-n', type=int, help='The total number of shares')
    parser.add_argument('-r', type=int, help='The threshold number of shares to reconstruct the image')
    parser.add_argument('-i', '--index', nargs='+', type=int, help='The index of shares to use for decoding')
    parser.add_argument('-c', '--compare', nargs=2, help='Compare two images')
    parser.add_argument('-w', '--watermark', help='The watermark to authenticate shares during decoding')
    args = parser.parse_args()

    if args.encode:
        start_time = time.time()
        print("\n=== Starting image encoding process ===")

        if not args.r:
            print("Error: Threshold number 'r' is required for decoding")
            return
        if not args.n:
            print("Error: Total number 'n' of shares is required for decoding")
            return
        if args.r > args.n:
            print("Error: Threshold 'r' cannot be greater than the total number 'n' of shares")
            return

        img_flattened, shape = preprocessing(args.encode)
        secret_imgs, imgs_extra = polynomial(img_flattened, n=args.n, r=args.r)
        to_save = secret_imgs.reshape(args.n, *shape)
        
        watermark = args.watermark if args.watermark else "default_watermark"
        
        for i, img in enumerate(to_save):
            secret_img_path = f"secret_{i + 1}.png"
            Image.fromarray(img.astype(np.uint8)).save(secret_img_path)
            img_extra = str(list((imgs_extra[i]))).encode()
            insert_text_chunk(secret_img_path, secret_img_path, img_extra)
            insert_text_chunk(secret_img_path, secret_img_path, watermark)  # Insert watermark in the share
            size = get_file_size(secret_img_path)
            print(f"{secret_img_path} saved.", size)

        end_time = time.time()
        time_elapsed = end_time - start_time
        print(f"=== Image encoding completed. Time elapsed: {time_elapsed:.2f} seconds===")

    if args.decode:
        start_time = time.time()
        print("\n=== Starting image decoding process ===")

        if not args.r:
            print("Error: Threshold number 'r' is required for decoding")
            return

        expected_watermark = args.watermark if args.watermark else "default_watermark"
        
        input_imgs = []
        input_imgs_extra = []
        for i in args.index:
            secret_img_path = f"secret_{i}.png"
            img_extra = read_text_chunk(secret_img_path)
            img, shape = preprocessing(secret_img_path)
            input_imgs.append(img)
            input_imgs_extra.append(img_extra)
        
        input_imgs = np.array(input_imgs)
        origin_img = decode_with_authentication(input_imgs, input_imgs_extra, args.index, r=args.r, expected_watermark=expected_watermark)
        
        if origin_img is not None:
            origin_img = origin_img.reshape(*shape)
            # Continue with further processing of origin_img
        else:
            print("Decoding failed due to authentication issues. Exiting program.")
            return
        Image.fromarray(origin_img.astype(np.uint8)).save(args.decode)
        size = get_file_size(args.decode)
        print(f"{args.decode} saved.", size)


        end_time = time.time()
        time_elapsed = end_time - start_time
        print(f"=== Image decoding completed. Time elapsed: {time_elapsed:.2f} seconds===")

    if args.compare:
        print("\n=== Comparing images ===")
        compare_images(args.compare[0], args.compare[1])

if __name__ == "__main__":
    main()

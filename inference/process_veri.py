from PIL import Image, ImageOps
from collections import Counter
import os

dataset = './data/VeRi'
target_dataset = './data/veri'
folders = ['image_test', 'image_query']

'''
gallery_files = os.listdir(f"{dataset}/image_test")
gallery_ids = [file[:4] for file in gallery_files]
unique_ids = list(set(gallery_ids))
qualified_ids = [id for id in unique_ids if gallery_ids.count(id) >= 16]

query_files = os.listdir(f"{dataset}/image_query")
query_ids = [file[:4] for file in query_files if file[:4] in qualified_ids]
counter = Counter(query_ids)
top_k_items = counter.most_common(150)

qualified_ids = [item[0] for item in top_k_items]
gallery_files = [f for f in gallery_files if f[:4] in qualified_ids]
query_files = [f for f in query_files if f[:4] in qualified_ids]

print(len(gallery_files))
print(len(query_files))
'''

#print(query_files)


for folder in folders:
    files = os.listdir(f"{dataset}/{folder}")
    os.makedirs(f"{target_dataset}/{folder}", exist_ok=True)
    for f in files:
        if f[-3:] != 'jpg':
            continue
        original_image = Image.open(f"{dataset}/{folder}/{f}")
        width, height = original_image.size
        

        # 2. pad the image
        width, height = original_image.size
        new_width, new_height = max(width, height), max(width, height)

        # Calculate the padding
        delta_width = new_width - width
        delta_height = new_height - height
        padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))

        # Add padding to the image
        new_image = ImageOps.expand(original_image, padding, fill='black')

        # Save or display the new image
        new_image.save(f"{target_dataset}/{folder}/{f}")

'''
train = self.get_datums(gallery_files, 'bounding_box_test')
test = self.get_datums(query_files, 'query')
train = self.generate_fewshot_dataset(train, num_shots=num_shots)
val = train



for folder in folders:
    files = os.listdir(f"{dataset}/{folder}")
    os.makedirs(f"{target_dataset}/{folder}", exist_ok=True)
    for f in files:
        if not f[-3:] == 'jpg':
            continue
        original_image = Image.open(f"{dataset}/{folder}/{f}")
        width, height = original_image.size
        
        # 1. crop to size (1:2.5 scale)
        if width >= int(height / 2.5):
            crop_size = (int(height / 2.5), height)
        else:
            crop_size = (width, int(width * 2.5))

        # Calculate the coordinates of the crop box
        def center_crop_coords(image_width, image_height, crop_width, crop_height):
            left = (image_width - crop_width) / 2
            top = (image_height - crop_height) / 2
            right = (image_width + crop_width) / 2
            bottom = (image_height + crop_height) / 2
            return left, top, right, bottom

        coords = center_crop_coords(original_image.width, original_image.height, *crop_size)

        # Crop the center of the image
        center_cropped_image = original_image.crop(coords)

        # 2. pad the image
        original_image = center_cropped_image
        assert abs(center_cropped_image.size[0] - crop_size[0]) <= 1
        assert abs(center_cropped_image.size[1] - crop_size[1]) <= 1
        width, height = original_image.size

        new_width, new_height = max(width, height), max(width, height)
        # Calculate the padding
        delta_width = new_width - width
        delta_height = new_height - height
        padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))

        # Add padding to the image
        new_image = ImageOps.expand(original_image, padding, fill='black')

        # Save or display the new image
        new_image.save(f"{target_dataset}/{folder}/{f}")
'''

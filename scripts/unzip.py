import os
import zipfile

path = '/data1/SIDD'

for filename in os.listdir(path):
    if filename.endswith('.zip'):
        full_path = os.path.join(path, filename)
        with zipfile.ZipFile(full_path, 'r') as zip_ref:
            print(f'unzip {filename}...')
            zip_ref.extractall(path)

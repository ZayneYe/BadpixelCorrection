import requests

for i in range(200):
    url = f'http://130.63.97.225/share/sidd_dng/000{i+1}_DNG.zip'
    print(f"Downloading 000{i+1}_DNG.zip...")
    response = requests.get(url)

    if response.status_code == 200:
        filename = url.split('/')[-1]
        
        with open(f'/data1/SIDD/{filename}', 'wb') as f:
            f.write(response.content)

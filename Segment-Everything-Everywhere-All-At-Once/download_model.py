import os
import requests
from tqdm import tqdm
import urllib3

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 直接下载链接
url = 'https://hf-mirror.com/xdecoder/SEEM/resolve/main/seem_focalt_v0.pt'
output_file = 'seem_focalt_v0.pt'

print(f'Downloading from: {url}')
print('Note: SSL verification disabled due to network issues')

try:
    # 禁用 SSL 验证
    response = requests.get(url, stream=True, verify=False, timeout=30, allow_redirects=True)

    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        print(f'File size: {total_size / 1024 / 1024:.2f} MB')

        with open(output_file, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f'\nDownloaded successfully to: {output_file}')
    else:
        print(f'Failed: HTTP {response.status_code}')

except Exception as e:
    print(f'Error: {e}')
    print('\nPlease download manually from:')
    print('https://huggingface.co/xdecoder/SEEM/blob/main/seem_focalt_v0.pt')

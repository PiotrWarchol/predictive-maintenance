import urllib.request
import os

print("Downloading NASA CMAPSS Turbofan Engine Dataset...")
print("=" * 50)

# Create data directory
os.makedirs("data/raw", exist_ok=True)

# NASA CMAPSS dataset files hosted on GitHub
base_url = "https://raw.githubusercontent.com/schwxd/LSTM-Keras-CMAPSS/master/data"

files = [
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt"
]

for filename in files:
    print(f"Downloading {filename}...")
    url = f"{base_url}/{filename}"
    try:
        urllib.request.urlretrieve(url, f"data/raw/{filename}")
        print(f"  Saved to data/raw/{filename}")
    except Exception as e:
        print(f"  Failed: {e}")

print()
print("Verifying downloads...")
for filename in files:
    path = f"data/raw/{filename}"
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  {filename}: {size:,} bytes")
    else:
        print(f"  {filename}: MISSING")

print()
print("Download complete!")
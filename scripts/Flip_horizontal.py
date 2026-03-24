import cv2, glob, os
folder = r'C:\Users\RohithSuryaCKM\Downloads\Projects\Image_detection\scripts\dataset\raw\scratch'
files = [f for f in glob.glob(folder + '/*.png')]
print(f'Found {len(files)} images to flip')
for p in files:
    img = cv2.imread(p)
    if img is not None:
        cv2.imwrite(p, cv2.flip(img, 1))
        print(f'  flipped: {os.path.basename(p)}')
print('Done.')
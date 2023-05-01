
from glob import glob
import os


splits = ['training','validation']

for split in splits:

    if not os.path.isfile(f'/home/abs/mmsegmentation/data/images/{split}/{split}.txt'):
        f = open(
            f"/home/abs/mmsegmentation/data/images/{split}/{split}.txt", "x")

    else:
        f = open(
            f"/home/abs/mmsegmentation/data/images/{split}/{split}.txt", "r+")
        f.truncate(0)
        f.close()

    fSave = open(
        (f'/home/abs/mmsegmentation/data/images/{split}/{split}.txt'), 'r+')

    for infile in glob(f'/home/abs/mmsegmentation/data/images/{split}/*.jpg'):
        fName = os.path.basename(infile)
        fSave.write(fName+'\n')

    fSave.close()

    fSave = open(
        (f'/home/abs/mmsegmentation/data/images/{split}/{split}.txt'), 'r+')

    total_files = len(fSave.readlines())
    print(total_files)
    # fSave.write("Dataset Length: "+str(total_files)+'\n')
    fSave.close()
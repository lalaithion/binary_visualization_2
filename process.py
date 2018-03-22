from PIL import Image
import os


def process_dir(path, start="jpg", end="png"):
    items = os.listdir(path)

    for i in items:
        name, ext, *_ = i.rsplit(".", maxsplit=1)
        fullpath = os.path.join(path, i)
        saveto = os.path.join(path, name + "." + end)
        if ext.lower() == start and not os.path.exists(saveto):
            im = Image.open(fullpath)
            print("saving", saveto)
            im.save(saveto)


if __name__ == "__main__":
    process_dir("./data/")

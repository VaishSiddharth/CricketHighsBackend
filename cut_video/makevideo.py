import time
import cv2
import mss
import numpy
import os

from os.path import isfile, join


def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    # for sorting the file names properly
    files.sort(key=lambda x: int(x[5:-4]))

    for i in range(len(files)):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        print(filename)
        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def mainvideos():
    pathIn = 'images/'
    pathOut = 'video.avi'
    fps = 25.0
    convert_frames_to_video(pathIn, pathOut, fps)



def save_images(*args, **kwargs):
    with mss.mss() as sct:
        # Part of the screen to capture
        sct.compression_level = 9
        monitor = {"top": 0, "left": 0, "width": 720, "height": 480}

        x = 0
        last_time = time.time()
        while x<500:
            # last_time = time.time()
            if (1 / (time.time() - last_time)) <= 30:
                last_time = time.time()
                # Grab the data
                sct_img = sct.grab(monitor)

                x = x + 1
                output = "images/image" + str(x) + ".png".format(**monitor)
                mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)

                # Get raw pixels from the screen, save it to a Numpy array
                img = numpy.array(sct_img)

                # Display the picture
                cv2.imshow("OpenCV/Numpy normal", img)

                # Display the picture in grayscale
                # cv2.imshow('OpenCV/Numpy grayscale',
                #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

                # print("fps: {}".format(1 / (time.time() - last_time)))

                # Press "q" to quit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break
            print("fps: {}".format(1 / (time.time() - last_time)))
        mainvideos()


def make_video(*args, **kwargs):
    image_folder = 'images'
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

print("working")
save_images()
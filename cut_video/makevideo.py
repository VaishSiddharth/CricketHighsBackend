import time
import cv2
import mss
import numpy
import os

from os.path import isfile, join

import sounddevice as sd
from scipy.io.wavfile import write
import moviepy.editor as mpe

def combine_audio_video():
    my_clip = mpe.VideoFileClip('video.avi')
    audio_background = mpe.AudioFileClip('output.wav')
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile("movie.mp4", fps=20)

def record_audio():
    fs = 44100  # Sample rate
    seconds = 21  # Duration of recording
    sd.default.device = 8

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording)  # Save as WAV file


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
    combine_audio_video()


def make_video():
    pathIn = 'images/'
    pathOut = 'video.avi'
    fps = 20.0
    convert_frames_to_video(pathIn, pathOut, fps)



def save_images(*args, **kwargs):
    with mss.mss() as sct:
        # Part of the screen to capture
        sct.compression_level = 9
        monitor = {"top": 0, "left": 0, "width": 720, "height": 480}

        x = 0
        fs = 44100  # Sample rate
        seconds = 25  # Duration of recording
        sd.default.device = 8

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        last_time = time.time()
        while x<500:
            # last_time = time.time()
            if (1 / (time.time() - last_time)) <= 20:
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
        write('output.wav', fs, myrecording)  # Save as WAV file
        make_video()

print("working")
save_images()
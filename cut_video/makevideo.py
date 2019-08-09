import time
import cv2
import mss
import numpy
import os
from multiprocessing import Process, Queue

import mss
import mss.tools

from os.path import isfile, join

import sounddevice as sd
from scipy.io.wavfile import write
import moviepy.editor as mpe

def save_images(*args, **kwargs):
    with mss.mss() as sct:
        # Part of the screen to capture
        sct.compression_level = 9
        monitor = {"top": 0, "left": 0, "width": 720, "height": 480}

        x = 0
        fs = 44100  # Sample rate
        seconds = 25  # Duration of recording
        # sd.default.device = 8 # change recording device

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        last_time = time.time()
        while x < 500:
            # last_time = time.time()
            if (1 / (time.time() - last_time)) <= 20:
                last_time = time.time()
                # Grab the data
                sct_img = sct.grab(monitor)

                x = x + 1
                output = "images/image" + str(x) + ".png".format(**monitor)

                # store images in png format
                mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)

                # Get raw pixels from the screen, save it to a Numpy array
                img = numpy.array(sct_img)

                # Display the picture
                cv2.imshow("OpenCV/Numpy normal", img)

                # print("fps: {}".format(1 / (time.time() - last_time)))

                # Press "q" to quit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break
            print("fps: {}".format(1 / (time.time() - last_time)))
        write('output.wav', fs, myrecording)  # Save as WAV file
        make_video()

def make_video():
    pathIn = 'images/'
    pathOut = 'video.avi'
    fps = 30.0
    convert_frames_to_video(pathIn, pathOut, fps)

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


def combine_audio_video():
    my_clip = mpe.VideoFileClip('video.avi')
    audio_background = mpe.AudioFileClip('output.wav')
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile("movie.mp4", fps=30)


def record_audio():
    import sounddevice as sd
    fs = 44100  # Sample rate
    seconds = 16  # Duration of recording
    # sd.default.device = 9

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording)  # Save as WAV file


print("working")
# save_images()

def grab(queue):
    # type: (Queue) -> None
    rect = {"top": 0, "left": 0, "width": 1920, "height": 1080}

    with mss.mss() as sct:
        last_time = time.time()
        x=0
        while x<500:
            if (1 / (time.time() - last_time)) <= 30:
                x=x+1
                last_time = time.time()
                queue.put(sct.grab(rect))
            print("fps: {}".format(1 / (time.time() - last_time)))

    # Tell the other worker to stop
    queue.put(None)


def save(queue):
    # type: (Queue) -> None

    number = 0
    output = "images/file_{}.png"
    to_png = mss.tools.to_png

    while "there are screenshots":
        img = queue.get()
        if img is None:
            break

        to_png(img.rgb, img.size, output=output.format(number))
        number += 1


    make_video()

if __name__ == "__main__":
    # The screenshots queue
    queue = Queue()  # type: Queue

    # 2 processes: one for grabing and one for saving PNG files
    Process(target=grab, args=(queue,)).start()
    Process(target=save, args=(queue,)).start()

    Process(target=record_audio(),args=()).start()
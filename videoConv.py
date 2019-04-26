import os
import cv2
import glob



def clear_directory(loc):
    ''' deletes all files in directory. if directory doesn't exist, creates directory '''

    if os.path.exists(loc):
        files = glob.glob(loc + '*')
        for f in files:
            os.remove(f)

    else:
        os.makedirs(loc)



def video_to_frames(input_dir, output_dir):
    ''' converts video to images '''

    clear_directory(output_dir)

    # get filenames
    filename = os.path.basename(input_dir)
    name, sep, ext = filename.rpartition('.')
    output = output_dir + name

    # load and save frames
    vidcap = cv2.VideoCapture(input_dir)
    hasFrames, image = vidcap.read()
    index = 0
    hasFrames = True
    while hasFrames:

        if index < 10:
            cv2.imwrite(str(output) + "-frame-0%d.jpg" % index, image)
            print('Saving Frame 0%d ' % index)
        else:
            cv2.imwrite(str(output) + "-frame-%d.jpg" % index, image)
            print('Saving Frame %d ' % index)
        hasFrames, image = vidcap.read()
        index += 1
      



def frames_to_video(input_dir, output_dir, fps):
    ''' converts images to video '''

    frame_array = []
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
 
    # sort file names
    files.sort()
 
    for i in range(len(files)):

        filename = input_dir + files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        frame_array.append(img)
 
    out = cv2.VideoWriter(output_dir, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for index in range(len(frame_array)):
        out.write(frame_array[index])
        if index < 10:
            print('Exporting Frame 0%d ' % index)
        else:
            print('Exporting Frame %d ' % index)

    out.release()



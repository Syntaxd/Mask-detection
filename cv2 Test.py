import PIL.Image
import cv2, IPython, PIL
from io import BytesIO
dir(PIL)


def get_frame(cam):
    # Capture frame-by-frame
    ret, frame = cam.read()

    #flip image for natural viewing
    frame = cv2.flip(frame, 1)
    return frame


cam = cv2.VideoCapture(1)
cam.isOpened()
# try:
#     while(True):
#         # Capture frame-by-frame
#         frame = get_frame(cam)
#         # Convert the image from OpenCV BGR format to matplotlib RGB format
#         # to display the image
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         showarray(frame)
#         print("%f FPS" % (1/(t2-t1)))
        
#         # Display the frame until new frame is available
#         clear_output(wait=True)
# except KeyboardInterrupt:
#     cam.release()

d = IPython.display.display("", display_id=1)
frame = get_frame(cam)

def array_to_image(a, fmt='jpeg'):
    #Create binary stream object
    f = BytesIO()

    #Convert array to binary stream object
    PIL.Image.fromarray(a).save(f, fmt)

    return IPython.display.Image(data=f.getvalue())

array_to_image(frame)

import cv2
import numpy as np
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib, GdkPixbuf
import time

start = time.time()
# icon record 
imgf = cv2.imread('icon/record.png')
# resize icon record
resf = cv2.resize(imgf,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
#open cam
cap = cv2.VideoCapture(0)
builder = Gtk.Builder()
builder.add_from_file("layout.glade")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('video/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'),
                      10, (frame_width,frame_height))
record_on = False


class Handler:
    def onDeleteWindow(self, *args):
        cap.release
        Gtk.main_quit(*args)

window = builder.get_object("window1")
image = builder.get_object("image1")
window.show_all()
builder.connect_signals(Handler())


def record(recordBut):
    global record_on
    record_on = True

def aboutInfo(popup):
    dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Recorder Info")
    dialog.format_secondary_text(
            """Press save to start record
            https://tik.tn
            """)
    dialog.run()
    dialog.destroy()
    

    
def show_frame(*args):
    ret, frame = cap.read()
    img = frame
    timer = time.time() - start
    if record_on :
        out.write(frame)
        rows,cols,channels = resf.shape
        rows1,cols1,channels1 = img.shape
        roi = img[0:rows, 0:cols ]
        img2gray = cv2.cvtColor(resf,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        img2_fg = cv2.bitwise_and(resf,resf,mask = mask)
        dst = cv2.add(img1_bg,img2_fg)
        img[0:rows, 0:cols ] = dst
        cv2.putText(img,str(timer)[0:4], (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,0,255))
        timer = time.time() - start 
        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pb = GdkPixbuf.Pixbuf.new_from_data(frame.tostring(),
                                        GdkPixbuf.Colorspace.RGB,
                                        False,
                                        8,
                                        frame.shape[1],
                                        frame.shape[0],
                                        frame.shape[2]*frame.shape[1])                            
    image.set_from_pixbuf(pb.copy())
    return True


recordBut = builder.get_object("s")
recordBut.connect("activate", record)

popup = builder.get_object("about")
popup.connect("activate", aboutInfo)

GLib.idle_add(show_frame)
Gtk.main()

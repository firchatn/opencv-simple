import cv2
import numpy as np
import gi
 
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib, GdkPixbuf
 
cap = cv2.VideoCapture(0)
builder = Gtk.Builder()
builder.add_from_file("layout.glade")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))



class Handler:
    def onDeleteWindow(self, *args):
        cap.release
        Gtk.main_quit(*args)

window = builder.get_object("window1")
image = builder.get_object("image1")
window.show_all()
builder.connect_signals(Handler())


def record(new_action):
    print('o')
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'),
                      10, (frame_width,frame_height))
    ret, frame = cap.read()
    out.write(frame)
    
new_action = builder.get_object("s")
new_action.connect("activate", record)


    
def show_frame(*args):
    ret, frame = cap.read()
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

GLib.idle_add(show_frame)
Gtk.main()

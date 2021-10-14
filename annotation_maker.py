import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import RectangleSelector
import glob

# global constants
img = None
tl_list = []
br_list = []
object_list = []
im_w=None
im_h=None

# constants
image_folder = 'data_dir/IRCP_dataset_1024X768'
savedir = 'data_dir/IRCP_dataset_annotation'
obj = ['0']


def line_select_callback(clk, rls):
    global tl_list
    global br_list
    global object_list
    tl_list.append((int(clk.xdata), int(clk.ydata)))
    br_list.append((int(rls.xdata), int(rls.ydata)))
    object_list.append(obj)


def onkeypress(event):
    global object_list
    global tl_list
    global br_list
    global img
    if event.key == 'q':
        print(tl_list,br_list)
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        save_path = os.path.join(savedir, img.name.replace('jpg', 'txt'))
        GT = open(save_path, 'w')
        for i in range(len(tl_list)):
            x_center=int((tl_list[i][0]+br_list[i][0])/2)/im_w
            y_center=int((tl_list[i][1]+br_list[i][1])/2)/im_h
            absolute_w=br_list[i][0]-tl_list[i][0]
            absolute_h=br_list[i][1]-tl_list[i][1]
            width=absolute_w/im_w
            height=absolute_h/im_h
            GT.write(obj[0]+" "+str(x_center)+ " " + str(y_center)+ " " + str(width)+ " " + str(height) + "\n")
        tl_list = []
        br_list = []
        object_list = []
        img = None


def toggle_selector(event):
    toggle_selector.RS.set_active(True)

def path_file_maker():
    current_dir = image_folder
    # Percentage of images to be used for the valid set
    percentage_test = 10;
    # Create train.txt and valid.txt
    file_train = open('train.txt', 'w')
    file_test = open('valid.txt', 'w')
    # Populate train.txt and valid.txt
    counter = 1
    index_test = round(100 / percentage_test)
    for file in glob.iglob(os.path.join(current_dir, '*.jpg')):
        title, ext = os.path.splitext(os.path.basename(file))
        if counter == index_test:
            counter = 1
            file_test.write(current_dir + "/" + title + '.jpg' + "\n")
        else:
            file_train.write(current_dir + "/" + title + '.jpg' + "\n")
            counter = counter + 1

if __name__ == '__main__':
    for n, image_file in enumerate(os.scandir(image_folder)):
        img = image_file
        fig, ax = plt.subplots(1, figsize=(10.5, 8))
        # mngr = plt.get_current_fig_manager()
        # mngr.window.setGeometry(250, 40, 800, 600)
        image = cv2.imread(image_file.path)
        try:
            im_w,im_h,dim=image.shape
        except:
            print("Error while reading image #:",n)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)

        toggle_selector.RS = RectangleSelector(
            ax, line_select_callback,
            drawtype='box', useblit=True,
            button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True,
        )
        bbox = plt.connect('key_press_event', toggle_selector)
        key = plt.connect('key_press_event', onkeypress)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
    path_file_maker()

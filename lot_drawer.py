import numpy as np

# ============================================================================
import cv2.cv2 as cv2

CANVAS_SIZE = (600, 800)

FINAL_LINE_COLOR = (15, 212, 15)  # green
WORKING_LINE_COLOR = (58, 14, 212) # red


# ============================================================================ #
#                                  CONFIG                                      #

basic_img = "lot_ny.jpg"
crop = {
    "y": 230, "x": 0, "h": 600, "w": 600
    }
index = 65  # starting index, to add e.g to exiting list --> starts with this number!

# ============================================================================ #


y = crop["y"]
x = crop["x"]
h = crop["h"]
w = crop["w"]
# basic_img = basic_img[y:y + h, x:x + w]

polygon_points = []
lot_list = []


class PolygonDrawer(object):
    def __init__(self, window_name):
        self.window_name = window_name  # Name for our window

        self.done = False  # Flag signalling we're done
        self.current = (0, 0)  # Current position, so we can draw the line-in-progress
        self.points = []  # List of points defining our polygon

    index = index

    def on_mouse(self, event, x, y, buttons, user_param, ind=index):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done:  # Polygon is finished - Nothing more to do
            poly = {"cords": self.points, "id": f"{PolygonDrawer.index}"}
            lot_list.append(poly)
            polygon_points.append(self.points)
            print(lot_list)
            PolygonDrawer.index += 1
            self.points = []
            self.done = False
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))

            if len(self.points) == 4:  # close polygon on 4 points
                self.done = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True

    def run(self):
        escape = False  # Flag to quit program
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        #cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))

        frame = cv2.imread(basic_img)
        frame = frame[y:y + h, x:x + w] # crop image
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while not escape:

            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            frame = cv2.imread(basic_img)
            frame = frame[y:y + h, x:x + w] # crop image
            canvas = frame
            if len(self.points) > 0:
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 3)
                # And also show what the current segment would look like
                cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR, 3)

            for i in range(len(polygon_points)):
                if len(polygon_points[i]) > 0:
                    cv2.fillPoly(canvas, np.array([polygon_points[i]]), FINAL_LINE_COLOR)

            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27:  # ESC hit
                # todo: maybe add txt export here
                self.done = True
                escape = True

        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        return canvas


# ============================================================================

if __name__ == "__main__":
    pd = PolygonDrawer("Polygon")
    image = pd.run()
    print("Lots = %s \n" % polygon_points)

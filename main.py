import cv2
import time
import mediapipe as mp
import numpy as np
import json
from matplotlib import pyplot as plt
import matplotlib.animation as animation


class InterfaceInput:
    def __init__(self):
        self.xlim = [-.5,.5]
        self.ylim = [-.5, .5]
        self.zlim = [-.5,.5]

    def do_action(self, hand_state, animator):
        if animator is None:
            return
        v = hand_state.state["index_direction_vector"]
        print(v)
        animator.clear()
        animator.plot(
            [0, v[0]],
            [0, v[1]],
            [0, v[2]]
        )
        animator.scatter([v[0]], [v[1]], [v[2]])
        animator.set_xlim3d(*self.xlim)
        animator.set_ylim3d(*self.ylim)
        animator.set_zlim3d(*self.zlim)
        animator.set_xlabel('$X$')
        animator.set_ylabel('$Y$')
        animator.set_zlabel('$Z$')

    def do_nothing(self, animator):
        if animator is None:
            return
        animator.clear()
        animator.plot(
            [0, 0],
            [0, 0],
            [0, 0]
        )
        animator.set_xlim3d(*self.xlim)
        animator.set_ylim3d(*self.ylim)
        animator.set_zlim3d(*self.zlim)
        animator.set_xlabel('$X$')
        animator.set_ylabel('$Y$')
        animator.set_zlabel('$Z$')


class HandState:
    def __init__(self):
        self.n_states = 30
        self.state = {}
        self.delta_t = 0.000001
        self.history = []
        self.last_time = 0

    def update(self, single_hand):
        self.delta_t = time.time() - self.last_time
        self.last_time = time.time()
        self.state = {
            **self.get_pointer_vector(single_hand),
            **self.get_cluser_center(single_hand),
            "delta_t": self.delta_t,
        }
        # print(self.state)
        # print(json.dumps({k: list(v) if isinstance(v, np.ndarray) else v for k,v in self.state.items()}, indent=4))
        if len(self.history) == self.n_states:
            self.history = self.history[1:]
        self.history.append(self.state)

    def get_pointer_vector(self, hand):
        knuckle = np.array([hand[0].x, hand[0].y, hand[0].z])
        tip = np.array([hand[8].x, hand[8].y, hand[8].z])
        vector = np.subtract(knuckle, tip)
        return {
            "index": tip,
            "index_direction_vector": vector,
            # "index_direction_vector_derivative": vector - self.history[-1]["index_direction_vector"]
            # if len(self.history) > 0 else np.zeros(3),
        }

    def get_cluser_center(self, hand):
        all_points = np.array([np.array([h.x, h.y, h.z])for h in hand])
        center = np.mean(all_points, axis=0)
        finger_tips = np.array([np.array([h.x, h.y, h.z]) for i, h in enumerate(hand) if i in [8, 12, 16, 20]])
        finger_tips_center = np.mean(finger_tips, axis=0)
        return {
            "center_cluster": center,
            "graph_center": np.subtract(np.array([.5, .5, .5]), center),
            "finger_tips_center": finger_tips_center,
            "hand_length": np.subtract(np.array([hand[0].x, hand[0].y, hand[0].z]), finger_tips_center),
            "finger_tip_mad": np.mean(np.linalg.norm(finger_tips - finger_tips_center, axis=1))
        }



# https://www.youtube.com/watch?v=Ercd-Ip5PfQ
class Runner:
    def __init__(self):
        self.will_plot = True
        self.cap = cv2.VideoCapture(0)
        self.hand_state = HandState()
        hands_model = mp.solutions.hands
        self.hands_model = hands_model.Hands(
            # static_image_mode=True,
            model_complexity=1,
            static_image_mode=False,
            max_num_hands=3,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.draw = mp.solutions.drawing_utils

        self.hand_states = HandState()
        self.interference = InterfaceInput()
        if self.will_plot:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            self.fig = fig
            self.ax = ax
            self.ax.set_xlim3d(-.5,.5)
            self.ax.set_ylim3d(-.5,.5)
            self.ax.set_zlim3d(-.5,.5)
            self.ani = animation.FuncAnimation(self.fig, self.process_frame, frames=1, repeat=True)
            plt.show()
        else:
            while True:
                self.process_frame(0)

    @staticmethod
    def show():
        print("calling show")
        plt.show()

    def process_frame(self, i):
        print(i)
        success, img = self.cap.read()
        # img = cv2.cvtColor(img)
        results = self.hands_model.process(img)
        results = results.multi_hand_landmarks
        if results:
            print("hand detected")
            hand = results[0].landmark
            self.hand_states.update(hand)
            print(self.hand_states.delta_t, " seconds")
            self.interference.do_action(self.hand_states, self.ax)
            self.draw.draw_landmarks(img, results[0])

            # draw the circle and color if grab
            x = int(self.hand_states.state["index"][0] * img.shape[1])
            y = int(self.hand_states.state["index"][1] * img.shape[0])
            # Define the radius and the color of the circle
            radius = 15
            color = (0, 0, 255)
            if self.hand_states.state["finger_tip_mad"] > 0.035:
                color = (0, 255, 0)
            # Draw the circle
            cv2.circle(img, (x, y), radius, color, -1)
        else:
            print("hand not detected")
            self.interference.do_nothing(self.ax)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
        return True


if __name__ == '__main__':
    r = Runner()
    r.show()

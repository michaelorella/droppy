import unittest
import unittest.mock

from io import StringIO

from contactangles.analysis import main

import functools

import pyautogui
import threading
import time


class TestFullAnalysisScript(unittest.TestCase):

    def automate_mouse_action(positions):
        def automation_wrapper(func):
            @functools.wraps(func)
            def decorator(*args, **kwargs):
                def thread_action(positions):
                    print('Thread starting')
                    time.sleep(10)
                    print('Starting moving')
                    pyautogui.moveTo(*positions[0])
                    pyautogui.dragTo(*positions[1], duration=0.5)
                    pyautogui.press('enter')

                thread = threading.Thread(target=thread_action,
                                          args=(positions,))
                thread.start()

                value = func(*args, **kwargs)

                return value
            return decorator
        return automation_wrapper

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    @automate_mouse_action([[84, 278], [444, 482]])
    def test_linear_fit(self):
        files = ['test_images/hydrophobic_flat.avi',
                 'test_images/hydrophilic_flat.avi',
                 'test_images/neutral_flat.avi']
        main(['test_images/hydrophobic_flat.avi'])

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    @automate_mouse_action([[84, 278], [444, 482]])
    def test_circular_fit(self):
        main(['test_images/hydrophobic_flat.avi','--fitType','circular'])

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    @automate_mouse_action([[84, 278], [544, 482]])
    def test_hydrophilic_linear_fit(self):
        main(['test_images/hydrophilic_flat.avi'])

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    @automate_mouse_action([[84, 278], [544, 482]])
    def test_hydrophilic_circular_fit(self):
        main(['test_images/hydrophilic_flat.avi','--fitType','circular',
              '--frequency','5'])

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    @automate_mouse_action([[144, 278], [494, 482]])
    def test_neutral_linear_fit(self):
        main(['test_images/neutral_flat.avi'])

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    @automate_mouse_action([[144, 278], [494, 482]])
    def test_neutral_circular_fit(self):
        main(['test_images/neutral_flat.avi','--fitType','circular'])

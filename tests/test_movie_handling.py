import unittest
import unittest.mock

import imageio

from droppy.moviehandling import extract_grayscale_frames

import logging
import sys

log = logging.getLogger('TestLog')
log.level = logging.DEBUG


class TestFrameExtraction(unittest.TestCase):
    def setUp(self):
        handle = logging.StreamHandler(sys.stdout)
        log.addHandler(handle)

    def tearDown(self):
        log.removeHandler(logging.StreamHandler(sys.stdout))

    def test_frame_counting(self):
        video = r'./test_images/hydrophobic_flat.avi'
        t, grayscale_images = extract_grayscale_frames(video,
                                                       start_time=0,
                                                       data_freq=1)

        self.assertEqual(len(t), 20)

        t, grayscale_images = extract_grayscale_frames(video,
                                                       start_time=0,
                                                       data_freq=2)
        self.assertEqual(len(t), 10)

        t, grayscale_images = extract_grayscale_frames(video,
                                                       start_time=0,
                                                       data_freq=1.5)
        self.assertEqual(len(t), 14)

        t, grayscale_images = extract_grayscale_frames(video,
                                                       start_time=0,
                                                       data_freq=0.3)
        self.assertEqual(len(t), 67)

        t, grayscale_images = extract_grayscale_frames(video,
                                                       start_time=0,
                                                       data_freq=0.1)
        self.assertEqual(len(t), 198)


@unittest.expectedFailure
class TestDataOutput(unittest.TestCase):
    def test_output_file_existence(self):
        self.fail('No test written')


@unittest.expectedFailure
class TestPlotting(unittest.TestCase):
    def test_plots_made(self):
        self.fail('No test written')

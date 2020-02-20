import unittest
import unittest.mock
import logging
import sys
import os
import pyautogui

from io import StringIO

import numpy as np

import skimage
from skimage import io

import matplotlib.pyplot as plt

from droppy.imageanalysis import (crop_points, auto_crop, get_crop,
                                  output_text, output_fits)

from droppy.edgedetection import (sigma_setter, extract_edges, CannyPlugin)
from droppy.linearfits import (fit_line, generate_vectors,
                               generate_droplet_width)
from droppy.circularfits import (dist, fit_circle, generate_circle_vectors,
                                 find_intersection)
from droppy.common import (L, R, T, B, positive_int, positive_float,
                           calculate_angle, baseF)
from droppy.bafits import (sim_bashforth_adams, fit_bashforth_adams)

import threading
import time

log = logging.getLogger('TestLog')
log.level = logging.DEBUG


class TestImageCropping(unittest.TestCase):
    def setUp(self):
        self.image = {}
        self.image['circle'] = io.imread(r'./test_images/circle.jpg',
                                         as_gray=True)
        self.image['triangle'] = io.imread(r'./test_images/triangle.jpg',
                                           as_gray=True)
        self.image['leaf'] = io.imread(r'./test_images/leaves900.jpg',
                                       as_gray=True)

        self.edge = {shape: extract_edges(self.image[shape])
                     for shape in self.image}

        handle = logging.StreamHandler(sys.stdout)
        log.addHandler(handle)

    def tearDown(self):
        del(self.image)
        del(self.edge)

        log.removeHandler(logging.StreamHandler(sys.stdout))

    def test_crop_size(self):
        edge = self.edge

        triangle = edge['triangle']
        circle = edge['circle']

        t_crop = crop_points(triangle, [200, 400, 200, 400])
        self.assertTrue(crop_points(circle, [200, 400, 200, 400]).size
                        == 0)
        self.assertFalse(199 in t_crop)
        self.assertFalse(0 in t_crop)
        self.assertTrue(200 in t_crop)
        self.assertTrue(t_crop.size == 604)

    def test_crop_above_line(self):
        edge_c = self.edge['circle']

        line = {}
        line[L] = lambda x, y: x
        line[R] = lambda x, y: x
        line[T] = lambda x, y: y
        line[B] = lambda x, y: y - x
        c_crop = crop_points(edge_c, [0, 600, 0, 0])

        self.assertTrue(all([y <= x for x, y in c_crop]))

    def test_crop_full(self):
        edge = self.edge

        triangle_size = self.image['triangle'].shape
        circle_size = self.image['circle'].shape

        triangle = edge['triangle']
        circle = edge['circle']

        t_crop = crop_points(triangle, [0, triangle_size[1],
                                             0, triangle_size[0]])
        c_crop = crop_points(circle, [0, circle_size[1],
                                           0, circle_size[0]])

        self.assertTrue(np.array_equal(t_crop, triangle))
        self.assertTrue(np.array_equal(c_crop, circle))

    def test_warnings(self):
        with self.assertWarns(Warning) as w:
            crop_points(self.edge['triangle'], [0, 100, 100, 0])

        self.assertIn('Check the order', str(w.warning))

        with self.assertWarns(Warning) as w:
            crop_points(self.edge['triangle'], [-100, 100, 0, 100])

        self.assertIn('All bounds must', str(w.warning))

    def test_get_crop(self):
        im = self.image['circle']

        def automate_input():
            time.sleep(1)
            pyautogui.moveTo(50, 70)
            pyautogui.dragTo(400, 250)
            pyautogui.press('enter')

        thread = threading.Thread(target=automate_input)
        thread.start()

        c = get_crop(im)

        self.assertEqual(len(c), 4)
        for i in range(len(c)):
            with self.subTest(i=i):
                self.assertGreater(c[i], 0)

        def automate_x_input():
            time.sleep(1)
            pyautogui.moveTo(50, 70)
            pyautogui.dragTo(400, 250)
            pyautogui.click(x=583, y=19)

        thread = threading.Thread(target=automate_x_input)
        thread.start()

        c2 = get_crop(im)

        self.assertListEqual(list(c), list(c2))

    def test_get_crop_bad_click(self):
        im = self.image['circle']

        def automate_bad_input():
            time.sleep(1)
            pyautogui.moveTo(50, 70)
            pyautogui.dragTo(1500, 800, duration=0.2)
            pyautogui.press('enter')

        thread = threading.Thread(target=automate_bad_input)
        thread.start()

        c = get_crop(im)
        log.debug(c)

        self.assertEqual(len(c), 4)

        im = self.image['leaf']

        def automate_dangerous_input():
            time.sleep(1)
            pyautogui.moveTo(50, 70)
            pyautogui.dragTo(850, 400, duration=0.2)
            pyautogui.press('enter')

        thread = threading.Thread(target=automate_bad_input)
        thread.start()

        c = get_crop(im)
        log.debug(c)

        self.assertEqual(len(c), 4)


class TestOutput(unittest.TestCase):
    @unittest.mock.patch('sys.stdout', new_callable=StringIO)
    def test_text_output(self, mock_out):
        output_text(0, {L: 180, R: 100}, 40, np.nan)
        self.assertIn(" 0.000", mock_out.getvalue())
        self.assertIn(" 180.000", mock_out.getvalue())
        self.assertIn(" 100.000", mock_out.getvalue())
        self.assertIn(" 140.000", mock_out.getvalue())
        self.assertIn(" 40.0", mock_out.getvalue())


class TestLineFitting(unittest.TestCase):
    def test_fit_line(self):
        points = (np.transpose(np.tile(np.arange(10), (2, 1)))
                  + np.random.rand(10, 2))
        result, *_ = fit_line(points)
        self.assertAlmostEqual(result[1], 1.0, delta=0.2)

        result, *_ = fit_line(points, order=2)
        self.assertAlmostEqual(result[2], 0.0, delta=0.1)

    def test_fit_circle(self):
        theta = np.linspace(0, 2*np.pi, num=500)
        x = np.cos(theta)
        y = np.sin(theta)

        res = fit_circle(np.transpose(np.vstack((x, y))))
        stat = res['success']
        fval = res['fun']
        out = res['x']
        *z, r = out

        log.debug(out)

        self.assertTrue(stat)
        self.assertLess(fval, 1e-5)
        self.assertAlmostEqual(r, 1, places=3)

        x = 10 + 5 * np.cos(theta)
        y = 13 + 5 * np.sin(theta)

        res = fit_circle(np.transpose(np.vstack((x, y))))
        stat = res['success']
        fval = res['fun']
        out = res['x']
        *z, r = out

        log.debug(out)

        self.assertTrue(stat)
        self.assertLess(fval, 1e-5)
        self.assertAlmostEqual(r, 5, places=3)
        self.assertAlmostEqual(z[0], 10, places=3)
        self.assertAlmostEqual(z[1], 13, places=3)


class TestEdgeExtraction(unittest.TestCase):
    def test_extract_edges(self):
        im = io.imread(r'./test_images/circle.jpg', as_gray=True)
        edges = extract_edges(im)

        for pt in edges:
            with self.subTest(pt=pt):
                pt_dist = (pt[0]-300)**2 + (pt[1]-300)**2
                r = 280**2
                δ = np.sqrt(np.abs(pt_dist - r))
                self.assertLessEqual(δ, 100.0)

    def test_generate_droplet_width(self):
        points = np.transpose([np.concatenate((np.linspace(0, 0.1, num=10),
                                               np.linspace(0.9, 1, num=10))),
                               np.ones(20)])
        bounds = [0, 1, -1, 1]

        lim = generate_droplet_width(points)
        self.assertEqual(lim[L], 0.0)
        self.assertEqual(lim[R], 1.0)


class TestCalculateDistances(unittest.TestCase):
    def test_calculate_angle(self):
        v1 = np.array([0, 1])
        v2 = np.array([1, 0])

        self.assertAlmostEqual(calculate_angle(v1, v2), 90, places=4)
        self.assertAlmostEqual(calculate_angle(v2, v1), 90, places=4)

        v1 = np.array([1, 1])
        v2 = np.array([1, 0])

        self.assertAlmostEqual(calculate_angle(v1, v2), 45, places=4)

        v1 = np.array([1, 0])
        v2 = np.array([1, 0])

        self.assertAlmostEqual(calculate_angle(v1, v2), 0, places=4)

        v1 = np.array([-1, 0])
        v2 = np.array([1, 0])

        self.assertAlmostEqual(calculate_angle(v1, v2), 180, places=4)

        v1 = np.array([1, 0])
        v2 = np.array([0, -1])

        self.assertAlmostEqual(calculate_angle(v1, v2), 90, places=4)

    def test_generate_vectors(self):
        # Test normal conditions (hydrophobic-ish)
        points = {L: np.transpose([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]),
                  R: np.transpose([[90, 91, 92, 93, 94, 95,
                                         96, 97, 98, 99, 100],
                                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])}
        limits = generate_droplet_width(np.vstack((points[L],
                                                        points[R])))
        eps = 1e-5
        a = [0, 0]

        v, b, m, bv, vertical = generate_vectors(points, limits, eps, a)

        self.assertAlmostEqual(v[L][0], 1)
        self.assertAlmostEqual(v[L][1], -1)
        self.assertAlmostEqual(v[R][0], 1)
        self.assertAlmostEqual(v[R][1], 1)

        self.assertAlmostEqual(b[L], 10.0)
        self.assertAlmostEqual(b[R], -90.0)

        self.assertAlmostEqual(m[L], -1)
        self.assertAlmostEqual(m[R], 1)

        self.assertListEqual(list(bv[L]), [1, 0])
        self.assertListEqual(list(bv[R]), [1, 0])

        self.assertFalse(vertical[L])
        self.assertFalse(vertical[R])

        # Test vertical line
        points = {L: np.transpose([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]),
                  R: np.transpose([[90, 90, 90, 90, 90, 90, 90, 90,
                                         90, 90, 90],
                                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])}
        limits = generate_droplet_width(np.vstack((points[L],
                                                        points[R])))
        eps = 1e-5
        a = [0, 0]

        v, b, m, bv, vertical = generate_vectors(points, limits, eps, a)

        self.assertAlmostEqual(v[L][0], 0)
        self.assertAlmostEqual(v[L][1], -1)
        self.assertAlmostEqual(v[R][0], 0)
        self.assertAlmostEqual(v[R][1], 1)

        self.assertAlmostEqual(b[L], 0)
        self.assertAlmostEqual(b[R], 90)

        self.assertAlmostEqual(m[L], 0)
        self.assertAlmostEqual(m[R], 0)

        self.assertListEqual(list(bv[L]), [1, 0])
        self.assertListEqual(list(bv[R]), [1, 0])

        self.assertTrue(vertical[L])
        self.assertTrue(vertical[R])

    def test_dist(self):
        *z, r = [0, 0, 1]
        points = np.array([[0, 0]])
        d = dist([*z, r], points)
        self.assertAlmostEqual(d, 1.0)

    def test_generate_circle_vectors(self):
        points = [[np.sqrt(2)/2, np.sqrt(2)/2],
                  [1/2, np.sqrt(3)/2],
                  [np.sqrt(3)/2, -1/2],
                  [1, 0]]

        good_v2s = np.array([[1/np.sqrt(2), -1/np.sqrt(2)],
                             [np.sqrt(3)/2, -1/2],
                             [-1/2, -np.sqrt(3)/2],
                             [0, 1]])

        for i, pt in enumerate(points):
            v1, v2 = generate_circle_vectors(pt)
            with self.subTest(msg=f'Circle vector = {v2}, '
                                  f'expected = {good_v2s[i, :]}',
                              circle_point=pt):
                self.assertAlmostEqual(np.linalg.norm(v1 - np.array([-1, 0])),
                                       0.0)
                self.assertAlmostEqual(np.linalg.norm(v2 - good_v2s[i, :]),
                                       0.0)

    def test_find_intersection(self):
        # Unit circle with x-axis
        b, m = [0, 0]
        *z, r = [0, 0, 1]

        x_t, y_t = find_intersection([b, m], [*z, r])
        self.assertEqual(y_t, 0)
        self.assertEqual(x_t, 1)

        # Unit circle with y = x
        b, m = [0, 1]
        *z, r = [0, 0, 1]

        x_t, y_t = find_intersection([b, m], [*z, r])
        self.assertAlmostEqual(y_t, 0)
        self.assertAlmostEqual(x_t, 1)

        b, m = [-0.5, 0]
        *z, r = [0, 0, 1]

        x_t, y_t = find_intersection([b, m], [*z, r])
        self.assertAlmostEqual(y_t, -0.5)
        self.assertAlmostEqual(x_t, np.sqrt(3) / 2)

        b, m = [0.5, 0]
        *z, r = [0, 0, 1]

        x_t, y_t = find_intersection([b, m], [*z, r])
        self.assertAlmostEqual(y_t, 0.5)
        self.assertAlmostEqual(x_t, np.sqrt(3) / 2)

        b, m = [1.5, 0]
        *z, r = [0, 0, 1]

        with self.assertRaises(ValueError):
            x_t, y_t = find_intersection([b, m], [*z, r])




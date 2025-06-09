# -*- coding: utf-8 -*-
""" Implementation of color transfer in python with mask support.

Papers: 
    Color Transfer between Images. (2001)
    Automated colour grading using colour distribution transfer. (2007) 
Referenced Implementations:
    https://github.com/chia56028/Color-Transfer-between-Images
    https://github.com/frcs/colour-transfer
"""

import os
import time

import cv2
import numpy as np
from python_color_transfer.utils import Rotations


class ColorTransfer:
    """ Methods for color transfer of images. """

    def __init__(self, eps=1e-6, m=6, c=3):
        """Hyper parameters.

        Attributes:
            c: dim of rotation matrix, 3 for ordinary img.
            m: num of random orthogonal rotation matrices.
            eps: prevents from zero dividing.
        """
        self.eps = eps
        if c == 3:
            self.rotation_matrices = Rotations.optimal_rotations()
        else:
            self.rotation_matrices = Rotations.random_rotations(m, c=c)
        self.RG = Regrain()

    def lab_transfer(self, img_arr_in=None, img_arr_ref=None, mask_in=None, mask_ref=None):
        """Convert img from RGB space to LAB space, apply mean std transfer,
        then convert back.
        Args:
            img_arr_in: BGR numpy array of input image.
            img_arr_ref: BGR numpy array of reference image.
            mask_in: Binary mask for the input image.
            mask_ref: Binary mask for the reference image.
        Returns:
            img_arr_out: Transferred BGR numpy array of input image.
        """
        lab_in = cv2.cvtColor(img_arr_in, cv2.COLOR_BGR2LAB)
        lab_ref = cv2.cvtColor(img_arr_ref, cv2.COLOR_BGR2LAB)

        if mask_in is not None and mask_ref is not None:
            lab_out = self.mean_std_transfer(lab_in, lab_ref, mask_in, mask_ref)
        else:
            lab_out = self.mean_std_transfer(lab_in, lab_ref)

        img_arr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
        return img_arr_out

    def mean_std_transfer(self, img_arr_in, img_arr_ref, mask_in=None, mask_ref=None):
        """Adapt img_arr_in's (mean, std) to img_arr_ref's (mean, std).

        img_o = (img_i - mean(img_i)) / std(img_i) * std(img_r) + mean(img_r).
        Args:
            img_arr_in: LAB numpy array of input image.
            img_arr_ref: LAB numpy array of reference image.
            mask_in: Binary mask for the input image.
            mask_ref: Binary mask for the reference image.
        Returns:
            img_arr_out: Transferred LAB numpy array of input image.
        """
        if mask_in is not None:
            masked_in = img_arr_in[mask_in > 0]
        else:
            masked_in = img_arr_in

        if mask_ref is not None:
            masked_ref = img_arr_ref[mask_ref > 0]
        else:
            masked_ref = img_arr_ref

        mean_in = np.mean(masked_in, axis=0)
        mean_ref = np.mean(masked_ref, axis=0)
        std_in = np.std(masked_in, axis=0)
        std_ref = np.std(masked_ref, axis=0)

        img_arr_out = (img_arr_in - mean_in) / (std_in + self.eps) * std_ref + mean_ref
        img_arr_out = np.clip(img_arr_out, 0, 255).astype("uint8")

        return img_arr_out

    def pdf_transfer(self, img_arr_in=None, img_arr_ref=None, mask_in=None, mask_ref=None, regrain=False):
        """Apply probability density function transfer.

        Args:
            img_arr_in: BGR numpy array of input image.
            img_arr_ref: BGR numpy array of reference image.
            mask_in: Binary mask for the input image.
            mask_ref: Binary mask for the reference image.
            regrain: Boolean indicating whether to apply regraining.
        Returns:
            img_arr_out: Transferred BGR numpy array of input image.
        """
        [h, w, c] = img_arr_in.shape
        reshape_arr_in = img_arr_in.reshape(-1, c).transpose() / 255.0
        reshape_arr_ref = img_arr_ref.reshape(-1, c).transpose() / 255.0

        if mask_in is not None:
            mask_in_flat = mask_in.flatten()
            reshape_arr_in = reshape_arr_in[:, mask_in_flat > 0]

        if mask_ref is not None:
            mask_ref_flat = mask_ref.flatten()
            reshape_arr_ref = reshape_arr_ref[:, mask_ref_flat > 0]

        reshape_arr_out = self.pdf_transfer_nd(arr_in=reshape_arr_in, arr_ref=reshape_arr_ref)

        # Create a full-size output array and populate only the masked region
        reshape_arr_out_full = np.zeros((c, h * w))
        if mask_in is not None:
            mask_in_flat = mask_in.flatten()
            reshape_arr_out_full[:, mask_in_flat > 0] = reshape_arr_out
        else:
            reshape_arr_out_full = reshape_arr_out

        reshape_arr_out = np.clip(reshape_arr_out_full, 0, 1) * 255.0
        img_arr_out = reshape_arr_out.transpose().reshape(h, w, c).astype("uint8")

        if regrain:
            img_arr_out = self.RG.regrain(img_arr_in=img_arr_in, img_arr_col=img_arr_out)
        return img_arr_out

    def pdf_transfer_nd(self, arr_in=None, arr_ref=None, step_size=1):
        """Apply n-dim probability density function transfer.

        Args:
            arr_in: shape=(n, x).
            arr_ref: shape=(n, x).
            step_size: arr = arr + step_size * delta_arr.
        Returns:
            arr_out: shape=(n, x).
        """
        # n times of 1d-pdf-transfer
        arr_out = np.array(arr_in)
        for rotation_matrix in self.rotation_matrices:
            rot_arr_in = np.matmul(rotation_matrix, arr_out)
            rot_arr_ref = np.matmul(rotation_matrix, arr_ref)
            rot_arr_out = np.zeros(rot_arr_in.shape)
            for i in range(rot_arr_out.shape[0]):
                rot_arr_out[i] = self._pdf_transfer_1d(rot_arr_in[i],
                                                       rot_arr_ref[i])
            rot_delta_arr = rot_arr_out - rot_arr_in
            delta_arr = np.matmul(
                rotation_matrix.transpose(), rot_delta_arr
            )
            arr_out = step_size * delta_arr + arr_out
        return arr_out

    def _pdf_transfer_1d(self, arr_in=None, arr_ref=None, n=300):
        """Apply 1-dim probability density function transfer.

        Args:
            arr_in: 1d numpy input array.
            arr_ref: 1d numpy reference array.
            n: discretization num of distribution of image's pixels.
        Returns:
            arr_out: Transferred input array.
        """

        arr = np.concatenate((arr_in, arr_ref))
        # discretization as histogram
        min_v = arr.min() - self.eps
        max_v = arr.max() + self.eps
        xs = np.array(
            [min_v + (max_v - min_v) * i / n for i in range(n + 1)])
        hist_in, _ = np.histogram(arr_in, xs)
        hist_ref, _ = np.histogram(arr_ref, xs)
        xs = xs[:-1]
        # compute probability distribution
        cum_in = np.cumsum(hist_in)
        cum_ref = np.cumsum(hist_ref)
        d_in = cum_in / cum_in[-1]
        d_ref = cum_ref / cum_ref[-1]
        # transfer
        t_d_in = np.interp(d_in, d_ref, xs)
        t_d_in[d_in <= d_ref[0]] = min_v
        t_d_in[d_in >= d_ref[-1]] = max_v
        arr_out = np.interp(arr_in, xs, t_d_in)
        return arr_out


class Regrain:

    def __init__(self, smoothness=1):
        """To understand the meaning of these params, refer to paper07."""
        self.nbits = [4, 16, 32, 64, 64, 64]
        self.smoothness = smoothness
        self.level = 0

    def regrain(self, img_arr_in=None, img_arr_col=None):
        """Keep gradient of img_arr_in and color of img_arr_col. """
        img_arr_in = img_arr_in / 255.0
        img_arr_col = img_arr_col / 255.0
        img_arr_out = np.array(img_arr_in)
        img_arr_out = self.regrain_rec(img_arr_out, img_arr_in, img_arr_col, self.nbits, self.level)
        img_arr_out = np.clip(img_arr_out, 0, 1) * 255.0
        return img_arr_out.astype("uint8")

    def regrain_rec(self, img_arr_out, img_arr_in, img_arr_col, nbits, level):
        """Direct translation of matlab code. """

        [h, w, _] = img_arr_in.shape
        h2 = (h + 1) // 2
        w2 = (w + 1) // 2
        if len(nbits) > 1 and h2 > 20 and w2 > 20:
            if img_arr_in.size == 0 or img_arr_col.size == 0 or img_arr_out.size == 0:
                return img_arr_out
            resize_arr_in = cv2.resize(img_arr_in, (w2, h2), interpolation=cv2.INTER_LINEAR)
            resize_arr_col = cv2.resize(img_arr_col, (w2, h2), interpolation=cv2.INTER_LINEAR)
            resize_arr_out = cv2.resize(img_arr_out, (w2, h2), interpolation=cv2.INTER_LINEAR)
            resize_arr_out = self.regrain_rec(resize_arr_out, resize_arr_in, resize_arr_col, nbits[1:], level + 1)
            img_arr_out = cv2.resize(resize_arr_out, (w, h), interpolation=cv2.INTER_LINEAR)
        img_arr_out = self.solve(img_arr_out, img_arr_in, img_arr_col, nbits[0], level)
        return img_arr_out

    def solve(self, img_arr_out, img_arr_in, img_arr_col, nbit, level, eps=1e-6):
        """Direct translation of matlab code. """

        [width, height, c] = img_arr_in.shape
        first_pad_0 = lambda arr: np.concatenate((arr[:1, :], arr[:-1, :]), axis=0)
        first_pad_1 = lambda arr: np.concatenate((arr[:, :1], arr[:, :-1]), axis=1)
        last_pad_0 = lambda arr: np.concatenate((arr[1:, :], arr[-1:, :]), axis=0)
        last_pad_1 = lambda arr: np.concatenate((arr[:, 1:], arr[:, -1:]), axis=1)

        delta_x = last_pad_1(img_arr_in) - first_pad_1(img_arr_in)
        delta_y = last_pad_0(img_arr_in) - first_pad_0(img_arr_in)
        delta = np.sqrt((delta_x**2 + delta_y**2).sum(axis=2, keepdims=True))

        psi = 256 * delta / 5
        psi[psi > 1] = 1
        phi = 30 * 2**(-level) / (1 + 10 * delta / self.smoothness)

        phi1 = (last_pad_1(phi) + phi) / 2
        phi2 = (last_pad_0(phi) + phi) / 2
        phi3 = (first_pad_1(phi) + phi) / 2
        phi4 = (first_pad_0(phi) + phi) / 2

        rho = 1 / 5.0
        for i in range(nbit):
            den = psi + phi1 + phi2 + phi3 + phi4
            num = (
                np.tile(psi, [1, 1, c]) * img_arr_col +
                np.tile(phi1, [1, 1, c]) *
                (last_pad_1(img_arr_out) - last_pad_1(img_arr_in) + img_arr_in)
                + np.tile(phi2, [1, 1, c]) *
                (last_pad_0(img_arr_out) - last_pad_0(img_arr_in) + img_arr_in)
                + np.tile(phi3, [1, 1, c]) *
                (first_pad_1(img_arr_out) - first_pad_1(img_arr_in) + img_arr_in) +
                np.tile(phi4, [1, 1, c]) *
                (first_pad_0(img_arr_out) - first_pad_0(img_arr_in) + img_arr_in))
            img_arr_out = (num / np.tile(den + eps, [1, 1, c]) * (1 - rho) + rho * img_arr_out)
        return img_arr_out

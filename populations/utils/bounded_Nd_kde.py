#!/opt/local/bin/python

####!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A bounded N-D KDE class for all of your bounded N-D KDE needs.
"""

import numpy as np
from scipy.stats import gaussian_kde as kde
import itertools

class Bounded_Nd_kde(kde):
    r"""Represents an N-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain."""

    def __init__(self, pts, bounds=None, *args, **kwargs):
        """Initialize with the given bounds.  ``bounds`` should have
        same dimensions as pts, or can be supplied as ``None``.  Extra
        parameters are passed to :class:`gaussian_kde`.

        :param pts must have shape (# of dims, # of data)

        :param bounds: The lower and upper domain boundary.
               Must be list of tuples with same number of dimensions 
               as pts if not None.
               For one-sided bounds, supply ``None``
        """

        pts = np.atleast_2d(pts)

        assert pts.ndim == 2, 'Bounded_kde pts array can only be two-dimensional'

        if bounds is not None:
            bounds = np.atleast_2d(bounds)
            assert np.asarray(bounds).shape == (pts.shape[0],2), 'If bounds are supplied, must have same dimensions as pts'    

        super(Bounded_Nd_kde, self).__init__(pts, *args, **kwargs)

        self._Ndim = pts.shape[0]
        self._bounds = bounds

    @property
    def Ndim(self):
        """The number of dimensions of the data."""
        return self._Ndim

    @property
    def bounds(self):
        """The lower and upper bounds of the N domains."""
        return self._bounds

    def evaluate(self, pts):
        """Return an estimate of the density evaluated at the given points."""
        pts = np.atleast_2d(pts)
        assert pts.ndim == 2, 'points must be two-dimensional'
        assert pts.shape[0] == self.Ndim, 'evaluation points must be same number of dimensions as kde'

        # inherit original instance of kde
        pdf = super(Bounded_Nd_kde, self).evaluate(pts)

        # keep track of extra volume analyzed
        original_vol = np.sum(pdf)
        extra_vol = 0

        ### first, reflect over the 2n sides of the hypercube
        if self.bounds is not None:
            for dim in np.arange(self.Ndim):
                for idx, edge in enumerate(self.bounds[dim]):
                    if edge is not None:
                        refl_pts = pts.copy() #Ndim x Npts
                        refl_pts[dim] = 2*edge - refl_pts[dim] 
                        new_pdf = super(Bounded_Nd_kde, self).evaluate(refl_pts)
                        pdf += new_pdf
                        extra_vol += np.sum(new_pdf)
                
        ### next, reflect over the 2^n vertices of the hypercube
        if self.bounds is not None:
            # keep track of lower (0) and upper (1) bounds
            which_bound = np.asarray([[0,1]]*len(self.bounds))
            # permutate over all the vertices
            for ul, vertex in zip(list(itertools.product(*which_bound)), \
                                list(itertools.product(*self.bounds))):
                # if any dimension at that vertex is unbounded, continue
                if any(np.asarray(vertex) == None):
                    continue
                # determine which points at that vertex are lower and upper bounds
                lower_bounds = (np.asarray(ul) == 0)
                upper_bounds = (np.asarray(ul) == 1)
                assert all(lower_bounds == ~upper_bounds), 'cannot have the same lower and upper bound'
                # get reflected points at this vertex
                refl_pts = pts.copy() #Ndim x Npts
                for dim in np.arange(self.Ndim):
                    if lower_bounds[dim]:
                        refl_pts[dim] = 2*self.bounds[dim,0] - refl_pts[dim]
                    else:
                        refl_pts[dim] = 2*self.bounds[dim,1] - refl_pts[dim]

                new_pdf = super(Bounded_Nd_kde, self).evaluate(refl_pts)
                pdf += new_pdf
                extra_vol += np.sum(new_pdf)

        # get the final normalization factor
        # FIXME this doesn't seem to be working, come back to this
        # FIXME also...tightly bounded KDE with reflect values 
        # over the other bound...
        if (self.bounds is not None) and any(self.bounds.flatten()!=None):
            norm_fac = 1
            #norm_fac = extra_vol / (original_vol + extra_vol)
        else:
            norm_fac = 1

        return pdf, norm_fac


    def bounded_resample(self, n=1):
        """Resample kde using scipy's default resampling method.
        Reflect out-of-bound points across the boundaries of the domain."""
        resamp_pts = self.resample(size = n)

        if self.bounds is not None:
            for dim in np.arange(self.Ndim):
                # reflect over the lower and upper bounds
                edge = self.bounds[dim,0]
                if edge is not None:
                    out_of_bounds = (resamp_pts[dim] < edge)
                    resamp_pts[dim][out_of_bounds] = 2*edge - resamp_pts[dim][out_of_bounds]
                # reflect over the upper bound
                edge = self.bounds[dim,1]
                if edge is not None:
                    out_of_bounds = (resamp_pts[dim] > edge)
                    resamp_pts[dim][out_of_bounds] = 2*edge - resamp_pts[dim][out_of_bounds]

        return resamp_pts


    def __call__(self, pts):
        pts = np.atleast_2d(pts)
        out_of_bounds = np.zeros(pts.shape[1], dtype='bool')

        if self.bounds is not None:
            for dim in np.arange(self.Ndim):
                if self.bounds[dim,0] is not None:
                    out_of_bounds[pts[dim, :] < self.bounds[dim,0]] = True
                if self.bounds[dim,1] is not None:
                    out_of_bounds[pts[dim, :] > self.bounds[dim,1]] = True

        results, norm_fac = self.evaluate(pts)
        # set out-of-bound evaluations to 0
        results[out_of_bounds] = 0.
        return norm_fac*results

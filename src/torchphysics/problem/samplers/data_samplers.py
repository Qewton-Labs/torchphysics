"""File with samplers that handle external created data.
E.g. measurements or validation data computed with other methods.
"""
import torch

from .sampler_base import PointSampler
from ..spaces import Points

import time

class DataSampler(PointSampler):
    """A sampler that processes external created data points.

    Parameters
    ----------
    points : torchphysics.spaces.points or dict
        The data points that this data sampler should pass to a condition.
        Either already a torchphysics.spaces.points object or in form of
        dictionary like: {'x': tensor_for_x, 't': tensor_for_t, .....}.
        For the dictionary all tensor need to have the same batch dimension.
    n_points : int, optional
        The number of points that should be sampled from the data. 
        If n_points < len(points) we do batch sampling, where the first 
        **sample_points** call will return the first n_points, the second call
        the next n_points, and so on. At the end the we wrap around and 
        starts at the beginning of the points again. If n_points is not a
        perfect divisor of the data set length, the last batch of points
        will be of shorter length
    """

    def __init__(self, points, n_points=-1):
        if isinstance(points, Points):
            self.points = points
        elif isinstance(points, dict):
            self.points = Points.from_coordinates(points)
        else:
            raise TypeError("points should be one of Points or dict.")
        
        self.total_len = len(self.points.as_tensor)
        if n_points < 1:
            n_points = self.total_len
        elif n_points > self.total_len:
            print(f"""Sampling number was set to {n_points} while only {self.total_len} data points are available. 
                      Will sample {self.total_len} points whenever called!""")
            n_points = self.total_len
        super().__init__(n_points=n_points)
        self.current_idx = 0

    def __len__(self):
        return self.n_points
    
    def sample_points(self, params=Points.empty(), device="cpu"):
        self.points = self.points.to(device)
        
        # Take a batch of points
        return_points = self.points[
            self.current_idx * self.n_points : min(self.total_len, (self.current_idx+1) * self.n_points)
            ]
        if (self.current_idx+1) * self.n_points >= self.total_len:
            self.current_idx = 0
        else:
            self.current_idx += 1
        
        # If sampler not coupled to other samplers or parameters, we can return:
        if params.isempty:
            return return_points

        repeated_params = self._repeat_params(params, len(self))
        repeated_points = return_points.repeat(len(params))
        return repeated_points.join(repeated_params)
        # Maybe given data has more dimensions than batch and space
        # (For example evaluation on quadrature points)
        # TODO: Make more general. What happens when parameters have higher dimension?
        # What when multiple dimension in both that do not fit?
        # start_time = time.time()
        # if len(self.points.as_tensor.shape) > 2:
        #     repeated_tensor = params.as_tensor
        #     for i in range(1, len(self.points.as_tensor.shape)-1):
        #         repeated_tensor = torch.repeat_interleave(repeated_tensor.unsqueeze(-1),
        #                                                 self.points.as_tensor.shape[i],
        #                                                 dim=i)
            
        #     repeated_params = Points(repeated_tensor, params.space)
        # print("Dimension thing took", time.time() - start_time)

        # # else we have to repeat data (mesh grid of both) and join the tensors together:
        # start_time = time.time()
        # repeated_params = self._repeat_params(repeated_params, len(self))
        # print("Repeating params took", time.time() - start_time)
        # start_time = time.time()
        # repeated_points = self.points.repeat(len(params))
        # print("Repeating points took", time.time() - start_time)

        # return repeated_points.join(repeated_params)
import yaml
import torch
import argparse
import numpy as np
from scipy import interpolate

mappers_dataset_folders = {'airs': 'AIRS', 'pascal': 'PASCAL', 'suim': 'SUIM', 'retouch': 'RETOUCH', 'oct': 'RETOUCH',
                           'spleen': 'SPLEEN'}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'T', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def retrieve_points_surface(surf, split_point, split_line, split_block, column='m_f_score'):
    new_surf = surf[(surf['point'] <= split_point) & (surf['line'] <= split_line) & (surf['block'] <= split_block)]
    point = new_surf['point'].to_numpy()
    line = new_surf['line'].to_numpy()
    block = new_surf['block'].to_numpy()
    values = new_surf[column].to_numpy()
    return point, line, block, values


class SplineInterpolation:

    def __init__(self, config, surf, column='Dice'):
        """
        Get interpolator for the splines.
        X: Classification
        Y: Segmentation
        Args:
            surf:
            column:

        Returns:

        """
        xy = surf[['split_cls', 'split_seg']].to_numpy()
        Z = surf[column].to_numpy()
        self.surf = surf
        kx = int(config['surface_interpolation']['degree'])
        ky = kx
        self.tck = interpolate.bisplrep(xy[:, 0], xy[:, 1], Z, kx=kx, ky=ky)

        cls_points = list(set(self.surf['split_cls'].to_numpy()))
        seg_points = list(set(self.surf['split_seg'].to_numpy()))
        cls_min = min(cls_points)
        seg_min = min(seg_points)
        self.min_coords = (cls_min, seg_min)
        self.max_coords = (100, 100)

    def interpolate_surf(self, num_points_coord, max_point=(100, 100)):
        xx = np.linspace(self.min_coords[0], max_point[0], num_points_coord)
        yy = np.linspace(self.min_coords[1], max_point[1], num_points_coord)
        xnew_edges, ynew_edges = np.meshgrid(xx, yy)
        xnew = xnew_edges[:-1, :-1] + np.diff(xnew_edges[:2, 0])[0] / 2.
        ynew = ynew_edges[:-1, :-1] + np.diff(ynew_edges[0, :2])[0] / 2.
        znew = interpolate.bisplev(xnew[0, :], ynew[:, 0], self.tck)
        return xnew, ynew, znew

    def interpolate_value(self, coords, return_tensor=False):
        if not self._check_point(coords):
            return None

        value = interpolate.bisplev(coords[0], coords[1], self.tck)
        if return_tensor:
            return torch.tensor([value], dtype=torch.float)
        return value

    def interpolate(self, pts):
        """
        There is something weird related to bisplev and needing a grid. I don't have that, so for loop. Sorry not sorry
        Args:
            pts:

        Returns:

        """
        values = []
        for p in pts:
            values.append(interpolate.bisplev(p[0], p[1], self.tck))
        return torch.tensor(values, dtype=torch.float)

    def _check_point(self, coords):
        if coords[0] > self.max_coords[0] or coords[0] < self.min_coords[0]:
            return False
        if coords[1] > self.max_coords[1] or coords[1] < self.min_coords[1]:
            return False
        return True

    def __call__(self, coords, **kwargs):
        return self.interpolate_value(coords, **kwargs)


class TriangleInterpolation:
    def __init__(self, config, surf, column='Dice'):
        self.surf = surf
        cls_points = list(set(surf['split_cls'].to_numpy()))
        seg_points = list(set(surf['split_seg'].to_numpy()))
        cls_points.sort()
        seg_points.sort()
        surf = surf.set_index(['split_cls', 'split_seg'])

        triangles = []
        triangle_values = []

        for i_cls in range(len(cls_points) - 1):
            for i_seg in range(len(seg_points) - 1):
                triangles.append([
                    [cls_points[i_cls], seg_points[i_seg]],
                    [cls_points[i_cls + 1], seg_points[i_seg]],
                    [cls_points[i_cls], seg_points[i_seg + 1]],
                ])
                triangle_values.append([
                    surf.loc[cls_points[i_cls], seg_points[i_seg]][column].item(),
                    surf.loc[cls_points[i_cls + 1], seg_points[i_seg]][column].item(),
                    surf.loc[cls_points[i_cls], seg_points[i_seg + 1]][column].item(),
                ])

                triangles.append([
                    [cls_points[i_cls + 1], seg_points[i_seg + 1]],
                    [cls_points[i_cls + 1], seg_points[i_seg]],
                    [cls_points[i_cls], seg_points[i_seg + 1]],
                ])
                triangle_values.append([
                    surf.loc[cls_points[i_cls + 1], seg_points[i_seg + 1]][column].item(),
                    surf.loc[cls_points[i_cls + 1], seg_points[i_seg]][column].item(),
                    surf.loc[cls_points[i_cls], seg_points[i_seg + 1]][column].item(),
                ])

        self.triangles = triangles
        self.triangle_values = triangle_values
        # return triangles, triangle_values

    def find_triangle_intersection(self, pt, triangle_vertices):
        # https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(pt, triangle_vertices[0], triangle_vertices[1])
        d2 = sign(pt, triangle_vertices[1], triangle_vertices[2])
        d3 = sign(pt, triangle_vertices[2], triangle_vertices[0])

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def test_triangles(self, pt):
        for i, triangle in enumerate(self.triangles):
            if self.find_triangle_intersection(pt, triangle):
                return i

    def barycentric(self, pt, vertices):
        v1 = np.array([
            vertices[2][0] - vertices[0][0],
            vertices[1][0] - vertices[0][0],
            vertices[0][0] - pt[0],
        ])
        v2 = np.array([
            vertices[2][1] - vertices[0][1],
            vertices[1][1] - vertices[0][1],
            vertices[0][1] - pt[1],
        ])

        u = np.cross(v1, v2)
        return np.array([1 - (u[0] + u[1]) / u[2], u[1] / u[2], u[0] / u[2]])

    def barycentric_interpolation(self, barycenter, vertex_val, return_tensor=False):
        den_value = np.sum(barycenter)
        value = (vertex_val[0] * barycenter[0] + vertex_val[1] * barycenter[1] + vertex_val[2] * barycenter[
            2]) / den_value
        if return_tensor:
            return torch.tensor([value], dtype=torch.float)
        return value

    def interpolate_value(self, coords, return_tensor=False):
        idx_triangle = self.test_triangles([coords[0], coords[1]])
        if idx_triangle is None:
            return None
        barycenter = self.barycentric([coords[0], coords[1]], self.triangles[idx_triangle])
        y_corner = self.barycentric_interpolation(barycenter, self.triangle_values[idx_triangle],
                                                  return_tensor=return_tensor)
        return y_corner

    def __call__(self, coords, **kwargs):
        return self.interpolate_value(coords, **kwargs)

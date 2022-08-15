#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper functions to construct matrices for CRA
"""

import numpy as np
import math as mt

from compas.geometry import cross_vectors
from scipy.sparse import csr_matrix

__author__ = "Gene Ting-Chun Kao"
__email__ = "kao@arch.ethz.ch"

__all__ = ['equilibrium_setup',
           'friction_setup',
           'external_force_setup',
           'make_aeq',
           'make_afr',
           'unit_basis',
           'num_vertices',
           'num_free',
           'free_nodes']


def equilibrium_setup(assembly, penalty=False):
    """Set up equilibrium matrix."""
    free = free_nodes(assembly)
    aeq = make_aeq(assembly, penalty=penalty)

    # TODO: make a function to remove fixed node from matrix
    aeq = aeq[[index * 6 + i for index in free for i in range(6)], :]
    print("Aeq: ", aeq.shape)

    return aeq


def external_force_setup(assembly, density):
    """Set up external force vector."""
    free = free_nodes(assembly)

    num_nodes = assembly.graph.number_of_nodes()
    key_index = {key: index for index, key in enumerate(assembly.graph.nodes())}

    p = [[0, 0, 0, 0, 0, 0] for i in range(num_nodes)]
    for node in assembly.graph.nodes():
        block = assembly.node_block(node)
        index = key_index[node]
        p[index][2] = -block.volume() * density

    p = np.array(p, dtype=float)
    p = p[free, :].reshape((-1, 1), order='C')

    return p


def friction_setup(assembly, mu, penalty=False):
    """Set up friction matrix."""
    v_count = num_vertices(assembly)
    afr = make_afr(v_count, fcon_number=8, mu=mu, penalty=penalty, friction_net=False)
    print("Afr: ", afr.shape)

    return afr


def num_free(assembly):
    """Return number of free blocks."""
    return len(list(key for key in assembly.graph.nodes_where({'is_support': False})))


def free_nodes(assembly):
    """Return free and fixed node list."""
    num_nodes = assembly.graph.number_of_nodes()
    key_index = {key: index for index, key in enumerate(assembly.graph.nodes())}

    fixed = list(key for key in assembly.graph.nodes_where({'is_support': True}))
    fixed = [key_index[key] for key in fixed]
    free = list(set(range(num_nodes)) - set(fixed))
    return free


def num_vertices(assembly):
    """Total number of vertices."""
    v_count = 0
    for b_j, b_k in assembly.graph.edges(False):
        for interface in assembly.graph.edge_attribute((b_j, b_k), 'interfaces'):
            v_count += len(interface.points)
    return v_count


def make_aeq(assembly, flip=False, penalty=False):
    """Create equilibrium matrix Aeq or penalty formulation matrix Aeq@B."""
    rows = []
    cols = []
    data = []

    count = 0
    shift = 3
    if penalty:
        shift = 4

    key_index = {key: index for index, key in enumerate(assembly.graph.nodes())}

    for b_j, b_k in assembly.graph.edges(False):
        bj_center = assembly.graph.node_attribute(b_j, 'block').center()
        bk_center = assembly.graph.node_attribute(b_k, 'block').center()

        for interface in assembly.graph.edge_attribute((b_j, b_k), 'interfaces'):
            # B_j
            block_rows, block_cols, block_data = aeq_block(interface, bj_center, not flip, penalty)
            # shift rows and cols
            rows += [row + 6 * key_index[b_j] for row in block_rows]
            cols += [col + shift * count for col in block_cols]
            data += block_data
            # B_k
            block_rows, block_cols, block_data = aeq_block(interface, bk_center, flip, penalty)
            # shift rows and cols
            rows += [row + 6 * key_index[b_k] for row in block_rows]
            cols += [col + shift * count for col in block_cols]
            data += block_data
            count += len(interface.points)

    return csr_matrix((data, (rows, cols)))


def aeq_block(interface, center, reverse, penalty=False):
    """Helper function to create Aeq and Aeq@B."""
    shift = 3
    if penalty:
        shift = 4

    rows, cols, data = [], [], []
    u = interface.frame.xaxis
    v = interface.frame.yaxis
    w = interface.frame.zaxis

    if reverse:
        u = [-1.0 * axis for axis in u]
        v = [-1.0 * axis for axis in v]
        w = [-1.0 * axis for axis in w]

    fx = [w[0], -w[0], u[0], v[0]] if penalty else [w[0], u[0], v[0]]
    fy = [w[1], -w[1], u[1], v[1]] if penalty else [w[1], u[1], v[1]]
    fz = [w[2], -w[2], u[2], v[2]] if penalty else [w[2], u[2], v[2]]

    for i, xyz in enumerate(interface.points):
        # coordinates of interface point relative to block mass center
        rxyz = [xyz[axis] - center[axis] for axis in range(3)]
        # moments
        mu = cross_vectors(rxyz, u)
        mv = cross_vectors(rxyz, v)
        mw = cross_vectors(rxyz, w)

        mx = [mw[0], -mw[0], mu[0], mv[0]] if penalty else [mw[0], mu[0], mv[0]]
        my = [mw[1], -mw[1], mu[1], mv[1]] if penalty else [mw[1], mu[1], mv[1]]
        mz = [mw[2], -mw[2], mu[2], mv[2]] if penalty else [mw[2], mu[2], mv[2]]

        for j in range(shift):
            col = j + (i * shift)
            if fx[j]:
                rows.append(0)
                cols.append(col)
                data.append(fx[j])
            if fy[j]:
                rows.append(1)
                cols.append(col)
                data.append(fy[j])
            if fz[j]:
                rows.append(2)
                cols.append(col)
                data.append(fz[j])
            if mx[j]:
                rows.append(3)
                cols.append(col)
                data.append(mx[j])
            if my[j]:
                rows.append(4)
                cols.append(col)
                data.append(my[j])
            if mz[j]:
                rows.append(5)
                cols.append(col)
                data.append(mz[j])

    return rows, cols, data


def unit_basis(assembly, penalty=False):
    """Create interface reference system as unit basis."""
    data = []
    for edge in assembly.graph.edges():
        interfaces = assembly.graph.edge_attribute(edge, 'interfaces')

        for interface in interfaces:
            u = interface.frame.xaxis
            v = interface.frame.yaxis
            w = interface.frame.zaxis
            for i in range(len(interface.points)):
                data.append([w[0], w[1], w[2]])
                if penalty:
                    data.append([-w[0], -w[1], -w[2]])
                data.append([u[0], u[1], u[2]])
                data.append([v[0], v[1], v[2]])
    return np.array(data)


def make_afr(total_vcount, fcon_number=8, mu=0.8, penalty=False, friction_net=False):
    """Create friction matrix Afr and Afr@B."""
    if penalty:
        return _make_afr_b(total_vcount, fcon_number=fcon_number, mu=mu, friction_net=friction_net)
    return make_afr(total_vcount, fcon_number=8, mu=mu)


def _make_afr(total_vcount, fcon_number=8, mu=0.8):
    """Create friction matrix Afr."""
    rows = []
    cols = []
    data = []
    c_8 = 1.0 / mt.sqrt(2.0)
    c_16max = mt.cos(mt.radians(22.5))
    c_16min = mt.sin(mt.radians(22.5))
    i, j = 0, 0

    for n in range(total_vcount):
        rows += [i + 0, i + 0, i + 1, i + 1, i + 2, i + 2, i + 3, i + 3]
        cols += [j, j + 1, j, j + 1, j, j + 2, j, j + 2]
        data += [-mu, 1, -mu, -1, -mu, 1, -mu, -1]

        if fcon_number != 8 and fcon_number != 16:
            i += 4

        if fcon_number == 8 or fcon_number == 16:
            rows += [i + 4, i + 4, i + 4]
            cols += [j, j + 1, j + 2]
            data += [-mu, c_8, c_8]
            rows += [i + 5, i + 5, i + 5]
            cols += [j, j + 1, j + 2]
            data += [-mu, -c_8, -c_8]
            rows += [i + 6, i + 6, i + 6]
            cols += [j, j + 1, j + 2]
            data += [-mu, c_8, -c_8]
            rows += [i + 7, i + 7, i + 7]
            cols += [j, j + 1, j + 2]
            data += [-mu, -c_8, c_8]
            i += 8

        if fcon_number == 16:
            rows += [i + 8, i + 8, i + 8]
            cols += [j, j + 1, j + 2]
            data += [-mu, c_16max, c_16min]
            rows += [i + 9, i + 9, i + 9]
            cols += [j, j + 1, j + 2]
            data += [-mu, c_16min, c_16max]
            rows += [i + 10, i + 10, i + 10]
            cols += [j, j + 1, j + 2]
            data += [-mu, -c_16min, c_16max]
            rows += [i + 11, i + 11, i + 11]
            cols += [j, j + 1, j + 2]
            data += [-mu, -c_16max, c_16min]
            rows += [i + 12, i + 12, i + 12]
            cols += [j, j + 1, j + 2]
            data += [-mu, -c_16max, -c_16min]
            rows += [i + 13, i + 13, i + 13]
            cols += [j, j + 1, j + 2]
            data += [-mu, -c_16min, -c_16max]
            rows += [i + 14, i + 14, i + 14]
            cols += [j, j + 1, j + 2]
            data += [-mu, c_16min, -c_16max]
            rows += [i + 15, i + 15, i + 15]
            cols += [j, j + 1, j + 2]
            data += [-mu, c_16max, -c_16min]
            i += 16

        j += 3

    return csr_matrix((data, (rows, cols)))


def _make_afr_b(total_vcount, fcon_number=8, mu=0.8, friction_net=False):
    """Create friction matrix Afr@B."""
    rows = []
    cols = []
    data = []
    c_8 = 1.0 / mt.sqrt(2.0)
    i = 0
    j = 0

    for n in range(total_vcount):
        # friction4
        if friction_net:
            rows += [i + 0, i + 0, i + 0, i + 1, i + 1, i + 1,
                     i + 2, i + 2, i + 2, i + 3, i + 3, i + 3]
            cols += [j, j + 1, j + 2, j, j + 1, j + 2,
                     j, j + 1, j + 3, j, j + 1, j + 3]
            data += [-mu, mu, 1, -mu, mu, -1, -mu, mu, 1, -mu, mu, -1]
        else:
            rows += [i + 0, i + 0, i + 1, i + 1, i + 2, i + 2, i + 3, i + 3]
            cols += [j, j + 2, j, j + 2, j, j + 3, j, j + 3]
            data += [-mu, 1, -mu, -1, -mu, 1, -mu, -1]

        if fcon_number != 8:
            i += 4
        if fcon_number == 8:
            if friction_net:
                rows += [i + 4, i + 4, i + 4, i + 4]
                cols += [j, j + 1, j + 2, j + 3]
                data += [-mu, mu, c_8, c_8]

                rows += [i + 5, i + 5, i + 5, i + 5]
                cols += [j, j + 1, j + 2, j + 3]
                data += [-mu, mu, -c_8, -c_8]

                rows += [i + 6, i + 6, i + 6, i + 6]
                cols += [j, j + 1, j + 2, j + 3]
                data += [-mu, mu, c_8, -c_8]

                rows += [i + 7, i + 7, i + 7, i + 7]
                cols += [j, j + 1, j + 2, j + 3]
                data += [-mu, mu, -c_8, c_8]
            else:
                rows += [i + 4, i + 4, i + 4]
                cols += [j, j + 2, j + 3]
                data += [-mu, c_8, c_8]

                rows += [i + 5, i + 5, i + 5]
                cols += [j, j + 2, j + 3]
                data += [-mu, -c_8, -c_8]

                rows += [i + 6, i + 6, i + 6]
                cols += [j, j + 2, j + 3]
                data += [-mu, c_8, -c_8]

                rows += [i + 7, i + 7, i + 7]
                cols += [j, j + 2, j + 3]
                data += [-mu, -c_8, c_8]

            i += 8

        j += 4

    return csr_matrix((data, (rows, cols)))


if __name__ == '__main__':
    pass

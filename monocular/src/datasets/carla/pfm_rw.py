#####################################################################
# This file is part of the 4D Light Field Benchmark.                #
#                                                                   #
# This work is licensed under the Creative Commons                  #
# Attribution-NonCommercial-ShareAlike 4.0 International License.   #
# To view a copy of this license,                                   #
# visit http://creativecommons.org/licenses/by-nc-sa/4.0/.          #
#####################################################################

import os
import sys
import torch
import numpy as np


def write_pfm(data, fpath, scale=1, file_identifier=b"Pf", dtype="float32"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    data = np.flipud(data)
    height, width = np.shape(data)[:2]
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    endianess = data.dtype.byteorder
    # print(endianess)

    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        scale *= -1

    with open(fpath, 'wb') as file:
        # print(file_identifier + b'\n')
        file.write(file_identifier + b'\n')
        file.write(b'%d %d\n' % (width, height))
        file.write(b'%d\n' % scale)
        file.write(values)


def read_pfm(fpath, expected_identifier=b"Pf"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    with open(fpath, 'rb') as f:
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (
                expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f).decode('ascii')
            # print(line_dimensions)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception('Could not parse dimensions: "%s". '
                            'Expected "width height", e.g. "512 512".' % line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise Exception('Could not parse max value / endianess information: "%s". '
                            'Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception(
                'Invalid binary values. Could not create %dx%d array from input.' % (height, width))

        return data


def _get_next_line(f):
    next_line = f.readline().rstrip()
    # ignore comments
    while next_line.startswith(b'#'):
        next_line = f.readline().rstrip()
    return next_line


if __name__ == '__main__':
    data = torch.rand(256, 256).mul(10000.0)
    data_np = data.numpy()
    write_pfm(data_np, 'temp.pfm')
    data_est_np = torch.from_numpy(read_pfm('temp.pfm').copy()).float()
    print(torch.min(torch.abs(data_est_np - data)))
    print(torch.max(torch.abs(data_est_np - data)))
    print(torch.mean(torch.abs(data_est_np - data)))
    print(data_est_np.min(), data.min())
    print(data_est_np.max(), data.max())

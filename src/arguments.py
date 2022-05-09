# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation script')
    # General arguments
    parser.add_argument('-c', '--config', default=None, type=str, metavar='NAME', help='the path of the config file')
    args = parser.parse_args()
    return args

#!/bin/bash
# Copyright 2015   David Snyder
#           2019   Lantian Li
#           2019   Yang Zhang
# Apache 2.0.

export KALDI_ROOT=${replace it by your kaldi root path}
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH

[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
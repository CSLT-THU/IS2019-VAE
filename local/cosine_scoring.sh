#!/bin/bash
# Copyright 2015   David Snyder
#           2019   Lantian Li
# Apache 2.0.
#
# This script trains an LDA transform and does cosine scoring.

#echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
    echo "Usage: $0 <enroll-data-dir> <test-data-dir> <trials-file> <scores-dir>"
fi

enroll_data_dir=$1
test_data_dir=$2
trials=$3
scores_dir=$4

mkdir -p $scores_dir/log
run.pl $scores_dir/log/cosine_scoring.log \
cat $trials \| awk '{print $1" "$2}' \| \
ivector-compute-dot-products - \
"ark:ivector-normalize-length ark:${enroll_data_dir}/xvector.ark ark:- |" \
"ark:ivector-normalize-length ark:${test_data_dir}/xvector.ark ark:- |" \
$scores_dir/cosine_scores || exit 1;

#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
#             2019   Tsinghua University (Author: Lantian Li)
#             2019   Yang Zhang
# Apache 2.0.
#
# This is an x-vector-based recipe for Speakers in the Wild (SITW).

. ./path.sh


for sub in dev eval; do
    # Cosine metric.
    echo "Test on SITW $sub:"
    
    local/cosine_scoring.sh data/sitw_$sub/enroll \
                            data/sitw_$sub/test \
                            data/sitw_$sub/test/core-core.lst \
                            data/sitw_$sub/foo
    
    eer=$(paste data/sitw_$sub/test/core-core.lst data/sitw_$sub/foo/cosine_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
    echo "Cosine EER: $eer%"
    
    # Create a PLDA model and do scoring.
    local/plda_scoring.sh   data/voxceleb_combined_200000 \
                            data/sitw_$sub/enroll \
                            data/sitw_$sub/test \
                            data/sitw_$sub/test/core-core.lst \
                            data/sitw_$sub/foo
    
    eer=$(paste data/sitw_$sub/test/core-core.lst data/sitw_$sub/foo/plda_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
    echo "PLDA EER: $eer%"
    
    # Create a LDA-PLDA model and do scoring.
    for lda_dim in 150;do
        
        local/lda_plda_scoring.sh --lda-dim $lda_dim --covar-factor 0.0 \
                                    data/voxceleb_combined_200000 \
                                    data/sitw_$sub/enroll \
                                    data/sitw_$sub/test \
                                    data/sitw_$sub/test/core-core.lst \
                                    data/sitw_$sub/foo
                                    eer=$(paste data/sitw_$sub/test/core-core.lst data/sitw_$sub/foo/lda_plda_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
                                    echo "LDA_PLDA EER(${lda_dim}): $eer%"
                                    
    done
    
    # Create a PCA-PLDA model and do scoring.
    for pca_dim in 150;do
        
        local/pca_plda_scoring.sh --pca-dim $pca_dim \
                                    data/voxceleb_combined_200000 \
                                    data/sitw_$sub/enroll \
                                    data/sitw_$sub/test \
                                    data/sitw_$sub/test/core-core.lst \
                                    data/sitw_$sub/foo
        
        eer=$(paste data/sitw_$sub/test/core-core.lst data/sitw_$sub/foo/pca_plda_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
        echo "PCA_PLDA EER(${pca_dim}): $eer%"
    done
    
    echo
done

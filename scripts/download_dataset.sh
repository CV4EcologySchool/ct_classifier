#!/bin/bash

# Downloads the Caltech Camera Traps dataset:
# https://lila.science/datasets/caltech-camera-traps
#
# using azcopy:
# https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10
#
# 2022 Benjamin Kellenberger

destFolder=datasets/CaltechCT



mkdir -p $destFolder
echo "Downloading images..."
azcopy copy https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz $destFolder/.

echo "Downloading metadata..."
azcopy copy https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_annotations.tar.gz $destFolder/.

echo "Unzipping..."
tar -xvf $destFolder/eccv_18_all_images_sm.tar.gz -C $destFolder/.
tar -xvf $destFolder/eccv_18_annotations.tar.gz -C $destFolder/.

echo "Cleaning up..."
rm $destFolder/eccv_18_all_images_sm.tar.gz
rm $destFolder/eccv_18_annotations.tar.gz
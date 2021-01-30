#!/bin/bash

PROCESSOR_TYPE=$(uname -p)

sudo apt install appstream

wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-${PROCESSOR_TYPE}.AppImage
chmod +x appimagetool-${PROCESSOR_TYPE}.AppImage 

APPIMAGETOOL_PATH=./appimagetool-${PROCESSOR_TYPE}.AppImage

ARCH=$PROCESSOR_TYPE ${APPIMAGETOOL_PATH} ./AppImage

rm -f ${APPIMAGETOOL_PATH}

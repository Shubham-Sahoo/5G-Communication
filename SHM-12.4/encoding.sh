#!/bin/bash

out_file=spac_qp_28_fr1.875
mkdir $out_file

#bin/./TAppEncoderStatic -c cfg/encoder_lowdelay_scalable.cfg -c cfg/layers.cfg -c cfg/per-sequence-svc/BasketballDrive-SNR.cfg -i0 cfg/BasketballDrive_176x144_50.yuv -i1 cfg/BasketballDrive_352x288_50.yuv 

bin/./TAppEncoderStatic -c cfg/encoder_randomaccess_scalable_8_10.cfg -c cfg/layers.cfg -c cfg/per-sequence-svc/BasketballDrive-SNR.cfg -i0 cfg/BasketballDrive_176x144_50.yuv -i1 cfg/BasketballDrive_352x288_50.yuv

cd $out_file

../bin/./TAppDecoderStatic -b ../str3.bin -o0 layer_t0.yuv -olsidx 0
../bin/./TAppDecoderStatic -b ../str3.bin -o1 layer_t1.yuv -olsidx 1


ffmpeg -f rawvideo -vcodec rawvideo -s 176x144 -r 1.875 -pix_fmt yuv420p -i layer_t0.yuv -c:v libx265 -preset ultrafast -qp 0 output_t0.mp4

ffmpeg -f rawvideo -vcodec rawvideo -s 352x288 -r 1.875 -pix_fmt yuv420p -i layer_t1.yuv -c:v libx265 -preset ultrafast -qp 0 output_t1.mp4

#bin/./TAppDecoderAnalyserStatic -b cfg/str1.bin -o0 layer_s0.yuv -ls 0
#bin/./TAppDecoderAnalyserStatic -b cfg/str1.bin -o1 layer_s1.yuv -ls 1

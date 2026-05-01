%% This is a demo code to show how to generate training and testing samples from the HSI %%
clc
clear
close all

%% Step 1: generate the training and testing images from the original HSI
load('/home/descfly/zly/datasets/Chikusei/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729.mat');%% Please down the Chikusei dataset (mat format) from https://www.sal.t.u-tokyo.ac.jp/hyperdata/
%% center crop this image to size 2048 x 2048
img = chikusei(107:2154,144:2191,:);
clear chikusei;
% normalization
img = img ./ max(max(max(img)));
img = single(img);
%% select first row as test images
[H, W, C] = size(img);
test_img_size = 512;
test_pic_num = floor(W / test_img_size);
mkdir ('test');
for i = 1:test_pic_num
    left = (i - 1) * test_img_size + 1;
    right = left + test_img_size - 1;
    test = img(1:test_img_size,left:right,:);
    save(strcat('./test/Chikusei_test_', int2str(i), '.mat'),'test');
end

%% the rest left for training
mkdir ('train');
img = img((test_img_size+1):end,:,:);
save('./train/Chikusei_train.mat', 'img');


%% This is a demo code to show how to generate training and testing samples from the HSI %%
clc
clear
close all

addpath('include');

%% Step 1: generate the training and testing images from the original HSI
load('/home/descfly/zly/datasets/Pavia/Pavia.mat');%% Please down the Pavia dataset (mat format) from https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
img = pavia(:,:,:);
clear pavia;
% normalization
img = img ./ 8000;
img = single(img);
size(img);
%% select first row as test images
[H, W, C] = size(img);
test_img_size = 216;
test_pic_num = 5;
mkdir ('test');
for i = 1:test_pic_num
    top = (i - 1) * test_img_size + 1;
    bottom = top + test_img_size - 1;
    test = img(top:bottom,1:test_img_size,:);
    save(strcat('./test/Pavia_test_', int2str(i), '.mat'),'test');
end

%% the rest left for training
mkdir ('train');
img = img(:, 224:end ,:);
save('./train/Pavia_train.mat', 'img');


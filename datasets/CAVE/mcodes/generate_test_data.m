clc
clear
close all
rng(10, 'twister');

fileFolder=fullfile('/home/descfly/zly/datasets/CAVE/mcodes/test/');
dirOutput=dir(fullfile(fileFolder,'*.mat'));
fileNames={dirOutput.name};
factor = 1/4;
img_size = 512;
if factor==1/3
    img_size=504;
end
bands = 31;
gt = zeros(numel(fileNames),img_size,img_size,bands);
ms = zeros(numel(fileNames),img_size*factor,img_size*factor,bands);
ms_bicubic = zeros(numel(fileNames),img_size,img_size,bands);
cd test;
for i = 1:numel(fileNames)
    load(fileNames{i},'gt');

    %% 1. Construct 19x19 Gaussian Kernel
    sigma = 0.2 + (3 - 0.2) * rand;     % Gaussian standard deviation
    kSize = 19;                         % Kernel size (odd)
    gaussKernel = fspecial('gaussian', kSize, sigma);   % 19x19, isotropic
    % imagesc(gaussKernel);             % Visualize with color map
    % axis image off; colormap hot; colorbar;
    % title('19x19 Gaussian Kernel (\sigma=3)');
    sigma
    % if factor==1/3
    %     gt = gt(5:508,5:508,:);
    % end
    size(gt)

    imgBlur = zeros(size(gt), 'single');      % Pre-allocate
    for b = 1:size(gt,3)
        imgBlur(:,:,b) = imfilter(gt(:,:,b), gaussKernel, ...
                                  'conv', 'replicate');
    end
    img_ms = single(imresize(imgBlur, factor));

    img_ms = img_ms;

    ms = img_ms;
    ms_bicubic = single(imresize(img_ms, 1/factor));
    gt = single(gt);
    lq = single(ms);
    bic = single(ms_bicubic);
    scale = round(1/factor);
    save(strcat('/home/descfly/zly/datasets/CAVE/mcodes/dataset/CAVE_x', num2str(scale), '/test/CAVE_x', num2str(scale), '_test', int2str(i), '.mat'),'gt','lq','bic');
end
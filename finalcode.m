clc;
clear;
close all;

% =========================================================================
% 1) Taking MRI image of brain as input
% =========================================================================
[files, path] = uigetfile({'*.tif', 'TIFF Files'}, ...
                         'Select input and ground truth images', ...
                         'MultiSelect', 'on');

if iscell(files)
    num_files = length(files);
else
    num_files = 1;
    files = {files};
end

if num_files ~= 2
    error('Please select exactly two images.');
end

str = fullfile(path, files{1});
strTr = fullfile(path, files{2});

I = imread(str);
IGndTr = imread(strTr);
Im = I;

figure('Name', 'Step 1: Input MRI');
imshow(I);
title('Figure 1: Original Input MRI Image');

% =========================================================================
% 2) Converting it to gray scale image
% =========================================================================
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);
I = uint8(0.2989*double(R) + 0.5870*double(G) + 0.1140*double(B));

figure('Name', 'Step 2: Grayscale Conversion');
imshow(I);
title('Figure 2: Grayscale Image (Luminosity Method)');

gsAdj = I;

% ---------------------------------------------------------------------
% Skull stripping
% ---------------------------------------------------------------------
imbw = gsAdj > 20;
imf = imfill(imbw, 'holes');
r = 20;
se = strel('disk', r);
erode_bw = imerode(imf, se);

erode_bw = bwareafilt(erode_bw, 1);

gsAdj = immultiply(gsAdj, erode_bw);

figure('Name', 'Step 2: Skull Stripping');
imshow(gsAdj);
title('Figure 3: Skull Stripped Image (largest component only)');

I = gsAdj;

% =========================================================================
% 3) High-pass filtering via Parks–McClellan unsharp mask
% =========================================================================
N = 30;
f = [0 0.1 0.15 1];
m = [0 0 1 1];
b_eq = firpm(N, f, m);
hpf2d = b_eq.' * b_eq;
Id = zeros(size(hpf2d));
mid = (size(hpf2d, 1) + 1) / 2;
Id(mid, mid) = 1;
alpha = 1;
H = Id + alpha * hpf2d;
sharpened = imfilter(gsAdj, H, 'replicate');

figure('Name', 'Step 3: High-pass Filtering');
imshow(sharpened);
title('Figure 4: Sharpened Image (Unsharp Mask with Parks–McClellan)');

% =========================================================================
% 4) Apply median filter
% =========================================================================
Median = medfilt2(sharpened);

figure('Name', 'Step 4: Median Filter');
imshow(Median);
title('Figure 5: Median Filtered Image (3x3 Kernel)');

% =========================================================================
% 5) Threshold segmentation
% =========================================================================
numClasses = 4; 
level = multithresh(Median, numClasses - 1);
seg_I = imquantize(Median, level);

figure('Name', 'Step 5: Multilevel Thresholding');
imshow(label2rgb(seg_I));
title('Figure 6: Thresholded Image (Multilevel)');

classMeans = -Inf(numClasses, 1);
for c = 1:numClasses
    classPixels = Median(seg_I == c);
    if ~isempty(classPixels)
        classMeans(c) = mean(classPixels);
    end
end
[~, brightestClass] = max(classMeans);
BW = seg_I == brightestClass;

figure('Name', 'Step 5: Thresholded Image Final');
imshow(BW);
title('Figure 7: Thresholded Image (Brightest-class Selection)');

% =========================================================================
% 6) Watershed segmentation
% =========================================================================
C = ~BW;
D = -bwdist(C);
L = watershed(D);
Wi = label2rgb(C, 'gray', 'w');

figure('Name', 'Step 6: Watershed Boundaries');
imshow(Wi);
title('Figure 8: Watershed Boundaries');

if size(Wi, 3) == 3
    Wi_gray = rgb2gray(Wi);
else
    Wi_gray = Wi;
end

lvl2 = graythresh(Wi_gray);
BW2 = imbinarize(Wi_gray, lvl2);

figure('Name', 'Step 6: Watershed Segmentation');
imshow(BW2);
title('Figure 9: Segmented Image via Watershed');

% =========================================================================
% 7) Morphological operations
% =========================================================================
sout = bwareaopen(BW2, 20); 

label = bwlabel(sout);
stats = regionprops(logical(sout), 'Solidity', 'Area', 'BoundingBox');
density = [stats.Solidity];
area = [stats.Area];
high_dense_area = density > 0.2;

no_tumor = 0;
if any(high_dense_area)
    max_area = max(area(high_dense_area));
    tumor_label = find(area == max_area);
    disp(['Maximum area: ', num2str(max_area)]);
    tumor = ismember(label, tumor_label);
else
    max_area = 0;
    tumor = false(size(sout));
end

if max_area > 500
    figure('Name', 'Step 7: Tumor Extraction');
    imshow(tumor);
    title('Figure 10: Detected Tumor Area');
else
    msgbox('No Tumor!!', 'Status');
    no_tumor = 1;
    tumor(tumor > 0) = 0;
end

BW3 = tumor;

% =========================================================================
% 8) Morphological closing
% =========================================================================
regionArea = nnz(BW3);
closeRadius = max(2, min(15, round(0.05 * sqrt(max(regionArea, 1)))));
se = strel('disk', closeRadius);
tumor = imclose(BW3, se);
tumor = imfill(tumor, 'holes');

figure('Name', 'Step 8: Morphological Closing');
imshow(tumor);
title('Figure 11: Closed Tumor Region');

% =========================================================================
% 9) Overlaying results
% =========================================================================
OLtumor = tumor;
[M, N] = size(OLtumor);
A = ones(M, N);
A(OLtumor == 1) = 0;

OLmask = IGndTr;
A2 = ones(M, N);
A2(OLmask == 1) = 0;

figure('Name', 'Step 9: Overlay Results');
subplot(1, 2, 1);
h = imshow(Im);
title('Figure 12a: MRI with Detected Tumor Overlay');
hold on;
set(h, 'AlphaData', A);
hold off;

subplot(1, 2, 2);
h1 = imshow(Im);
title('Figure 12b: MRI with Ground Truth Mask');
hold on;
set(h1, 'AlphaData', A2);
hold off;

% =========================================================================
% 10) Result evaluation
% =========================================================================
predictedImage = tumor;
groundTruthImage = IGndTr;
predicted = logical(predictedImage);
groundTruth = logical(groundTruthImage);

if no_tumor == 1
    intersection = nnz(predicted == groundTruth);
    diceCoefficient = 2 * intersection / 131072;
else
    intersection = nnz(predicted & groundTruth);
    diceCoefficient = 2 * intersection / (nnz(predicted) + nnz(groundTruth));
end

disp(['Dice Coefficient: ', num2str(diceCoefficient)]);

if no_tumor == 1
    union = 131072 - intersection;
    iouScore = intersection / union;
else
    union = nnz(predicted | groundTruth);
    iouScore = intersection / union;
end

disp(['IoU Score: ', num2str(iouScore)]);

if no_tumor == 1
    truePositives  = nnz(groundTruth == predicted);
    falsePositives = nnz(~groundTruth == predicted);
    falseNegatives = nnz(groundTruth == ~predicted);
else
    truePositives  = nnz(groundTruth & predicted);
    falsePositives = nnz(~groundTruth & predicted);
    falseNegatives = nnz(groundTruth & ~predicted);
end

precision = truePositives / (truePositives + falsePositives);
recall = truePositives / (truePositives + falseNegatives);
if precision + recall == 0
    f1Score = 0;
else
    f1Score = 2 * (precision * recall) / (precision + recall);
end

disp(['F1 Score: ', num2str(f1Score)]);
cFile = 'scan9FN2BCOS_Oddball_AA_ROS-1706_600_600_2_uint16.dat';
cPath = '/datajoint-data/data/aeltona/Oddball_AA_ROS-1706_2024-03-12_scan9FN2BCOS_sess9FN2ANVG';
fID = fopen(fullfile(cPath, cFile)); %old binary file

dType = 'uint16';
sRate = 5;
n_frames = 12916; n_channels = 2; height = 600; width = 600;
imSize = [600 600];
cFrames = 12916*2;
dimCnt = 200;
cData = fread(fID, height*width*n_frames*n_channels, [dType '=>' dType]);
cData = reshape(cData, height, width, n_channels, n_frames);

hemoData = single(squeeze(cData(:,:,1,:)));
blueData = single(squeeze(cData(:,:,2,:)));

blueMean = nanmean(blueData,3);
hemoMean = nanmean(hemoData,3);

% get reference images
blueRef = fft2(single(blueMean)); %blue reference for alignment
hemoRef = fft2(single(hemoMean)); %blue reference for alignment

% do alignment
for iFrames = 1:size(blueData,3)
    [~, temp] = dftregistration(blueRef, fft2(blueData(:, :, iFrames)), 10);
    blueData(:, :, iFrames) = abs(ifft2(temp));
    if rem(iFrames,100) == 0
        disp(iFrames)
    end
end
for iFrames = 1:size(hemoData,3)
    [~, temp] = dftregistration(hemoRef, fft2(hemoData(:, :, iFrames)), 10);
    hemoData(:, :, iFrames) = abs(ifft2(temp));
    if rem(iFrames,100) == 0
        disp(iFrames)
    end
end

% mean correction
blueData = bsxfun(@minus, blueData, blueMean);
blueData = bsxfun(@rdivide, blueData, blueMean);
hemoData = bsxfun(@minus, hemoData, hemoMean);
hemoData = bsxfun(@rdivide, hemoData, hemoMean);

% run SVD
nrPixels = height*width;
cData = reshape(cat(3, blueData, hemoData), nrPixels, []);
[cU, cS] = fsvd(cData, min([size(cData), dimCnt]));
U = cU * cS; %combine eigenvalues and spatial components
U = reshape(U, height,width,[]);
clear cData

% get temporal components
blueV = gather(pinv(U) * reshape(blueData, nrPixels, []));
hemoV = gather(pinv(U) * reshape(hemoData, nrPixels, []));

% do hemodynamic correction
[newVc, regC, T, hemoVar] = cc_SvdHemoCorrect(U, blueV, hemoV, sRate, length(blueV));
save(fullfile(cPath, strrep(cFile, '.dat', '_Vc.mat')), 'U', 'newVc' , 'regC', 'T', 'hemoVar', '-v7.3');
    


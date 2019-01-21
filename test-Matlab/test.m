addpath(genpath('/home/wangxiangxue/caffe-master/matlab'));
addpath(genpath('/home/wangxiangxue/caffe-master/examples/glasses/glassesmodel/m/matlabPyrTools'));
addpath(genpath('/home/wangxiangxue/caffe-master/examples/glasses/glassesmodel/m/matlabPyrTools/MEX'))

clear;
blobname = 'noGlass';
if ~exist('model_inited','var')||~model_inited
    caffe.reset_all();
    caffe.set_mode_gpu(); 
    caffe.set_device(0);
    net = caffe.Net('/home/wangxiangxue/caffe-master/examples/SRCNN/SRCNN_train/SRCNN_net.prototxt','/home/wangxiangxue/caffe-master/examples/SRCNN/SRCNN_train/models/SRCNN_solver_iter_6000000.caffemodel', 'test');
    model_inited = true;
end;

% tic
root_folder = '/home/wangxiangxue/data/glasses/test/';  
fid = fopen('/home/wangxiangxue/caffe-master/examples/SRCNN/SRCNN_train/2test15216.txt');
tline = fgetl(fid);
iN = 0;
Int = zeros(15216,1);
Rp = Int;
Rs = Int;
Rm = Int;
Ri = Int;
sumtime = 0;
while ischar(tline)
    S = regexp(tline,'*','split');
    Img = imread([root_folder,tline]);
    II = rgb2gray(Img);
    imshow(II);
    Img2 = imread([root_folder,S{2}]);
    if size(Img,3)==1
        Img = repmat(Img,[1,1,3]);
    end
    if size(Img2,3)==1
        Img2 = repmat(Img2,[1,1,3]);
    end
    Img2 = double(Img2)*0.00392156862; 
    Img = Img(:,:,[3,2,1]);
    Img = double(Img);
    Img = Img*0.00392156862; 
    Img = permute(Img, [2,1,3]);  
    tic
    t1=clock;
    net.forward({Img}); 
    FV = net.blobs(blobname).get_data();
    FV = permute(FV, [2,1,3]); 
    FV = double(FV(:,:,[3,2,1]));
    t2=clock;
    eachtime = etime(t2,t1);
    sumtime = sumtime + eachtime;
    iN = iN+1;
    Rp(iN) = psnr(FV,Img2);
    Rs(iN) = ssim(FV,Img2);
    tmp  = sqrt((FV-Img2).^2);
    Rm(iN) = mean(tmp(:));  
    Ri(iN) = ifcvec(mean(FV,3),mean(Img2,3));
    if mod(iN,1000)==0
        fprintf('Fea1 %d, time: %0.00f s\n',iN,sumtime);
    end  
    tline = fgetl(fid);
end
sumtime
fprintf('finish %d, time: %0.0f s\n',iN,toc);
fprintf('psnr  ssim  mse  ifc\n%0.2f %0.4f %0.7f %0.2f\n',sum(Rp)/iN,sum(Rs)/iN,sum(Rm)/iN,sum(Ri)/iN);
kn = iN/3;
for i=1:3
    fprintf('%0.2f %0.4f %0.7f %0.2f\n',sum(Rp((i-1)*kn+1:i*kn))/kn,sum(Rs((i-1)*kn+1:i*kn))/kn,sum(Rm((i-1)*kn+1:i*kn))/kn,sum(Ri((i-1)*kn+1:i*kn))/kn);
end
fclose(fid);

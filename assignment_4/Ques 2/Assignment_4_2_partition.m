
%input the image
data = imread("bw.jpg");
%disp(data);

data=im2double(data);

%spectral clustering
[C1, C2]=spectral_clustering_2(data, k);
size(C1)
size(C2)
%gscatter(C1, C2)

%k-means clustering
idx=kmeans(data(:),k);
idx(idx==2)=[0];
re=reshape(idx, [75,100]);
imshow(re)


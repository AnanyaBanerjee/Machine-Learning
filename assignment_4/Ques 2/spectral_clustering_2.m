
function [C1, C2]=spectral_clustering_2(data, k)
%load sonar_train
%train_data=load("sonar_train.csv");
%store all relevant columns in data
%data=train_data(:,1:60)


%store all target column values in labels
%labels=train_data(:,61)

size_of_dataset=size(data);  %size gives(n,p)
                %n=number of rows
                %m=number of columns
n=size(data, 1);
m=size(data,2);

disp("n is")
disp(n)
%find A
%sigma=[0,01, 0.1, 1, 10,100]
sigma=0.1;
s=-1/(2*sigma^2);

A=zeros(n,n);

for i=1:n
    for j=1:n
        %disp(data(i,:))
        f=(norm(data(i,:)-data(j,:)))^2;
        f1=exp(s*f);
        A(i,j)=f1;
    end
end

%disp(A)
%loop over the entire dataset, find mean of each column
%find mean of every column and store it in m

%initialize D
D=zeros(n,n);

%compute D= sum(Aij) 
%subtract mean of col c_j from all elements of the column c_j
for z= 1:n
    D(z,z)=sum(A(:,i))%/n;
end


%disp(n)
%disp(size(A))
%disp(size(D))
%computed laplacian
L=D-A ;%(104,60)

%to find eigen values and vectors, we find L*L'
%Lap=L*L';
Lap=L;
%disp(Lap)
%disp("diag d is")
%disp(D)

% Lap*v= v*d
%d: diagonal matrix of eigen values
%v=columns are corresponding right eigen vector
[v,d]=eig(Lap);

%diag(d) gives which eigen vector has most contribution
Dd=diag(d);

%disp("Diag is ")
%disp(Dd)

%error idea given by Lap*v-v*d
%disp(Lap*v- v*d)

%get ideal value of k from PCA
%for testing: k=3
%k=3
%constructing matrix V with k-smallest eigen values of L
V=zeros(n,k); %(104,3);

%k smallest eigen values imply top k eigen values and vectors
for i=1:k
    V(:,i)=v(:,i);
end

%disp("all eigen vec are")
%disp(v)
%%disp("V is")
%disp(V)
%kmeans(V,k) performs k-means clustering to partition the observations of the n-by-p data matrix V into k clusters, 
%and returns an n-by-1 vector (idx) containing cluster indices of each eigen vector y. 
%now applying k-means to V
idx=kmeans(V,k);  %size(idx)=(104,1)

%idx assigns clusters S_i to each eigen vector y_i

%final list of cluster assignments 
Cl=zeros(1,size(idx,1));
%traverse idx and assign clusters to x depedning on what cluster their
%correspondonding eigen vectors have
for i=1:size(idx,1)
    Cl(i)=idx(i);
end


%list of clusters
C=unique(Cl) ;

%no of unique clusters
%no_of_clusters=size(C,2);


%assign datapoints to clusters
%make number_of clusters clusters

C1=find(idx==1);
C2=find(idx==2);
%C3=find(idx==3);

%C1=data(C1_,:);
%disp C1
%C2=data(C2_,:);
 
end


% 
% C1=int16.empty(104,0)  %cluster corresponding to 1
% C2=int16.empty(104,0)  %cluster corresponding to 2
% C3=int16.empty(104,0)  %cluster corresponding to 3
% a=1
% b=1
% d=1
% for j=1:n
%     if idx(j)==1
%         C1(a)=j
%         a=a+1
%     elseif idx(j)==2
%         C2(b)=j
%         b=b+1
%     else
%         C3(d)=j
%         d=d+1
%     end
% end
%   
% final_list_of_clusters={C1;C2;C3}

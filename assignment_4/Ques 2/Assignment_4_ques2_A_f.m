%applying function circs
X=circs();

X=X';


%introducing k value
k=2;

%applying spectral clustering
[C1, C2]=spectral_clustering_2(X,k);

%scatter plot for spectral clustering
C1_=X(C1,:);
C2_=X(C2,:);

figure(1)
c='g'
gscatter(C1_(:,1), C1_(:,2),[],c)
hold on 
c='r'
gscatter(C2_(:,1), C2_(:,2),[],c)
title('Spectral Clustering');


%applying k-means clustering
idx=kmeans(X,k);
%scatter plot for k-means
%final list of cluster assignments acc to -k-means
K1=find(idx==1);
K2=find(idx==2);
K1_=X(K1,:);
K2_=X(K2,:);

figure(2);
c='b'
gscatter(K1_(:,1), K1_(:,2),[],c)
hold on
c='y'
gscatter(K2_(:,1), K2_(:,2),[],c)
title('K-Means Clustering');









% part1_ Dimensional reduction by PCA

% dimention reduction using principle component analyse
%% load orl face database 
clc
clear all
 close all
load('ORL.mat');
ndata=fea.' ;     % data
data=im2double(ndata);
label=gnd ;    %labels
cnt=0;
cnt2=0;
%% dividing dataset to train and test 
tr=7; % number of image  per person for train 
for i=0:39
    for j=1:10
        if j<=tr
        cnt=cnt+1;
       trdt(cnt,:)= data(10*i+j,:);  % train data 
       trlb(cnt,:)=label(10*i+j);    % train labels 
        else 
            cnt2=cnt2+1;
            tstdt(cnt2,:)=data(10*i+j,:); %  test data
             tstlb(cnt2,:)=label(10*i+j); %  test labels
        end
    end
end
   %%  nirmalization and standardization    
 m=mean(trdt);    % mean vector of train data 
  s=std(trdt);    % standard deviation
  N=280;          % number of train data
  M=120;          % number of test data 
   for i=1:N
          nstdt(i,:)=((trdt(i,:)-m)./s)/255;   % normal standard data
  end 
  for i=1:M
      nstsdt(i,:)=((tstdt(i,:)-m)./s)/255; 
  end
   %% scatter matrix 
  c1=(N-1)*cov(nstdt); % scatter matrix of train data 
  %% eigen values and eigen vectors of scatter matrix 
  [u,v]=eig(c1);
  %%
  v1=sort(v*ones(10304,1),'descend');
  %%
  k=1:10304;
  plot(k,v1);
  %%
  p=400; %new dimension
  u7=importdata('u7.mat');
   f=u7(:,10304-(p-1):10304); %eigen vectors corresponding to maximum  eigen values  
%% projection to new spce
m2=mean(nstsdt); 
       a=nstdt*f; %projected train  samples
 
        for i=1:120
      newt(i,:)=nstsdt(i,:)-m2;
        end
       at=newt*f;  %projected test samples 
       
% Part2_ classification with SVM

% support vector machine classifier 
%% svm using toolbox 
   svmmodel=fitcecoc(a,trlb);
 %%
  label= predict(svmmodel,at);
  %%
   count=0;
       for i=1:120
        if label(i)==tstlb(i)
           count=count+1;
        end
        end
        acc=count/120;
%% multiclass one vs one svm without toolbox
   s1=[];
   indx=0;
   a2=[];
   for i=1:39
      for j=i+1:40
           cnt2=0;
          indx=indx+1;
          for k=1:280
              if trlb(k)==i || trlb(k)==j
                  cnt2=cnt2+1;
                  trlb2(cnt2)=trlb(k);
                  a2(cnt2,:)=a(k,:);
              end
          end
                  model{indx}=fitcsvm(a2,trlb2,'KernelScale','auto','Standardize',false,...
            'KernelFunction','rbf','BoxConstraint',1);
        [label,score]=predict(model{indx},at);
             s1(:,indx)=score(:,1);
        l(:,indx)=label;
        clear a2
        clear trlb2
      end
   end
   %% labeling and evaluation
    count=0;
   for i=1:120
    [m,n]=max(abs(s1(i,:)));
    flbl(i)=l(i,n);
    if flbl(i)==tstlb(i)
           count=count+1;
    end
   end
        svmacc=count/120; % accuracy 


%Part3_ classification with KNN

% knn classification 
%% classification by toolbox
 knnmodel=fitcknn(a,trlb);
 count=0;
label2=predict(knnmodel,at);
%% accuracy measurement
  for i=1:120
      if label2(i)==tstlb(i)
          count=count+1;
      end
  end
      
      knnacc=count/120;
      %% nearest neighbor code (without toolbox ) 
      for i=1:120
          for j=1:280
              dist(j)=sqrt(sum((at(i,:)-a(j,:)).^2));
          end
          [m,indx]=min(dist);
          label2(i)=trlb(indx);
          dist=0;
      end
for i=1:120;
         
         if label2(i)==tstlb(i)
           count=count+1;
       end
   end
       
      knnacc=count/120;
          
%Part4_ classification with LDA

% linear discriminant analyse classification  
 ldamodel=fitcdiscr(a,trlb,'discrimType','linear');
 %%
 lbl=predict(ldamodel,at);
 %%
 count=0;
  for i=1:120
        if lbl(i)==tstlb(i)
            count=count+1;
       end
    end
        
        ldaacc=count/120;



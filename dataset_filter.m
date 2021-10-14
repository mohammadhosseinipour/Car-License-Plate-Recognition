clc;close all; clear;
Dir="C:\Users\mohammad\Desktop\LPR\darkflow-master";
imDir = fullfile(Dir,'imds2');
imds = imageDatastore(imDir);
test_size=size(imds.Files);
p1=test_size;
progressBar = waitbar(0,  'Performing dataset filtering...');
for i=1:test_size
    I = rgb2gray(readimage(imds,i));
    I_size=size(I);
%     r = randi(I_size(1),50,1);
%     c = randi(I_size(2),50,1);
    for j=i+1:test_size
        try
            J = rgb2gray(readimage(imds,j));
            flag=1;
            if I_size==size(J)
                if I(50:60,:)==J(50:60,:)
                    flag=0;
                end
                
%                 for h=1:50
%                     if I(r(h),c(h))==J(r(h),c(h))
%                         flag=flag-1;
%                         break;
%                     end
%                 end
            end
            if flag==0
                imds.Files = setdiff(imds.Files,imds.Files(j));
                test_size=size(imds.Files);
                if j==test_size(1)
                    break;
                end
            end 
        catch
            test_size=size(imds.Files);
            if j<=test_size(1)
                disp(string(size(imds.Files)));
                disp("not supported:"+string(imds.Files(j)));
                imds.Files = setdiff(imds.Files,imds.Files(j));            
            end
            
        end
        
    end
disp("picrure #"+string(i));
    if i==test_size(1)
       break;
    end
waitbar(i /test_size(1), progressBar);
end
writeall(imds,"C:\Users\mohammad\Desktop\LPR\darkflow-master\imds",'Folderlayout','flatten')
close(progressBar);
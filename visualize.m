function visualize(train,N)

Dimage=[];
Fim=[];
k=0;
for i=1:size(train,1)
Im = reshape(train(i,:),20,20);
Dimage = [Dimage Im];
k=k+1;
if (k==N)
    
   Fim = [Fim Dimage'];
    Dimage = [];
    k=0;
end


end
imshow(Fim');
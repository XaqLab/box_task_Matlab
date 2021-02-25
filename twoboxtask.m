
twoboxtask_init;

Tx = kronn(Tb1,Tb2);
px0 = kronn(pb10,pb20)';

tmax=100;
prL=repmat(pr0,[tmax,1])';
plL=repmat(pl0,[tmax,1])';
pb1L=repmat(pb10,[tmax,1])';
pb2L=repmat(pb20,[tmax,1])';
phL=repmat(ph0,[1,tmax]);

pxL=repmat(px0,[1,tmax]);

for t=2:tmax
    prL(:,t)=Tr*prL(:,t-1);
    plL(:,t)=Tl*plL(:,t-1);
    pb1L(:,t)=Tb1*pb1L(:,t-1);
    pb2L(:,t)=Tb2*pb2L(:,t-1);
    
    pxL(:,t)=Tx*pxL(:,t-1);
    
    phL(:,t)=Th*phL(:,t-1);
end




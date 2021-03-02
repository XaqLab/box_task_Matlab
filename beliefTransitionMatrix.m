function Tqqq = beliefTransitionMatrix(p_appear, p_disappear, nq, w)
%function beliefTransitionMatrix(p_appear, p_disappear, nq)
% create transition matrix between nq belief states q to q'
% w is width of diffusive noise

dq = 1/nq;
a = 1 - p_disappear - p_appear;
%L = dq*sqrt(1+1/a^2);

for i=1:nq
    for j=1:nq

        q=(i-1)*dq;
        qq=(j-1)*dq;
        
        bm = (qq - p_appear) / a;
        bp = (qq+dq - p_appear) / a;
        
        Tqqq(j,i) = max(0, min(q+dq, bp) - max(q,bm) );
    end
end
Tqqq = Tqqq ./ repmat(sum(Tqqq,1),[nq 1]);

nt=20;
d=w/nt;
dD = toeplitz([-2*d,d,zeros(1,nq-2)]);
dD(2,1) = 2*d;
dD(end-1,end) = 2*d;
D=expm(dD*nt);
D = D ./ repmat(sum(D,1),[nq 1]);

Tqqq=D*Tqqq;


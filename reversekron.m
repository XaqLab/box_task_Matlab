function BA=reversekron(AB,n)
%function BA=reversekron(AB,n)
%   Takes a kronecker product AB and returns the kronecker product BA
%   A and B are assumed to be square, with sizes n(1) and n(2)
BA = col2im(...
            im2col(AB, n(2)*[1,1],'distinct')', ...
        n(1)*[1,1],prod(n)*[1,1],'distinct');

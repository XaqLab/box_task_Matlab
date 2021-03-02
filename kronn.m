function Z = kronn(varargin)
%function Z = kronn(varargin)
% returns multidimensional kronecker product of all matrices in list

Z = varargin{1};
for i=2:length(varargin)
    Z = kron(Z,varargin{i});
end

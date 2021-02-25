function Z = tensorsumm(varargin)
%function Z = tensorsumm(varargin)
% returns multidimensional kronecker sum of all matrices in list

Z = varargin{1};
for i=2:length(varargin)
    Z = tensorsum(Z,varargin{i});
end

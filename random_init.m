% random initial state

pr0 = rand(size(pr0)); % kronecker product of these transition matrices
pr0 = pr0 / sum(pr0);
pl0 = rand(size(pl0)); % kronecker product of these transition matrices
pl0 = pl0 / sum(pl0);
pb10 = rand(size(pb10)); % kronecker product of these transition matrices
pb10 = pb10 / sum(pb10);
ph0 = rand(size(ph0)); % kronecker product of these transition matrices
ph0 = ph0 / sum(ph0);

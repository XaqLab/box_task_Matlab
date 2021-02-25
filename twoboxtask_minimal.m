
twoboxtask_init;                                % set parameters and define transition matrices

% setup action sequence
tblock = 50;                                    % time for block of actions
aseq = [a0 pb g2 pb a0 g1 pb g2];               % action sequence
tmax = tblock * length(aseq);                   % total duration of actions
aL = reshape(repmat(aseq,[tblock 1]),[tmax,1]); % repeat each action tblock times.


phL=repmat(ph0,[1,tmax]);                       % initialize p(state)
pht=zeros([nq,nr,nq,nl,tmax]);                  % initialize time sequence of marginals
pht(:,:,:,:,1) = reshape(phL(:,1),[nq,nr,nq,nl]); % joint distribution. Format: b2 * r * b1 * l

% main loop: dynamics!
for t=2:tmax
    phL(:,t) = ThA(:,:,aL(t))' * phL(:,t-1); % update state according to action
    pht(:,:,:,:,t) = reshape(phL(:,t),[nq,nr,nq,nl]); % joint distribution. Format: b2 * r * b1 * l
end

imagesc(phL); title('p(state) evolution'); xlabel('time'); ylabel('state');

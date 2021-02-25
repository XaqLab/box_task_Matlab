
twoboxtask_init;                                % set parameters and define transition matrices

tblock = 50;                                    % time for block of actions
aseq = [a0 pb g2 pb a0 g1 pb g2];               % action sequence
tmax = tblock * length(aseq);                   % total duration of actions
aL = reshape(repmat(aseq,[tblock 1]),[tmax,1]); % repeat each action tblock times.

phL=repmat(ph0,[1,tmax]);                       % initialize p(state)

pht=zeros([nq,nr,nq,nl,tmax]);                  % initialize time sequence of marginals
prt=zeros([nr,tmax]);
plt=zeros([nl,tmax]);
pb1b2t=zeros([nq,nq,tmax]);
pb1t=zeros([nq,tmax]);
pb2t=zeros([nq,tmax]);
% joint and marginals
pht(:,:,:,:,1) = reshape(phL(:,1),[nq,nr,nq,nl]); % joint distribution. Format: b2 * r * b1 * l
prt(:,1) = squeeze(sum(sum(sum(pht(:,:,:,:,1),1),3),4)); % marginal over reward
plt(:,1) = squeeze(sum(sum(sum(pht(:,:,:,:,1),1),2),3)); % marginal over location
pb1t(:,1) = squeeze(sum(sum(sum(pht(:,:,:,:,1),1),2),4)); % marginal over beliefs b1
pb2t(:,1) = squeeze(sum(sum(sum(pht(:,:,:,:,1),2),3),4)); % marginal over beliefs b2
pb1b2t(:,:,1) = squeeze(sum(sum(pht(:,:,:,:,1),2),4)); % bivariate marginal over beliefs b1,b2

for t=2:tmax
    
    % update state according to action
    phL(:,t) = ThA{aL(t)} * phL(:,t-1);
    
    % reformat joint distribution
    pht(:,:,:,:,t) = reshape(phL(:,t),[nq,nr,nq,nl]); % joint distribution. Format: b2 * r * b1 * l
    prt(:,t) = squeeze(sum(sum(sum(pht(:,:,:,:,t),1),3),4));
    plt(:,t) = squeeze(sum(sum(sum(pht(:,:,:,:,t),1),2),3));
    pb1t(:,t) = squeeze(sum(sum(sum(pht(:,:,:,:,t),1),2),4));
    pb2t(:,t) = squeeze(sum(sum(sum(pht(:,:,:,:,t),2),3),4));
    pb1b2t(:,:,t) = squeeze(sum(sum(pht(:,:,:,:,t),2),4)); % joint distribution over beliefs
end

% display results
figure;
subplot(5,1,1); imagesc(log(phL)); title('p(state) evolution'); xlabel('time'); ylabel('state');
subplot(5,1,2); plot(prt(2,:)); ylabel('p(R=1)'); xlabel('time');
subplot(5,1,3); plot(plt'); ylabel('p(loc)'); xlabel('time');
subplot(5,1,4); imagesc(1:tmax,bL,pb1t(end:-1:1,:)); title('p(belief(reward available @ 1))'); ylabel('b(A1)'); xlabel('time');
subplot(5,1,5); imagesc(1:tmax,bL,pb2t(end:-1:1,:)); title('p(belief(reward available @ 2))'); ylabel('b(A2)'); xlabel('time');

% conditioning

T = 200;
mT = 1/gamma;   % mean time to next event

nlooks = 20;
mT_obs = mT / 2; % look twice as often as average reward arrival

ntrials = 100; % number of trials
tL = exprnd(mT,[ntrials,1]); % time of food arrival
tobsL = cumsum(exprnd(mT_obs,[nlooks 1])); % list of possible looking times. Stay consistent over iterations
r = zeros(nlooks,ntrials);

for trial=1:ntrials % keep going until we get nr rewards
    
    i = find( tobsL > tL(trial), 1); % first observation after food available

    r(i,trial) = 1;

end


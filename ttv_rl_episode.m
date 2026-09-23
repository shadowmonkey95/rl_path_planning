function [reward, rewards, log] = ttv_rl_episode(pathInput, cfg)
%TTV_RL_EPISODE Run one RL path-evaluation episode for a tractor-trailer.
%
%   [reward, rewards, log] = ttv_rl_episode(pathInput, cfg)
%
% The RL action is assumed to have already been converted into a geometric
% path. This function closes the loop as follows:
%
%   path -> linear MPC -> nonlinear tractor-trailer plant -> reward
%
% INPUT
%   pathInput : Either an N-by-2 numeric array [x, y], or a structure with
%               fields .x and .y. The points must describe the path in
%               traversal order. The path is internally translated and
%               rotated into a road-fixed frame whose origin and heading
%               equal those of the first path point.
%   cfg       : Optional configuration structure. Call with cfg = struct()
%               to use the defaults returned by ttv_default_config below.
%
% OUTPUT
%   reward    : Scalar episode reward for an RL agent.
%   rewards   : Reward components and termination information.
%   log       : Closed-loop histories and tracking/articulation metrics.
%
% The default reward intentionally follows Feher et al.'s RL path-planning
% structure: on a successful episode it combines a tractor-tire-slip term
% and a path-curvature smoothness term with equal weights. Articulation is
% logged but is not yet included in the reward. A failed episode receives
% -1 before the final path section and -0.75 in the final section.
%
% Example
%   x = linspace(0,120,601)';
%   y = 1.75*(tanh(0.12*(x-35))-tanh(0.12*(x-80)));
%   [r, parts, data] = ttv_rl_episode([x,y], struct());
%
% Requirements
%   CasADi 3.7.x on the MATLAB path, or cfg.casadiPath pointing to it.

    if nargin < 2 || isempty(cfg)
        cfg = struct();
    end
    cfg = ttv_default_config(cfg);
    ensure_casadi(cfg);

    path = prepare_path(pathInput);
    if path.s(end) <= cfg.minPathLength
        error('ttv_rl_episode:ShortPath', ...
            'The usable path length must exceed %.3f m.', cfg.minPathLength);
    end

    persistent cache
    cacheKey = make_cache_key(cfg);
    if isempty(cache) || ~isfield(cache,'key') || ~strcmp(cache.key,cacheKey)
        cache = build_environment(cfg,cacheKey);
    end
s
    % Nonlinear-plant state:
    % xp = [X1; Y1; psi1; v1; r1; phi; q; delta_actual].
    if isempty(cfg.initialPlantState)
        xp = zeros(8,1);
        xp(1) = path.x(1);
        xp(2) = path.y(1);
        xp(3) = path.psi(1);
    else
        xp = cfg.initialPlantState(:);
        if numel(xp) ~= 8
            error('ttv_rl_episode:InitialState', ...
                'cfg.initialPlantState must contain eight elements.');
        end
    end

    nx = 7;
    N = cfg.N;
    maxSteps = max(1,ceil(cfg.maxEpisodeTime/cfg.T));

    % Warm starts. The CasADi solver is persistent; these guesses remain
    % local to the episode and are shifted after every MPC solution.
    xm = plant_to_mpc_state(xp,cfg.V);
    Xguess = repmat(xm,1,N+1);
    Uguess = zeros(1,N);

    % Preallocate histories to avoid repeated allocation during RL training.
    tHist       = nan(1,maxSteps+1);
    plantHist   = nan(8,maxSteps+1);
    mpcHist     = nan(nx,maxSteps+1);
    uHist       = nan(1,maxSteps);
    refYHist    = nan(1,maxSteps);
    refPsiHist  = nan(1,maxSteps);
    eLatHist    = nan(1,maxSteps+1);
    ePsiHist    = nan(1,maxSteps+1);
    distHist    = nan(1,maxSteps+1);
    alphaHist   = nan(3,maxSteps+1);
    forceHist   = nan(3,maxSteps+1);
    sHist       = nan(1,maxSteps+1);
    if cfg.logPhysicalDiagnostics
        massResidualHist = nan(1,maxSteps+1);
        massRcondHist = nan(1,maxSteps+1);
        yawRateIdentityHist = nan(1,maxSteps+1);
        articulationRateIdentityHist = nan(1,maxSteps+1);
    end

    [sNow,eLat,ePsi,dist] = path_errors(xp,path);
    [alpha,Fy] = plant_outputs(cache,xp,0);
    tHist(1) = 0;
    plantHist(:,1) = xp;
    mpcHist(:,1) = xm;
    eLatHist(1) = eLat;
    ePsiHist(1) = ePsi;
    distHist(1) = dist;
    alphaHist(:,1) = alpha;
    forceHist(:,1) = Fy;
    sHist(1) = sNow;
    if cfg.logPhysicalDiagnostics
        diagnostics = plant_diagnostics(cache,xp,0);
        massResidualHist(1) = diagnostics.massRelativeResidual;
        massRcondHist(1) = diagnostics.massRcond;
        yawRateIdentityHist(1) = diagnostics.yawRateIdentity;
        articulationRateIdentityHist(1) = diagnostics.articulationRateIdentity;
    end

    failed = false;
    passed = false;
    failureReason = '';
    finalStep = 0;

    for step = 1:maxSteps
        xm = plant_to_mpc_state(xp,cfg.V);

        % Preview the geometric path by arc length. Path information enters
        % the MPC cost, not the state equation x_dot = A*x + B*u.
        sPreview = sNow + cfg.V*cfg.T*(1:N);
        sPreview = min(max(sPreview,path.s(1)),path.s(end));
        yRef = interp1(path.s,path.y,sPreview,'pchip');
        psiRef = interp1(path.s,path.psi,sPreview,'linear');

        p = zeros(nx+2*N,1);
        p(1:nx) = xm;
        for j = 1:N
            p(nx+2*j-1) = yRef(j);
            p(nx+2*j) = psiRef(j);
        end

        w0 = [Xguess(:); Uguess(:)];
        args = struct('x0',w0,'lbx',cache.lbx,'ubx',cache.ubx, ...
            'lbg',cache.lbg,'ubg',cache.ubg,'p',p);

        try
            sol = cache.solver('x0',args.x0,'lbx',args.lbx,'ubx',args.ubx, ...
                'lbg',args.lbg,'ubg',args.ubg,'p',args.p);
            solverStats = cache.solver.stats();
            solverOK = isfield(solverStats,'success') && solverStats.success;
        catch solverException
            solverOK = false;
            solverStats = struct('return_status',solverException.message);
        end

        if ~solverOK
            failed = true;
            failureReason = ['MPC solver: ',solver_status(solverStats)];
            finalStep = step-1;
            break;
        end

        wOpt = full(sol.x);
        Xopt = reshape(wOpt(1:nx*(N+1)),nx,N+1);
        Uopt = reshape(wOpt(nx*(N+1)+1:end),1,N);
        deltaCmd = min(max(Uopt(1),-cfg.deltaMax),cfg.deltaMax);

        % Integrate the nonlinear plant with smaller RK4 substeps.
        h = cfg.T/cfg.plantSubsteps;
        for substep = 1:cfg.plantSubsteps
            xp = rk4_step(cache.plantDynamics,xp,deltaCmd,h);
        end
        xp(8) = min(max(xp(8),-cfg.deltaMax),cfg.deltaMax);

        [sNow,eLat,ePsi,dist] = path_errors(xp,path);
        [alpha,Fy] = plant_outputs(cache,xp,deltaCmd);
        xm = plant_to_mpc_state(xp,cfg.V);

        tHist(step+1) = step*cfg.T;
        plantHist(:,step+1) = xp;
        mpcHist(:,step+1) = xm;
        uHist(step) = deltaCmd;
        refYHist(step) = yRef(1);
        refPsiHist(step) = psiRef(1);
        eLatHist(step+1) = eLat;
        ePsiHist(step+1) = ePsi;
        distHist(step+1) = dist;
        alphaHist(:,step+1) = alpha;
        forceHist(:,step+1) = Fy;
        sHist(step+1) = sNow;
        if cfg.logPhysicalDiagnostics
            diagnostics = plant_diagnostics(cache,xp,deltaCmd);
            massResidualHist(step+1) = diagnostics.massRelativeResidual;
            massRcondHist(step+1) = diagnostics.massRcond;
            yawRateIdentityHist(step+1) = diagnostics.yawRateIdentity;
            articulationRateIdentityHist(step+1) = diagnostics.articulationRateIdentity;
        end
        finalStep = step;

        % Warm-start shift for the next MPC call.
        Xguess = [Xopt(:,2:end),Xopt(:,end)];
        Uguess = [Uopt(2:end),Uopt(end)];
        Xguess(:,1) = xm;

        % The original paper's failure checks that are meaningful with a
        % path-only input: lateral slip, distance error and heading error.
        if max(abs(alpha(1:2))) > cfg.reward.maxLateralSlip
            failed = true;
            failureReason = 'tractor lateral-slip limit';
        elseif dist > cfg.reward.maxDistanceError
            failed = true;
            failureReason = 'path-distance limit';
        elseif abs(ePsi) > cfg.reward.maxHeadingError
            failed = true;
            failureReason = 'path-heading limit';
        elseif any(~isfinite(xp))
            failed = true;
            failureReason = 'non-finite plant state';
        end

        if failed
            break;
        end

        if sNow >= path.s(end)-cfg.finishTolerance
            passed = true;
            break;
        end
    end

    if ~failed && ~passed
        failed = true;
        failureReason = 'episode time limit';
    end

    nStateSamples = finalStep+1;
    nControlSamples = finalStep;
    stateRange = 1:nStateSamples;
    controlRange = 1:nControlSamples;

    alphaUsed = alphaHist(:,stateRange);
    maxSlipFront = max(abs(alphaUsed(1,:)));
    maxSlipRear = max(abs(alphaUsed(2,:)));
    maxSlipTrailer = max(abs(alphaUsed(3,:)));

    % Feher et al.-style successful-episode reward. In the paper v0 is in
    % km/h in the fitted speed-dependent lateral-slip reference.
    speedKmh = 3.6*cfg.V;
    slipReference = 0.0037*exp(0.0693*speedKmh);
    rewardSlip = 2*slipReference-maxSlipFront-maxSlipRear;

    kappaDD = path.kappaDD;
    rewardCurvature = cfg.reward.cKappaDD ...
        -abs(max(kappaDD))-abs(min(kappaDD));

    progress = min(max(sHist(nStateSamples)/path.s(end),0),1);
    if passed
        rewardPenalty = 0;
        reward = cfg.reward.wCurvature*rewardCurvature ...
            +cfg.reward.wSlip*rewardSlip;
    else
        if progress >= cfg.reward.lastSectionFraction
            rewardPenalty = cfg.reward.failurePenaltyLastSection;
        else
            rewardPenalty = cfg.reward.failurePenalty;
        end
        reward = rewardPenalty;
    end

    rewards = struct();
    rewards.total = reward;
    rewards.slip = rewardSlip;
    rewards.curvature = rewardCurvature;
    rewards.penalty = rewardPenalty;
    rewards.slipReference = slipReference;
    rewards.passed = passed;
    rewards.failed = failed;
    rewards.failureReason = failureReason;
    rewards.progress = progress;
    rewards.maxSlipFront = maxSlipFront;
    rewards.maxSlipRear = maxSlipRear;
    rewards.maxSlipTrailer = maxSlipTrailer;

    log = struct();
    log.time = tHist(stateRange);
    log.plantState = plantHist(:,stateRange);
    log.mpcState = mpcHist(:,stateRange);
    log.deltaCommand = uHist(controlRange);
    log.referenceY = refYHist(controlRange);
    log.referencePsi = refPsiHist(controlRange);
    log.lateralError = eLatHist(stateRange);
    log.headingError = ePsiHist(stateRange);
    log.distanceError = distHist(stateRange);
    log.slipAngles = alphaHist(:,stateRange);
    log.lateralForces = forceHist(:,stateRange);
    log.pathProgress = sHist(stateRange);
    if cfg.logPhysicalDiagnostics
        log.physicalDiagnostics = struct( ...
            'massRelativeResidual',massResidualHist(stateRange), ...
            'massRcond',massRcondHist(stateRange), ...
            'yawRateIdentity',yawRateIdentityHist(stateRange), ...
            'articulationRateIdentity',articulationRateIdentityHist(stateRange));
    end
    log.path = path;
    log.rewards = rewards;
    log.metrics = make_metrics(log);
    log.config = cfg;
end


function cfg = ttv_default_config(cfg)
    cfg = set_default(cfg,'casadiPath','');
    cfg = set_default(cfg,'logPhysicalDiagnostics',false);
    cfg = set_default(cfg,'T',0.1);
    cfg = set_default(cfg,'N',10);
    cfg = set_default(cfg,'V',20);
    cfg = set_default(cfg,'maxEpisodeTime',30);
    cfg = set_default(cfg,'plantSubsteps',5);
    cfg = set_default(cfg,'minPathLength',5);
    cfg = set_default(cfg,'finishTolerance',0.5);
    cfg = set_default(cfg,'initialPlantState',[]);

    cfg = set_default(cfg,'deltaMax',0.5);
    cfg = set_default(cfg,'deltaRateMax',0.6);
    cfg = set_default(cfg,'steeringTimeConstant',0.15);
    cfg = set_default(cfg,'phiMax',inf);
    cfg = set_default(cfg,'qMax',inf);

    cfg = set_default(cfg,'m1',5760);
    cfg = set_default(cfg,'m2',6640);
    cfg = set_default(cfg,'a1',1.10);
    cfg = set_default(cfg,'a2',5.21);
    cfg = set_default(cfg,'b1',2.39);
    cfg = set_default(cfg,'b2',3.28);
    cfg = set_default(cfg,'c',1.64);
    cfg = set_default(cfg,'Iz1',34823);
    cfg = set_default(cfg,'Iz2',179992);
    cfg = set_default(cfg,'C1',223281);
    cfg = set_default(cfg,'C2',223281);
    cfg = set_default(cfg,'C3',223281);
    cfg = set_default(cfg,'mu',0.90);
    cfg = set_default(cfg,'g',9.81);

    if ~isfield(cfg,'Fz1') || ~isfield(cfg,'Fz2') || ~isfield(cfg,'Fz3') ...
            || isempty(cfg.Fz1) || isempty(cfg.Fz2) || isempty(cfg.Fz3)
        hitchLoad = cfg.m2*cfg.g*cfg.b2/(cfg.a2+cfg.b2);
        trailerAxleLoad = cfg.m2*cfg.g*cfg.a2/(cfg.a2+cfg.b2);
        tractorFrontLoad = (cfg.b1*cfg.m1*cfg.g ...
            +(cfg.b1-cfg.c)*hitchLoad)/(cfg.a1+cfg.b1);
        tractorRearLoad = cfg.m1*cfg.g+hitchLoad-tractorFrontLoad;
        cfg.Fz1 = tractorFrontLoad;
        cfg.Fz2 = tractorRearLoad;
        cfg.Fz3 = trailerAxleLoad;
    end

    if ~isfield(cfg,'mpc') || isempty(cfg.mpc), cfg.mpc = struct(); end
    cfg.mpc = set_default(cfg.mpc,'Qy',2.0);
    cfg.mpc = set_default(cfg.mpc,'Qpsi',1.0);
    cfg.mpc = set_default(cfg.mpc,'Qphi',0.0);
    cfg.mpc = set_default(cfg.mpc,'Qq',0.0);
    cfg.mpc = set_default(cfg.mpc,'Rdelta',0.1);
    cfg.mpc = set_default(cfg.mpc,'maxIterations',300);
    cfg.mpc = set_default(cfg.mpc,'acceptableTolerance',1e-7);

    if ~isfield(cfg,'reward') || isempty(cfg.reward), cfg.reward = struct(); end
    cfg.reward = set_default(cfg.reward,'wCurvature',0.5);
    cfg.reward = set_default(cfg.reward,'wSlip',0.5);
    cfg.reward = set_default(cfg.reward,'cKappaDD',0.0);
    cfg.reward = set_default(cfg.reward,'maxLateralSlip',0.2);
    cfg.reward = set_default(cfg.reward,'maxDistanceError',1.0);
    cfg.reward = set_default(cfg.reward,'maxHeadingError',deg2rad(20));
    cfg.reward = set_default(cfg.reward,'failurePenalty',-1.0);
    cfg.reward = set_default(cfg.reward,'failurePenaltyLastSection',-0.75);
    cfg.reward = set_default(cfg.reward,'lastSectionFraction',0.80);

    validateattributes(cfg.T,{'double'},{'scalar','positive','finite'});
    validateattributes(cfg.N,{'double'},{'scalar','integer','>=',2});
    validateattributes(cfg.V,{'double'},{'scalar','positive','finite'});
    validateattributes(cfg.logPhysicalDiagnostics,{'logical','numeric'}, {'scalar'});
    cfg.logPhysicalDiagnostics = logical(cfg.logPhysicalDiagnostics);
    validateattributes(cfg.plantSubsteps,{'double'},{'scalar','integer','>=',1});
    if abs(cfg.reward.wCurvature+cfg.reward.wSlip-1) > 1e-10
        error('ttv_rl_episode:RewardWeights', ...
            'cfg.reward.wCurvature + cfg.reward.wSlip must equal one.');
    end
end


function ensure_casadi(cfg)
    if ~isempty(cfg.casadiPath)
        addpath(cfg.casadiPath);
    end
    try
        import casadi.* %#ok<IMPORT>
        casadi.SX.sym('casadi_check');
    catch
        error('ttv_rl_episode:CasadiMissing', ...
            ['CasADi is not available. Add it to the MATLAB path or set ', ...
             'cfg.casadiPath to the CasADi MATLAB directory.']);
    end
end


function cache = build_environment(cfg,key)
    import casadi.*

    [Ac,Bc] = build_linear_prediction_model(cfg);
    nx = size(Ac,1);
    aug = expm([Ac,Bc;zeros(1,nx+1)]*cfg.T);
    Ad = aug(1:nx,1:nx);
    Bd = aug(1:nx,nx+1);

    X = SX.sym('X',nx,cfg.N+1);
    U = SX.sym('U',1,cfg.N);
    P = SX.sym('P',nx+2*cfg.N,1);
    objective = 0;
    constraints = X(:,1)-P(1:nx);

    for k = 1:cfg.N
        yRef = P(nx+2*k-1);
        psiRef = P(nx+2*k);
        eY = X(1,k+1)-yRef;
        ePsi = X(3,k+1)-psiRef;
        objective = objective ...
            +cfg.mpc.Qy*eY^2 ...
            +cfg.mpc.Qpsi*ePsi^2 ...
            +cfg.mpc.Qphi*X(5,k+1)^2 ...
            +cfg.mpc.Qq*X(6,k+1)^2 ...
            +cfg.mpc.Rdelta*U(k)^2;
        constraints = [constraints; X(:,k+1)-(Ad*X(:,k)+Bd*U(k))]; %#ok<AGROW>
    end

    decision = [X(:);U(:)];
    nlp = struct('f',objective,'x',decision,'g',constraints,'p',P);
    opts = struct;
    opts.ipopt.max_iter = cfg.mpc.maxIterations;
    opts.ipopt.print_level = 0;
    opts.ipopt.acceptable_tol = cfg.mpc.acceptableTolerance;
    opts.ipopt.acceptable_obj_change_tol = 1e-6;
    opts.print_time = false;
    solver = nlpsol('ttv_mpc_solver','ipopt',nlp,opts);

    nw = nx*(cfg.N+1)+cfg.N;
    lbx = -inf(nw,1);
    ubx = inf(nw,1);
    for k = 0:cfg.N
        stateOffset = k*nx;
        lbx(stateOffset+7) = -cfg.deltaMax;
        ubx(stateOffset+7) = cfg.deltaMax;
        if isfinite(cfg.phiMax)
            lbx(stateOffset+5) = -cfg.phiMax;
            ubx(stateOffset+5) = cfg.phiMax;
        end
        if isfinite(cfg.qMax)
            lbx(stateOffset+6) = -cfg.qMax;
            ubx(stateOffset+6) = cfg.qMax;
        end
    end
    controlStart = nx*(cfg.N+1)+1;
    lbx(controlStart:end) = -cfg.deltaMax;
    ubx(controlStart:end) = cfg.deltaMax;

    ng = nx*(cfg.N+1);
    lbg = zeros(ng,1);
    ubg = zeros(ng,1);

    [plantDynamics,plantOutput,plantDiagnostics] = build_nonlinear_plant(cfg);

    cache = struct('key',key,'Ad',Ad,'Bd',Bd,'solver',solver, ...
        'lbx',lbx,'ubx',ubx,'lbg',lbg,'ubg',ubg, ...
        'plantDynamics',plantDynamics,'plantOutput',plantOutput, ...
        'plantDiagnostics',plantDiagnostics);
end


function [Ac,Bc] = build_linear_prediction_model(cfg)
    % The descriptor matrices use the sign convention of the source AHV
    % paper: Fy_i = C_i*alpha_i with signed C_i < 0.
    C1s = -abs(cfg.C1);
    C2s = -abs(cfg.C2);
    C3s = -abs(cfg.C3);
    V = cfg.V;

    M = [cfg.Iz1, cfg.m1*cfg.c*V, 0, 0;
        -cfg.m2*(cfg.a2+cfg.c), (cfg.m1+cfg.m2)*V, -cfg.m2*cfg.a2, C3s*(cfg.a2+cfg.b2)/V;
        cfg.Iz2, cfg.m1*cfg.a2*V, cfg.Iz2, -C3s*cfg.b2*(cfg.a2+cfg.b2)/V;
        0, 0, 0, 1];

    K = [cfg.m1*cfg.c*V-(C1s*cfg.a1*(cfg.a1+cfg.c)+C2s*cfg.b1*(cfg.b1-cfg.c))/V, ...
        -C1s*(cfg.a1+cfg.c)+C2s*(cfg.b1-cfg.c), 0, 0;
        (cfg.m1+cfg.m2)*V+(-C1s*cfg.a1+C2s*cfg.b1+C3s*(cfg.a2+cfg.b2+cfg.c))/V, ...
        -(C1s+C2s+C3s), 0, C3s;
        cfg.m1*cfg.a2*V+(-C1s*cfg.a1*cfg.a2+C2s*cfg.b1*cfg.a2-C3s*cfg.b2*(cfg.a2+cfg.b2+cfg.c))/V, ...
        -(C1s*cfg.a2+C2s*cfg.a2-C3s*cfg.b2), 0, -C3s*cfg.b2;
        0, 0, -1, 0];

    G = [-C1s*(cfg.a1+cfg.c);-C1s;-C1s*cfg.a2;0];
    Av = -(M\K);
    Bv = M\G;

    % MPC state: [y; y_dot; psi; r1; phi; q; delta_actual].
    Ac = zeros(7,7);
    Ac(1,2) = 1;
    Ac(2,:) = [0, Av(2,2), -V*Av(2,2), V*(Av(2,1)+1), ...
        V*Av(2,4), V*Av(2,3), V*Bv(2)];
    Ac(3,4) = 1;
    Ac(4,:) = [0, Av(1,2)/V, -Av(1,2), Av(1,1), ...
        Av(1,4), Av(1,3), Bv(1)];
    Ac(5,6) = 1;
    Ac(6,:) = [0, Av(3,2)/V, -Av(3,2), Av(3,1), ...
        Av(3,4), Av(3,3), Bv(3)];
    Ac(7,7) = -1/cfg.steeringTimeConstant;

    Bc = zeros(7,1);
    Bc(7) = 1/cfg.steeringTimeConstant;
end


function [plantDynamics,plantOutput,plantDiagnostics] = build_nonlinear_plant(cfg)
    import casadi.*

    xp = SX.sym('xp',8,1);
    deltaCmd = SX.sym('delta_cmd');

    X1 = xp(1); %#ok<NASGU>
    Y1 = xp(2); %#ok<NASGU>
    psi1 = xp(3);
    v1 = xp(4);
    r1 = xp(5);
    phi = xp(6);
    q = xp(7);
    delta = xp(8);
    r2 = r1+q;
    V = cfg.V;

    hitchLateralVelocity = v1-cfg.c*r1;
    V2 = V*cos(phi)+hitchLateralVelocity*sin(phi);
    v2 = -V*sin(phi)+hitchLateralVelocity*cos(phi)-cfg.a2*r2;

    alpha1 = delta-atan2(v1+cfg.a1*r1,V);
    alpha2 = -atan2(v1-cfg.b1*r1,V);
    alpha3 = -atan2(v2-cfg.b2*r2,V2);
    alpha = [alpha1;alpha2;alpha3];

    Fy1 = cfg.mu*cfg.Fz1*tanh(cfg.C1*alpha1/(cfg.mu*cfg.Fz1));
    Fy2 = cfg.mu*cfg.Fz2*tanh(cfg.C2*alpha2/(cfg.mu*cfg.Fz2));
    Fy3 = cfg.mu*cfg.Fz3*tanh(cfg.C3*alpha3/(cfg.mu*cfg.Fz3));
    Fy = [Fy1;Fy2;Fy3];

    Mplant = [cfg.m1, 0, 0, 0, 1;
        0, cfg.Iz1, 0, 0, -cfg.c;
        cfg.m2*sin(phi), -cfg.m2*cfg.c*sin(phi), 0, -cos(phi), -sin(phi);
        cfg.m2*cos(phi), -cfg.m2*(cfg.c*cos(phi)+cfg.a2), -cfg.m2*cfg.a2, sin(phi), -cos(phi);
        0, cfg.Iz2, cfg.Iz2, cfg.a2*sin(phi), -cfg.a2*cos(phi)];

    bplant = [Fy1*cos(delta)+Fy2-cfg.m1*V*r1;
        cfg.a1*Fy1*cos(delta)-cfg.b1*Fy2;
        cfg.m2*(v2*r1-cfg.a2*r2*q);
        Fy3-cfg.m2*V2*r1;
        -cfg.b2*Fy3];

    zeta = solve(Mplant,bplant);
    v1Dot = zeta(1);
    r1Dot = zeta(2);
    qDot = zeta(3);

    rawDeltaRate = (deltaCmd-delta)/cfg.steeringTimeConstant;
    deltaDot = cfg.deltaRateMax*tanh(rawDeltaRate/cfg.deltaRateMax);

    xDot = [V*cos(psi1)-v1*sin(psi1);
        V*sin(psi1)+v1*cos(psi1);
        r1;
        v1Dot;
        r1Dot;
        q;
        qDot;
        deltaDot];

    plantDynamics = Function('ttv_plant_dynamics',{xp,deltaCmd},{xDot});
    plantOutput = Function('ttv_plant_output',{xp,deltaCmd}, ...
        {alpha,Fy,[V2;v2;zeta(4);zeta(5)]});
    plantDiagnostics = Function('ttv_plant_diagnostics',{xp,deltaCmd}, ...
        {Mplant,zeta,bplant,xDot});
end


function nextState = rk4_step(dynamics,state,input,h)
    k1 = full(dynamics(state,input));
    k2 = full(dynamics(state+0.5*h*k1,input));
    k3 = full(dynamics(state+0.5*h*k2,input));
    k4 = full(dynamics(state+h*k3,input));
    nextState = state+h*(k1+2*k2+2*k3+k4)/6;
end


function xm = plant_to_mpc_state(xp,V)
    psi = xp(3);
    v1 = xp(4);
    yDot = V*sin(psi)+v1*cos(psi);
    xm = [xp(2);yDot;psi;xp(5);xp(6);xp(7);xp(8)];
end


function [alpha,Fy] = plant_outputs(cache,xp,input)
    [alphaCasadi,FyCasadi] = cache.plantOutput(xp,input);
    alpha = full(alphaCasadi);
    Fy = full(FyCasadi);
end


function diagnostics = plant_diagnostics(cache,xp,input)
    [Mcasadi,zetaCasadi,bCasadi,xDotCasadi] = cache.plantDiagnostics(xp,input);
    Mplant = full(Mcasadi);
    zeta = full(zetaCasadi);
    bplant = full(bCasadi);
    xDot = full(xDotCasadi);
    residual = Mplant*zeta-bplant;
    scale = max([1,norm(Mplant*zeta,2),norm(bplant,2)]);

    diagnostics = struct();
    diagnostics.massRelativeResidual = norm(residual,2)/scale;
    diagnostics.massRcond = rcond(Mplant);
    diagnostics.yawRateIdentity = (xp(5)+xp(7))-xp(5)-xp(7);
    diagnostics.articulationRateIdentity = xDot(6)-xp(7);
end


function [sNow,eLat,ePsi,dist] = path_errors(xp,path)
    distanceSquared = (path.x-xp(1)).^2+(path.y-xp(2)).^2;
    [~,index] = min(distanceSquared);
    sNow = path.s(index);
    dx = xp(1)-path.x(index);
    dy = xp(2)-path.y(index);
    eLat = -sin(path.psi(index))*dx+cos(path.psi(index))*dy;
    ePsi = wrap_angle(xp(3)-path.psi(index));
    dist = sqrt(distanceSquared(index));
end


function path = prepare_path(pathInput)
    if isnumeric(pathInput)
        if size(pathInput,2) ~= 2
            error('ttv_rl_episode:PathFormat', ...
                'A numeric path must be an N-by-2 array [x,y].');
        end
        x = pathInput(:,1);
        y = pathInput(:,2);
    elseif isstruct(pathInput) && isfield(pathInput,'x') && isfield(pathInput,'y')
        x = pathInput.x(:);
        y = pathInput.y(:);
    else
        error('ttv_rl_episode:PathFormat', ...
            'pathInput must be [x,y] or a structure with fields x and y.');
    end

    if numel(x) ~= numel(y) || numel(x) < 7
        error('ttv_rl_episode:PathSize', ...
            'The path must contain at least seven paired x-y points.');
    end
    if any(~isfinite(x)) || any(~isfinite(y))
        error('ttv_rl_episode:PathFinite','Path coordinates must be finite.');
    end

    segmentLength = hypot(diff(x),diff(y));
    keep = [true;segmentLength > 1e-8];
    x = x(keep);
    y = y(keep);
    if numel(x) < 7
        error('ttv_rl_episode:PathUnique', ...
            'The path contains too few distinct points.');
    end

    % Normalize to the initial path pose. This keeps the small-angle linear
    % MPC model valid for double-lane-change paths with arbitrary placement.
    psi0 = atan2(y(2)-y(1),x(2)-x(1));
    rotation = [cos(psi0),sin(psi0);-sin(psi0),cos(psi0)];
    localPoints = rotation*[x-x(1),y-y(1)]';
    x = localPoints(1,:)';
    y = localPoints(2,:)';

    s = [0;cumsum(hypot(diff(x),diff(y)))];
    dxds = gradient(x,s);
    dyds = gradient(y,s);
    psi = unwrap(atan2(dyds,dxds));
    kappa = gradient(psi,s);
    kappaPrime = gradient(kappa,s);
    kappaDD = gradient(kappaPrime,s);

    path = struct('x',x,'y',y,'s',s,'psi',psi, ...
        'kappa',kappa,'kappaPrime',kappaPrime,'kappaDD',kappaDD);
end


function metrics = make_metrics(log)
    metrics = struct();
    metrics.rmsLateralError = sqrt(mean(log.lateralError.^2));
    metrics.peakLateralError = max(abs(log.lateralError));
    metrics.rmsHeadingError = sqrt(mean(log.headingError.^2));
    metrics.peakHeadingError = max(abs(log.headingError));
    metrics.peakArticulationAngle = max(abs(log.plantState(6,:)));
    metrics.peakArticulationRate = max(abs(log.plantState(7,:)));
    metrics.rmsArticulationRate = sqrt(mean(log.plantState(7,:).^2));
    metrics.peakSteeringCommand = max(abs(log.deltaCommand));
    metrics.peakActualSteering = max(abs(log.plantState(8,:)));
    metrics.maxSlipFront = max(abs(log.slipAngles(1,:)));
    metrics.maxSlipRear = max(abs(log.slipAngles(2,:)));
    metrics.maxSlipTrailer = max(abs(log.slipAngles(3,:)));
end


function status = solver_status(stats)
    if isfield(stats,'return_status')
        status = char(stats.return_status);
    else
        status = 'unknown failure';
    end
end


function angle = wrap_angle(angle)
    angle = atan2(sin(angle),cos(angle));
end


function value = set_default(value,fieldName,defaultValue)
    if ~isfield(value,fieldName) || isempty(value.(fieldName))
        value.(fieldName) = defaultValue;
    end
end


function key = make_cache_key(cfg)
    values = [cfg.T,cfg.N,cfg.V,cfg.plantSubsteps,cfg.deltaMax, ...
        cfg.deltaRateMax,cfg.steeringTimeConstant,cfg.phiMax,cfg.qMax, ...
        cfg.m1,cfg.m2,cfg.a1,cfg.a2,cfg.b1,cfg.b2,cfg.c,cfg.Iz1,cfg.Iz2, ...
        cfg.C1,cfg.C2,cfg.C3,cfg.mu,cfg.Fz1,cfg.Fz2,cfg.Fz3, ...
        cfg.mpc.Qy,cfg.mpc.Qpsi,cfg.mpc.Qphi,cfg.mpc.Qq, ...
        cfg.mpc.Rdelta,cfg.mpc.maxIterations,cfg.mpc.acceptableTolerance];
    key = sprintf('%.16g,',values);
end

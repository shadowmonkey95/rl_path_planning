% TTV_RL_EPISODE Run one complete closed-loop RL evaluation episode.
%
% Inputs
%   pathInput : N-by-2 [x,y] path or struct with fields x and y.
%   cfg       : Optional configuration struct. Missing/empty fields receive
%               defaults from ttv_default_config().
%
% Outputs
%   reward    : One scalar episode reward returned to the RL algorithm.
%   rewards   : Struct containing reward components, success/failure status,
%               progress, and peak slip values.
%   log       : Struct containing the full simulated trajectory and metrics.
%
% Main data flow
%   pathInput -> prepare_path -> MPC preview -> steering command
%             -> nonlinear tractor-trailer plant -> tracking/slip metrics
%             -> episode reward.
%
% Important design point
%   The RL agent is assumed to have already generated the geometric path.
%   This function therefore evaluates path quality; it is not itself the
%   neural-network/RL policy.
%
% State conventions
%   Plant state xp = [X1,Y1,psi1,v1,r1,phi,q,delta_actual].'
%   MPC state   xm = [Y,Y_dot,psi,r1,phi,q,delta_actual].'
%   X1,Y1        : tractor position [m]
%   psi1         : tractor heading [rad]
%   v1           : tractor lateral velocity in body frame [m/s]
%   r1           : tractor yaw rate [rad/s]
%   phi          : tractor-trailer articulation angle [rad]
%   q            : articulation-rate variable [rad/s]
%   delta_actual : actual front steering angle [rad]

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

    % nargin = number of input arguments supplied by the caller.
    % Here cfg is optional: no cfg, or an empty cfg, means use defaults.
    if nargin < 2 || isempty(cfg)
        cfg = struct();
    end
    % Fill every missing configuration field before any model is built.
    cfg = ttv_default_config(cfg);
    ensure_casadi(cfg);

    % Convert raw [x,y] points into the internally used road-fixed path struct.
    path = prepare_path(pathInput);
    if path.s(end) <= cfg.minPathLength
        error('ttv_rl_episode:ShortPath', ...
            'The usable path length must exceed %.3f m.', cfg.minPathLength);
    end

    % persistent keeps the expensive CasADi solver/model alive between calls.
    persistent cache
    % Cache key changes when model/MPC parameters change.
    cacheKey = make_cache_key(cfg);
    if isempty(cache) || ~isfield(cache,'key') || ~strcmp(cache.key,cacheKey)
        cache = build_environment(cfg,cacheKey);
    end

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

    % nx = number of MPC state variables: y, y_dot, psi, r1, phi, q, delta.
    nx = 7;
    % N = number of future control intervals in the MPC horizon.
    N = cfg.N;
    % Number of MPC updates allowed: ceil(total episode time / update period).
    maxSteps = max(1,ceil(cfg.maxEpisodeTime/cfg.T));

    % Warm starts. The CasADi solver is persistent; these guesses remain
    % local to the episode and are shifted after every MPC solution.
    xm = plant_to_mpc_state(xp,cfg.V);
    % Initial optimization guess: repeat current state over the full horizon.
    Xguess = repmat(xm,1,N+1);
    % Initial steering-command guess is zero at every future step.
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
        % Future arc-length preview assumes constant forward speed V: s ~= sNow + V*T*k.
        sPreview = sNow + cfg.V*cfg.T*(1:N);
        sPreview = min(max(sPreview,path.s(1)),path.s(end));
        yRef = interp1(path.s,path.y,sPreview,'pchip');
        psiRef = interp1(path.s,path.psi,sPreview,'linear');

        % p contains current MPC state plus N pairs of [reference y, reference psi].
        p = zeros(nx+2*N,1);
        p(1:nx) = xm;
        for j = 1:N
            p(nx+2*j-1) = yRef(j);
            p(nx+2*j) = psiRef(j);
        end

        % CasADi decision vector = all predicted states followed by all controls.
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
        % First nx*(N+1) values are the predicted state trajectory X.
        Xopt = reshape(wOpt(1:nx*(N+1)),nx,N+1);
        % Remaining N values are the predicted steering commands U.
        Uopt = reshape(wOpt(nx*(N+1)+1:end),1,N);
        % Receding-horizon control: only the first optimized command is applied.
        deltaCmd = min(max(Uopt(1),-cfg.deltaMax),cfg.deltaMax);

        % Integrate the nonlinear plant with smaller RK4 substeps.
        % Integrate the nonlinear plant with smaller steps than the MPC period.
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
    % Convert m/s to km/h because the fitted slip-reference formula uses km/h.
    speedKmh = 3.6*cfg.V;
    % Empirical speed-dependent target slip angle.
    slipReference = 0.0037*exp(0.0693*speedKmh);
    % Higher reward when front/rear tractor slip stays below the reference.
    rewardSlip = 2*slipReference-maxSlipFront-maxSlipRear;

    kappaDD = path.kappaDD;
    % Smoothness reward: penalize both positive and negative extreme curvature second derivative.
    rewardCurvature = cfg.reward.cKappaDD ...
        -abs(max(kappaDD))-abs(min(kappaDD));

    % Normalize traveled path arc length to [0,1].
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


% TTV_DEFAULT_CONFIG Fill a configuration struct with simulation/MPC defaults.
%
% Input
%   cfg : User-supplied configuration struct. Existing non-empty fields are
%         preserved; missing or empty fields are assigned defaults.
%
% Output
%   cfg : Completed configuration struct.
%
% Key parameters
%   T              : MPC/control update period [s].
%   N              : MPC prediction horizon [steps].
%   V              : Constant longitudinal speed used by the model [m/s].
%   deltaMax       : Steering-angle bound [rad].
%   deltaRateMax   : Steering-rate saturation bound [rad/s].
%   plantSubsteps  : Number of RK4 substeps inside one MPC interval.
%   m1,m2          : Tractor/trailer masses [kg].
%   a1,b1          : Tractor CG distances to front/rear axles [m].
%   a2,b2          : Trailer/hitch geometry distances [m].
%   c              : Tractor CG-to-hitch geometry parameter [m].
%   Iz1,Iz2        : Yaw inertias [kg m^2].
%   C1,C2,C3       : Tire cornering-stiffness magnitudes [N/rad].
%   Fz1,Fz2,Fz3    : Normal tire loads [N].
%   mu             : Tire-road friction coefficient [-].
%   g              : Gravity [m/s^2].
%
% Derived static loads
%   hitchLoad = m2*g*b2/(a2+b2)
%   trailerAxleLoad = m2*g*a2/(a2+b2)
%   tractorFrontLoad = [b1*m1*g + (b1-c)*hitchLoad]/(a1+b1)
%   tractorRearLoad = m1*g + hitchLoad - tractorFrontLoad
%   These distribute the static weight and hitch load among the three
%   modeled lateral tire/axle locations.

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


% ENSURE_CASADI Verify that the CasADi MATLAB interface is available.
%
% Input
%   cfg : Configuration struct; cfg.casadiPath optionally contains the
%         directory containing CasADi's MATLAB bindings.
%
% Side effect
%   Adds cfg.casadiPath to MATLAB's search path when it is non-empty.
%
% Why this is needed
%   The MPC NLP and nonlinear plant are created with CasADi symbolic objects
%   (SX) and functions. The episode cannot run without CasADi.

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


% BUILD_ENVIRONMENT Build and cache the MPC NLP plus nonlinear plant model.
%
% Inputs
%   cfg : Completed simulation/MPC configuration.
%   key : String uniquely describing model parameters used for caching.
%
% Output
%   cache : Struct containing discrete MPC matrices, CasADi solver, bounds,
%           and nonlinear plant/output/diagnostic functions.
%
% MPC optimization problem
%   Decision variables are X(1:N+1) and U(1:N).
%   Objective approximately follows
%       J = sum_k [Qy*eY^2 + Qpsi*ePsi^2 + Qphi*phi^2
%                  + Qq*q^2 + Rdelta*delta_cmd^2]
%   subject to the discrete linear prediction model
%       X_{k+1} = Ad*X_k + Bd*U_k.
%
% The equality constraints also force X(:,1) to equal the measured current
% MPC state supplied through parameter vector P.

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


% BUILD_LINEAR_PREDICTION_MODEL Construct the continuous-time linear MPC model.
%
% Input
%   cfg : Vehicle parameters and speed.
%
% Outputs
%   Ac : 7x7 continuous-time state matrix.
%   Bc : 7x1 continuous-time steering-input matrix.
%
% State ordering
%   xm = [y, y_dot, psi, r1, phi, q, delta_actual].'
%
% Tire convention
%   Fy_i = C_i*alpha_i, with C_i made negative here. The sign convention is
%   inherited from the source model used by this file.
%
% Continuous linear model
%   x_dot = Ac*x + Bc*delta_cmd.
%
% The intermediate descriptor form is M*z_dot = K*z + G*delta, so MATLAB's
% backslash operator solves z_dot = -(M\K)z + (M\G)delta.

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


% BUILD_NONLINEAR_PLANT Create CasADi functions for the nonlinear vehicle model.
%
% Input
%   cfg : Vehicle geometry, mass, tire, friction, speed, and steering data.
%
% Outputs
%   plantDynamics    : Function (xp,deltaCmd) -> x_dot.
%   plantOutput      : Function -> tire slip angles, lateral forces, and
%                      selected trailer/hitch quantities.
%   plantDiagnostics : Function -> mass matrix M, solved accelerations,
%                      right-hand side b, and complete x_dot.
%
% Nonlinear state
%   xp = [X1,Y1,psi1,v1,r1,phi,q,delta_actual].'
%
% Vehicle kinematics
%   r2 = r1 + q                       trailer yaw rate
%   hitchLateralVelocity = v1-c*r1    lateral velocity at hitch reference
%   V2 = V*cos(phi) + hitchV*sin(phi) trailer longitudinal velocity
%   v2 = -V*sin(phi) + hitchV*cos(phi) - a2*r2
%
% Tire slip angles
%   alpha1 = delta - atan2(v1+a1*r1,V)
%   alpha2 = -atan2(v1-b1*r1,V)
%   alpha3 = -atan2(v2-b2*r2,V2)
%
% Saturating tire-force law
%   Fy_i = mu*Fz_i*tanh(C_i*alpha_i/(mu*Fz_i))
%   For small slip, tanh(x) ~= x, so the force is approximately linear;
%   for large slip, tanh prevents the force from growing without bound.
%
% Dynamics
%   Mplant*zeta = bplant, solved using zeta = Mplant\bplant.
%   zeta contains the accelerations needed to construct x_dot.
%
% Steering actuator
%   rawDeltaRate = (deltaCmd-delta)/steeringTimeConstant
%   deltaDot = deltaRateMax*tanh(rawDeltaRate/deltaRateMax)
%   This behaves like a first-order steering actuator with a smooth rate
%   saturation.

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


% RK4_STEP Advance a nonlinear ODE by one Runge-Kutta 4th-order step.
%
% Inputs
%   dynamics : Function returning x_dot = f(x,u).
%   state    : Current state vector x_k.
%   input    : Held control input u over this integration step.
%   h        : Integration time step [s].
%
% Output
%   nextState : Approximation of x(t+h).
%
% Formula
%   k1 = f(x_k,u)
%   k2 = f(x_k + h*k1/2,u)
%   k3 = f(x_k + h*k2/2,u)
%   k4 = f(x_k + h*k3,u)
%   x_{k+1} = x_k + h*(k1 + 2*k2 + 2*k3 + k4)/6.

function nextState = rk4_step(dynamics,state,input,h)
    k1 = full(dynamics(state,input));
    k2 = full(dynamics(state+0.5*h*k1,input));
    k3 = full(dynamics(state+0.5*h*k2,input));
    k4 = full(dynamics(state+h*k3,input));
    nextState = state+h*(k1+2*k2+2*k3+k4)/6;
end


% PLANT_TO_MPC_STATE Convert the full nonlinear plant state to MPC state.
%
% Inputs
%   xp : 8x1 nonlinear plant state.
%   V  : Constant longitudinal speed [m/s].
%
% Output
%   xm : 7x1 MPC state [Y,Y_dot,psi,r1,phi,q,delta_actual].'
%
% Kinematic conversion
%   Y_dot = V*sin(psi) + v1*cos(psi).
%   This rotates the body-frame lateral velocity into the road/global Y
%   direction and adds the contribution from forward motion.

function xm = plant_to_mpc_state(xp,V)
    psi = xp(3);
    v1 = xp(4);
    yDot = V*sin(psi)+v1*cos(psi);
    xm = [xp(2);yDot;psi;xp(5);xp(6);xp(7);xp(8)];
end


% PLANT_OUTPUTS Evaluate tire slip angles and lateral tire forces.
%
% Inputs
%   cache : Environment cache containing CasADi plantOutput function.
%   xp    : Current nonlinear plant state.
%   input : Current steering command.
%
% Outputs
%   alpha : 3x1 slip-angle vector [front; tractor-rear; trailer] [rad].
%   Fy    : 3x1 lateral-force vector [N].

function [alpha,Fy] = plant_outputs(cache,xp,input)
    [alphaCasadi,FyCasadi] = cache.plantOutput(xp,input);
    alpha = full(alphaCasadi);
    Fy = full(FyCasadi);
end


% PLANT_DIAGNOSTICS Compute numerical consistency checks for the plant.
%
% Inputs
%   cache : Environment cache.
%   xp    : Current nonlinear state.
%   input : Steering command.
%
% Output
%   diagnostics : Struct containing residual/conditioning and kinematic
%                 identity checks.
%
% Checks
%   residual = M*zeta-b should be near zero because zeta solves M*zeta=b.
%   massRelativeResidual normalizes its 2-norm by the largest relevant scale.
%   massRcond = rcond(M) estimates conditioning; very small values indicate
%   that the mass matrix is close to singular.

function diagnostics = plant_diagnostics(cache,xp,input)
    [Mcasadi,zetaCasadi,bCasadi,xDotCasadi] = cache.plantDiagnostics(xp,input);
    Mplant = full(Mcasadi);
    zeta = full(zetaCasadi);
    bplant = full(bCasadi);
    xDot = full(xDotCasadi);
    % Numerical residual of the solved linear system M*zeta=b.
    residual = Mplant*zeta-bplant;
    scale = max([1,norm(Mplant*zeta,2),norm(bplant,2)]);

    diagnostics = struct();
    diagnostics.massRelativeResidual = norm(residual,2)/scale;
    % Reciprocal condition estimate; small rcond means Mplant may be ill-conditioned.
    diagnostics.massRcond = rcond(Mplant);
    diagnostics.yawRateIdentity = (xp(5)+xp(7))-xp(5)-xp(7);
    diagnostics.articulationRateIdentity = xDot(6)-xp(7);
end


% PATH_ERRORS Find the closest path point and calculate tracking errors.
%
% Inputs
%   xp   : Nonlinear plant state; xp(1:2) are tractor X/Y position.
%   path : Prepared path containing x,y,s,psi.
%
% Outputs
%   sNow : Arc-length coordinate of the closest path point [m].
%   eLat : Signed lateral/path-normal error [m].
%   ePsi : Wrapped heading error [rad].
%   dist : Euclidean distance to the closest path point [m].
%
% Formula
%   eLat = -sin(psi_ref)*(X-X_ref) + cos(psi_ref)*(Y-Y_ref).
%   This is the position error projected onto the path's left-normal vector.

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


% PREPARE_PATH Validate, clean, transform, and differentiate a geometric path.
%
% Input
%   pathInput : N-by-2 [x,y] numeric array or struct with x/y fields.
%
% Output
%   path : Struct containing x,y, arc length s, heading psi, curvature
%          kappa, curvature derivative kappaPrime, and second derivative
%          kappaDD.
%
% Processing
%   1. Validate shape, size, and finite values.
%   2. Remove consecutive points separated by less than 1e-8 m.
%   3. Translate first point to the origin and rotate the path so its initial
%      heading becomes zero.
%   4. Compute arc length s.
%   5. Estimate heading from atan2(dy/ds,dx/ds).
%   6. Estimate curvature kappa=d(psi)/ds and its derivatives numerically.

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

    % Distance between consecutive path samples: sqrt(dx^2 + dy^2).
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
    % Initial path heading = atan2(dy,dx).
    psi0 = atan2(y(2)-y(1),x(2)-x(1));
    rotation = [cos(psi0),sin(psi0);-sin(psi0),cos(psi0)];
    localPoints = rotation*[x-x(1),y-y(1)]';
    x = localPoints(1,:)';
    y = localPoints(2,:)';

    % Arc length s_i = sum of all segment lengths up to point i.
    s = [0;cumsum(hypot(diff(x),diff(y)))];
    dxds = gradient(x,s);
    dyds = gradient(y,s);
    % Path heading psi(s) is tangent direction; unwrap removes 2*pi jumps.
    psi = unwrap(atan2(dyds,dxds));
    % Curvature kappa = d(psi)/ds.
    kappa = gradient(psi,s);
    % First curvature derivative = dkappa/ds.
    kappaPrime = gradient(kappa,s);
    % Second curvature derivative = d^2(kappa)/ds^2; used in reward.
    kappaDD = gradient(kappaPrime,s);

    path = struct('x',x,'y',y,'s',s,'psi',psi, ...
        'kappa',kappa,'kappaPrime',kappaPrime,'kappaDD',kappaDD);
end


% MAKE_METRICS Summarize tracking, articulation, steering, and slip histories.
%
% Input
%   log : Episode log struct produced by ttv_rl_episode.
%
% Output
%   metrics : Scalar RMS/peak metrics useful for analysis and RL evaluation.
%
% RMS formula
%   RMS(x) = sqrt(mean(x.^2)).

function metrics = make_metrics(log)
    metrics = struct();
    % RMS lateral error = sqrt(mean(e_lat^2)).
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


% SOLVER_STATUS Convert CasADi/Ipopt solver status into readable text.
%
% Input
%   stats : Struct returned by cache.solver.stats().
%
% Output
%   status : Character string from stats.return_status, or a fallback string.

function status = solver_status(stats)
    if isfield(stats,'return_status')
        status = char(stats.return_status);
    else
        status = 'unknown failure';
    end
end


% WRAP_ANGLE Wrap an angle to the interval [-pi, pi].
%
% Formula
%   atan2(sin(theta),cos(theta)) returns the equivalent principal angle.

function angle = wrap_angle(angle)
    % atan2(sin(theta),cos(theta)) gives theta wrapped to [-pi,pi].
    angle = atan2(sin(angle),cos(angle));
end


% SET_DEFAULT Assign a struct field only when it is missing or empty.
%
% Inputs
%   value       : Struct being modified.
%   fieldName   : Name of the field to inspect.
%   defaultValue: Value to assign when the field is absent/empty.
%
% Output
%   value       : Same struct, with the requested field guaranteed non-empty.

function value = set_default(value,fieldName,defaultValue)
    if ~isfield(value,fieldName) || isempty(value.(fieldName))
        value.(fieldName) = defaultValue;
    end
end


% MAKE_CACHE_KEY Build a string identifying the model/MPC parameters.
%
% Input
%   cfg : Completed configuration struct.
%
% Output
%   key : Numeric-parameter string used to decide whether a persistent
%         CasADi environment can safely be reused.
%
% Why
%   Building the CasADi NLP is expensive. If all parameters that affect the
%   model/solver are unchanged, the existing persistent cache can be reused.

function key = make_cache_key(cfg)
    values = [cfg.T,cfg.N,cfg.V,cfg.plantSubsteps,cfg.deltaMax, ...
        cfg.deltaRateMax,cfg.steeringTimeConstant,cfg.phiMax,cfg.qMax, ...
        cfg.m1,cfg.m2,cfg.a1,cfg.a2,cfg.b1,cfg.b2,cfg.c,cfg.Iz1,cfg.Iz2, ...
        cfg.C1,cfg.C2,cfg.C3,cfg.mu,cfg.Fz1,cfg.Fz2,cfg.Fz3, ...
        cfg.mpc.Qy,cfg.mpc.Qpsi,cfg.mpc.Qphi,cfg.mpc.Qq, ...
        cfg.mpc.Rdelta,cfg.mpc.maxIterations,cfg.mpc.acceptableTolerance];
    key = sprintf('%.16g,',values);
end

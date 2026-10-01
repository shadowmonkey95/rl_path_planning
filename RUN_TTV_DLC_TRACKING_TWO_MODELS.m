%% TRACTOR-TRAILER DOUBLE-LANE-CHANGE TRACKING WITH TWO MODELS
% Model 1: linear 3-DOF tractor-trailer model inside the MPC.
% Model 2: nonlinear 3-DOF tractor-trailer plant used as the simulated vehicle.
%
% The double-lane-change reference is generated inside this script.
% There is no external path input and no RL reward. The purpose is to
% inspect tracking, articulation and the difference between the prediction
% model and the nonlinear plant before connecting the framework to RL.

clear;
close all;
clc;

%% 1. CASADI PATH
% Change this line if CasADi is stored elsewhere.
casadiPath = 'casadi-3.7.2-windows64-matlab2018b';
if isfolder(casadiPath)
    addpath(casadiPath);
end
import casadi.*

%% 2. MPC AND SIMULATION SETTINGS
T = 0.1;                    % MPC sampling time, s
N = 20;                     % prediction horizon
simTime = 20;               % total simulation time, s
plantSubsteps = 5;          % RK4 substeps within one MPC sample

deltaMax = 0.5;             % steering-angle limit, rad
deltaRateMax = 0.6;         % physical steering-rate limit, rad/s
Tsteer = 0.15;              % steering actuator time constant, s

% The original code mainly penalized lateral displacement and steering.
% Qpsi, Qphi and Qq are initially zero so that the structure remains close
% to that code. They can be activated later for controlled comparisons.
Qy = 2.0;
Qpsi = 0.0;
Qphi = 0.0;
Qq = 0.0;
Rdelta = 0.1;

%% 3. TRACTOR-TRAILER PARAMETERS
m1 = 5760;                  % tractor mass, kg
m2 = 6640;                  % trailer mass, kg
a1 = 1.10;                  % tractor CG to front axle, m
b1 = 2.39;                  % tractor CG to rear axle, m
c = 1.64;                   % tractor CG to fifth wheel, m
a2 = 5.21;                  % trailer CG to fifth wheel, m
b2 = 3.28;                  % trailer CG to trailer axle, m
Iz1 = 34823;                % tractor yaw inertia, kg m^2
Iz2 = 179992;               % trailer yaw inertia, kg m^2

% Positive stiffness magnitudes. The linear model below converts them to
% the signed convention used by the source 3-DOF model.
C1 = 223281;                % tractor front axle cornering stiffness, N/rad
C2 = 223281;                % tractor rear axle cornering stiffness, N/rad
C3 = 223281;                % trailer axle cornering stiffness, N/rad

V = 20;                     % constant tractor longitudinal speed, m/s
mu = 0.90;                  % tire-road friction coefficient
g0 = 9.81;

% Static vertical loads used by the nonlinear saturated-tire model.
hitchLoad = m2*g0*b2/(a2+b2);
Fz3 = m2*g0*a2/(a2+b2);
Fz1 = (b1*m1*g0+(b1-c)*hitchLoad)/(a1+b1);
Fz2 = m1*g0+hitchLoad-Fz1;

%% 4. INTERNAL DOUBLE-LANE-CHANGE REFERENCE
% This is the same tanh structure used in the previous script. It moves
% from y = 0 to approximately y = 3.5 m and then returns to y = 0.
Yref = @(time) 1.75*(tanh(1.5*time-6)-tanh(1.5*time-16.5));

% Heading reference derived from dy/dx = (dy/dt)/V. Qpsi is zero by
% default, but this signal is retained for observation and later tests.
dYref = @(time) 2.625*(1./cosh(1.5*time-6).^2 ...
                      -1./cosh(1.5*time-16.5).^2);
Psiref = @(time) atan2(dYref(time),V);

%% 5. MODEL 1: LINEAR PREDICTION MODEL FOR MPC
% Base dynamic state:
% xv = [r1; beta1; q; phi], where q = phi_dot = r2-r1.
%
% Descriptor form:
% Mv*xv_dot + Kv*xv = Gv*delta_actual.
C1s = -abs(C1);
C2s = -abs(C2);
C3s = -abs(C3);

Mv = [Iz1, m1*c*V, 0, 0;
     -m2*(a2+c), (m1+m2)*V, -m2*a2, C3s*(a2+b2)/V;
      Iz2, m1*a2*V, Iz2, -C3s*b2*(a2+b2)/V;
      0, 0, 0, 1];

Kv = [m1*c*V-(C1s*a1*(a1+c)+C2s*b1*(b1-c))/V, ...
      -C1s*(a1+c)+C2s*(b1-c), 0, 0;
      (m1+m2)*V+(-C1s*a1+C2s*b1+C3s*(a2+b2+c))/V, ...
      -(C1s+C2s+C3s), 0, C3s;
      m1*a2*V+(-C1s*a1*a2+C2s*b1*a2-C3s*b2*(a2+b2+c))/V, ...
      -(C1s*a2+C2s*a2-C3s*b2), 0, -C3s*b2;
      0, 0, -1, 0];

Gv = [-C1s*(a1+c);-C1s;-C1s*a2;0];

Av = -(Mv\Kv);
Bv = Mv\Gv;

% MPC state:
% xm = [Y; Ydot; psi1; r1; phi; q; delta_actual].
nx = 7;
nu = 1;
Ac = zeros(nx,nx);
Ac(1,2) = 1;
Ac(2,:) = [0, Av(2,2), -V*Av(2,2), V*(Av(2,1)+1), ...
           V*Av(2,4), V*Av(2,3), V*Bv(2)];
Ac(3,4) = 1;
Ac(4,:) = [0, Av(1,2)/V, -Av(1,2), Av(1,1), ...
           Av(1,4), Av(1,3), Bv(1)];
Ac(5,6) = 1;
Ac(6,:) = [0, Av(3,2)/V, -Av(3,2), Av(3,1), ...
           Av(3,4), Av(3,3), Bv(3)];
Ac(7,7) = -1/Tsteer;

Bc = zeros(nx,1);
Bc(7) = 1/Tsteer;

% Exact zero-order-hold discretization of the linear prediction model.
expAug = expm([Ac,Bc;zeros(1,nx+1)]*T);
Ad = expAug(1:nx,1:nx);
Bd = expAug(1:nx,nx+1);

%% 6. CASADI MULTIPLE-SHOOTING MPC
X = SX.sym('X',nx,N+1);
U = SX.sym('U',nu,N);

% P contains the current MPC state followed by N pairs [Yref; Psiref].
P = SX.sym('P',nx+2*N,1);

objective = 0;
constraints = X(:,1)-P(1:nx);

for k = 1:N
    yReference = P(nx+2*k-1);
    psiReference = P(nx+2*k);

    lateralError = X(1,k+1)-yReference;
    headingError = X(3,k+1)-psiReference;

    objective = objective ...
        +Qy*lateralError^2 ...
        +Qpsi*headingError^2 ...
        +Qphi*X(5,k+1)^2 ...
        +Qq*X(6,k+1)^2 ...
        +Rdelta*U(k)^2;

    nextState = Ad*X(:,k)+Bd*U(k);
    constraints = [constraints;X(:,k+1)-nextState]; %#ok<AGROW>
end

decisionVariables = [X(:);U(:)];
nlpProblem = struct('f',objective,'x',decisionVariables, ...
                    'g',constraints,'p',P);

solverOptions = struct;
solverOptions.ipopt.max_iter = 500;
solverOptions.ipopt.print_level = 0;
solverOptions.ipopt.acceptable_tol = 1e-8;
solverOptions.ipopt.acceptable_obj_change_tol = 1e-6;
solverOptions.print_time = false;

solver = nlpsol('solver','ipopt',nlpProblem,solverOptions);

% Bounds for all predicted states and control inputs.
nDecision = nx*(N+1)+nu*N;
args.lbx = -inf(nDecision,1);
args.ubx = inf(nDecision,1);

for k = 0:N
    stateOffset = k*nx;
    args.lbx(stateOffset+7) = -deltaMax;
    args.ubx(stateOffset+7) = deltaMax;
end

controlStart = nx*(N+1)+1;
args.lbx(controlStart:end) = -deltaMax;
args.ubx(controlStart:end) = deltaMax;

nConstraints = nx*(N+1);
args.lbg = zeros(nConstraints,1);
args.ubg = zeros(nConstraints,1);

%% 7. MODEL 2: NONLINEAR PLANT
% Nonlinear plant state:
% xp = [X1; Y1; psi1; v1; r1; phi; q; delta_actual].
xpSymbol = SX.sym('xp',8,1);
deltaCommandSymbol = SX.sym('delta_command');

psi1Symbol = xpSymbol(3);
v1Symbol = xpSymbol(4);
r1Symbol = xpSymbol(5);
phiSymbol = xpSymbol(6);
qSymbol = xpSymbol(7);
deltaSymbol = xpSymbol(8);
r2Symbol = r1Symbol+qSymbol;

% Exact fifth-wheel velocity compatibility.
hitchLateralVelocity = v1Symbol-c*r1Symbol;
V2Symbol = V*cos(phiSymbol)+hitchLateralVelocity*sin(phiSymbol);
v2Symbol = -V*sin(phiSymbol)+hitchLateralVelocity*cos(phiSymbol) ...
           -a2*r2Symbol;

% Nonlinear tire slip angles.
alpha1Symbol = deltaSymbol-atan2(v1Symbol+a1*r1Symbol,V);
alpha2Symbol = -atan2(v1Symbol-b1*r1Symbol,V);
alpha3Symbol = -atan2(v2Symbol-b2*r2Symbol,V2Symbol);

% Smooth saturated tire forces. For small slip, Fy approximately equals
% C*alpha; for large slip, |Fy| approaches mu*Fz.
Fy1Symbol = mu*Fz1*tanh(C1*alpha1Symbol/(mu*Fz1));
Fy2Symbol = mu*Fz2*tanh(C2*alpha2Symbol/(mu*Fz2));
Fy3Symbol = mu*Fz3*tanh(C3*alpha3Symbol/(mu*Fz3));

% Unknown vector:
% zeta = [v1_dot; r1_dot; q_dot; Hx; Hy].
Mplant = [m1, 0, 0, 0, 1;
          0, Iz1, 0, 0, -c;
          m2*sin(phiSymbol), -m2*c*sin(phiSymbol), 0, ...
              -cos(phiSymbol), -sin(phiSymbol);
          m2*cos(phiSymbol), -m2*(c*cos(phiSymbol)+a2), -m2*a2, ...
              sin(phiSymbol), -cos(phiSymbol);
          0, Iz2, Iz2, a2*sin(phiSymbol), -a2*cos(phiSymbol)];

bplant = [Fy1Symbol*cos(deltaSymbol)+Fy2Symbol-m1*V*r1Symbol;
          a1*Fy1Symbol*cos(deltaSymbol)-b1*Fy2Symbol;
          m2*(v2Symbol*r1Symbol-a2*r2Symbol*qSymbol);
          Fy3Symbol-m2*V2Symbol*r1Symbol;
          -b2*Fy3Symbol];

zetaSymbol = solve(Mplant,bplant);
v1DotSymbol = zetaSymbol(1);
r1DotSymbol = zetaSymbol(2);
qDotSymbol = zetaSymbol(3);

rawSteeringRate = (deltaCommandSymbol-deltaSymbol)/Tsteer;
deltaDotSymbol = deltaRateMax*tanh(rawSteeringRate/deltaRateMax);

plantRhs = [V*cos(psi1Symbol)-v1Symbol*sin(psi1Symbol);
            V*sin(psi1Symbol)+v1Symbol*cos(psi1Symbol);
            r1Symbol;
            v1DotSymbol;
            r1DotSymbol;
            qSymbol;
            qDotSymbol;
            deltaDotSymbol];

plantFunction = Function('plantFunction', ...
    {xpSymbol,deltaCommandSymbol},{plantRhs});
plantOutputFunction = Function('plantOutputFunction', ...
    {xpSymbol,deltaCommandSymbol}, ...
    {[alpha1Symbol;alpha2Symbol;alpha3Symbol], ...
     [Fy1Symbol;Fy2Symbol;Fy3Symbol]});

%% 8. INITIAL CONDITIONS AND HISTORY ARRAYS
nSteps = round(simTime/T);
time = 0:T:simTime;

% Plant initial condition:
% [X1; Y1; psi1; v1; r1; phi; q; delta_actual].
xp = zeros(8,1);

% Initial MPC state obtained from the nonlinear plant.
xm = plant_to_mpc_state(xp,V);

plantHistory = zeros(8,nSteps+1);
mpcHistory = zeros(7,nSteps+1);
controlHistory = zeros(1,nSteps);
referenceYHistory = zeros(1,nSteps+1);
referencePsiHistory = zeros(1,nSteps+1);
slipHistory = zeros(3,nSteps+1);
forceHistory = zeros(3,nSteps+1);
predictionErrorHistory = zeros(7,nSteps);
solverTimeHistory = zeros(1,nSteps);

plantHistory(:,1) = xp;
mpcHistory(:,1) = xm;
referenceYHistory(1) = Yref(0);
referencePsiHistory(1) = Psiref(0);
[initialSlip,initialForce] = plant_outputs(plantOutputFunction,xp,0);
slipHistory(:,1) = initialSlip;
forceHistory(:,1) = initialForce;

% Initial guesses for multiple shooting.
Xguess = repmat(xm,1,N+1);
Uguess = zeros(1,N);

%% 9. CLOSED-LOOP MPC SIMULATION
completedSteps = nSteps;

for mpcIter = 1:nSteps
    currentTime = (mpcIter-1)*T;
    xm = plant_to_mpc_state(xp,V);

    args.p = zeros(nx+2*N,1);
    args.p(1:nx) = xm;

    for k = 1:N
        predictedTime = currentTime+k*T;
        args.p(nx+2*k-1) = Yref(predictedTime);
        args.p(nx+2*k) = Psiref(predictedTime);
    end

    args.x0 = [Xguess(:);Uguess(:)];

    solveTimer = tic;
    solution = solver('x0',args.x0,'lbx',args.lbx,'ubx',args.ubx, ...
                      'lbg',args.lbg,'ubg',args.ubg,'p',args.p);
    solverTimeHistory(mpcIter) = toc(solveTimer);

    solverStats = solver.stats();
    if ~solverStats.success
        warning('MPC solver failed at step %d: %s', ...
            mpcIter,solverStats.return_status);
        completedSteps = mpcIter-1;
        break;
    end

    optimalDecision = full(solution.x);
    Xoptimal = reshape(optimalDecision(1:nx*(N+1)),nx,N+1);
    Uoptimal = reshape(optimalDecision(nx*(N+1)+1:end),1,N);
    deltaCommand = Uoptimal(1);

    % Linear-model one-step prediction retained for model-mismatch plots.
    oneStepPrediction = Ad*xm+Bd*deltaCommand;

    % Apply the MPC command to the nonlinear plant using RK4.
    integrationStep = T/plantSubsteps;
    for substep = 1:plantSubsteps
        xp = rk4_plant_step(plantFunction,xp,deltaCommand,integrationStep);
    end
    xp(8) = min(max(xp(8),-deltaMax),deltaMax);

    xmNext = plant_to_mpc_state(xp,V);
    [slipAngles,lateralForces] = plant_outputs( ...
        plantOutputFunction,xp,deltaCommand);

    plantHistory(:,mpcIter+1) = xp;
    mpcHistory(:,mpcIter+1) = xmNext;
    controlHistory(mpcIter) = deltaCommand;
    referenceYHistory(mpcIter+1) = Yref(currentTime+T);
    referencePsiHistory(mpcIter+1) = Psiref(currentTime+T);
    slipHistory(:,mpcIter+1) = slipAngles;
    forceHistory(:,mpcIter+1) = lateralForces;
    predictionErrorHistory(:,mpcIter) = xmNext-oneStepPrediction;

    % Shift the previous optimum to warm-start the next MPC problem.
    Xguess = [Xoptimal(:,2:end),Xoptimal(:,end)];
    Xguess(:,1) = xmNext;
    Uguess = [Uoptimal(2:end),Uoptimal(end)];
end

%% 10. TRIM HISTORIES IF THE SOLVER STOPPED EARLY
stateIndices = 1:completedSteps+1;
controlIndices = 1:completedSteps;
timeState = time(stateIndices);
timeControl = time(controlIndices);

plantHistory = plantHistory(:,stateIndices);
mpcHistory = mpcHistory(:,stateIndices);
referenceYHistory = referenceYHistory(stateIndices);
referencePsiHistory = referencePsiHistory(stateIndices);
slipHistory = slipHistory(:,stateIndices);
forceHistory = forceHistory(:,stateIndices);
controlHistory = controlHistory(controlIndices);
predictionErrorHistory = predictionErrorHistory(:,controlIndices);
solverTimeHistory = solverTimeHistory(controlIndices);

% Derived variables for observation.
r1History = plantHistory(5,:);
qHistory = plantHistory(7,:);
r2History = r1History+qHistory;
beta1History = atan2(plantHistory(4,:),V);
trackingError = plantHistory(2,:)-referenceYHistory;

if completedSteps > 0
    peakSteeringCommand = max(abs(controlHistory));
    meanSolveTime = mean(solverTimeHistory);
    maximumSolveTime = max(solverTimeHistory);
else
    peakSteeringCommand = NaN;
    meanSolveTime = NaN;
    maximumSolveTime = NaN;
end

%% 11. NUMERICAL SUMMARY
fprintf('\n===== TTV DOUBLE-LANE-CHANGE TRACKING =====\n');
fprintf('Completed MPC steps      : %d / %d\n',completedSteps,nSteps);
fprintf('RMS lateral error        : %.6f m\n',sqrt(mean(trackingError.^2)));
fprintf('Peak lateral error       : %.6f m\n',max(abs(trackingError)));
fprintf('Peak articulation angle  : %.6f rad\n',max(abs(plantHistory(6,:))));
fprintf('Peak articulation rate   : %.6f rad/s\n',max(abs(qHistory)));
fprintf('Peak steering command    : %.6f rad\n',peakSteeringCommand);
fprintf('Peak actual steering     : %.6f rad\n',max(abs(plantHistory(8,:))));
fprintf('Max front slip           : %.6f rad\n',max(abs(slipHistory(1,:))));
fprintf('Max rear slip            : %.6f rad\n',max(abs(slipHistory(2,:))));
fprintf('Max trailer slip         : %.6f rad\n',max(abs(slipHistory(3,:))));
fprintf('Mean MPC solve time      : %.6f s\n',meanSolveTime);
fprintf('Maximum MPC solve time   : %.6f s\n',maximumSolveTime);

%% 12. PLOTS
figure('Name','Path tracking','Color','w');
subplot(2,1,1);
plot(timeState,referenceYHistory,'b--','LineWidth',1.6); hold on;
plot(timeState,plantHistory(2,:),'r','LineWidth',1.4);
grid on;
xlabel('Time (s)'); ylabel('Y (m)');
legend('Y reference','Nonlinear plant','Location','best');
title('Double-lane-change lateral tracking');

subplot(2,1,2);
plot(timeState,trackingError,'k','LineWidth',1.4);
grid on;
xlabel('Time (s)'); ylabel('Y-Y_{ref} (m)');
title('Lateral tracking error');

figure('Name','Global trajectory','Color','w');
plot(V*timeState,referenceYHistory,'b--','LineWidth',1.6); hold on;
plot(plantHistory(1,:),plantHistory(2,:),'r','LineWidth',1.4);
grid on; axis equal;
xlabel('X (m)'); ylabel('Y (m)');
legend('Reference','Tractor CG','Location','best');
title('Reference and nonlinear-plant trajectory');

figure('Name','Vehicle dynamic states','Color','w');
subplot(3,2,1);
plot(timeState,beta1History,'LineWidth',1.3); grid on;
ylabel('\beta_1 (rad)'); title('Tractor sideslip');

subplot(3,2,2);
plot(timeState,r1History,'LineWidth',1.3); hold on;
plot(timeState,r2History,'--','LineWidth',1.3); grid on;
ylabel('Yaw rate (rad/s)'); legend('r_1','r_2');

subplot(3,2,3);
plot(timeState,plantHistory(6,:),'LineWidth',1.3); grid on;
ylabel('\phi (rad)'); title('Articulation angle');

subplot(3,2,4);
plot(timeState,qHistory,'LineWidth',1.3); grid on;
ylabel('q = d\phi/dt (rad/s)'); title('Articulation rate');

subplot(3,2,5);
plot(timeState,plantHistory(3,:),'LineWidth',1.3); hold on;
plot(timeState,referencePsiHistory,'--','LineWidth',1.3); grid on;
xlabel('Time (s)'); ylabel('\psi_1 (rad)');
legend('\psi_1','\psi_{ref}');

subplot(3,2,6);
plot(timeState,plantHistory(4,:),'LineWidth',1.3); grid on;
xlabel('Time (s)'); ylabel('v_1 (m/s)');
title('Tractor lateral velocity');

figure('Name','Steering','Color','w');
hDeltaCommand = stairs(timeControl,controlHistory,'b','LineWidth',1.3); hold on;
hDeltaActual = plot(timeState,plantHistory(8,:),'r','LineWidth',1.3);
yline(deltaMax,'k--','HandleVisibility','off');
yline(-deltaMax,'k--','HandleVisibility','off');
grid on;
xlabel('Time (s)'); ylabel('Steering angle (rad)');
legend([hDeltaCommand,hDeltaActual], ...
       {'\delta_{cmd}','\delta_{actual}'},'Location','best');
title('MPC command and nonlinear steering actuator');

figure('Name','Tire slip angles','Color','w');
plot(timeState,slipHistory(1,:),'LineWidth',1.3); hold on;
plot(timeState,slipHistory(2,:),'LineWidth',1.3);
plot(timeState,slipHistory(3,:),'LineWidth',1.3);
yline(0.2,'k--','HandleVisibility','off');
yline(-0.2,'k--','HandleVisibility','off');
grid on;
xlabel('Time (s)'); ylabel('Slip angle (rad)');
legend('\alpha_1','\alpha_2','\alpha_3','Location','best');
title('Nonlinear-plant tire slip angles');

figure('Name','Linear prediction versus nonlinear plant','Color','w');
stateLabels = {'Y','Ydot','psi_1','r_1','phi','q','delta'};
for stateIndex = 1:7
    subplot(4,2,stateIndex);
    plot(timeControl,predictionErrorHistory(stateIndex,:),'LineWidth',1.2);
    grid on;
    xlabel('Time (s)');
    ylabel(['e_{pred}(',stateLabels{stateIndex},')']);
end
sgtitle('One-step error: nonlinear plant minus linear prediction');

figure('Name','MPC computation time','Color','w');
plot(timeControl,1000*solverTimeHistory,'LineWidth',1.3); grid on;
xlabel('Time (s)'); ylabel('Solve time (ms)');
title('MPC solver time');

%% LOCAL FUNCTIONS
function xm = plant_to_mpc_state(xp,V)
% Map nonlinear-plant states to the linear MPC state.
    psi1 = xp(3);
    v1 = xp(4);
    Ydot = V*sin(psi1)+v1*cos(psi1);
    xm = [xp(2);Ydot;psi1;xp(5);xp(6);xp(7);xp(8)];
end


function nextState = rk4_plant_step(plantFunction,state,input,h)
% One fixed-step RK4 integration step for the nonlinear plant.
    k1 = full(plantFunction(state,input));
    k2 = full(plantFunction(state+0.5*h*k1,input));
    k3 = full(plantFunction(state+0.5*h*k2,input));
    k4 = full(plantFunction(state+h*k3,input));
    nextState = state+h*(k1+2*k2+2*k3+k4)/6;
end


function [slipAngles,lateralForces] = plant_outputs( ...
        plantOutputFunction,state,input)
% Evaluate tire-slip angles and lateral tire forces for plotting.
    [slipCasadi,forceCasadi] = plantOutputFunction(state,input);
    slipAngles = full(slipCasadi);
    lateralForces = full(forceCasadi);
end

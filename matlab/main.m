%% Genetic Algorithm Optimization
% created by: Gino Bertollucci Colherinhas 08/2016
% updated on: 23/10/2019
% Genetic optimization based on floating points.

%% Used evolutionary strategies:
% Selection: Roullete-Wheel; 
% Crossover: BLX-alpha.
% Deterministic elistism and decimation.

clc; clear; clf; close all; addpath fix examples

%% Selecting the function to be optimized

[filename] = uigetfile('examples\*.m','Select the function to be optimized');
if isequal(filename,0)
    error('Select the function to be optimized')
else
    disp(['The selected function to be optimized was: ', fullfile(filename)])
    fit_function = str2func(filename(1:end-2));
    [bounds,~]=fit_function(ones(1e6,1));
end

%% Optimization Parameters
N_ger = 500;    % Number of generations
N_ind = 500;    % Number of chromosomes/generation
N_diz = 20;     % Decimation step
p_diz = 0.2;    % Decimation probability
p_elit = 0.02;  % Elitism probability
p_m = 0.02;     % Mutation probability
p_c = 0.6;      % Crossover probability
    
%% Lower and Upper limits (L_1, L_2, ..., L_N/ U_1, U_2, ..., U_N):

CromLim = bounds;
   
%% Pre-allocating memory
melhor = zeros(N_ger,1); medio = zeros(N_ger,1);

%% Initial Population
pop = newpop(N_ind,CromLim);
t=tic;
%% Optimization
for i=1:N_ger
    %Fitness of each chromosome
    [OUTPUT,fit] = fitness(pop,fit_function);

    % Evolutionary Strategies
    pop = evolution_strategies(pop,fit,p_elit,p_m,p_c,CromLim);

    % Best individual and its location
    [best,best_pos] = max(fit);
    melhor(i) = best;
    medio(i) = mean(fit);
    fittest = [pop(best_pos,:) OUTPUT(best_pos)];

    % Decimation
    if mod(i,N_diz)==0
        pop(round(N_ind*(1-p_diz))+1:N_ind,:) = newpop(N_ind-round(N_ind*(1-p_diz)),CromLim);
    end
end
toc(t)

disp(' ')
fprintf('Para as entradas: f(')
fprintf('[%4.4f]',fittest(1:end-1))
fprintf(')')
fprintf('\nO valor ótimo encontrado foi: f = %4.4f',fittest(end))
disp(' ')

%% Results Analysis
% Optimization curve:
figure(1)
plot(1:i,melhor(1:i)./max(melhor(1:i)),'-ks','LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor','w','MarkerSize',5.5); hold on    
plot(1:i,medio(1:i)./max(melhor(1:i)),'-k*','LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor','w','MarkerSize',5.5); hold on
ylabel('Fitness')
xlabel('Generations')  

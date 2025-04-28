%% Fitness Function
% this function reads the fitness function
% pop: population to be coded
% INPUTS: entries
% OUTPUTS: descedent of the generation

function [OUTPUT,fit] = fitness(pop,fit_function)
    
    % Pre-allocating memory
    Nind=length(pop(:,1));
    OUTPUT=zeros(Nind,1);
    fit=zeros(Nind,1);
    
    for i=1:Nind
        INPUT = pop(i,:);
        [~,OUTPUT(i)] = fit_function(INPUT);
        fit(i) = 1/(OUTPUT(i)+10);
    end
end

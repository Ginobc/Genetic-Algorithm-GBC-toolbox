%% Seleção, Elitismo, Cruzamento e Mutação
function new_pop = evolution_strategies(pop,fit,p_elit,p_m,p_c,CromLim)

    N_ind = length(fit);
    q = zeros(N_ind,1);
    N_elit = round(N_ind*p_elit);
    new_pop = zeros(size(pop));
    
    %% Seleção (Roullete-Wheel):
    P = fit/sum(fit);
    r = rand(N_ind,1);
    j = 1;
    for i = 1:N_ind
        q(i) = sum(P(1:i));     % Probabilidades Acumuladas
    end
    seleciona=find(r<q)';
    j=length(seleciona)+1;
    
    %% Elitismo:
    [~, pos] = sort(fit,'descend');
    for i = 1:N_elit
       new_pop(i,:) = pop(pos(i),:);
    end
    
    %% Cruzamento e Mutação
    i = i+1;
    while i <= N_ind
        if p_m <= rand
            pos_pai = seleciona(round(rand.*((j-1)-1)+1));
            pos_mae = seleciona(round(rand.*((j-1)-1)+1));
            pai = pop(pos_pai,:);
            mae = pop(pos_mae,:);
            if p_c >= rand
                % BLX-alpha
%                 a1 = 0;
%                 b1 = 0.05;
%                 alpha = a1 + (b1-a1).*rand;
                alpha = 0.25;
                a2 = -alpha; b2 = 1+alpha;
                beta = a2 + (b2-a2).*rand;
                new_pop(i,:) = pai+beta*(mae-pai);
                for k=1:size(CromLim,1)
                    if new_pop(i,k)<CromLim(k,1)
                        new_pop(i,k)=CromLim(k,1);
                    end
                    if new_pop(i,k)>CromLim(k,2)
                        new_pop(i,k)=CromLim(k,2);
                    end
                end
            else
                if fit(pos_pai)>fit(pos_mae)
                    new_pop(i,:) = pai;
                else
                    new_pop(i,:) = mae;
                end
            end
        else
            new_pop(i,:) = newpop(1,CromLim);
        end
        i=i+1;
    end
